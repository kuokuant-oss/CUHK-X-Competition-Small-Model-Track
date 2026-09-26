"""B0/B1/B2 matched transductive replay, explicit jobs only; no submission chain."""

# Role: fit(), the training loop of member B (per-sample loss weights on soft targets, mixup,
#   mixed precision, per-epoch checkpoints), and main(), the per-fold validation of B's
#   pseudo-label recipe.
# Used by: 103_b_final_student.py calls fit() and assert_identity(), 124_c_full_student.py calls
#   assert_identity(); main() is run by hand; training (main() is validation only).
# main(), per requested fold k: the fold-k models of the R(2+1)D-34 and ir-CSN-152 60-epoch runs
#   (neither saw fold k) label a test-like subset of fold k (one contiguous run per take, run
#   lengths drawn from the test take sizes, fixed in work/b-matched-preflight.json). The
#   R(2+1)D-34 fold model is then fine-tuned for 10 epochs on the other folds' labelled clips
#   plus extra samples, in three variants: B0 adds as many labelled clips as the subset holds,
#   drawn at random (control), B1 the subset with the R(2+1)D-34 posteriors, B2 the subset with
#   the mean of both teachers (the recipe of the delivered B). Each is scored on the subset
#   before and after training.
# Produces, under models_root/<--out>/: fold<k>/ with identity.json, peer-probs.npz, before.npz,
#   teacher-scores.json and paired.json; fold<k>/<variant>/ with identity.json, checkpoint.pt,
#   epoch-NN.json, student.pt, after.npz and result.json; paired-evaluation.json.
# Inputs, not included in the repository (TRAINING.md lists them): work/b-matched-preflight.json;
#   under models_root, result.json, oof.npz and folds/fold<k>.pt of both runs, and
#   abc-a-20260910/fold<k>/before.npz from 47_matched_continuation.py; both runs' training
#   caches; folds.parquet, takes_train.parquet and takes_test.parquet. models_root is not defined
#   in the shipped configs/paths.yaml.
# Each fold's identity.json records the SHA256 of the source files, and every later invocation
#   for that fold must reproduce it; without the inputs above, main() documents the procedure
#   rather than running out of the box.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Make cuhkx (src/), scripts/ and the training modules importable; module names that start with
# a digit are loaded with import_module.
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"),str(ROOT/'training')]
# Deterministic cuBLAS kernels need this workspace setting before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
base = import_module("47_matched_continuation")


# SHA256 of a value's JSON form with sorted keys; identifies an identity record in checkpoints.
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# Writes the identity record on the first start and refuses to continue if a later start
# differs. The JSON round trip makes paths and tuples compare in the form in which they are stored.
def assert_identity(path, value):
    value = json.loads(json.dumps(value, default=str))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"identity changed: {path}")
    else:
        base.write_json(path, value)


# Trains make_model() on a ReplayConcat of ReplayDatasets whose items are (inputs, soft target,
# loss weight). Writes dest/checkpoint.pt and dest/epoch-NN.json after every epoch and
# dest/student.pt at the end, and returns (model in eval mode, per-epoch records). identity
# supplies the variant name, the number of pseudo-labelled samples and their planned weight share.
def fit(make_model, dataset, config, dest, identity):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from cuhkx.replay import epoch_batches, epoch_mass_loss, weighted_targets
    from cuhkx.train import (
        _apply_freeze_bn,
        _atomic_save,
        _loader_kwargs,
        _param_groups,
        _restore_rng,
        _rng_state,
        seed_everything,
    )

    # Gradient accumulation and EMA/SWA weight averaging are not implemented in this loop.
    if config.grad_accum != 1 or config.ema_decay or config.swa_start:
        raise ValueError("this registered replay supports no accumulation/EMA/SWA")
    seed_everything(config.seed, deterministic=config.deterministic)
    model = make_model().to(config.device)
    # Share the epoch number with DataLoader workers: each sample's frames and augmentation come
    # from a generator keyed on (seed, epoch, clip).
    dataset.enable_epoch_sync()
    # The shuffling order has its own seeded generator, saved in the checkpoint so that a resumed
    # run keeps the same order. Default collation yields (inputs, targets (batch, 40), weights
    # (batch,)); the last batch may be short.
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **_loader_kwargs(config),
    )
    # Parameter groups from cuhkx.train (the pretrained backbone at lr × backbone_lr_mult), AdamW,
    # and a one-cycle schedule stepped once per batch.
    groups, max_lrs = _param_groups(model, config)
    optimizer = torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=config.epochs * len(loader),
        pct_start=config.pct_start,
    )
    # Mixed precision only when configured and on CUDA.
    use_amp = config.amp and config.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Total loss weight of one epoch; for the delivered B, 2,931 × 1 + 405 × 0.5 = 3,133.5.
    epoch_weight = sum(float(ds.weights.sum(dtype=np.float64)) for ds in dataset.datasets)
    # Pseudo-labelled samples expected per epoch; the control B0 has none.
    expected_pseudo = int(identity["n_pseudo"]) if identity["arm"] != "B0" else 0
    checkpoint = dest / "checkpoint.pt"
    start_epoch = 1
    # Resume after the last completed epoch; the checkpoint must carry the same identity digest.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity_digest"] != digest(identity):
            raise ValueError("replay checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        scaler.load_state_dict(saved["scaler"])
        _restore_rng(saved["rng"])
        generator.set_state(saved["loader_rng"])
        start_epoch = saved["epoch"] + 1
    rows = []
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.monotonic()
            model.train()
            _apply_freeze_bn(model, config.freeze_bn)
            # Per-epoch counters: samples, pseudo-labelled samples, steps skipped by the gradient
            # scaler and batches; loss weight of pseudo-labelled and of labelled samples; loss.
            seen = pseudo_seen = skips = attempts = 0
            pseudo_mass = true_mass = sum_loss = 0.0
            # epoch_batches sets the epoch before the iterator is created, so prefetching
            # workers already use it.
            for inputs, target, weight in epoch_batches(loader, dataset, epoch):
                # Pseudo-labelled samples are the ones whose loss weight is not 1.
                is_pseudo = weight != 1
                pseudo_seen += int(is_pseudo.sum())
                pseudo_mass += float(weight[is_pseudo].double().sum())
                true_mass += float(weight[~is_pseudo].double().sum())
                inputs = tuple(t.to(config.device) for t in inputs)
                target, weight = target.to(config.device), weight.to(config.device)
                optimizer.zero_grad(set_to_none=True)
                # Mixup with one λ ~ Beta(α, α) per batch (mixup_alpha must be positive): inputs
                # are blended with a permuted copy, and so are the weight-scaled targets, so each
                # sample keeps its loss weight.
                lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                perm = torch.randperm(len(weight), device=config.device)
                mixed = tuple(lam * t.float() + (1 - lam) * t[perm].float() for t in inputs)
                mass = weighted_targets(target, weight, perm, lam)
                # Cross-entropy against the mixed target mass, times batches per epoch / epoch
                # weight: over an epoch its mean equals the weighted loss of all samples divided
                # by their total weight, also with a short last batch.
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(*mixed)
                    loss = epoch_mass_loss(
                        logits, mass, epoch_weight=epoch_weight, steps_per_epoch=len(loader)
                    )
                # A non-finite loss stops training; the last completed epoch's checkpoint stays.
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss; checkpoint preserved")
                scaler.scale(loss).backward()
                # The scaler skips the optimizer step when gradients overflow and then lowers its
                # scale, so a lower scale after update() counts one skipped step. The schedule
                # advances either way.
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skips += int(scaler.get_scale() < old_scale)
                schedule.step()
                sum_loss += float(loss.detach())
                seen += len(weight)
                attempts += 1
            # The epoch must have shown every sample, the expected number of pseudo-labelled
            # ones and the planned pseudo-labelled share of the (unmixed) loss weight.
            if seen != len(dataset) or pseudo_seen != expected_pseudo:
                raise ValueError("unexpected replay exposure")
            measured_ratio = pseudo_mass / true_mass
            wanted_ratio = identity["pseudo_to_true_mass_ratio"] if expected_pseudo else 0
            if not math.isclose(measured_ratio, wanted_ratio, abs_tol=1e-7):
                raise ValueError("effective pseudo/true mass ratio differs from registration")
            row = {
                "epoch": epoch,
                "samples": seen,
                "true_presentations": seen - pseudo_seen,
                "pseudo_presentations": pseudo_seen,
                "true_mass": true_mass,
                "pseudo_mass": pseudo_mass,
                "pseudo_to_true_mass_ratio": measured_ratio,
                "optimizer_attempts": attempts,
                "scaler_skips": skips,
                "optimizer_updates": attempts - skips,
                "loss": sum_loss / len(loader),
                "seconds": time.monotonic() - started,
                "heldout_score": None,
            }
            # Atomic checkpoint with optimizer, schedule, scaler and all random states, so that a
            # resumed run keeps the same random streams. heldout_score stays None: no held-out
            # data is scored here.
            _atomic_save(
                {
                    "epoch": epoch,
                    "identity_digest": digest(identity),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": _rng_state(),
                    "loader_rng": generator.get_state(),
                },
                checkpoint,
            )
            base.write_json(dest / f"epoch-{epoch:02d}.json", row)
            rows.append(row)
            print(json.dumps({"arm": identity["arm"], "fold": identity["fold"], **row}), flush=True)
    finally:
        # Stop persistent DataLoader workers even when training fails.
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    # Final weights on the CPU; the later steps read student.pt.
    _atomic_save({k: v.cpu() for k, v in model.state_dict().items()}, dest / "student.pt")
    return model.eval(), rows


# Accuracy differences in percentage points (raw and decoded) for each listed pair whose two
# entries are present; "before" is the fold model without fine-tuning.
def contrasts(arms):
    return {
        f"{a}-{b}": {m: 100 * (arms[a][m] - arms[b][m]) for m in ("raw", "decoded")}
        for a, b in (
            ("B2", "B0"),
            ("B1", "B0"),
            ("B2", "B1"),
            ("B0", "before"),
            ("B1", "before"),
            ("B2", "before"),
        )
        if a in arms and b in arms
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 3])
    ap.add_argument("--arms", nargs="+", choices=["B0", "B1", "B2"], default=["B0", "B1", "B2"])
    ap.add_argument("--out", default="abc-b-20260911")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    base.assert_idle()
    import numpy as np
    import pandas as pd

    from cuhkx.paths import p

    # Teacher 1, which is also the starting model, comes from the R(2+1)D-34 60-epoch run on the
    # det view; teacher 2 from the ir-CSN-152 run.
    student_run, peer_run = "ig65m-t32-det-60ep", "csn152-4f224-clip40"
    # Per-fold settings fixed before training: the test-like subset of fold k (clips per take),
    # its size, the pseudo-label weight, the planned weight ratio and the steps per epoch.
    registration = json.loads((ROOT / "work/b-matched-preflight.json").read_text())
    registered = {f["fold"]: f for f in registration["folds"]}
    models = p("models_root")
    output = models / args.out
    # The 2,931 labelled clips in out-of-fold order; labels and folds must agree with the record.
    ref = np.load(models / student_run / "oof.npz")
    table = pd.read_parquet(p("processed_root") / "folds.parquet").set_index("clip_id")
    table = table.loc[ref["clip_ids"]]
    assert np.array_equal(table.action_id.to_numpy(), ref["labels"])
    assert np.array_equal(table.fold.to_numpy(), ref["folds"])
    # Training takes as ordered clip lists, and the action of every labelled clip.
    take_table = pd.read_parquet(p("processed_root") / "takes_train.parquet")
    take_table = take_table[take_table.clip_id.isin(table.index)]
    all_takes = {
        t: g.sort_values("position").clip_id.tolist() for t, g in take_table.groupby("take_id")
    }
    truth = table.action_id.to_dict()
    # For both runs: the out-of-fold record must match the fold table, the training cache must
    # exist, and each requested fold's model must have held out exactly that fold's subjects.
    configs, provenance = {}, {}
    for run in (student_run, peer_run):
        document = json.loads((models / run / "result.json").read_text(encoding="utf-8"))
        cfg = document["config"]
        cfg["extra"].pop("subjects", None)
        configs[run] = cfg
        oof = np.load(models / run / "oof.npz")
        assert set(oof["clip_ids"]) == set(table.index)
        assert np.array_equal(table.loc[oof["clip_ids"], "action_id"], oof["labels"])
        assert np.array_equal(table.loc[oof["clip_ids"], "fold"], oof["folds"])
        cache = p("cache_root") / cfg["extra"]["cache"]
        for suffix in ("_data.npy", "_meta.npz"):
            if not cache.with_name(cache.stem + suffix).exists():
                raise FileNotFoundError(cache.with_name(cache.stem + suffix))
        provenance[run] = {}
        for fold in args.folds:
            expected = sorted(table[table.fold == fold].user.unique())
            records = [r for r in document["folds"] if r["fold"] == fold]
            if len(records) != 1 or sorted(records[0]["val_subjects"]) != expected:
                raise ValueError(f"unverified held-out teacher subjects: {run}/{fold}")
            source = models / run / "folds" / f"fold{fold}.pt"
            provenance[run][fold] = {
                "source": str(source),
                "source_sha256": base.sha256(source),
                "config": cfg,
                "heldout_subjects": expected,
            }
    # SHA256 of the source files this validation depends on; recorded in every fold's identity.
    code_files = [
        Path(__file__),
        ROOT / "src/cuhkx/replay.py",
        ROOT / "src/cuhkx/train.py",
        ROOT / "src/cuhkx/fused.py",
        ROOT / "src/cuhkx/tta.py",
        ROOT / "training/44_self_training.py",
        ROOT / "training/48_evaluate_matched.py",
    ]
    code_hashes = {str(q.relative_to(ROOT)): base.sha256(q) for q in code_files}
    # The planned subset of fold k consists of distinct fold-k clips, and the planned number of
    # labelled clips equals the number of clips in the other folds.
    for fold in args.folds:
        r = registered[fold]
        assert r["n_real"] == int((table.fold != fold).sum())
        picked = [c for m in r["pseudo_takes"].values() for c in m]
        assert len(set(picked)) == len(picked) == r["n_pseudo"]
        assert (table.loc[picked, "fold"] == fold).all()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "provenance": provenance,
                    "registration": registration,
                    "code_sha256": code_hashes,
                },
                default=str,
            ),
            flush=True,
        )
        return

    import torch

    from cuhkx.fused import FusedFrameDataset, labels_by_row, load_fused, make_fused_transform
    from cuhkx.replay import ReplayConcat, ReplayDataset
    from cuhkx.takes import decode_all, transition_matrix
    from cuhkx.train import TrainConfig
    from cuhkx.tta import predict

    helper = import_module("44_self_training")
    evaluator = import_module("48_evaluate_matched")
    # Step 1: open the student's training cache (memory-mapped), index the labelled clips by
    # cache row, and set up the training and evaluation transforms of the 60-epoch recipe.
    cfg = configs[student_run]
    extra = cfg["extra"]
    data, offsets, ids, _ = load_fused(p("cache_root") / extra["cache"])
    assert isinstance(data, np.memmap)
    # Labelled clips in cache order (names), their cache rows (order), actions and folds;
    # row_labels is indexed by cache row.
    row_of = {c: i for i, c in enumerate(ids)}
    names = np.array([c for c in ids if c in table.index])
    order = np.array([row_of[c] for c in names])
    labels, fold_ids = table.loc[names, "action_id"].to_numpy(), table.loc[names, "fold"].to_numpy()
    row_labels = labels_by_row(len(ids), order, labels)
    train_tf = make_fused_transform(
        extra["crop"],
        erase_prob=extra["erase_prob"],
        flip_prob=extra["flip_prob"],
        scale_range=extra.get("scale_jitter"),
    )
    eval_tf = make_fused_transform(extra["crop"], erase_prob=0, flip_prob=0)
    output.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        # Step 2, per fold k: the test-like subset (picked) and its positions among fold k's
        # clips (picked_indices).
        reg = registered[fold]
        picked = [c for m in reg["pseudo_takes"].values() for c in m]
        mask = fold_ids == fold
        fold_names = names[mask]
        names_to_index = {c: i for i, c in enumerate(fold_names)}
        picked_indices = np.array([names_to_index[c] for c in picked])
        source = Path(provenance[student_run][fold]["source"])
        directory = output / f"fold{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / a).exists() for a in args.arms) and not args.resume:
            raise FileExistsError(f"existing arm under {directory}; --resume required")
        # The fold's identity record: plan, teacher provenance and source hashes.
        fold_identity = {
            "registered": reg,
            "provenance": {r: provenance[r][fold] for r in provenance},
            "code_sha256": code_hashes,
            "transductive": True,
        }
        assert_identity(directory / "identity.json", fold_identity)
        # Teacher 1: four-pass posteriors of the fold-k R(2+1)D-34 model on fold k, written by
        # 47_matched_continuation.py from the same weights.
        original = np.load(models / "abc-a-20260910" / f"fold{fold}" / "before.npz")
        assert str(original["source_sha256"]) == provenance[student_run][fold]["source_sha256"]
        assert np.array_equal(original["clip_ids"], fold_names)
        own_probs = original["probs"]
        # Teacher 2: four-pass posteriors of the ir-CSN-152 fold-k model on fold k, from its own
        # cache, crop and frame count; computed once and kept in peer-probs.npz.
        peer_file = directory / "peer-probs.npz"
        if not peer_file.exists():
            peer_cfg = configs[peer_run]
            peer_data, peer_offsets, peer_ids, _ = load_fused(
                p("cache_root") / peer_cfg["extra"]["cache"]
            )
            assert isinstance(peer_data, np.memmap)
            peer_rows = {c: i for i, c in enumerate(peer_ids)}
            ds = FusedFrameDataset(
                peer_data,
                peer_offsets,
                np.zeros(len(peer_ids), dtype=np.int64),
                np.array([peer_rows[c] for c in fold_names]),
                peer_cfg["n_frames"],
                False,
                make_fused_transform(peer_cfg["extra"]["crop"], erase_prob=0, flip_prob=0),
                peer_cfg["seed"],
            )
            net = (
                helper.load_state(
                    helper.build_model(peer_cfg, peer_data.shape[-1], "cpu"),
                    Path(provenance[peer_run][fold]["source"]),
                )
                .to(cfg["device"])
                .eval()
            )
            started = time.monotonic()
            print(
                f"TEACHER {peer_run} fold{fold} own crop/cache/T: {peer_cfg['extra']['crop']}",
                flush=True,
            )
            probs = predict(net, ds, cfg["device"], peer_cfg["eval_batch_size"], True)
            np.savez(
                peer_file,
                clip_ids=fold_names,
                probs=probs,
                source_sha256=provenance[peer_run][fold]["source_sha256"],
            )
            print(f"teacher seconds={time.monotonic() - started:.1f}", flush=True)
            del net, ds, peer_data
            torch.cuda.empty_cache()
        peer = np.load(peer_file)
        assert np.array_equal(peer["clip_ids"], fold_names)
        assert str(peer["source_sha256"]) == provenance[peer_run][fold]["source_sha256"]
        # The mean of both teachers is the pseudo-label target of variant B2. Teacher 1's
        # posteriors and the mean must be finite distributions over the 40 classes, one row per
        # fold-k clip.
        heterogeneous = (own_probs + peer["probs"]) / 2
        for probs in (own_probs, heterogeneous):
            assert probs.shape == (len(fold_names), 40) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(1), 1, atol=1e-5)
        np.savez(directory / "before.npz", clip_ids=fold_names, probs=own_probs)
        # Transition prior counted on the other three folds' takes only.
        transitions = transition_matrix(
            {t: m for t, m in all_takes.items() if int(table.loc[m[0], "fold"]) != fold}, truth
        )

        # Accuracy on the test-like subset by per-clip argmax (raw) and after joint decoding of
        # its takes at transition weight 0.75 (decoded), overall and per subject. The default
        # arguments bind this fold's values.
        def score_subset(
            probs,
            fold_names=fold_names,
            reg=reg,
            transitions=transitions,
            picked=picked,
            picked_indices=picked_indices,
        ):
            prob_map = dict(zip(fold_names, probs, strict=True))
            decoded = decode_all(reg["pseudo_takes"], prob_map, transitions, 0.75)
            y = table.loc[picked, "action_id"].to_numpy()
            raw = probs[picked_indices].argmax(1)
            dec = np.array([decoded[c] for c in picked])
            users = table.loc[picked, "user"].to_numpy()

            def counts(which):
                return {
                    "n": int(which.sum()),
                    "raw": float((raw[which] == y[which]).mean()),
                    "decoded": float((dec[which] == y[which]).mean()),
                    "raw_correct": int((raw[which] == y[which]).sum()),
                    "decoded_correct": int((dec[which] == y[which]).sum()),
                }

            return {
                **counts(np.ones(len(y), dtype=bool)),
                "subjects": {u: counts(users == u) for u in sorted(set(users))},
            }

        # Accuracy of the teachers themselves on the subset, not a gain of the student.
        base.write_json(
            directory / "teacher-scores.json",
            {
                "own": score_subset(own_probs),
                "heterogeneous": score_subset(heterogeneous),
                "not_student_gain": True,
            },
        )
        # Step 3: the labelled training clips (other folds) with label-smoothed one-hot targets,
        # 1 − ε on the true class plus ε / 40 on every class.
        true_rows = order[~mask]
        true_targets = (
            np.eye(40, dtype=np.float32)[labels[~mask]] * (1 - cfg["label_smoothing"])
            + cfg["label_smoothing"] / 40
        )
        # Fold k's labels are hidden (-1) in the training datasets; the targets are the only
        # supervision.
        no_heldout_labels = row_labels.copy()
        no_heldout_labels[order[mask]] = -1
        # All of fold k with segment-centre frames and the centre crop, for the predictions
        # after training.
        validation = FusedFrameDataset(
            data, offsets, row_labels, order[mask], cfg["n_frames"], False, eval_tf, cfg["seed"]
        )
        for arm in args.arms:
            # Step 4, per variant: the 60-epoch recipe with 10 epochs, peak learning rate 3e-5 and
            # fold k.
            dest = directory / arm
            dest.mkdir(parents=True, exist_ok=True)
            config = TrainConfig(**copy.deepcopy(cfg))
            config.epochs, config.lr, config.fold = 10, 3e-5, fold
            config.checkpoint_path, config.metrics_path = dest / "checkpoint.pt", None
            # B0's extra samples are n_pseudo labelled rows drawn with replacement (seed 1042 + k);
            # B1 and B2 add the subset with teacher targets at the planned pseudo-label weight.
            sample = np.random.default_rng(1042 + fold).integers(0, len(true_rows), reg["n_pseudo"])
            if arm == "B0":
                extra_rows, targets = true_rows[sample], true_targets[sample]
                weights = np.ones(len(extra_rows), dtype=np.float32)
            else:
                extra_rows = np.array([row_of[c] for c in picked])
                targets = (own_probs if arm == "B1" else heterogeneous)[picked_indices].astype(
                    np.float32
                )
                weights = np.full(len(extra_rows), reg["pseudo_weight"], dtype=np.float32)

            # Both parts use random frame picks and the training augmentation.
            def dataset(rows, targets, weights, training_labels=no_heldout_labels):
                images = FusedFrameDataset(
                    data,
                    offsets,
                    training_labels,
                    rows,
                    cfg["n_frames"],
                    True,
                    train_tf,
                    cfg["seed"],
                )
                return ReplayDataset(images, targets, weights)

            # Labelled clips at weight 1, followed by the variant's extra samples.
            replay = ReplayConcat(
                [
                    dataset(true_rows, true_targets, np.ones(len(true_rows), dtype=np.float32)),
                    dataset(extra_rows, targets, weights),
                ]
            )
            from dataclasses import asdict

            # The variant's identity record; its steps per epoch, ceil(samples / 8), must equal
            # the planned value.
            identity = {
                "fold": fold,
                "arm": arm,
                "fold_identity_sha256": digest(fold_identity),
                "config": asdict(config),
                "n_pseudo": reg["n_pseudo"],
                "pseudo_to_true_mass_ratio": reg["pseudo_to_true_mass_ratio"],
                "sample_count": len(replay),
                "steps_per_epoch": math.ceil(len(replay) / 8),
                "source_sha256": provenance[student_run][fold]["source_sha256"],
            }
            assert identity["steps_per_epoch"] == reg["steps_per_epoch"]
            assert_identity(dest / "identity.json", identity)
            if (dest / "result.json").exists():
                print(f"completed arm preserved {dest}", flush=True)
                continue

            # Every variant starts from the fold-k R(2+1)D-34 model.
            def make_model(source_path=source):
                return helper.load_state(
                    helper.build_model(cfg, data.shape[-1], "cpu"), source_path
                )

            print(
                f"START {arm} fold{fold}, 10ep/{reg['optimizer_attempts_10ep']} attempts",
                flush=True,
            )
            started = time.monotonic()
            net, _ = fit(make_model, replay, config, dest, identity)
            trained = time.monotonic() - started
            # Step 5: four-pass predictions on all of fold k; scores on the subset (primary) and,
            # as a secondary estimate, 48_evaluate_matched.evaluate_fold on 20 thinnings of fold
            # k's takes.
            started = time.monotonic()
            after = predict(net, validation, config.device, config.eval_batch_size, True)
            elapsed = time.monotonic() - started
            np.savez(dest / "after.npz", clip_ids=fold_names, probs=after)
            primary = {"before": score_subset(own_probs), arm: score_subset(after)}
            sensitivity = evaluator.evaluate_fold(
                fold, fold_names, {"before": own_probs, arm: after}
            )
            sensitivity["transductive"] = True
            sensitivity["warning"] = "20 thinning diagnostics; trained on ONE fixed subset only"
            result = {
                "fold": fold,
                "arm": arm,
                "transductive": True,
                "primary": primary,
                "contrasts_pp": contrasts(primary),
                "sensitivity_20repeats": sensitivity,
                "training_seconds": trained,
                "ms_per_clip_tta": elapsed * 1000 / len(fold_names),
                "weights_bytes_fp32": (dest / "student.pt").stat().st_size,
                "deliverable_bytes": None,
                "identity": identity,
            }
            base.write_json(dest / "result.json", result)
            print(
                json.dumps({"fold": fold, "arm": arm, "contrasts": result["contrasts_pp"]}),
                flush=True,
            )
            del net, replay
            torch.cuda.empty_cache()
        # Paired comparison for this fold over every variant finished so far, including those
        # of earlier invocations.
        primary = {"before": score_subset(own_probs)}
        for arm in ("B0", "B1", "B2"):
            path = directory / arm / "after.npz"
            if path.exists():
                blob = np.load(path)
                assert np.array_equal(blob["clip_ids"], fold_names)
                primary[arm] = score_subset(blob["probs"])
        base.write_json(
            directory / "paired.json",
            {"fold": fold, "primary": primary, "contrasts_pp": contrasts(primary)},
        )
    # Mean contrasts over all folds that have a paired.json under the output directory.
    pairs = [json.loads(q.read_text()) for q in sorted(output.glob("fold*/paired.json"))]
    summary = {
        "folds": pairs,
        "transductive": True,
        "means_pp": {
            name: {
                m: float(
                    np.mean(
                        [f["contrasts_pp"][name][m] for f in pairs if name in f["contrasts_pp"]]
                    )
                )
                for m in ("raw", "decoded")
            }
            for name in sorted({name for f in pairs for name in f["contrasts_pp"]})
        },
    }
    base.write_json(output / "paired-evaluation.json", summary)
    print(json.dumps(summary["means_pp"]), flush=True)


if __name__ == "__main__":
    main()
