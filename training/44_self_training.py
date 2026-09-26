"""Transductive self-training: pseudo-label the unseen subjects, then fine-tune on them.

    # gate (v4 section 1.2): a held-out fold stands in for the test set
    uv run python -u scripts/44_self_training.py simulate --run ig65m-t32-det-60ep
        --peer ig65m-mixup-det-60ep --folds 0 1 2 3

    # the real thing, once the gate passes
    uv run python -u scripts/44_self_training.py apply --run ig65m-t32-det-60ep
        --base models/t32-soup8/soup.pt --pseudo models/t32-soup8/test_probs.npz
        --out models/t32-soup8-st

This is the only lever the project has that attacks the subject shift directly, and the
organisers permit it explicitly (organizer-correspondence.md C-6, C-11). The 2026-09-05
attempt scored -4.0 pp, under conditions that no longer hold: a much weaker model, hard
labels, a conf > 0.8 filter that kept 85 clips, and no take decoding.

WHICH MODEL MAY PRODUCE THE PSEUDO-LABELS IS THE WHOLE EXPERIMENT.

``train.py`` sets ``is_val = folds == fold`` and trains on the complement, so
``folds/foldN.pt`` is the model that HELD OUT fold N. For fold f to stand in for the test
set, its clips must be labelled by a model that never saw fold f's subjects -- which is
``fold_f.pt``, from any run. It is NOT the other three folds: ``fold_g`` for g != f trained
on everything except g, fold f included, so labelling fold f with those means labelling it
with models that memorised it. The gate would then report a large positive delta that is
memorisation, which is the 2026-09-07 failure repeated one level up.

So the source is fold f's models across runs (``--peer`` adds the second run's fold f), and
the model fine-tuned is fold f's too. ``simulate`` refuses any source that has seen the fold
it is labelling, rather than trusting the caller to pass the right paths.

The soft target mixes the source posterior with the take decoder's answer, because the
decoder carries structure the per-clip posterior does not -- it is worth about six clips on
the leaderboard. Equal weight, registered before running and not swept: ``blend`` is
0.5 * posterior + 0.5 * one-hot(decoded); ``probs`` and ``decoded`` are the two ends, kept
for attribution rather than for tuning.
"""
# Role: development experiment for self-training on held-out folds: each fold's takes are thinned
# to test-like partial takes, pseudo-labelled by models that never saw that fold, and that fold's
# model is fine-tuned on them plus the other folds' labels. Also provides build_model, load_state.
# Used by: training/47_matched_continuation.py, 101_matched_replay.py and 103_b_final_student.py
# (build_model, load_state), 108_finalize_calibrated.py and 124_c_full_student.py (build_model);
# training only, not used by the delivered run.
# Member B itself was trained by training/103_b_final_student.py (cuhkx/replay.py), not by this
# script. Importing this module also imports 95_fusion_shape_audit.py and, through it,
# 52_simulate_incomplete_takes.py. The usage lines above name scripts/; the file is in training/.
# Main steps of "simulate" ("apply" is disabled), for each fold f: 1) thin fold f's takes;
# 2) pseudo-label the kept clips with the models that held fold f out, and decode them per take;
# 3) fine-tune the --run model among them on the other folds' labels plus the pseudo-labels;
# 4) score the kept clips before and after, raw and decoded. All folds' results go to --out.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make the cuhkx package (src/) importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.fused import (  # noqa: E402
    FusedFrameDataset,
    export_memmap,
    labels_by_row,
    load_fused,
    make_fused_transform,
)
from cuhkx.models import Classifier, PretrainedVideo3D  # noqa: E402
from cuhkx.paths import p  # noqa: E402
from cuhkx.soup import held_out_fold  # noqa: E402
from cuhkx.takes import decode_all, transition_matrix  # noqa: E402
from cuhkx.train import seed_everything  # noqa: E402
from cuhkx.tta import predict as tta_predict  # noqa: E402

# training/ itself, so that digit-named sibling modules can be loaded with import_module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

# Thinning is not a refinement here, it is the whole measurement. Decoding fold f's COMPLETE
# training takes (3.7 clips each, exact structure) makes the take decoder far stronger than
# it will ever be at delivery, where test takes average 2.41 clips and 47 of 168 are
# singletons. Measured consequence: on fold 0 the teacher decoded to 0.8303 against a raw
# 0.7434, and the student recovered 4.08 of that 8.69 pp -- which is the student learning the
# decoder, not learning the subjects. The decoder then runs again at delivery anyway, which
# is exactly what decoded delta = +0.00 was saying. Thinning to the real test size
# distribution breaks the saturation AND yields the quantity delivery actually produces.
# subsample_contiguous keeps one contiguous run of clips per take, with a length drawn from the
# take sizes it is given (below: the clip counts of the real test takes).
subsample_contiguous = import_module("95_fusion_shape_audit").subsample_contiguous

N_CLASSES = 40


class SoftTargetDataset(Dataset):
    """A FusedFrameDataset whose items carry a distribution and a per-sample loss weight.

    Real labels and pseudo-labels differ only in the weight, so one loader covers both and
    the mixing ratio is a property of the data rather than of the training loop.
    """

    def __init__(self, base, targets: np.ndarray, weights: np.ndarray):
        if len(targets) != len(base) or len(weights) != len(base):
            raise ValueError(
                f"targets {len(targets)} and weights {len(weights)} must match the "
                f"{len(base)} items of the underlying dataset"
            )
        self.base, self.targets, self.weights = base, targets, weights

    def __len__(self) -> int:
        return len(self.base)

    # The epoch hooks are forwarded, so the base dataset's per-epoch augmentation seeding still
    # works through this wrapper.
    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def enable_epoch_sync(self) -> None:
        self.base.enable_epoch_sync()

    def __getitem__(self, i):
        # `inputs` is a 1-tuple holding one (T, C, H, W) uint8 clip. The base item's integer label
        # is dropped; a (40,) float32 target row and a scalar loss weight take its place.
        inputs, _ = self.base[i]
        return inputs, self.targets[i].astype(np.float32), np.float32(self.weights[i])


def soft_cross_entropy(logits, target, weight):
    """Weighted cross entropy against a distribution instead of an index."""
    # Per sample: -sum_k target_k * log softmax(logits)_k, times the sample's weight; then the
    # plain batch mean (not divided by the sum of the weights).
    return (weight * -(target * F.log_softmax(logits, dim=-1)).sum(-1)).mean()


# Rebuild a run's network from its training config: PretrainedVideo3D (backbone named by
# cfg["extra"]["arch"], stem widened to `in_channels`) under a dropout + linear 40-class head.
# pretrained=False skips the Kinetics download, because the caller loads a checkpoint into it.
def build_model(cfg: dict, in_channels: int, device: str) -> Classifier:
    return Classifier(
        PretrainedVideo3D(
            arch=cfg["extra"]["arch"],
            in_channels=in_channels,
            dropout=cfg["dropout"],
            pretrained=False,
            attention_pool=cfg["extra"].get("attention_pool", "none"),
        )
    ).to(device)


# Load a state dict saved with torch.save (weights_only=True: tensors and plain containers only)
# and cast every tensor to float32 before copying it into the model.
def load_state(model: Classifier, path: Path) -> Classifier:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in blob.items()})
    return model


def decoded_accuracy(probs, clip_names, truth, takes, transitions, decode_weight: float) -> float:
    """Take-decoded accuracy, i.e. the quantity every gate in this project is written on.

    Module level rather than a closure: as a closure it captured the fold loop's variables,
    and the two calls that matter compare a before and an after inside that loop.
    """
    prob_map = dict(zip(clip_names, probs, strict=True))
    decoded_map = decode_all(takes, prob_map, transitions, decode_weight)
    # A clip that no take covers keeps its per-clip argmax.
    pred = np.array([decoded_map.get(c, int(prob_map[c].argmax())) for c in clip_names])
    return float((pred == truth).mean())


# Soft targets, one (40,) row per clip that sums to 1, for the --pseudo-target modes described in
# the module docstring: the posterior, the one-hot decoded label, or their equal blend.
def make_targets(probs: np.ndarray, decoded: np.ndarray, mode: str) -> np.ndarray:
    onehot = np.eye(N_CLASSES, dtype=np.float64)[decoded]
    if mode == "probs":
        return probs
    if mode == "decoded":
        return onehot
    return 0.5 * probs + 0.5 * onehot


def finetune(model, loader, epochs: int, lr: float, device: str, cfg: dict):
    """Short fine-tune at a low LR. BatchNorm stays live, as in the recipe being extended."""
    # Two parameter groups: the backbone (names starting with "encoder") at lr times
    # backbone_lr_mult (default 0.1), the classifier head at lr. Both rates stay constant.
    backbone, head = [], []
    for name, param in model.named_parameters():
        (backbone if name.startswith("encoder") else head).append(param)
    mult = cfg["extra"].get("backbone_lr_mult", 0.1)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": lr * mult}, {"params": head, "lr": lr}],
        weight_decay=cfg["weight_decay"],
    )
    # Mixed precision with loss scaling on CUDA only; on the CPU both are disabled.
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    model.train()
    # A ConcatDataset exposes its parts as .datasets. Each part gets a shared epoch counter, so
    # DataLoader worker processes draw the augmentations of the current epoch.
    datasets = getattr(loader.dataset, "datasets", [loader.dataset])
    for dataset in datasets:
        dataset.enable_epoch_sync()
    for epoch in range(epochs):
        # Epochs are numbered from 1 in the per-sample seed (seed, epoch, cache row).
        for dataset in datasets:
            dataset.set_epoch(epoch + 1)
        total, seen = 0.0, 0
        for inputs, target, weight in loader:
            inputs = tuple(t.to(device) for t in inputs)
            target, weight = target.to(device), weight.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = soft_cross_entropy(model(*inputs), target, weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * target.size(0)
            seen += target.size(0)
        print(f"    epoch {epoch + 1}/{epochs}  loss {total / max(seen, 1):.4f}", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "apply"):
        q = sub.add_parser(name)
        q.add_argument("--run", required=True, help="run supplying the config and fold models")
        q.add_argument("--epochs", type=int, default=10)
        q.add_argument("--lr", type=float, default=3e-5)
        q.add_argument("--pseudo-weight", type=float, default=0.5)
        q.add_argument("--pseudo-target", choices=["blend", "probs", "decoded"], default="blend")
        q.add_argument("--batch-size", type=int, default=8)
        q.add_argument("--num-workers", type=int, default=4)
        q.add_argument("--decode-weight", type=float, default=0.75)
    sim = sub.choices["simulate"]
    sim.add_argument("--peer", default=None, help="second run whose fold f joins the source")
    sim.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    sim.add_argument("--out", default=".scratch/self_training_gate.json")
    args = ap.parse_args()

    # Only "simulate" runs; "apply" stops here with an explanation.
    if args.cmd != "simulate":
        raise SystemExit(
            "apply is deliberately not reachable until the simulate gate has passed "
            "(v4 section 1.2: paired delta >= +1.0 pp and >= 3/4 folds positive)"
        )

    # Step 1: the run's training config, its fused training cache and the two transforms.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((p("models_root") / args.run / "result.json").read_text(encoding="utf-8"))[
        "config"
    ]
    extra = cfg["extra"]
    seed_everything(cfg["seed"])

    cache = p("cache_root") / extra["cache"]
    # Write a memory-mappable .npy copy beside the cache if it is missing, then map it. `data` is
    # (total frames, H, W, C) uint8; clip j's frames are data[offsets[j]:offsets[j + 1]].
    export_memmap(cache)
    data, offsets, clip_ids, _ = load_fused(cache)
    in_channels = data.shape[-1]
    # Evaluation: centre crop only. Training: random crop, horizontal flip (p 0.5) and random
    # erasing (p 0.25), unless the config sets other probabilities.
    eval_tf = make_fused_transform(extra["crop"], erase_prob=0.0, flip_prob=0.0)
    train_tf = make_fused_transform(
        extra["crop"],
        erase_prob=extra.get("erase_prob", 0.25),
        flip_prob=extra.get("flip_prob", 0.5),
    )

    # Step 2: align the fold table with the cache rows. `order` holds the cache rows of the
    # labelled clips, `names`, `labels` and `fold_of` follow it, and `row_labels` is indexed by
    # cache row (-1 for rows without a label).
    folds_table = pd.read_parquet(p("processed_root") / "folds.parquet").set_index("clip_id")
    keep = [i for i, c in enumerate(clip_ids) if c in folds_table.index]
    order = np.array(keep)
    names = [clip_ids[i] for i in keep]
    labels = folds_table.loc[names, "action_id"].to_numpy()
    fold_of = folds_table.loc[names, "fold"].to_numpy()
    row_labels = labels_by_row(len(clip_ids), order, labels)

    # Training takes: take id -> clip ids in recording order.
    takes_table = pd.read_parquet(p("processed_root") / "takes_train.parquet")
    takes = {
        k: g.sort_values("position")["clip_id"].tolist() for k, g in takes_table.groupby("take_id")
    }
    truth_all = dict(zip(names, labels, strict=True))
    fold_by_name = dict(zip(names, fold_of.tolist(), strict=True))
    # A take belongs to one subject, so all its clips share a fold; the first member decides.
    take_fold = {t: fold_by_name[m[0]] for t, m in takes.items() if m and m[0] in fold_by_name}
    # The real test's take size distribution, which is what fold f gets thinned to.
    target_sizes = (
        pd.read_parquet(p("processed_root") / "takes_test.parquet")
        .groupby("take_id")
        .size()
        .to_numpy()
    )
    print(
        f"target take sizes from the real test: {len(target_sizes)} takes, "
        f"mean {target_sizes.mean():.2f} clips"
    )

    # Step 3, for each fold f: check the sources, thin, pseudo-label, fine-tune and score.
    results = []
    for f in args.folds:
        # Pseudo-label sources: this run's model that held fold f out, and the peer run's, if any.
        sources = [p("models_root") / args.run / "folds" / f"fold{f}.pt"]
        if args.peer:
            sources.append(p("models_root") / args.peer / "folds" / f"fold{f}.pt")
        for q in sources:
            if not q.exists():
                raise SystemExit(f"missing source model: {q}")
            # Fail closed. A source that trained on the fold it labels makes the gate report
            # memorisation as a gain, and nothing downstream could tell.
            if held_out_fold(q) != f:
                raise SystemExit(
                    f"REFUSED: {q} did not hold out fold {f}, so it trained on the subjects "
                    f"it would be labelling. That measures memorisation, not self-training."
                )

        # Thin fold f's takes to the real test's size distribution. The kept clips ARE the
        # simulated test set: they are pseudo-labelled, fine-tuned on, and scored, all on the
        # same thinned structure, which is what delivery actually sees. The dropped clips are
        # not added back with true labels -- they belong to held-out subjects.
        rng = np.random.default_rng(cfg["seed"] + f)
        mine = {t: m for t, m in takes.items() if take_fold.get(t) == f}
        thinned = subsample_contiguous(mine, target_sizes, rng)
        kept = [c for members in thinned.values() for c in members]
        kept_set = set(kept)
        # Transitions from the OTHER folds only, as 70_mock_public_test.py does: a prior
        # estimated partly on fold f's own labels is a leak, small but free to avoid.
        transitions = transition_matrix(
            {t: m for t, m in takes.items() if take_fold.get(t) != f}, truth_all
        )

        # is_f marks the kept clips of fold f within `names`; full_f counts all of fold f's clips.
        is_f = np.array([n in kept_set for n in names])
        n_f = int(is_f.sum())
        full_f = int((fold_of == f).sum())
        print(
            f"fold {f}: thinned {len(mine)} takes / {full_f} clips -> "
            f"{len(thinned)} takes / {n_f} clips ({n_f / max(full_f, 1):.0%}), "
            f"mean {n_f / max(len(thinned), 1):.2f} clips per take",
            flush=True,
        )
        # Deterministic view of the kept clips: segment-centre frames and a centre crop.
        eval_ds = FusedFrameDataset(
            data, offsets, row_labels, order[is_f], cfg["n_frames"],
            train=False, transform=eval_tf, seed=cfg["seed"],
        )  # fmt: skip

        # --- pseudo-labels, from models that never saw fold f -----------------------------
        # Each source's four-pass TTA softmax, averaged over the sources: (n_kept, 40).
        stack = []
        for q in sources:
            model = load_state(build_model(cfg, in_channels, device), q).eval()
            stack.append(tta_predict(model, eval_ds, device, args.batch_size, True))
            del model
        fused = np.mean(stack, axis=0)
        fold_names = [names[i] for i in np.where(is_f)[0]]
        prob_map = dict(zip(fold_names, fused, strict=True))
        # Take-decoded labels of the kept clips. The two accuracies below score the pseudo-labels
        # against the true labels, which exist here because fold f is a training fold.
        decoded_map = decode_all(thinned, prob_map, transitions, args.decode_weight)
        decoded = np.array([decoded_map.get(c, int(prob_map[c].argmax())) for c in fold_names])
        truth_f = labels[is_f]
        pseudo_acc = float((decoded == truth_f).mean())
        raw_acc = float((fused.argmax(1) == truth_f).mean())

        # --- fine-tune fold f's own model on real labels + fold f's pseudo-labels ---------
        # NOT ~is_f. Since thinning, is_f means "fold f's KEPT clips", so ~is_f would also
        # contain fold f's DROPPED clips -- held-out subjects, with their true labels, in the
        # training set. That is a leak, and it would have inflated the very number this gate
        # exists to produce.
        is_other = fold_of != f
        real_rows = order[is_other]
        real_targets = np.eye(N_CLASSES)[labels[is_other]]
        # Labelled clips of the other folds: one-hot targets at weight 1, training augmentation.
        # The kept clips of fold f follow with their soft targets at --pseudo-weight (default 0.5).
        real_ds = SoftTargetDataset(
            FusedFrameDataset(
                data,
                offsets,
                row_labels,
                real_rows,
                cfg["n_frames"],
                train=True,
                transform=train_tf,
                seed=cfg["seed"],
            ),  # fmt: skip
            real_targets,
            np.ones(len(real_rows)),
        )
        pseudo_ds = SoftTargetDataset(
            FusedFrameDataset(
                data,
                offsets,
                row_labels,
                order[is_f],
                cfg["n_frames"],
                train=True,
                transform=train_tf,
                seed=cfg["seed"],
            ),  # fmt: skip
            make_targets(fused, decoded, args.pseudo_target),
            np.full(n_f, args.pseudo_weight),
        )
        mixed = torch.utils.data.ConcatDataset([real_ds, pseudo_ds])
        train_loader = DataLoader(
            mixed,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        # The student starts from the first source, the --run model that held fold f out; `before`
        # and `after` are its four-pass predictions on the kept clips.
        student = load_state(build_model(cfg, in_channels, device), sources[0])
        before = tta_predict(student.eval(), eval_ds, device, args.batch_size, True)
        student = finetune(student, train_loader, args.epochs, args.lr, device, cfg)
        after = tta_predict(student.eval(), eval_ds, device, args.batch_size, True)

        # Every probability array is kept, so any later question -- a different thinning
        # draw, a different tw, a different fusion -- is answered by reloading these rather
        # than by another two hours of GPU. The first version of this script stored only
        # scalars, and that is precisely why the thinned reading could not be recovered
        # after the fact.
        Path(".scratch").mkdir(exist_ok=True)
        np.savez(
            f".scratch/st_probs_fold{f}.npz",
            clip_ids=np.array(fold_names),
            truth=truth_f,
            teacher=fused,
            before=before,
            after=after,
            kept_takes=np.array(list(thinned.keys())),
        )

        a0 = decoded_accuracy(before, fold_names, truth_f, thinned, transitions, args.decode_weight)
        a1 = decoded_accuracy(after, fold_names, truth_f, thinned, transitions, args.decode_weight)
        # Undecoded too, and it is the informative one here. On a TRAINING fold the take
        # decoder saturates: measured on fold 0, the two-model pseudo-label source, the
        # student before and the student after all decoded to exactly 0.8303 while the raw
        # number was 0.7434. decode_take forces one action per take, and train takes average
        # 3.7 clips with exact structure, so it erases the differences between similar
        # models. Test takes average 2.4 with 47 singletons and do no such thing -- so a
        # decoded-only delta of +0.00 here would be a property of the instrument, and
        # reporting it as "self-training does nothing" would be wrong.
        r0 = float((before.argmax(1) == truth_f).mean())
        r1 = float((after.argmax(1) == truth_f).mean())
        results.append(
            {
                "fold": f,
                "n_pseudo": n_f,
                "sources": [str(q) for q in sources],
                "pseudo_label_accuracy_decoded": pseudo_acc,
                "pseudo_label_accuracy_raw": raw_acc,
                "decoded_before": a0,
                "decoded_after": a1,
                "raw_before": r0,
                "raw_after": r1,
                "delta_raw_pp": 100 * (r1 - r0),
                "delta_pp": 100 * (a1 - a0),
            }
        )
        print(
            f"fold {f}: {n_f} clips; pseudo-label acc {pseudo_acc:.4f} (raw {raw_acc:.4f}); "
            f"decoded {a0:.4f} -> {a1:.4f} ({100 * (a1 - a0):+.2f} pp)  |  "
            f"raw {r0:.4f} -> {r1:.4f} ({100 * (r1 - r0):+.2f} pp)",
            flush=True,
        )
        del student

    # Step 4: summary over folds and the pass rule printed below: a mean decoded change of at
    # least +1.0 pp with at least 3 folds positive.
    raw_deltas = [r["delta_raw_pp"] for r in results]
    print()
    print(
        f"undecoded mean delta {float(np.mean(raw_deltas)):+.2f} pp, "
        f"{sum(1 for d in raw_deltas if d > 0)}/{len(raw_deltas)} folds positive"
    )
    deltas = [r["delta_pp"] for r in results]
    mean_delta = float(np.mean(deltas))
    positive = sum(1 for d in deltas if d > 0)
    print(f"\nmean delta {mean_delta:+.2f} pp over {len(deltas)} folds; {positive} positive")
    print("registered gate: mean delta >= +1.0 pp AND >= 3/4 folds positive")
    print("GATE PASSES" if mean_delta >= 1.0 and positive >= 3 else "GATE DOES NOT PASS")
    Path(args.out).write_text(
        json.dumps(
            {"folds": results, "mean_delta_pp": mean_delta, "folds_positive": positive},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
