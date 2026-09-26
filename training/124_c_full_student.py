"""Transfer the qualified C recipe to one full-data shared-view student and int8 pack."""

# Role: Trains member C: fine-tunes the R(2+1)D-34 weight average for 10 epochs on all 2,931
#   labelled clips, drawing the det or miw view per clip and epoch, re-estimates its BatchNorm
#   statistics on 1,600 training clips and writes an int8 pack. C is not trained on test clips.
# Used by: run by hand; training. Its pack (abc-c-final-20260911/deliverable-int8.pt) is member C
#   before it was merged into weights/model.pt.
# Main steps: 1) check the four-fold validation records, source hashes and recipe; 2) open both
#   training caches and check the test caches' build records; 3) choose the training and
#   calibration clips; 4) write or compare the identity record; 5) check the compute budget;
#   6) train with train_one_fold; 7) re-estimate BatchNorm; 8) pack with the person detector.
# Produces, under models_root/abc-c-final-20260911/: identity.json, checkpoint.pt, metrics.jsonl,
#   student-raw.pt, training.json, student.pt, calibration.json, deliverable-int8.pt and
#   result.json; it also updates the time records in abc-c-20260911/. --preflight-only runs the
#   checks and writes only work/c-full-preflight.json.
# Inputs, not included in the repository (TRAINING.md lists them): under models_root, the C
#   validation records (abc-c-20260911/), the continuation configurations written by
#   47_matched_continuation.py (abc-a-20260910/fold<k>/A1/), the weight average and its int8 pack
#   (t32-soup/), B's identity records (abc-b2-final-20260911/, abc-b2-bncal-final-20260911/) and
#   the det and miw run records; the det and miw training and test caches; folds.parquet;
#   work/c-worker-equivalence/result.json and docs/plans/2026-09-11-c-full-delivery.md, whose
#   SHA256 enters the identity. models_root is not defined in the shipped configs/paths.yaml.
# Every source file listed in the four fold records must still have its recorded SHA256, so the
#   script documents the procedure rather than running out of the box.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Make cuhkx (src/), scripts/ and the training modules importable; module names that start with
# a digit are loaded with import_module.
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"),str(ROOT/'training')]
base = import_module("47_matched_continuation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    # Refuse to start while any other Python process runs.
    base.assert_idle()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from cuhkx.budget import pack_deliverable, save_deliverable
    from cuhkx.fused import FusedFrameDataset, labels_by_row, load_fused, make_fused_transform
    from cuhkx.interpolation import calibrate_bn
    from cuhkx.models import PretrainedVideo3D
    from cuhkx.paths import p
    from cuhkx.shared_views import SharedViewDataset
    from cuhkx.train import TrainConfig, _atomic_save, _collate, seed_everything, train_one_fold

    # Step 1: abc-c-20260911 holds the four-fold validation of the C recipe; its summary must
    # record that the recipe met its acceptance criteria, including with int8 weights.
    campaign = p("models_root") / "abc-c-20260911"
    dest = p("models_root") / "abc-c-final-20260911"
    gate_path = campaign / "quant-fourfold.json"
    gate = json.loads(gate_path.read_text())
    assert gate["decision"]["eligible"] and gate["decision"]["quantization_passed"]
    # A separate check must have shown that the changed number of DataLoader workers (2, see
    # operational_change below) gives the same batches, labels and main-process random state.
    worker_proof = ROOT / "work/c-worker-equivalence/result.json"
    assert json.loads(worker_proof.read_text())["same_batch_tensors_labels_main_rng"]
    # Every source file recorded by the four fold runs (training and int8 check) must still have
    # its recorded SHA256, and the four A1 configurations of 47_matched_continuation.py must agree
    # apart from fold-specific fields; the first one becomes the template.
    template = None
    ignored = {"fold", "checkpoint_path", "metrics_path", "recipe_hash"}
    for fold in range(4):
        for rel in (f"fold{fold}/identity.json", f"fold{fold}/quant-audit/identity.json"):
            ident = json.loads((campaign / rel).read_text())
            for path, digest in ident["code_sha256"].items():
                assert base.sha256(ROOT / path) == digest, path
        aid = json.loads(
            (p("models_root") / f"abc-a-20260910/fold{fold}/A1/identity.json").read_text()
        )
        recipe = {k: v for k, v in aid["config"].items() if k not in ignored}
        if template is None:
            template = copy.deepcopy(aid["config"])
            reference = recipe
        else:
            assert recipe == reference
    # C starts from the same R(2+1)D-34 weight average as member B.
    source = p("models_root") / "t32-soup/soup.pt"
    bid = json.loads((p("models_root") / "abc-b2-final-20260911/identity.json").read_text())
    assert base.sha256(source) == bid["provenance"]["own"]["source_sha256"]
    # Full-data version of that configuration: no fold, outputs under dest, 2 DataLoader workers;
    # the recipe hash is set later. The asserts pin 10 epochs, batch 8, seed 42, evaluation batch
    # 6 and no EMA or SWA weight averaging.
    template.update(
        fold=None,
        checkpoint_path=dest / "checkpoint.pt",
        metrics_path=dest / "metrics.jsonl",
        recipe_hash=None,
        num_workers=2,
    )
    config = TrainConfig(**template)
    assert config.epochs == 10 and config.batch_size == 8 and config.seed == 42
    assert config.eval_batch_size == 6 and config.ema_decay == config.swa_start == 0
    # The 60-epoch det run's configuration supplies the architecture, the augmentation settings
    # and the det cache; a run on the miw view supplies the miw cache.
    raw_cfg = json.loads((p("models_root") / "ig65m-t32-det-60ep/result.json").read_text())[
        "config"
    ]
    raw_cfg["extra"].pop("subjects", None)
    local_cfg = json.loads((p("models_root") / "ig65m-t32-miw-f0/result.json").read_text())[
        "config"
    ]
    # Step 2: open both training caches memory-mapped: all frames as (frames, 144, 144, 4) uint8
    # plus per-clip offsets. Each build record must equal the one its run recorded, and both
    # caches must list the same clips in the same order.
    caches, metadata = {}, {}
    for view, cfg in (("det", raw_cfg), ("miw", local_cfg)):
        cache = p("cache_root") / cfg["extra"]["cache"]
        data, offsets, ids, build = load_fused(cache)
        assert isinstance(data, np.memmap) and data.shape[-3:] == (144, 144, 4)
        assert build == cfg["extra"]["cache_build"] and len(ids) == len(set(ids))
        caches[view] = (data, offsets, np.asarray(ids))
        metadata[view] = dict(
            cache=str(cache),
            build=build,
            meta_sha256=base.sha256(cache.with_name(cache.stem + "_meta.npz")),
        )
    assert np.array_equal(caches["det"][2], caches["miw"][2])
    # The test caches of both views must have been built with the training caches' recipe, and
    # their build records must convert back into a build command (26_build_test_caches.py); the
    # records go into the pack's metadata. Only the build records of the test caches are read.
    test_views = {}
    builder = import_module("26_build_test_caches")
    for view in ("det", "miw"):
        test_name = f"fused-{view}_test.npz"
        _, _, _, test_build = load_fused(p("cache_root") / test_name)
        clashes, _ = builder.compare_builds(metadata[view]["build"], list(test_build))
        assert not clashes, clashes
        prefix, suffix = builder.split_cache_name(test_name)
        builder.argv_from_build(list(test_build), prefix, suffix)
        test_views[view] = dict(test_cache=test_name, test_cache_build=list(test_build))
    # Step 3: the 2,931 labelled clips (their fold-table labels must match the 60-epoch run's
    # out-of-fold record), their cache rows (order), and 1,600 of them drawn with seed 42 for
    # BatchNorm re-estimation, the same clips as for member B (108_finalize_calibrated.py).
    # labels is indexed by cache row.
    ids = caches["det"][2]
    ref = np.load(p("models_root") / "ig65m-t32-det-60ep/oof.npz")
    table = (
        pd.read_parquet(p("processed_root") / "folds.parquet")
        .set_index("clip_id")
        .loc[ref["clip_ids"]]
    )
    assert len(table) == 2931 and np.array_equal(table.action_id, ref["labels"])
    order = np.array([i for i, c in enumerate(ids) if c in table.index])
    assert len(order) == 2931
    chosen = np.random.default_rng(42).permutation(order)[:1600]
    old_bn = json.loads(
        (p("models_root") / "abc-b2-bncal-final-20260911/identity.json").read_text()
    )
    assert ids[chosen].tolist() == old_bn["calibration_ids"]
    labels = labels_by_row(len(ids), order, table.loc[ids[order], "action_id"].to_numpy())
    # Training: random 128² crop of the 144² view with the recorded flip, erasing and scale-jitter
    # settings; evaluation: centre 128² crop.
    ex = raw_cfg["extra"]
    train_tf = make_fused_transform(
        128,
        erase_prob=ex["erase_prob"],
        flip_prob=ex["flip_prob"],
        scale_range=ex.get("scale_jitter"),
    )
    eval_tf = make_fused_transform(128, erase_prob=0, flip_prob=0)

    # 32 frames per clip: one random frame per temporal segment when training, the segment
    # centres otherwise.
    def dataset(view, rows, train=False):
        data, offsets, _ = caches[view]
        return FusedFrameDataset(
            data, offsets, labels, rows, 32, train, train_tf if train else eval_tf, 42
        )

    # Per clip and epoch, SharedViewDataset returns the det or the miw sample, chosen by a hash of
    # (seed, epoch, clip id); a clip that lacks one view always uses the other.
    def shared(rows, train=False):
        return SharedViewDataset(
            dataset("det", rows, train), dataset("miw", rows, train), ids[rows]
        )

    # The training set is all 2,931 labelled clips; every one of them must have det frames.
    train_ds = shared(order, True)
    assert train_ds.det_present.all()
    # Step 4: the identity record: starting weights, configuration, SHA256 of this script, the
    # planning note, the worker check and the library files, cache records, clip lists and the
    # det/miw counts per epoch. 3,670 optimizer steps are 10 epochs of 367 batches of 8. There is
    # no held-out score; the trainer's validation set is only a diagnostic (Step 6).
    files = [Path(__file__), ROOT / "docs/plans/2026-09-11-c-full-delivery.md", worker_proof]
    files += [
        ROOT / q
        for q in (
            "src/cuhkx/train.py",
            "src/cuhkx/fused.py",
            "src/cuhkx/shared_views.py",
            "src/cuhkx/models.py",
            "src/cuhkx/interpolation.py",
            "src/cuhkx/budget.py",
        )
    ]
    identity = dict(
        source=str(source),
        source_sha256=base.sha256(source),
        config=asdict(config),
        code_sha256={str(q.relative_to(ROOT)): base.sha256(q) for q in files},
        quant_gate_sha256=base.sha256(gate_path),
        cache_metadata=metadata,
        test_views=test_views,
        train_ids=ids[order].tolist(),
        calibration_ids=ids[chosen].tolist(),
        training_views=[train_ds.view_counts(e) for e in range(1, 11)],
        calibration_views=shared(chosen).view_counts(0),
        optimizer_attempts=3670,
        heldout_accuracy=None,
        validation_is_train_diagnostic=True,
        operational_change="8 to 2 workers; exact CPU integration proof",
        stage_limit_seconds=4200,
        total_limit_seconds=32400,
    )
    # --preflight-only stops here and writes the identity to work/c-full-preflight.json.
    if args.preflight_only:
        base.write_json(ROOT / "work/c-full-preflight.json", identity)
        print("C full metadata/source/recipe/calibration/worker preflight passed", flush=True)
        return
    # A later start needs --resume and must reproduce the identity record; a finished run
    # (result.json present) is left as it is.
    if dest.exists() and not args.resume:
        raise FileExistsError("preserve existing C final attempt")
    dest.mkdir(exist_ok=True)
    import_module("101_matched_replay").assert_identity(dest / "identity.json", identity)
    if (dest / "result.json").exists():
        print("Completed C student and pack preserved", flush=True)
        return
    # The identity's hash is stored in every checkpoint; train_one_fold refuses to resume a
    # checkpoint with a different hash.
    config.recipe_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    # Step 5: compute-time limit. compute-budget.json holds the seconds already spent on C; this
    # full-data stage may add at most 4,200 s, counted from its first start
    # (full-budget-origin.json), and the total may not exceed 32,400 s. The finally block below
    # adds this invocation's time.
    budget_path = campaign / "compute-budget.json"
    used = json.loads(budget_path.read_text())["seconds"]
    origin_path = campaign / "full-budget-origin.json"
    if not origin_path.exists():
        base.write_json(
            origin_path, dict(seconds_before_full=used, gate_sha256=base.sha256(gate_path))
        )
    origin = json.loads(origin_path.read_text())
    assert origin["gate_sha256"] == base.sha256(gate_path)
    limit = min(32400, origin["seconds_before_full"] + 4200)
    started = time.monotonic()
    helper = import_module("44_self_training")
    unpack = import_module("69_infer_deliverable").to_fp32_state

    # R(2+1)D-34 with a 4-channel stem, weights converted to FP32 (to_fp32_state also expands
    # int8 states). The two input-normalisation buffers must hold the Kinetics mean and standard
    # deviation on channels 0-2 and their average on the IR channel.
    def make_model(path=source):
        net = helper.build_model(raw_cfg, 4, "cpu")
        net.load_state_dict(unpack(torch.load(path, map_location="cpu", weights_only=True)))
        count = 0
        for key, value in net.state_dict().items():
            if key.endswith(".mean") or key.endswith(".std"):
                preset = (
                    PretrainedVideo3D.KINETICS_MEAN
                    if key.endswith(".mean")
                    else PretrainedVideo3D.KINETICS_STD
                )
                assert np.allclose(value.flatten().numpy(), (*preset, sum(preset) / 3), atol=1e-3)
                count += 1
        assert count == 2
        return net

    try:
        # Step 6: train unless student-raw.pt exists. At least 3,600 s must remain; the dataset
        # refuses to start an epoch less than 240 s before a deadline set 600 s before the limit,
        # and the checkpoint of the last completed epoch is kept.
        raw = dest / "student-raw.pt"
        if not raw.exists():
            assert used + 3600 <= limit, "insufficient full training reserve"
            train_ds.deadline = started + limit - used - 600
            print("C FULL START 2931 true / 10ep / 3670 attempts / 2 workers", flush=True)
            train_start = time.monotonic()
            # The trainer needs a validation set; 8 labelled clips in the det view serve as a
            # diagnostic only. It returns the model, the diagnostic accuracy, its probabilities
            # and the accuracy on the un-augmented training clips (fit).
            net, _, _, fit = train_one_fold(make_model, train_ds, dataset("det", order[:8]), config)
            training_seconds = time.monotonic() - train_start
            # The last checkpoint must be epoch 10 of 10 with 3,670 scheduled steps, and the
            # returned weights must equal it. AdamW's per-parameter step counts give the number
            # of updates applied (steps skipped by the gradient scaler do not count).
            saved = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
            assert saved["epoch"] == saved["epochs"] == 10
            assert saved["schedule"]["total_steps"] == 3670
            assert all(torch.equal(v.cpu(), saved["model"][k]) for k, v in net.state_dict().items())
            updates = [int(v["step"]) for v in saved["optimizer"]["state"].values() if "step" in v]
            # student-raw.pt: the trained weights before BatchNorm re-estimation.
            _atomic_save({k: v.cpu() for k, v in net.state_dict().items()}, raw)
            base.write_json(
                dest / "training.json",
                dict(
                    training_seconds=training_seconds,
                    fixed_epochs=10,
                    optimizer_attempts=3670,
                    observed_updates_min=min(updates),
                    observed_updates_max=max(updates),
                    exact_epoch10_tensor_match=True,
                    student_sha256=base.sha256(raw),
                    fit=fit,
                    heldout_accuracy=None,
                ),
            )
            del net, saved
            torch.cuda.empty_cache()
        # Step 7: BatchNorm re-estimation, with at least 300 s left: statistics reset, then a
        # cumulative average over 200 batches of 8 of the 1,600 calibration clips, with
        # inference-time frames and crops, dropout off and no gradients. Each clip uses the view
        # drawn for epoch 0. The learned parameters must stay unchanged.
        assert used + time.monotonic() - started + 300 <= limit
        calibrated = dest / "student.pt"
        if not calibrated.exists():
            seed_everything(42, deterministic=True)
            net = make_model(raw)
            loader = DataLoader(shared(chosen), batch_size=8, shuffle=False, collate_fn=_collate)
            stats = calibrate_bn(net.to("cuda"), loader, max_batches=200, device="cuda")
            assert stats["batches"] == 200 and stats["samples"] == 1600
            net.cpu()
            original = torch.load(raw, map_location="cpu", weights_only=True)
            assert all(torch.equal(v, original[k]) for k, v in net.named_parameters())
            _atomic_save(net.state_dict(), calibrated)
            base.write_json(
                dest / "calibration.json",
                dict(**stats, source_sha256=base.sha256(raw), learned_parameters_unchanged=True),
            )
            del net, original, loader
            torch.cuda.empty_cache()
        # Step 8: the pack. The unquantised person detector and the base metadata come from the
        # int8 pack of the weight average. The metadata adds the views with their test-cache build
        # records, equal det/miw weights, four-pass test-time augmentation, the decoding weight
        # 0.75 and the identity hash; graded_entrypoint names a development script that is not in
        # this repository.
        floor = torch.load(
            p("models_root") / "t32-soup/deliverable-int8.pt", map_location="cpu", weights_only=True
        )
        detector = floor["components"]["detector"]
        assert all(
            not isinstance(v, dict) and (not v.is_floating_point() or v.dtype == torch.float32)
            for v in detector.values()
        )
        meta = copy.deepcopy(floor["meta"])
        meta.pop("soup", None)
        meta.update(
            format="cuhkx-shared-C-v1",
            run=dest.name,
            precision="int8",
            views=test_views,
            graded_entrypoint="scripts/125_c_delivery.py",
            view_weights={"det": 0.5, "miw": 0.5},
            batch_size=6,
            tta="4-pass",
            decode_weight=0.75,
            class_order=list(range(40)),
            identity_sha256=base.sha256(dest / "identity.json"),
        )
        # The calibrated network is stored with int8 per-output-channel weights (other
        # floating-point tensors in FP16) next to the detector; pack_deliverable joins the
        # directory into one file, which must be under 100,000,000 bytes and must load back into
        # the network.
        state = torch.load(calibrated, map_location="cpu", weights_only=True)
        save_deliverable({"image": state}, dest / "deliverable-int8", meta=meta, int8=True)
        save_deliverable({"detector": detector}, dest / "deliverable-int8")
        packed = pack_deliverable(dest / "deliverable-int8")
        assert packed["total_bytes"] < 100_000_000
        actual = torch.load(dest / "deliverable-int8.pt", map_location="cpu", weights_only=True)
        assert set(actual["components"]) == {"image", "detector"}
        check = helper.build_model(raw_cfg, 4, "cpu")
        check.load_state_dict(unpack(actual["components"]["image"]))
        assert used + time.monotonic() - started <= limit
        # submission_ready stays False; pending lists the checks still outstanding at that point.
        result = dict(
            packed=packed,
            pack_sha256=base.sha256(dest / "deliverable-int8.pt"),
            calibrated_student_sha256=base.sha256(calibrated),
            submission_ready=False,
            pending=[
                "full dual-view raw chain",
                "two cold starts",
                "missing-local fallback",
                "preregistration and fresh quota",
            ],
            heldout_accuracy=None,
            single_student=True,
        )
        base.write_json(dest / "result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        # Always add this invocation's time to the compute budget, also after a failure.
        base.write_json(
            budget_path,
            dict(
                seconds=used + time.monotonic() - started,
                prior_seconds=used,
                full_invocation_seconds=time.monotonic() - started,
                full_phase_limit_seconds=4200,
                total_limit_seconds=32400,
            ),
        )


if __name__ == "__main__":
    main()
