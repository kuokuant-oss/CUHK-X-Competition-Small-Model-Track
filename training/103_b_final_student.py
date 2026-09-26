"""One registered full-data B2 student; teachers stay outside the inference package."""

# Role: Trains member B: fine-tunes the R(2+1)D-34 weight average for 10 epochs on the det view of
#   the 2,931 labelled clips plus the 405 test clips, whose soft pseudo-labels are the mean
#   four-pass softmax of two teachers (the R(2+1)D-34 and ir-CSN-152 weight averages).
# Used by: run by hand; training. Its student.pt is the input of 108_finalize_calibrated.py
#   B2-BNcal, which re-estimates BatchNorm and writes the pack that became member B.
# Main steps: 1) check the four-fold validation summary and source hashes; 2) teacher posteriors
#   on the test clips; 3) the weighted dataset (labelled clips at weight 1 with label-smoothed
#   one-hot targets, test clips at weight 0.5 with the pseudo-labels); 4) training with
#   101_matched_replay.fit; 5) an int8 pack of this student, before BatchNorm re-estimation.
# Produces, under models_root/abc-b2-final-20260911/: identity.json, teacher-own.npz,
#   teacher-peer.npz, pseudo.npz, checkpoint.pt, epoch-NN.json, student.pt, deliverable-int8.pt
#   and result.json. --dry-run only prints the identity record.
# Inputs, not included in the repository (TRAINING.md lists them): under models_root, the
#   validation summary abc-b-20260911/quant-fourfold.json, both weight averages with their run
#   configurations, ig65m-t32-det-60ep/oof.npz and the int8 pack t32-soup/deliverable-int8.pt;
#   the det training cache and the teachers' test caches (det, det248); folds.parquet; the test
#   CSV (paths key test_csv). models_root and test_csv are not defined in the shipped
#   configs/paths.yaml.
# Every source file listed in the validation summary must still have its recorded SHA256, so the
#   script documents the procedure rather than running out of the box.

from __future__ import annotations

import argparse
import copy
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
replay = import_module("101_matched_replay")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Refuse to start while any other Python process runs.
    base.assert_idle()
    import numpy as np
    import pandas as pd

    from cuhkx.paths import p

    # Step 1: the four-fold validation summary of the B recipe with int8 weights
    # (abc-b-20260911/quant-fourfold.json) must record a pass, and every source file it lists
    # must still have its recorded SHA256.
    models = p("models_root")
    campaign = models / "abc-b-20260911"
    gate_file = campaign / "quant-fourfold.json"
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if not gate["pilot_gate"]:
        raise ValueError("B quantized accuracy gate not passed")
    for path, value in gate["code_sha256"].items():
        assert base.sha256(ROOT / path) == value
    dest = models / "abc-b2-final-20260911"
    if dest.exists() and not args.resume:
        raise FileExistsError(dest)
    # Teacher 1 ("own") is the R(2+1)D-34 weight average, whose 60-epoch run also supplies B's
    # recipe; teacher 2 ("peer") is the ir-CSN-152 weight average. Each entry names the run with
    # the fold models and the directory of their average.
    sources = {
        "own": ("ig65m-t32-det-60ep", "t32-soup"),
        "peer": ("csn152-4f224-clip40", "csn152-4f224-clip40-soup"),
    }
    # Per teacher: the run configuration, the weight average (which must average all four fold
    # models of that run) and the test cache of the teacher's own view.
    provenance = {}
    for name, (run, soup) in sources.items():
        config_file = models / run / "result.json"
        cfg = json.loads(config_file.read_text(encoding="utf-8"))["config"]
        cfg["extra"].pop("subjects", None)
        source = models / soup / "soup.pt"
        soup_meta = json.loads((models / soup / "soup_meta.json").read_text(encoding="utf-8"))
        assert soup_meta["source"] == run and soup_meta["members"] == [0, 1, 2, 3]
        cache = p("cache_root") / cfg["extra"]["cache"].replace("_train", "_test")
        assert cache.with_name(cache.stem + "_data.npy").is_file()
        cache_meta = cache.with_name(cache.stem + "_meta.npz")
        provenance[name] = dict(
            source=str(source),
            source_sha256=base.sha256(source),
            config=cfg,
            config_sha256=base.sha256(config_file),
            soup_meta=soup_meta,
            test_cache=str(cache),
            test_cache_meta_sha256=base.sha256(cache_meta),
        )
    # B's configuration: teacher 1's run recipe with 10 epochs, peak learning rate 3e-5 and no
    # fold.
    cfg = copy.deepcopy(provenance["own"]["config"])
    cfg.update(
        epochs=10,
        lr=3e-5,
        fold=None,
        checkpoint_path=str(dest / "checkpoint.pt"),
        metrics_path=None,
    )
    # The 2,931 labelled clips in out-of-fold order; their labels must agree with the fold table.
    table = pd.read_parquet(p("processed_root") / "folds.parquet").set_index("clip_id")
    ref = np.load(models / sources["own"][0] / "oof.npz")
    table = table.loc[ref["clip_ids"]]
    assert len(table) == 2931 and np.array_equal(table.action_id, ref["labels"])
    # The path column identifies clips; the prediction column is never read as a label.
    test_paths = pd.read_csv(p("test_csv"), usecols=["path"])["path"]
    # Test clip names (the last path component, SM_test_XXXX) in CSV order.
    test_ids = [str(x).replace("\\", "/").rstrip("/").split("/")[-1] for x in test_paths]
    assert len(set(test_ids)) == len(test_ids) == 405
    train_cache = p("cache_root") / cfg["extra"]["cache"]
    assert train_cache.with_name(train_cache.stem + "_data.npy").is_file()
    # Source files whose SHA256 enters the identity record.
    code_files = [
        Path(__file__),
        ROOT / "training/101_matched_replay.py",
        ROOT / "training/44_self_training.py",
        ROOT / "src/cuhkx/replay.py",
        ROOT / "src/cuhkx/train.py",
        ROOT / "src/cuhkx/fused.py",
        ROOT / "src/cuhkx/tta.py",
        ROOT / "src/cuhkx/budget.py",
    ]
    # Identity record: 2,931 + 405 = 3,336 samples in 417 batches of 8 per epoch, the
    # pseudo-label share of the loss weight (0.5 × 405 / 2,931 = 6.9%), configuration, teacher
    # provenance, test clip order and hashes. fit() checks the sample counts and the share in
    # every epoch.
    identity = dict(
        fold=None,
        arm="B2",
        n_pseudo=405,
        n_true=2931,
        sample_count=3336,
        steps_per_epoch=417,
        pseudo_to_true_mass_ratio=0.5 * 405 / 2931,
        config=cfg,
        provenance=provenance,
        test_ids=test_ids,
        code_sha256={str(q.relative_to(ROOT)): base.sha256(q) for q in code_files},
        quant_gate_sha256=base.sha256(gate_file),
        train_cache_meta_sha256=base.sha256(train_cache.with_name(train_cache.stem + "_meta.npz")),
        floor_sha256=base.sha256(models / "t32-soup/deliverable-int8.pt"),
        heldout_accuracy=None,
        transductive=True,
    )
    if args.dry_run:
        print(json.dumps(identity), flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    # Written on the first start; every later start (--resume) must reproduce it.
    replay.assert_identity(dest / "identity.json", identity)
    if (dest / "result.json").exists():
        print("completed final student preserved", flush=True)
        return
    import torch

    from cuhkx.budget import pack_deliverable, save_deliverable
    from cuhkx.fused import FusedFrameDataset, labels_by_row, load_fused, make_fused_transform
    from cuhkx.replay import ReplayConcat, ReplayDataset
    from cuhkx.train import TrainConfig
    from cuhkx.tta import predict

    # Step 2: each teacher's four-pass softmax (plain, flip, frame roll by ±1) on the 405 test
    # clips in CSV order, from its own test cache with segment-centre frames and the centre crop;
    # kept in teacher-<name>.npz and reused on a later start.
    helper = import_module("44_self_training")
    posteriors = {}
    teacher_seconds = 0.0
    for name in sources:
        info = provenance[name]
        cache = Path(info["test_cache"])
        data, offsets, ids, _ = load_fused(cache)
        assert isinstance(data, np.memmap) and set(ids) == set(test_ids) and len(ids) == 405
        indexes = {c: i for i, c in enumerate(ids)}
        order = np.array([indexes[c] for c in test_ids])
        teacher_file = dest / f"teacher-{name}.npz"
        if teacher_file.exists():
            saved = np.load(teacher_file)
            assert np.array_equal(saved["clip_ids"], test_ids)
            assert str(saved["source_sha256"]) == info["source_sha256"]
            posteriors[name] = saved["probs"]
            continue
        tc = info["config"]
        dataset = FusedFrameDataset(
            data,
            offsets,
            np.zeros(len(ids), dtype=np.int64),
            order,
            tc["n_frames"],
            False,
            make_fused_transform(tc["extra"]["crop"], erase_prob=0, flip_prob=0),
            tc["seed"],
        )
        net = helper.load_state(helper.build_model(tc, data.shape[-1], "cpu"), Path(info["source"]))
        net.to(tc["device"]).eval()
        start = time.monotonic()
        # Batch 4 for ir-CSN-152 and 6 for R(2+1)D-34; the result is (405 clips, 40 classes).
        probs = predict(net, dataset, tc["device"], 4 if name == "peer" else 6, True)
        seconds = time.monotonic() - start
        teacher_seconds += seconds
        assert probs.shape == (405, 40) and np.isfinite(probs).all()
        assert np.allclose(probs.sum(1), 1, atol=1e-5)
        np.savez(
            teacher_file,
            clip_ids=np.array(test_ids),
            probs=probs,
            source_sha256=info["source_sha256"],
        )
        posteriors[name] = probs
        print(json.dumps(dict(teacher=name, seconds=seconds, clips=405)), flush=True)
        del net, dataset, data
        torch.cuda.empty_cache()
    # Soft pseudo-label of a test clip: the plain mean of both teachers' posteriors, with no
    # confidence threshold.
    pseudo = ((posteriors["own"] + posteriors["peer"]) * 0.5).astype(np.float32)
    np.savez(dest / "pseudo.npz", clip_ids=np.array(test_ids), probs=pseudo)
    # Step 3: labelled clips from the det training cache and test clips from the det test cache,
    # both with random frame picks and the recipe's training augmentation.
    data, offsets, ids, _ = load_fused(train_cache)
    assert isinstance(data, np.memmap)
    order = np.array([i for i, c in enumerate(ids) if c in table.index])
    labels = table.loc[[ids[i] for i in order], "action_id"].to_numpy()
    assert len(order) == 2931
    extra = cfg["extra"]
    tf = make_fused_transform(
        extra["crop"],
        erase_prob=extra["erase_prob"],
        flip_prob=extra["flip_prob"],
        scale_range=extra.get("scale_jitter"),
    )
    true_images = FusedFrameDataset(
        data,
        offsets,
        labels_by_row(len(ids), order, labels),
        order,
        cfg["n_frames"],
        True,
        tf,
        cfg["seed"],
    )
    test_data, test_offsets, names, _ = load_fused(Path(provenance["own"]["test_cache"]))
    test_rows = {c: i for i, c in enumerate(names)}
    # Test clips carry no label (-1); like the labelled clips, they are supervised only by the
    # targets given to ReplayDataset below.
    pseudo_images = FusedFrameDataset(
        test_data,
        test_offsets,
        np.full(len(names), -1, dtype=np.int64),
        np.array([test_rows[c] for c in test_ids]),
        cfg["n_frames"],
        True,
        tf,
        cfg["seed"],
    )
    # Label-smoothed one-hot targets of the labelled clips: 1 − ε on the true class plus ε / 40
    # on every class.
    targets = (
        np.eye(40, dtype=np.float32)[labels] * (1 - cfg["label_smoothing"])
        + cfg["label_smoothing"] / 40
    )
    # Loss weight 1 per labelled clip and 0.5 per test clip.
    dataset = ReplayConcat(
        [
            ReplayDataset(true_images, targets, np.ones(2931, dtype=np.float32)),
            ReplayDataset(pseudo_images, pseudo, np.full(405, 0.5, dtype=np.float32)),
        ]
    )
    config = TrainConfig(**cfg)
    assert asdict(config)["epochs"] == 10 and len(dataset) == 3336

    # B starts from teacher 1's weights, the R(2+1)D-34 weight average.
    def make_model():
        return helper.load_state(
            helper.build_model(cfg, data.shape[-1], "cpu"), Path(provenance["own"]["source"])
        )

    # Step 4: 10 epochs of 417 batches (4,170 optimizer steps) with 101_matched_replay.fit, which
    # writes checkpoint.pt, epoch-NN.json and student.pt.
    start = time.monotonic()
    print(
        "FULL B2 STUDENT: 2931 true + 405 pseudo; 10ep, 4170 attempts; no train accuracy gate",
        flush=True,
    )
    net, _ = replay.fit(make_model, dataset, config, dest, identity)
    training_seconds = time.monotonic() - start
    del net, dataset, true_images, pseudo_images
    torch.cuda.empty_cache()
    # Step 5: int8 pack of student.pt with the unquantised person detector and the metadata of the
    # weight average's int8 pack. This pack precedes BatchNorm re-estimation; the pack that became
    # member B is written by 108_finalize_calibrated.py.
    floor = torch.load(
        models / "t32-soup/deliverable-int8.pt", map_location="cpu", weights_only=True
    )
    detector = floor["components"]["detector"]
    assert all(
        not isinstance(v, dict) and (not v.is_floating_point() or v.dtype == torch.float32)
        for v in detector.values()
    )
    meta = copy.deepcopy(floor["meta"])
    meta.pop("soup", None)
    meta.update(
        run=dest.name,
        precision="int8",
        continuation_identity_sha256=base.sha256(dest / "identity.json"),
        transductive_training=True,
    )
    state = torch.load(dest / "student.pt", map_location="cpu", weights_only=True)
    save_deliverable({"image": state}, dest / "deliverable-int8", meta=meta, int8=True)
    save_deliverable({"detector": detector}, dest / "deliverable-int8")
    packed = pack_deliverable(dest / "deliverable-int8")
    assert packed["total_bytes"] < 100_000_000
    # submission_ready stays False; pending lists the checks still outstanding at that point.
    result = dict(
        training_seconds=training_seconds,
        teacher_seconds=teacher_seconds,
        packed=packed,
        quant_fold_evidence=str(gate_file),
        heldout_accuracy=None,
        submission_ready=False,
        pending=["complete inference.sh chain", "two cold starts", "live Kaggle quota"],
    )
    base.write_json(dest / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
