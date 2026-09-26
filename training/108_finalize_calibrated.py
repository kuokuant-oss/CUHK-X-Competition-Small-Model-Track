"""Transfer a quantization-qualified BNcal/T2 recipe to one full-data checkpoint."""

# Role: BatchNorm re-estimation of member B (option B2-BNcal): B's weights from
#   103_b_final_student.py keep their learned parameters, the running statistics are re-estimated
#   on 1,600 training clips, and the result is written as student.pt and an int8 pack.
# Used by: run by hand after 103_b_final_student.py; training. The B2-BNcal pack
#   (abc-b2-bncal-final-20260911/deliverable-int8.pt) is member B before it was merged into
#   weights/model.pt. Option T2 first averages B 50/50 with a second full-data model trained from
#   the same weight average; T2 is not part of the delivered model.
# Produces, under models_root/abc-b2-bncal-final-20260911/ (T2: abc-t2-final-20260911/):
#   identity.json, student.pt, deliverable-int8.pt and result.json; it also adds its run time to
#   abc-interp-20260911/compute-budget.json.
# Inputs, not included in the repository (TRAINING.md lists them): under models_root, the
#   four-fold validation records and compute budget in abc-interp-20260911/, B's student.pt and
#   training-audit.json in abc-b2-final-20260911/, ig65m-t32-det-60ep/result.json and oof.npz,
#   the int8 pack t32-soup/deliverable-int8.pt and, for T2, abc-a1-for-t2-20260911/ and
#   t32-soup/soup.pt; the det training cache; folds.parquet. models_root is not defined in the
#   shipped configs/paths.yaml.
# The source files listed in the four fold records must still have their recorded SHA256, so the
#   script documents the procedure rather than running out of the box.

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Make cuhkx (src/), scripts/ and the training modules importable; module names that start with
# a digit are loaded with import_module.
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"),str(ROOT/'training')]
base = import_module("47_matched_continuation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # B2-BNcal: re-estimate B's BatchNorm statistics (member B). T2: average B with a second
    # full-data model first (not used by the delivered model).
    ap.add_argument("arm", choices=["B2-BNcal", "T2"])
    args = ap.parse_args()
    # Refuse to start while any other Python process runs; the whole invocation's time is added
    # to the shared compute budget at the end.
    base.assert_idle()
    invocation_started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from cuhkx.budget import pack_deliverable, save_deliverable
    from cuhkx.fused import FusedFrameDataset, labels_by_row, load_fused, make_fused_transform
    from cuhkx.interpolation import calibrate_bn, interpolate_state
    from cuhkx.paths import p
    from cuhkx.train import _atomic_save, _collate, seed_everything

    # Step 1: at least 180 s must remain in the compute budget shared by both options, and the
    # four-fold validation summary must mark the option's int8 version as eligible.
    models = p("models_root")
    budget_file = models / "abc-interp-20260911/compute-budget.json"
    budget = json.loads(budget_file.read_text())
    if budget["seconds"] + 180 > budget["limit_seconds"]:
        raise ValueError("insufficient registered T1/T2 budget for final calibration and pack")
    gate_file = models / "abc-interp-20260911/quant-fourfold.json"
    gate = json.loads(gate_file.read_text())
    if not gate["decisions"][args.arm + "-int8"]["eligible"]:
        raise ValueError("candidate did not pass the registered quantized accuracy gate")
    # All fold sources and measured implementation remain the ones that passed the gate.
    for fold in range(4):
        identity_file = (
            models / f"abc-interp-20260911/fold{fold}/{args.arm}/quant-audit/identity.json"
        )
        identity = json.loads(identity_file.read_text())
        for path, digest in identity["code_sha256"].items():
            assert base.sha256(ROOT / path) == digest
    # Step 2: B's weights from 103_b_final_student.py must match the hash in its training record.
    # For B2-BNcal both ends of the average are B; for T2 the second model and B.
    b_dir = models / "abc-b2-final-20260911"
    b_source = b_dir / "student.pt"
    assert (
        base.sha256(b_source)
        == json.loads((b_dir / "training-audit.json").read_text())["student_sha256"]
    )
    left_source = (
        b_source if args.arm == "B2-BNcal" else models / "abc-a1-for-t2-20260911/student.pt"
    )
    if not left_source.is_file():
        raise FileNotFoundError("T2 requires its independently trained full-data A1")
    # T2's second model must match its recorded hash, have been trained for 10 epochs from the
    # R(2+1)D-34 weight average within its compute budget, and name the same validation summary.
    if args.arm == "T2":
        a_result = json.loads((left_source.parent / "result.json").read_text())
        a_identity = json.loads((left_source.parent / "identity.json").read_text())
        assert base.sha256(left_source) == a_result["student_sha256"]
        assert a_result["fixed_epoch"] == 10 and a_result["within_compute_budget"]
        assert a_identity["quant_gate_sha256"] == base.sha256(gate_file)
        assert a_identity["source_sha256"] == base.sha256(models / "t32-soup/soup.pt")
    dest = models / (
        "abc-b2-bncal-final-20260911" if args.arm == "B2-BNcal" else "abc-t2-final-20260911"
    )
    # No resume: an existing output directory is never overwritten.
    if dest.exists():
        raise FileExistsError("preserve existing final candidate")
    # Step 3: the 60-epoch run's configuration (architecture and cache), the det training cache
    # (memory-mapped), the 2,931 labelled clips and 1,600 of them drawn with seed 42 for the
    # re-estimation; 124_c_full_student.py checks that C uses the same 1,600 clips.
    cfg = json.loads((models / "ig65m-t32-det-60ep/result.json").read_text())["config"]
    cfg["extra"].pop("subjects", None)
    cache = p("cache_root") / cfg["extra"]["cache"]
    for suffix in ("_data.npy", "_meta.npz"):
        assert cache.with_name(cache.stem + suffix).is_file()
    data, offsets, ids, _ = load_fused(cache)
    assert isinstance(data, np.memmap)
    ref = np.load(models / "ig65m-t32-det-60ep/oof.npz")
    table = (
        pd.read_parquet(p("processed_root") / "folds.parquet")
        .set_index("clip_id")
        .loc[ref["clip_ids"]]
    )
    assert len(table) == 2931 and np.array_equal(table.action_id, ref["labels"])
    order = np.array([i for i, c in enumerate(ids) if c in table.index])
    assert len(order) == 2931
    chosen = np.random.default_rng(42).permutation(order)[:1600]
    labels = table.loc[[ids[i] for i in order], "action_id"].to_numpy()
    code_files = [
        Path(__file__),
        ROOT / "src/cuhkx/interpolation.py",
        ROOT / "src/cuhkx/fused.py",
        ROOT / "src/cuhkx/models.py",
        ROOT / "src/cuhkx/budget.py",
    ]
    # Identity record: both ends and their hashes, alpha = 0.5, the calibration clips and
    # settings, and the SHA256 of the source files; written before the work starts.
    identity = dict(
        arm=args.arm,
        left=str(left_source),
        right=str(b_source),
        left_sha256=base.sha256(left_source),
        right_sha256=base.sha256(b_source),
        alpha=0.5,
        quant_gate_sha256=base.sha256(gate_file),
        calibration_ids=[ids[i] for i in chosen],
        seed=42,
        batch=8,
        max_batches=200,
        calibration="train only, reset+cumulative average, eval frames/crop, dropout off",
        code_sha256={str(q.relative_to(ROOT)): base.sha256(q) for q in code_files},
        heldout_accuracy=None,
    )
    dest.mkdir()
    base.write_json(dest / "identity.json", identity)
    seed_everything(42, deterministic=True)
    # Step 4: the network with the parameter average of both ends. interpolate_state resets the
    # BatchNorm running statistics (mean 0, variance 1) and copies the other buffers, which must
    # agree; for B2-BNcal both ends are B, so the learned parameters stay B's.
    helper = import_module("44_self_training")
    net = helper.build_model(cfg, data.shape[-1], "cpu")
    left = torch.load(left_source, map_location="cpu", weights_only=True)
    right = torch.load(b_source, map_location="cpu", weights_only=True)
    net.load_state_dict(interpolate_state(net, left, right, 0.5))
    del left, right
    net.to(cfg["device"])
    # Calibration data: the 1,600 clips with 32 segment-centre frames and a centre 128² crop, as
    # at inference; no augmentation.
    dataset = FusedFrameDataset(
        data,
        offsets,
        labels_by_row(len(ids), order, labels),
        chosen,
        32,
        False,
        make_fused_transform(128, erase_prob=0, flip_prob=0),
        42,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=_collate)
    started = time.monotonic()
    # Running statistics as a cumulative average over 200 batches of 8, with dropout off and no
    # gradients; the learned parameters are not touched.
    calibration = calibrate_bn(net, loader, max_batches=200, device=cfg["device"])
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    _atomic_save(state, dest / "student.pt")
    del net, loader, dataset
    torch.cuda.empty_cache()
    # Step 5: int8 pack of the calibrated network with the unquantised person detector and the
    # metadata of the weight average's int8 pack; it must be under 100,000,000 bytes.
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
        calibrated_identity_sha256=base.sha256(dest / "identity.json"),
    )
    save_deliverable({"image": state}, dest / "deliverable-int8", meta=meta, int8=True)
    save_deliverable({"detector": detector}, dest / "deliverable-int8")
    packed = pack_deliverable(dest / "deliverable-int8")
    assert packed["total_bytes"] < 100_000_000
    # submission_ready stays False; pending lists the checks still outstanding at that point.
    result = dict(
        arm=args.arm,
        seconds=time.monotonic() - started,
        calibration=calibration,
        packed=packed,
        quantized_fold_evidence=str(gate_file),
        heldout_accuracy=None,
        submission_ready=False,
        pending=["full inference chain", "two cold starts", "live quota and preregistration"],
    )
    # Add this invocation's time to the shared compute budget.
    budget["seconds"] += time.monotonic() - invocation_started
    result["shared_budget_seconds"] = budget["seconds"]
    base.write_json(budget_file, budget)
    base.write_json(dest / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
