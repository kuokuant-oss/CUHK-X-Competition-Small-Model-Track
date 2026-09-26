"""Detect one person crop window per clip, from IR, and write it to a parquet.

    uv run python -u scripts/22_person_windows.py --split train test

Writes ``data/processed/person_windows_<split>.parquet`` with one row per clip:
``clip_id, top, left, bottom, right, n_detected, source``. ``21_build_fused_cache.py
--windows`` reads it.

Two passes rather than detecting inside the cache builder, for three reasons: the detector
wants the GPU while the cache builder is a thread pool of PNG decodes; the windows are the
thing worth eyeballing and diffing, so they should be a file; and a rebuilt cache with a
different resize should reuse the same boxes rather than re-detect and drift.

The fallback chain is explicit and recorded per clip, so ``source`` counts are auditable:

    IR, full frame  ->  IR, four overlapping tiles + contrast stretch  ->  motion box
                                                                         ->  whole frame

The middle rung exists because the misses are not random: measured on the 123 train clips the
full-frame pass lost, every one has readable IR and all of them are the same scene -- a dim
room with the actor ~50 px tall, i.e. ~25 px after the detector's 320 px resize. Tiling makes
the actor 1.7x larger relative to that input and recovers 89.4% of them. It costs four extra
forward passes and about 4% of clips reach it. ``--no-tile-rescue`` is the control arm.

The motion rung is ``depth.motion_bbox``, i.e. exactly what every earlier fused cache used,
so a clip that the detector misses is no worse off than it was before this script existed.
See ``cuhkx.person`` for why IR, why not Depth_Color, and why not YOLO.
"""
# Role: one square person window per clip from the IR frames, through the detector cascade of
#   cuhkx.person (full frame, four contrast-stretched tiles, depth motion box, whole frame).
# Used by: scripts/el25r_p1_repeat_runtime.py (in-process, test split, detector taken from
#   weights/model.pt); both (the training caches were cropped with its windows too).

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cuhkx library in src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx import depth as depth_codec  # noqa: E402
from cuhkx import person as person_mod  # noqa: E402
from cuhkx.data import IMAGE_SUFFIXES, _decode_frames  # noqa: E402
from cuhkx.fused import square_window  # noqa: E402
from cuhkx.paths import p  # noqa: E402


# Sorted image files in one modality folder; empty when the folder is missing.
def frame_files(clip_dir: Path) -> list[Path]:
    if not clip_dir.is_dir():
        return []
    return sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


# {clip id: modality folder}. Test: every SM_test_* folder under test_root. Training: the clips
# that have frames of this modality according to the clip index.
def clip_dirs(split: str, modality: str) -> dict[str, Path]:
    if split == "test":
        root = p("test_root")
        return {c.name: c / modality for c in sorted(root.glob("SM_test_*")) if c.is_dir()}
    index = pd.read_parquet(p("processed_root") / "clip_index_listing.parquet")
    root = p("train_root") / modality
    present = index[index[f"n_{modality}"] > 0]
    return {row.clip_id: root / row.clip_id for row in present.itertuples()}


def motion_window(depth_dir: Path, table: np.ndarray) -> tuple[tuple[int, int, int, int], bool]:
    """The pre-existing fallback: ``depth.motion_bbox`` on LUT-decoded depth, squared.

    Returns ``(window, degenerate)``. Identical parameters to ``fused.load_clip_fused`` --
    step 2, ``min_side=16`` in strided pixels -- so a clip that lands here gets the same
    crop it would have got before this script existed.
    """
    # Without depth frames, or with any unreadable one, the window is the whole 480x640 frame,
    # flagged as degenerate.
    files = frame_files(depth_dir)
    if not files:
        return (0, 0, 480, 640), True
    # Same cap as the cache builder: at most 32 evenly spaced frames.
    if len(files) > 32:
        files = [files[i] for i in np.linspace(0, len(files) - 1, 32).round().astype(int)]
    frames, skipped = _decode_frames(files, convert="RGB")
    if skipped or not frames:
        return (0, 0, 480, 640), True
    # Decode depth with the palette table and find the motion box on every 2nd pixel; the box is
    # scaled back to full resolution and made square (scale 1.0).
    stack = np.stack(frames)
    decoded = np.stack([depth_codec.decode(f, table)[0] for f in stack])
    valid = decoded != depth_codec.INVALID
    step = 2
    top, left, bottom, right = depth_codec.motion_bbox(
        decoded[:, ::step, ::step], valid[:, ::step, ::step], min_side=32 // step
    )
    # motion_bbox returns the whole (strided) frame when the motion cue fails.
    whole = (0, 0, decoded.shape[1] // step, decoded.shape[2] // step)
    degenerate = (top, left, bottom, right) == whole
    window = square_window(top * step, left * step, bottom * step, right * step, 1.0)
    return window, degenerate


def main() -> None:
    # Step 1: options. The defaults come from cuhkx.person: 8 probe frames, margin 1.4, window
    # side at least 0.35 x frame width, minimum score 0.10. --batch is accepted but not used.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    ap.add_argument("--probe-frames", type=int, default=person_mod.DETECTION_FRAMES)
    ap.add_argument("--margin", type=float, default=person_mod.CROP_MARGIN)
    ap.add_argument("--min-side-fraction", type=float, default=person_mod.MIN_SIDE_FRACTION)
    ap.add_argument("--min-score", type=float, default=person_mod.MIN_SCORE)
    ap.add_argument("--batch", type=int, default=16, help="probe frames per detector call")
    ap.add_argument(
        "--no-tile-rescue",
        action="store_true",
        help="skip the four-tile rescue pass for clips the plain pass misses -- the control "
        "arm for what it is worth, and the way to reproduce the pre-2026-09-03 windows",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument(
        "--detector-weights",
        default=None,
        help="load the detector from this checkpoint instead of torchvision's download "
        "cache. This is the graded path: the packaged detector is the one the organisers "
        "count inside the 100 MB budget, so it has to be the one that actually runs, and "
        "the finals machine cannot be assumed to have the download cache or a network",
    )
    args = ap.parse_args()

    # Step 2: the detector (from --detector-weights when given, else torchvision's COCO weights)
    # and the depth palette table for the motion fallback.
    model, device = person_mod.load_detector(weights_path=args.detector_weights)
    table = depth_codec.load_lut(p("processed_root") / "depth_lut.npz")
    source = args.detector_weights or "torchvision COCO_V1 (download cache)"
    print(f"detector ssdlite320_mobilenet_v3_large on {device}  weights: {source}")

    # Step 3: per split, every clip with IR or Depth_Color frames (test: every SM_test_* folder).
    for split in args.split:
        ir_dirs = clip_dirs(split, "IR")
        depth_dirs = clip_dirs(split, "Depth_Color")
        clip_ids = sorted(ir_dirs.keys() | depth_dirs.keys())
        if args.limit:
            clip_ids = clip_ids[: args.limit]
        start = time.time()

        # Probe frames of one clip: up to --probe-frames (8) evenly spaced IR frames as grayscale,
        # stacked to (n, H, W) uint8; an empty array when none is readable. The default argument
        # binds this split's folders.
        def load_probes(clip_id: str, ir_dirs: dict[str, Path] = ir_dirs):
            files = frame_files(ir_dirs.get(clip_id, Path("__missing__")))
            if not files:
                return clip_id, np.zeros((0, 0, 0), dtype=np.uint8)
            picks = person_mod.probe_indices(len(files), args.probe_frames)
            frames, _ = _decode_frames([files[i] for i in picks], convert="L")
            if not frames:
                return clip_id, np.zeros((0, 0, 0), dtype=np.uint8)
            return clip_id, np.stack(frames)

        # Step 4: the cascade per clip; counts tallies where each window came from.
        rows = []
        counts = {"strong": 0, "agree": 0, "tiled": 0, "motion": 0, "whole": 0}
        # Decode ahead of the GPU on a thread pool; the detector call is the serial part.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (clip_id, probes) in enumerate(pool.map(load_probes, clip_ids)):
                # Frame size from the probes (480x640 when there are none); full-frame pass: the
                # best person box per probe frame, then the per-clip accept rule ('strong' score
                # or 'agree': weak boxes that stay in place).
                height, width = (probes.shape[1], probes.shape[2]) if len(probes) else (480, 640)
                boxes, scores = person_mod.detect_boxes(model, device, probes, args.min_score)
                kept, reason = person_mod.accept_boxes(boxes, scores, width, args.probe_frames)
                if not len(kept) and not args.no_tile_rescue:
                    # Small, distant actor in a dark scene: four overlapping tiles at higher
                    # effective resolution. Costs four extra passes, reached by ~4% of clips.
                    boxes, scores = person_mod.detect_boxes_tiled(
                        model, device, probes, args.min_score
                    )
                    kept, reason = person_mod.accept_boxes(boxes, scores, width, args.probe_frames)
                    if len(kept):
                        reason = "tiled"
                # Accepted boxes give a square window on their median centre with side
                # max(margin x largest box side, min-side fraction x frame width), not clipped
                # to the frame; otherwise the depth motion box, or the whole frame if that fails.
                if len(kept):
                    window = person_mod.window_from_boxes(
                        kept, height, width, args.margin, args.min_side_fraction
                    )
                    source, n_hit = reason, len(kept)
                else:
                    window, degenerate = motion_window(
                        depth_dirs.get(clip_id, Path("__missing__")), table
                    )
                    source, n_hit = ("whole" if degenerate else "motion"), 0
                # Row: clip id, window (top, left, bottom, right), accepted boxes, source, and
                # the best score of the last detector pass.
                counts[source] += 1
                rows.append(
                    (
                        clip_id,
                        *window,
                        n_hit,
                        source,
                        float(scores.max()) if len(scores) else 0.0,
                    )
                )
                if (i + 1) % 250 == 0:
                    rate = (i + 1) / (time.time() - start)
                    print(f"  {i + 1}/{len(clip_ids)}  {rate:.1f} clip/s  {counts}", flush=True)

        # Step 5: write person_windows<suffix>_<split>.parquet to processed_root and print a
        # summary (sources, fallback rate, window sides, probe frames hit).
        table_out = pd.DataFrame(
            rows,
            columns=[
                "clip_id",
                "top",
                "left",
                "bottom",
                "right",
                "n_detected",
                "source",
                "best_score",
            ],
        )
        out = p("processed_root") / f"person_windows{args.suffix}_{split}.parquet"
        table_out.to_parquet(out, index=False)

        n = len(table_out)
        sides = (table_out["bottom"] - table_out["top"]).to_numpy()
        fallback = table_out["source"].isin(["motion", "whole"]).mean()
        print(f"\n=== person windows {split} -> {out.name} ===")
        print(f"  clips {n}   {time.time() - start:.0f}s")
        for key in ("strong", "agree", "tiled", "motion", "whole"):
            print(f"  {key:9s} {counts[key]:5d}  ({counts[key] / max(1, n) * 100:.1f}%)")
        print(f"  FALLBACK RATE (not detector): {fallback * 100:.1f}%   gate is <= 3.0%")
        print(
            f"  side px  min {sides.min()}  p10 {np.percentile(sides, 10):.0f}  "
            f"med {int(np.median(sides))}  p90 {np.percentile(sides, 90):.0f}  max {sides.max()}"
        )
        hits = table_out.loc[table_out["source"].isin(["strong", "agree", "tiled"]), "n_detected"]
        if len(hits):
            print(
                f"  probe frames hit (of {args.probe_frames})  med {int(hits.median())}  "
                f"all-8 {(hits == args.probe_frames).mean() * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
