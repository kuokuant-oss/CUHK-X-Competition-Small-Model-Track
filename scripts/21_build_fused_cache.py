"""Build the pixel-aligned Depth+IR cache that early fusion needs.

    uv run python -u scripts/21_build_fused_cache.py --split train test --bbox-scale 1.4

Writes ``data/interim/fused<suffix>_<split>.npz`` as ``(n, H, W, 4)`` uint8 -- Depth_Color
RGB in channels 0-2, IR in channel 3, one shared crop window and one shared set of frame
picks. See ``cuhkx.fused`` for why any of that matters.

This is a *new* script rather than a branch inside ``20_build_cache.py`` on purpose: the
existing caches are what ``models/v5`` and ``submissions/cand_b.csv`` (LB 0.56716) were
built from, and that fallback must stay byte-reproducible while this line is unproven.

Every knob that changes the cache is recorded in the npz itself, because the project has
already been bitten once by a setting that survived only in a run directory's name.
"""
# Role: builds one fused cache for the training or test clips: uint8 frames (Depth_Color R, G, B
#   and IR) cut with one window per clip and resized, stored with the recipe that built them.
# Used by: scripts/fd18_raw_builder.py, which the inference entry point runs once per view (det,
#   miw, det248); both (the training caches were built with it).

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cuhkx library in src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.fused import FUSED_SIZE, MAX_FRAMES, build_fused_cache  # noqa: E402
from cuhkx.paths import p  # noqa: E402


# {clip id: folder} for the training clips that have frames of this modality (clip index).
def train_dirs(modality_dir: str) -> dict[str, Path]:
    index = pd.read_parquet(p("processed_root") / "clip_index_listing.parquet")
    root = p("train_root") / modality_dir
    present = index[index[f"n_{modality_dir}"] > 0]
    return {row.clip_id: root / row.clip_id for row in present.itertuples()}


# {clip id: folder} for every SM_test_* clip folder under test_root.
def test_dirs(modality_dir: str) -> dict[str, Path]:
    root = p("test_root")
    return {
        clip.name: clip / modality_dir for clip in sorted(root.glob("SM_test_*")) if clip.is_dir()
    }


def existing_size(out_dir: Path, split: str) -> int | None:
    """The ``size=`` a previously built cache for this split recorded, if there is one.

    Read from the memmap sidecar when it exists, because that is a few hundred bytes; the
    npz would be gigabytes.
    """
    # Only the cache without a --suffix (fused_<split>) is consulted.
    for candidate in (
        out_dir / f"fused_{split}_meta.npz",
        out_dir / f"fused_{split}.npz",
    ):
        if not candidate.exists():
            continue
        try:
            blob = np.load(candidate, allow_pickle=False)
        except (OSError, ValueError):
            continue
        for entry in (str(x) for x in blob.get("build", [])):
            if entry.startswith("size="):
                return int(entry.split("=", 1)[1])
    return None


def _resolve_size(args, out_dir: Path) -> int:
    """Default ``--size`` to the existing cache's, and refuse a silent change.

    This exists because of a near miss on 2026-09-03. The deployed cache is **144 px** and
    training passes ``--crop 128``; the 16 px of slack *is* the spatial augmentation, since
    ``fused.make_fused_transform`` random-crops from cache size down to crop size. Rebuilding
    at ``FUSED_SIZE[0]`` (128) would have made ``random_crop(frames, 128)`` a no-op on a
    128 px frame -- **spatial augmentation silently switched off**, no error, no warning, and
    a fold-0 number no longer comparable to the 0.6658 it is meant to be measured against.

    The class of mistake is the one this project keeps paying for: a default that is right in
    isolation and wrong against the artefacts already on disk. So the default is now read
    from those artefacts, and a deliberate change has to say so.
    """
    known = {s: existing_size(out_dir, s) for s in args.split}
    seen = {v for v in known.values() if v is not None}
    if args.size is None:
        if len(seen) > 1:
            raise SystemExit(
                f"existing caches disagree on size {sorted(seen)}; pass --size explicitly"
            )
        resolved = seen.pop() if seen else FUSED_SIZE[0]
        print(f"  size={resolved} (from existing cache)" if seen or known else "")
        return resolved
    mismatched = {s: v for s, v in known.items() if v is not None and v != args.size}
    if mismatched and not args.allow_size_change:
        raise SystemExit(
            f"--size {args.size} differs from the existing cache(s) {mismatched}. The spatial "
            f"augmentation is the gap between cache size and --crop, so changing this "
            f"silently removes or changes it. Pass --allow-size-change if that is the intent."
        )
    return args.size


def main() -> None:
    # Step 1: options. The inference entry point passes the recipe stored in the checkpoint,
    # e.g. --size 144 --windows person_windows for det (see scripts/26_build_test_caches.py).
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    ap.add_argument(
        "--size",
        type=int,
        default=None,
        help="output side in pixels. Defaults to whatever the existing cache for this split "
        "was built at, so a rebuild stays comparable to the runs that used it; pass a value "
        "only when deliberately changing the resolution (and expect --crop to move with it)",
    )
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="multiply the motion box about its centre. 1.0 reproduces the existing crop; "
        "larger pulls in the static scene objects that define several classes",
    )
    ap.add_argument(
        "--whole-frame",
        action="store_true",
        help="skip the crop entirely -- the other end of the --bbox-scale ablation",
    )
    ap.add_argument(
        "--windows",
        default=None,
        help="stem of the person-window tables from scripts/22_person_windows.py (e.g. "
        "'person_windows'); each clip is then cropped to its detector box instead of its "
        "motion box. --bbox-scale does not apply to those clips -- the margin is already "
        "baked in at detection time (person.CROP_MARGIN), and applying both would compound "
        "two margins with nothing recording that it happened",
    )
    ap.add_argument(
        "--motion-margin",
        type=float,
        default=0.25,
        help="padding of the motion box as a fraction of its side (depth.motion_bbox default "
        "0.25). 0.05 builds the tight 'what moves' view: hands and object for seated classes",
    )
    ap.add_argument(
        "--motion-quantile",
        type=float,
        default=0.98,
        help="motion-magnitude quantile that defines the moving region (depth.motion_bbox)",
    )
    ap.add_argument(
        "--motion-in-window",
        action="store_true",
        help="with --windows: crop to the moving region found INSIDE each detector box "
        "(hands/arms/object for seated classes); falls back to the box when degenerate",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--suffix", default="", help="appended to the cache filename")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on this many clips")
    ap.add_argument(
        "--allow-size-change",
        action="store_true",
        help="permit a --size that differs from the existing cache. Required, because the "
        "spatial augmentation lives in the gap between cache size and --crop, and closing "
        "that gap silently removes it",
    )
    args = ap.parse_args()

    # Step 2: output folder, and the side every frame is resized to.
    out_dir = p("cache_root")
    args.size = _resolve_size(args, out_dir)
    size = (args.size, args.size)

    # Step 3: per split, the Depth_Color and IR folders; the clip list follows Depth_Color, and
    # --limit keeps its first clips.
    for split in args.split:
        depth = train_dirs("Depth_Color") if split == "train" else test_dirs("Depth_Color")
        ir = train_dirs("IR") if split == "train" else test_dirs("IR")
        if args.limit is not None:
            depth = dict(list(depth.items())[: args.limit])

        # Step 4: crop windows. With --windows, the per-clip person windows (top, left, bottom,
        # right) from processed_root/<windows>_<split>.parquet, and the build record keeps the
        # table name plus the tally of window sources. Without --windows (and for a clip with no
        # row), the depth motion box, or the whole frame with --whole-frame.
        windows, window_note = None, "windows=motion"
        if args.windows:
            table = pd.read_parquet(p("processed_root") / f"{args.windows}_{split}.parquet")
            windows = {
                row.clip_id: (int(row.top), int(row.left), int(row.bottom), int(row.right))
                for row in table.itertuples()
            }
            missing = sorted(set(depth) - set(windows))
            shares = table["source"].value_counts().to_dict()
            window_note = f"windows={args.windows}:{shares}"
            print(f"  crop windows from {args.windows}_{split}.parquet: {shares}")
            if missing:
                # Not fatal: a clip with no row falls through to the motion box, which is
                # exactly the crop it had before this option existed. Say so rather than
                # letting a silently unmatched clip_id convention turn the whole run into
                # the old behaviour with a new filename.
                print(
                    f"  !! {len(missing)} clips have no window row, using motion box: {missing[:5]}"
                )

        # Step 5: decode, crop and resize every clip on a thread pool (fused.build_fused_cache):
        # one uint8 array of all frames, shape (frames, size, size, 4), per-clip offsets and a
        # build report.
        start = time.time()
        cache, report = build_fused_cache(
            depth,
            ir,
            p("processed_root") / "depth_lut.npz",
            size=size,
            workers=args.workers,
            max_frames=args.max_frames,
            bbox_scale=args.bbox_scale,
            whole_frame=args.whole_frame,
            windows=windows,
            motion_margin=args.motion_margin,
            motion_quantile=args.motion_quantile,
            motion_in_window=args.motion_in_window,
        )

        # Step 6: save cache_root/fused<suffix>_<split>.npz: the frames, the offsets (clip i owns
        # rows offsets[i]:offsets[i + 1]), the clip ids, and the build record that the inference
        # entry point compares with the recipe stored in the checkpoint.
        out = out_dir / f"fused{args.suffix}_{split}.npz"
        np.savez(
            out,
            data=cache.data,
            offsets=cache.offsets,
            clip_ids=np.array(cache.clip_ids),
            build=np.array(
                [
                    f"size={args.size}",
                    f"max_frames={args.max_frames}",
                    f"bbox_scale={args.bbox_scale}",
                    f"whole_frame={args.whole_frame}",
                    f"motion_margin={args.motion_margin}",
                    f"motion_quantile={args.motion_quantile}",
                    f"motion_in_window={args.motion_in_window}",
                    window_note,
                    "channels=depth_color_rgb+ir",
                ]
            ),
        )

        # Step 7: print a summary of the build report (frame counts, crop sides, motion-box
        # failures, unreadable IR, unaligned or empty clips).
        native = np.array(report["native"])
        sides = np.array(report["sides"]) if report["sides"] else np.array([0])
        held = np.array([len(cache.get(i)) for i in range(len(cache))])
        print(
            f"\n=== fused {split} -> {out.name}  {out.stat().st_size / 1e6:.1f} MB  "
            f"{time.time() - start:.0f}s ==="
        )
        print(f"  clips {len(cache)}   empty {int((held == 0).sum())}   rows {len(cache.data)}")
        print(
            f"  native frames/clip  min {native.min()}  med {int(np.median(native))}  "
            f"max {native.max()}   capped at {args.max_frames}: "
            f"{int((native > args.max_frames).sum())} clips "
            f"({(native > args.max_frames).mean() * 100:.1f}%)"
        )
        print(
            f"  crop side (px, pre-resize)  min {sides.min()}  med {int(np.median(sides))}  "
            f"max {sides.max()}"
        )
        print(
            f"  degenerate boxes (motion cue failed -> whole frame): "
            f"{len(report['degenerate'])} "
            f"({len(report['degenerate']) / max(1, len(cache)) * 100:.1f}%)"
        )
        if report["ir_dead"]:
            print(f"  IR unreadable, channel zeroed, clip KEPT: {report['ir_dead']}")
        if report["unaligned"]:
            print(
                f"  !! depth/IR frame counts differ or frames unreadable: "
                f"{len(report['unaligned'])} clips"
            )
            for row in report["unaligned"][:10]:
                print(f"       {row[0]}  depth={row[1]} ir={row[2]} skipped={row[3]}")
            if len(report["unaligned"]) > 10:
                print(f"       ... and {len(report['unaligned']) - 10} more")
        if report["empty"]:
            print(f"  !! empty clips ({len(report['empty'])}): {report['empty'][:10]}")


if __name__ == "__main__":
    main()
