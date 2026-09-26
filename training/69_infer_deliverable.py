"""Predict the test set from a shipped deliverable directory, and nothing else.

    uv run python scripts/69_infer_deliverable.py --run csn152-fulldata-e30/deliverable-fp16

This is the path ``inference.sh`` runs, and until 2026-09-08 it did not exist for the model
this project actually ships. What existed was ``scripts/65_infer_fused.py``, which hardcodes
``PretrainedTSN``, expects seed-suffixed ``<member>-s*.pt`` weights and defaults to
``fused_test.npz`` -- the shape of the ``fused-v1`` line from early September. Every score
since 2026-09-04 comes from ``PretrainedVideo3D`` (IG65M R(2+1)D-34, then ir-CSN-152) saved
by ``budget.save_deliverable`` as a single ``image.pt``. So the graded path targeted an
architecture family and a file layout the deliverable does not use, and the S6 gate that
would have caught this had never been run.

Three things this script refuses to do:

* **Guess the architecture.** ``arch``, ``in_channels``, ``n_frames`` and ``crop`` are read
  from ``meta.json`` and missing keys are fatal. A deliverable that cannot say how to
  rebuild its own model is not a deliverable, and defaulting any of these would reproduce
  the class of bug where a wrong crop silently costs several points.
* **Recompute channel statistics.** ``PretrainedVideo3D`` registers the Kinetics mean/std as
  *persistent* buffers (``src/cuhkx/models.py``), so normalisation travels inside the
  checkpoint. Recomputing anything from the test cache would be a train/test mismatch that
  only shows up as a leaderboard score disagreeing with the local one.
* **Reimplement the averaging.** The 4-pass TTA comes from ``cuhkx.tta``, the same functions
  ``scripts/90_infer_tta.py`` uses, because the S6 gate is D = 0 against a csv that script
  produced.

Precision is read from ``meta.json`` too: ``fp32``/``fp16`` load directly and ``int8`` is
unpacked with ``budget.dequantize_state``. That last branch matters -- the incumbent soup
only fits the 100 MB cap at int8 (its fp16 is 127.24 MB), and before this script no code in
the repo loaded an int8 deliverable at all, so its weights had been measured but never read
back.
"""
# Role: development tool that rebuilds one PretrainedVideo3D classifier from a deliverable (a
# directory with meta.json and <component>.pt, or one packed file with 'meta' and 'components')
# and saves its four-pass probabilities for the clips of a fused test cache to an npz.
# Used by: training/124_c_full_student.py imports to_fp32_state (turns a possibly quantised state
# dict into float32); training only, not used by the delivered run.
# Where the docstring says inference.sh runs this path, it means an earlier inference.sh. The
# delivered inference.sh runs scripts/el25r_p1_repeat_runtime.py and does not import this file.
# This tool cannot read weights/model.pt either: its top-level dict is a zlib envelope without
# 'meta' or 'components' (cuhkx/fd18_codec.py).
# The usage line above names scripts/; the file is in training/.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the cuhkx package (src/) importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.budget import dequantize_state  # noqa: E402
from cuhkx.fused import (  # noqa: E402
    FusedFrameDataset,
    export_memmap,
    load_fused,
    make_fused_transform,
)
from cuhkx.models import Classifier, PretrainedVideo3D  # noqa: E402
from cuhkx.paths import p  # noqa: E402
from cuhkx.tta import predict  # noqa: E402

# Meta keys needed to rebuild the model: backbone name, input channels, frames per clip, crop side.
REQUIRED = ("arch", "in_channels", "n_frames", "crop")


def to_fp32_state(state: dict) -> dict:
    """An fp32 state dict, whether or not the components arrived quantised.

    The test is **structural**, not the meta's precision string: quantize_state stores each
    quantised tensor as a dict ({"_int8", "_scale"} at 8 bits, {"_packed", "_scale", "_bits",
    "_shape"} below), and the meta says "int8" for the first form but "int6"/"int5" for the
    second. Branching on `precision == "int8"` therefore skipped dequantisation for every
    bit-packed deliverable, and the raw dict reached `.is_floating_point()` and died there.
    The packer had five unit tests and had still never been run through the graded entrypoint.
    """
    if any(isinstance(v, dict) for v in state.values()):
        state = dequantize_state(state)
    # Floating tensors become float32; integer tensors such as BatchNorm counters are unchanged.
    return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}


def load_weights(path: Path, precision: str) -> dict:
    """Read one component file back into an fp32 state dict."""
    del precision  # kept for call-site compatibility; the format is read off the state itself
    return to_fp32_state(torch.load(path, map_location="cpu", weights_only=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="deliverable directory, relative to models/")
    ap.add_argument("--cache", default=None, help="override meta.json's test cache")
    ap.add_argument("--component", default="image", help="<name>.pt inside the deliverable")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="npz path; default test_probs.npz in the run")
    ap.add_argument(
        "--no-tta",
        action="store_true",
        help="single pass instead of the 4-pass average. Off by default because every csv "
        "this project has submitted since 2026-09-06 used the four passes",
    )
    args = ap.parse_args()

    # Step 1: locate the deliverable and read its meta (and, for a packed file, its components).
    # A deliverable is either the directory budget.save_deliverable writes, or the single
    # checkpoint file budget.pack_deliverable collapses it into. The organisers' C-4 wording
    # asks for one file, so the packed form is what ships; the directory form stays readable
    # because it is what the size gate and the drill were built against.
    # Accept either a name under models/ ("t32-soup/deliverable-int8") or a path that already
    # points at one ("models/t32-soup/deliverable-int8"). Both conventions are in use --
    # 26_build_test_caches.py and 49_cold_start_drill.py take real paths, this took a name --
    # and joining a path onto models_root silently produced models/models/... , which then
    # failed as "not a deliverable directory" and read like a packaging fault rather than a
    # path one. Prefer the literal path when it exists so neither caller has to know.
    target = Path(args.run)
    if not target.exists():
        target = p("models_root") / args.run
    packed = None
    if target.is_file():
        # Packed form: one torch file {'meta': dict, 'components': {name: state dict}}.
        blob = torch.load(target, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or "meta" not in blob or "components" not in blob:
            raise SystemExit(
                f"{target.name} is not a packed deliverable (expected keys 'meta' and 'components')"
            )
        meta, packed = blob["meta"], blob["components"]
        run_dir = target.parent
        print(f"packed deliverable {target.name}: components {sorted(packed)}")
    else:
        # Directory form: meta.json plus one <component>.pt per component.
        run_dir = target
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"no meta.json in {run_dir}; this is not a deliverable directory")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Step 2: check the meta, then load the test cache it names (or the one given by --cache).
    missing = [k for k in REQUIRED if meta.get(k) is None]
    if missing:
        raise SystemExit(
            f"meta.json is missing {missing}; refusing to guess. A deliverable has to "
            f"record how to rebuild its own model."
        )

    precision = str(meta.get("precision", "fp32"))
    cache_name = args.cache or meta.get("test_cache")
    if not cache_name:
        raise SystemExit("no test cache: pass --cache or record test_cache in meta.json")

    cache_path = p("cache_root") / cache_name
    # Make a memory-mappable .npy copy beside the cache if it is missing, then map it. `data` is
    # (total frames, H, W, C) uint8; clip j's frames are data[offsets[j]:offsets[j + 1]], and
    # `build` lists the "key=value" settings the cache was built with.
    export_memmap(cache_path)
    data, offsets, clip_ids, build = load_fused(cache_path)
    print(f"cache {cache_name} {build}\n  clips {len(clip_ids)} rows {len(data)}")

    expected_build = meta.get("test_cache_build")
    if expected_build and build:
        # Compare the keys the two records share, not the lists verbatim -- the same rule
        # scripts/26_build_test_caches.py applies, and for the same reason. Two things differ
        # innocently: the window-source tally is an observation about this particular test set,
        # and the record format has gained fields over time (the 2026-09-03 fused-det cache
        # predates motion_margin / motion_quantile / motion_in_window). Verbatim comparison
        # printed a loud mismatch on a cache that was in fact built to the right recipe, on the
        # first genuine cold-start run. A warning that cries wolf on the correct case is worse
        # than none: on the day, it trains the operator to ignore the one that matters.
        # Loud but not fatal, because a submission that is probably fine beats no submission
        # inside a two-hour window -- unlike at build time, where there is room to fix it.
        # "key=value" entries -> {key: value}. For "windows" only the text before ":" is kept;
        # the rest is the tally of window sources on this particular test set.
        def _keys(entries):
            out = {}
            for entry in (str(e) for e in entries):
                key, _, value = entry.partition("=")
                out[key] = value.split(":", 1)[0] if key == "windows" else value
            return out

        want, got = _keys(expected_build), _keys(build)
        clashes = [k for k in want.keys() & got.keys() if want[k] != got[k]]
        clashes += [f"{k} (absent from the built cache)" for k in want.keys() - got.keys()]
        if clashes:
            print("!! test cache build differs from the one recorded in the deliverable")
            print(f"   disagrees on: {sorted(clashes)}")
            print(f"   recorded: {expected_build}")
            print(f"   actual:   {build}")

    n_channels = int(meta["in_channels"])
    if n_channels != data.shape[-1]:
        raise SystemExit(f"deliverable expects {n_channels} channels, cache has {data.shape[-1]}")

    # Step 3: rebuild the network from the meta and load the requested component's weights,
    # dequantised to float32 where they are stored quantised.
    model = Classifier(
        PretrainedVideo3D(
            arch=str(meta["arch"]),
            in_channels=n_channels,
            dropout=float(meta.get("dropout", 0.3)),
            pretrained=False,  # weights are about to be loaded; skip the download
        )
    )
    if packed is not None:
        if args.component not in packed:
            raise SystemExit(f"no component {args.component!r}; file has {sorted(packed)}")
        model.load_state_dict(to_fp32_state(packed[args.component]))
        source = f"{target.name}[{args.component}]"
    else:
        weights_path = run_dir / f"{args.component}.pt"
        if not weights_path.exists():
            raise SystemExit(f"no {weights_path.name} in {run_dir}")
        model.load_state_dict(load_weights(weights_path, precision))
        source = weights_path.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"model {meta['arch']}  {source} ({precision})  T={meta['n_frames']} crop={meta['crop']}")

    # Step 4: four-pass test-time averaging (one pass with --no-tta) over every cache row, with
    # segment-centre frames and a centre crop; the dataset's labels are placeholders. `probs` is
    # (n_clips, 40).
    transform = make_fused_transform(int(meta["crop"]), erase_prob=0.0, flip_prob=0.0)
    dataset = FusedFrameDataset(
        data,
        offsets,
        np.zeros(len(clip_ids), dtype=np.int64),
        np.arange(len(clip_ids)),
        int(meta["n_frames"]),
        train=False,
        transform=transform,
    )
    probs = predict(model, dataset, device, args.batch_size, tta=not args.no_tta)

    # A clip the cache holds no frames for is fed a block of zeros (fused.empty_clip), so the
    # model returns a confident-looking distribution derived from nothing. Averaging that into
    # a fusion is worse than having no opinion at all, so those rows are dropped and the clip
    # is simply absent from this member -- which is what 64_combine_probs.py --allow-partial
    # is for. 10 of the 405 test clips ship no Thermal directory; the Depth+IR caches cover
    # all 405, so for every candidate scored so far this is a no-op.
    # covered[j] is True when cache row j holds at least one frame.
    covered = np.asarray(offsets[1:] > offsets[:-1])
    dropped = [c for c, ok in zip(clip_ids, covered, strict=True) if not ok]
    if dropped:
        probs = probs[covered]
        clip_ids = [c for c, ok in zip(clip_ids, covered, strict=True) if ok]
        print(
            f"\n!! {len(dropped)} clip(s) have no cached frames and are OMITTED from this "
            f"member rather than predicted from zeros: {dropped[:12]}"
        )
        print(
            "   fuse with scripts/64_combine_probs.py --allow-partial. This member alone "
            "cannot make a submission: it is short of a full set of clips."
        )

    # Step 5: save the probabilities (under two keys), the clip ids and a JSON summary of the run.
    out = Path(args.out) if args.out else run_dir / "test_probs.npz"
    np.savez(
        out,
        **{"member:image_tta": probs},
        # `fused` is what scripts/62_infer_from_deliverable.py --reuse-probs reads, and it is
        # what scripts/78_fallback_drill.py drills. With one member the fused distribution IS
        # that member, so this is the same array under the name the decoding path expects --
        # not a placeholder.
        fused=probs,
        clip_ids=np.array(clip_ids),
        meta=np.array(
            [
                json.dumps(
                    {
                        "deliverable": str(args.run),
                        "component": args.component,
                        "precision": precision,
                        "arch": meta["arch"],
                        "n_frames": meta["n_frames"],
                        "crop": meta["crop"],
                        "cache": cache_name,
                        "cache_build": build,
                        "omitted_empty_clips": dropped,
                        "tta": "plain+hflip+roll+1+roll-1, softmax mean"
                        if not args.no_tta
                        else "single pass",
                    }
                )
            ]
        ),
    )
    print(f"\n  {probs.shape} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
