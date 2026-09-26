"""Build exactly the test cache(s) a deliverable says it needs, from its own meta.json.

    uv run python -u scripts/26_build_test_caches.py --deliverable models/<run>/deliverable-int8
    uv run python -u scripts/26_build_test_caches.py --deliverable a.pt b/ --dry-run

This exists because ``inference.sh`` and the deliverable disagreed about which cache to build,
and nothing checked. ``69_infer_deliverable.py:114`` reads ``test_cache`` out of meta.json --
``fused-det248_test.npz`` for every CSN candidate, ``fused-det_test.npz`` (144 px) for the
floor soup -- while ``inference.sh`` hardcoded ``IMG_CACHE_ARGS`` to build ``-det176``. On a
cold start the file the model then asks for does not exist. Every D=0 reproduction to date
ran with ``SKIP_CACHE=1``, which is precisely the flag that skips this code path, so the
mismatch has never once been exercised.

The fix is to stop writing the build arguments down twice. ``21_build_fused_cache.py`` and
``25_build_thermal_cache.py`` both record every knob into the npz's ``build`` array, and
``save_deliverable`` copies that array into meta.json as ``test_cache_build``. So the
deliverable already carries its own recipe; this script just inverts it back into argv.

After building, the recorded recipe is compared against what actually landed in the new npz,
and a mismatch is **fatal here** -- unlike the same comparison in ``69_infer_deliverable.py``,
which only warns. A warning is right at inference time (a submission that is probably fine
beats no submission at all, on a two-hour clock). It is wrong at build time, where there is
still time to fix the cause.
"""
# Role: turns the cache recipe stored in a model's metadata (the 'test_cache_build' list of
#   "key=value" entries) back into a 21_build_fused_cache.py command line, and compares a built
#   cache's recorded recipe with it.
# Used by: scripts/el25r_p1_repeat_runtime.py and cuhkx.fd13_inference import its helper
#   functions; inference. Its own main() is not run by the delivered pipeline.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Make the cuhkx library in src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.paths import p  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: ``build`` entry -> (CLI flag, kind). "store_true" entries emit the flag only when the
#: recorded value is True; "skip" entries describe the cache without being settable.
FUSED_FLAGS = {
    "size": ("--size", "value"),
    "max_frames": ("--max-frames", "value"),
    "bbox_scale": ("--bbox-scale", "value"),
    "whole_frame": ("--whole-frame", "store_true"),
    "motion_margin": ("--motion-margin", "value"),
    "motion_quantile": ("--motion-quantile", "value"),
    "motion_in_window": ("--motion-in-window", "store_true"),
    "windows": ("--windows", "windows"),
    "channels": (None, "skip"),
}
THERMAL_FLAGS = {
    "size": ("--size", "value"),
    "max_frames": ("--max-frames", "value"),
    "whole_frame": ("--whole-frame", "store_true"),
    "window_frac": ("--window-frac", "value"),
    "hot_quantile": ("--hot-quantile", "value"),
    "y_bias": ("--y-bias", "value"),
    "window": (None, "skip"),
    "channels": (None, "skip"),
}
BUILDERS = {
    "fused": ("scripts/21_build_fused_cache.py", FUSED_FLAGS),
    # The thermal builder script is not part of this package.
    "thermal": ("scripts/25_build_thermal_cache.py", THERMAL_FLAGS),
}


# Deliverable metadata (used by main() only).
def read_meta(target: Path) -> dict:
    """meta.json out of either a deliverable directory or a packed single-file checkpoint."""
    if target.is_dir():
        meta_path = target / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"no meta.json in {target}; not a deliverable directory")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    if target.suffix == ".pt":
        import torch

        blob = torch.load(target, map_location="cpu", weights_only=False)
        if "meta" not in blob:
            raise SystemExit(f"{target} is not a packed deliverable (no 'meta' key)")
        return blob["meta"]
    raise SystemExit(f"{target} is neither a directory nor a .pt packed deliverable")


# Recipe helpers. The inference entry point uses split_cache_name, argv_from_build,
# recorded_build and compare_builds; cuhkx.fd13_inference uses compare_builds.
def split_cache_name(cache_name: str) -> tuple[str, str]:
    """``fused-det248_test.npz`` -> ``("fused", "-det248")``.

    The prefix picks the builder and the remainder is its ``--suffix``, which is how the two
    builders name their output in the first place.
    """
    stem = Path(cache_name).stem
    if not stem.endswith("_test"):
        raise SystemExit(f"cache name {cache_name!r} is not a *_test cache")
    stem = stem[: -len("_test")]
    for prefix in BUILDERS:
        if stem == prefix or stem.startswith(prefix):
            return prefix, stem[len(prefix) :]
    raise SystemExit(f"cache name {cache_name!r} matches no known builder {sorted(BUILDERS)}")


def argv_from_build(build: list[str], prefix: str, suffix: str) -> list[str]:
    """Invert a recorded ``build`` array back into the argv that produces it."""
    script, flags = BUILDERS[prefix]
    # "--suffix=" with an equals sign, always: argparse reads a bare "--suffix -det248" as a
    # flag followed by an unknown option and the cache lands under the wrong name. That cost
    # a build on 2026-09-09.
    argv = [sys.executable, "-u", script, "--split", "test", f"--suffix={suffix}"]
    for entry in build:
        if "=" not in entry:
            raise SystemExit(f"unparsable build entry {entry!r} in {prefix}{suffix}")
        key, value = entry.split("=", 1)
        if key not in flags:
            raise SystemExit(
                f"build entry {key!r} has no CLI flag for the {prefix} builder; refusing to "
                f"guess, because a dropped knob is a silently different cache"
            )
        flag, kind = flags[key]
        if kind == "skip":
            continue
        if kind == "store_true":
            if value == "True":
                argv.append(flag)
        elif kind == "windows":
            # Recorded as "person_windows:{'strong': 385, ...}" -- the stem is the argument,
            # the source tally after the colon is a build-time observation, not a setting.
            argv += [flag, value.split(":", 1)[0]]
        else:
            argv += [flag, value]
    if prefix == "fused" and any(e.startswith("size=") for e in build):
        # The size guard compares against the *default* cache, not this suffix, so a
        # deliberate 248 trips it. The size is not a free variable here: it is whatever the
        # deliverable was trained at, and disagreeing with it is what the guard is for.
        argv.append("--allow-size-change")
    return argv


def comparable(build: list[str]) -> list[str]:
    """Drop the parts of a ``build`` record that describe the *test set*, not the recipe.

    ``windows=person_windows:{'strong': 385, 'agree': 11, ...}`` carries a tally of how each
    clip's crop window was sourced. That is an observation about the 405 Kaggle test clips,
    not a setting -- on the finals set the same recipe produces a different tally by
    definition. Comparing it verbatim would abort a correct build on the one day that matters.
    """
    return [e.split(":", 1)[0] if e.startswith("windows=") else e for e in build]


# A recipe as {key: value}, after dropping the window-source tally.
def as_dict(build: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in comparable(build):
        key, _, value = entry.partition("=")
        out[key] = value
    return out


def compare_builds(deliverable: list[str], built: list[str]) -> tuple[list[str], list[str]]:
    """``(disagreements, record-only differences)`` between two build records.

    Compares the keys the two have **in common**, rather than the lists verbatim. The build
    record is a format that has gained fields over time: the 2026-09-03 ``fused-det_test``
    cache predates ``motion_margin`` / ``motion_quantile`` / ``motion_in_window``, so today's
    builder writes three keys that cache never had. A verbatim comparison calls that a recipe
    change and aborts, which is what it did on the first real cold-start drill -- a false
    alarm on the exact run whose job is to prove the path works.

    A key the *deliverable* names and the build does not is still fatal: that direction means
    the deliverable asked for something this builder no longer produces. The reverse is only
    the record growing, so it is reported and not enforced.
    """
    want, got = as_dict(deliverable), as_dict(built)
    disagree = [
        f"{k}: deliverable={want[k]!r} built={got[k]!r}"
        for k in sorted(want.keys() & got.keys())
        if want[k] != got[k]
    ]
    disagree += [
        f"{k}: deliverable={want[k]!r} but the build records no such key"
        for k in sorted(want.keys() - got.keys())
    ]
    record_only = sorted(got.keys() - want.keys())
    return disagree, record_only


# The build record of a cache: from its memory-map sidecar <name>_meta.npz when that exists,
# else from the .npz itself; None when neither holds one.
def recorded_build(cache_path: Path) -> list[str] | None:
    for candidate in (
        cache_path.with_name(cache_path.stem + "_meta.npz"),
        cache_path,
    ):
        if not candidate.exists():
            continue
        blob = np.load(candidate, allow_pickle=False)
        if "build" in blob:
            return [str(x) for x in blob["build"]]
    return None


# Command-line tool: build the caches that one or more deliverables name. It reads a deliverable
# folder with meta.json or a .pt file with a top-level 'meta' key; weights/model.pt keeps its
# metadata inside a compressed envelope, so the entry point calls the helpers above instead.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--deliverable",
        nargs="+",
        required=True,
        help="deliverable directories and/or packed .pt files. An ensemble names one per "
        "member; caches needed by more than one member are built once",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave a cache alone when one with the recorded build already exists. Off by "
        "default: on the day, the test set is new and every cache has to be rebuilt",
    )
    args = ap.parse_args()

    # Step 1: per deliverable, read the cache name and recipe, note a packaged detector, and turn
    # the recipe into a command line; deliverables that share a cache must agree on it.
    wanted: dict[str, tuple[list[str], list[str]]] = {}
    needs_windows = False
    detectors: set[Path] = set()
    for name in args.deliverable:
        meta = read_meta(Path(name))
        # A detector shipped inside the deliverable is the one that must run. Packaging it
        # and then silently detecting with torchvision's download cache would count bytes
        # for a model that never executes, and would leave the graded run depending on the
        # network. The detector loader also accepts the detector component in a packed file.
        candidate = Path(name) if Path(name).is_file() else Path(name) / "detector.pt"
        if candidate.exists():
            detectors.add(candidate)
        cache_name = meta.get("test_cache")
        build = meta.get("test_cache_build")
        if not cache_name:
            raise SystemExit(f"{name}: meta.json records no test_cache")
        if not build:
            raise SystemExit(
                f"{name}: meta.json records test_cache={cache_name} but no test_cache_build, "
                f"so there is no way to rebuild it. Refusing to guess."
            )
        prefix, suffix = split_cache_name(cache_name)
        argv = argv_from_build(list(build), prefix, suffix)
        if any(str(e).startswith("windows=") for e in build):
            needs_windows = True
        if cache_name in wanted and wanted[cache_name][0] != argv:
            raise SystemExit(
                f"two deliverables disagree about how to build {cache_name}:\n"
                f"  {' '.join(wanted[cache_name][0])}\n  {' '.join(argv)}"
            )
        wanted[cache_name] = (argv, list(build))
        print(f"{name}\n  needs {cache_name}")

    # Step 2: when a recipe crops to person windows, compute them first.
    if needs_windows:
        # The detector boxes are an input to the cache, not a cached artefact of their own,
        # and person_windows_test.parquet on this machine predates the 2026-09-05 move. On a
        # new test set it does not exist at all.
        windows_cmd = [
            sys.executable, "-u", "scripts/22_person_windows.py", "--split", "test",
        ]  # fmt: skip
        if len(detectors) > 1:
            raise SystemExit(
                f"the deliverables ship different detectors {sorted(map(str, detectors))}; "
                f"they would place different crop windows, so there is no single cache to build"
            )
        if detectors:
            windows_cmd += ["--detector-weights", str(next(iter(detectors)))]
        else:
            print(
                "  !! no detector.pt in any deliverable: falling back to torchvision's "
                "download cache. That is fine here, but a graded package must ship its own "
                "detector -- scripts/27_export_detector.py --out <deliverable>"
            )
        print(f"\n--- person windows ---\n  {' '.join(windows_cmd)}")
        if not args.dry_run:
            subprocess.run(windows_cmd, cwd=REPO, check=True)

    # Step 3: build each cache, then compare its build record with the recipe (fatal on any
    # disagreement) and report fields only the record has and a changed window-source tally.
    for cache_name, (argv, build) in wanted.items():
        cache_path = p("cache_root") / cache_name
        print(f"\n--- {cache_name} ---\n  {' '.join(argv)}")
        existing = recorded_build(cache_path)
        if (
            args.skip_existing
            and existing is not None
            and comparable(existing) == comparable(build)
        ):
            print("  already built with this exact recipe; skipping")
            continue
        if args.dry_run:
            continue
        subprocess.run(argv, cwd=REPO, check=True)
        actual = recorded_build(cache_path)
        if actual is None:
            raise SystemExit(f"{cache_name} has no build record after building")
        disagree, record_only = compare_builds(build, actual)
        if disagree:
            raise SystemExit(
                f"{cache_name} was built with a recipe that differs from the deliverable's:\n"
                + "".join(f"    {d}\n" for d in disagree)
                + "Fix the cause; this is a build-time error, not something to warn past."
            )
        print("  build record agrees with the deliverable on every shared key")
        if record_only:
            print(
                f"  (the build record has gained {record_only} since this deliverable's cache "
                f"was made, so there is nothing to compare them against)"
            )
        # Compare the tally itself, not the whole record: the record also differs whenever it
        # has gained a field, and reporting "the tally differs" while printing two identical
        # tallies is the kind of noise that trains people to skim past this section.
        want_tally = [e for e in build if e.startswith("windows=")]
        got_tally = [e for e in actual if e.startswith("windows=")]
        if want_tally != got_tally:
            # Same recipe, different test set. Expected on the finals set, and worth printing,
            # because a large swing in how many clips fell back from "strong" detection says
            # something about the new data.
            print(
                f"  (window-source tally differs, as it must on a new test set)\n"
                f"   was: {want_tally}\n   now: {got_tally}"
            )

    if args.dry_run:
        print("\n(dry run: nothing was built)")


if __name__ == "__main__":
    main()
