"""Recover which recording take each clip came from, and its position within that take.

    uv run python scripts/12_build_take_index.py

Writes ``data/processed/takes_train.parquet`` and ``takes_test.parquet`` with
``clip_id, take_id, position, take_size, date, seconds, frame`` — the structure ADR 0007
depends on.

Training takes are known outright (one ``user/trial`` is one continuous recording), while
test takes must be reconstructed from the timestamps left in frame filenames. That
difference is an opportunity, not a nuisance: the reconstruction is run on the training
clips too and scored against the known answer, so the load-bearing assumption is measured
rather than assumed.
"""
# Role: take tables for the training clips (known takes: one user/trial each) and the test clips
#   (cuhkx.takes.group_takes on filename timestamps: a new take at a new date, a frame counter
#   that goes backwards, or a gap over 60 s), plus a check of that grouping on the training clips.
# Used by: training (scripts in training/ read takes_train.parquet); not used by the delivered
#   run, whose entry point groups test takes by recorder start time (cuhkx.el22_clock_adapter).

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the cuhkx library in src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.paths import p  # noqa: E402
from cuhkx.takes import clip_span, group_takes, group_train_takes, parse_stamp  # noqa: E402


def train_spans() -> dict[str, tuple[str, float, int, float, int]]:
    """Frame span per training clip, read from the Skeleton tree (the smallest one)."""
    index = pd.read_parquet(p("processed_root") / "clip_index_listing.parquet")
    root = p("train_root") / "Skeleton"
    # One stamp (date, seconds since midnight, frame counter) per Skeleton prediction file; the
    # span runs from the earliest stamp to the latest.
    spans = {}
    for clip_id in index.loc[index["n_Skeleton"] > 0, "clip_id"]:
        stamps = [
            stamp
            for path in (root / clip_id / "predictions").glob("*.json")
            if (stamp := parse_stamp(path.name)) is not None
        ]
        if stamps:
            first, last = min(stamps), max(stamps)
            spans[clip_id] = (first[0], first[1], first[2], last[1], last[2])
    return spans


# (date, start seconds, start frame, end seconds, end frame) per test clip, from its frame file
# names (cuhkx.takes.clip_span); clips without stamps are left out.
def test_spans() -> dict[str, tuple[str, float, int, float, int]]:
    return {
        clip.name: span
        for clip in sorted(p("test_root").glob("SM_test_*"))
        if clip.is_dir() and (span := clip_span(clip)) is not None
    }


def score_reconstruction(reconstructed: dict[str, list[str]], truth: dict[str, list[str]]) -> None:
    """How well timestamp grouping recovers the takes we actually know.

    Reported as pair agreement: over every pair of clips, does the reconstruction agree with
    the truth about whether they share a take? That is the property the decoder relies on —
    a merged pair invents a false distinctness constraint, a split pair forfeits a real one.
    """
    assigned = {c: t for t, members in reconstructed.items() for c in members}
    actual = {c: t for t, members in truth.items() for c in members}
    clips = sorted(set(assigned) & set(actual))

    # Over all clip pairs: same take in the truth, same take in the reconstruction, and both.
    same_true = same_pred = both = 0
    for i, a in enumerate(clips):
        for b in clips[i + 1 :]:
            t = actual[a] == actual[b]
            q = assigned[a] == assigned[b]
            same_true += t
            same_pred += q
            both += t and q

    precision = both / same_pred if same_pred else 0.0
    recall = both / same_true if same_true else 0.0
    # A known take is exactly recovered when all its clips fall in one reconstructed take that
    # holds no other known clip.
    exact = sum(
        1
        for members in truth.values()
        if len({assigned[c] for c in members if c in assigned}) == 1
        and sum(1 for c in reconstructed.get(assigned[members[0]], []) if c in actual)
        == len(members)
    )
    print(
        f"  reconstruction vs known takes: pair precision {precision:.4f}  recall {recall:.4f}  "
        f"exactly recovered {exact}/{len(truth)} takes"
    )


# One row per clip: take id, position in the take, take size, and the clip's first stamp (date,
# seconds since midnight, frame counter).
def to_frame(takes: dict[str, list[str]], spans: dict) -> pd.DataFrame:
    rows = []
    for take_id, members in takes.items():
        for position, clip_id in enumerate(members):
            date, seconds, frame = spans[clip_id][0], spans[clip_id][1], spans[clip_id][2]
            rows.append(
                {
                    "clip_id": clip_id,
                    "take_id": take_id,
                    "position": position,
                    "take_size": len(members),
                    "date": date,
                    "seconds": seconds,
                    "frame": frame,
                }
            )
    return pd.DataFrame(rows)


# Print the clip and take counts and how many takes have each size.
def describe(name: str, frame: pd.DataFrame) -> None:
    sizes = frame.groupby("take_id").size().value_counts().sort_index().to_dict()
    print(f"{name}: {len(frame):5d} clips in {frame['take_id'].nunique():4d} takes  sizes {sizes}")


def main() -> None:
    out_dir = p("processed_root")

    # Step 1: training takes from user/trial, ordered by the first frame counter of each clip.
    spans = train_spans()
    truth = group_train_takes(list(spans), {c: s[2] for c, s in spans.items()})
    frame = to_frame(truth, spans)
    frame.to_parquet(out_dir / "takes_train.parquet", index=False)
    describe("train", frame)

    # Step 2: pair precision and recall of the timestamp grouping on the training clips.
    # Validate the test-side algorithm on data where the answer is known.
    score_reconstruction(group_takes(spans), truth)

    # Step 3: test takes from the filename timestamps.
    spans = test_spans()
    frame = to_frame(group_takes(spans), spans)
    frame.to_parquet(out_dir / "takes_test.parquet", index=False)
    describe("test ", frame)


if __name__ == "__main__":
    main()
