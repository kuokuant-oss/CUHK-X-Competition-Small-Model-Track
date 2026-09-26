"""Test the decoder under the conditions that actually hold at test time.

The take decoder was validated on *training* takes, which are complete. The 405 test clips
are a **sample** of the recordings, so a reconstructed test take usually has clips missing
from the middle — and the transition prior ``P(next | previous)`` assumes the two clips it
relates were adjacent in the original recording. When clips are missing, that assumption is
false, and the prior is applied to pairs that were two or three steps apart.

This script subsamples training takes down to the test take-size distribution and re-runs
the decoder, which measures the gain under the conditions that will actually apply instead
of the ones that were convenient to measure.

    uv run python scripts/52_simulate_incomplete_takes.py --run skeleton-T16-e120-s42

Distinctness should survive subsampling — a subset of distinct actions is still distinct.
The transition prior should not.
"""
# Role: development check of the take decoder on out-of-fold training probabilities: complete
# takes against takes thinned to the test take sizes (random positions, order kept), at several
# transition weights. Also provides subsample, the random thinner.
# Used by: training/95_fusion_shape_audit.py imports subsample, so this module is also loaded
# whenever 44_self_training.py or 48_evaluate_matched.py is imported; training only, not used by
# the delivered run. Inputs (not distributed): <models_root>/<run>/oof.npz and the take tables.
# The usage line above names scripts/; the file is in training/.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cuhkx package (src/) importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.paths import p  # noqa: E402
from cuhkx.takes import decode_all, transition_matrix  # noqa: E402


# Accuracy over the clips that have both a prediction and a label (0.0 if there are none).
def accuracy(predictions: dict[str, int], truth: dict[str, int]) -> float:
    shared = [c for c in predictions if c in truth]
    return float(np.mean([predictions[c] == truth[c] for c in shared])) if shared else 0.0


def subsample(takes: dict[str, list[str]], target: np.ndarray, rng) -> dict[str, list[str]]:
    """Thin each take to a size drawn from the test distribution, keeping the order."""
    # Per take, two draws from `rng`: the kept size (a random entry of `target`, at most the
    # take's length), then that many distinct positions, sorted so the recording order is kept.
    out = {}
    for take_id, members in takes.items():
        keep = min(len(members), int(rng.choice(target)))
        if keep <= 0:
            continue
        chosen = sorted(rng.choice(len(members), size=keep, replace=False))
        out[take_id] = [members[i] for i in chosen]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    # Step 1: out-of-fold probabilities, labels and folds of the training clips, keyed by clip id.
    blob = np.load(p("models_root") / args.run / "oof.npz", allow_pickle=False)
    clip_ids = [str(c) for c in blob["clip_ids"]]
    probs = {c: blob["probs"][i] for i, c in enumerate(clip_ids)}
    truth = {c: int(blob["labels"][i]) for i, c in enumerate(clip_ids)}
    fold_of = {c: int(blob["folds"][i]) for i, c in enumerate(clip_ids)}

    # Step 2: complete training takes (clip ids in recording order), each assigned the fold of its
    # first clip, and the clip counts of the test takes that the thinning draws from.
    train_table = pd.read_parquet(p("processed_root") / "takes_train.parquet")
    train_table = train_table[train_table["clip_id"].isin(probs)]
    full_takes = {
        take_id: group.sort_values("position")["clip_id"].tolist()
        for take_id, group in train_table.groupby("take_id")
    }
    take_fold = {t: fold_of[m[0]] for t, m in full_takes.items()}

    test_table = pd.read_parquet(p("processed_root") / "takes_test.parquet")
    target_sizes = test_table.groupby("take_id").size().to_numpy()
    print(
        f"test take sizes: mean {target_sizes.mean():.2f}, "
        f"train take sizes: mean {np.mean([len(v) for v in full_takes.values()]):.2f}"
    )

    # Baseline without take decoding: per-clip argmax.
    raw = {c: int(v.argmax()) for c, v in probs.items()}
    print(f"\nper-clip argmax (no decoding): {accuracy(raw, truth):.4f}")

    # Decode each fold's takes with a transition prior counted on the other folds' complete takes;
    # weight 0 uses no prior (pairwise distinct labels only, solved as an assignment problem).
    # Accuracy covers the decoded clips only, so with thinned takes only the kept clips count.
    def evaluate(takes: dict[str, list[str]], weight: float) -> float:
        out: dict[str, int] = {}
        for fold in sorted(set(take_fold.values())):
            train_takes = {t: m for t, m in full_takes.items() if take_fold[t] != fold}
            transitions = transition_matrix(train_takes, truth) if weight > 0 else None
            val_takes = {t: m for t, m in takes.items() if take_fold[t] == fold}
            out |= decode_all(val_takes, probs, transitions, weight)
        return accuracy(out, truth)

    # Step 3: per transition weight, complete takes against the mean and standard deviation over
    # --repeats random thinnings (generator seeds 0, 1, ...).
    print("\n  weight   complete takes   subsampled to test sizes")
    for weight in args.weights:
        complete = evaluate(full_takes, weight)
        trials = [
            evaluate(subsample(full_takes, target_sizes, np.random.default_rng(seed)), weight)
            for seed in range(args.repeats)
        ]
        label = "distinctness only" if weight == 0 else f"transitions w={weight}"
        print(
            f"  {label:20s}  {complete:.4f}        {np.mean(trials):.4f} +/- {np.std(trials):.4f}"
        )


if __name__ == "__main__":
    main()
