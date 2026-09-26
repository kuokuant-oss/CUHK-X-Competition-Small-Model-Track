"""Paired designated estimator, with complete per-subject and per-repeat evidence (CPU)."""
# Role: scores the probabilities of several model variants for one held-out fold on the same 20
# test-like thinnings of that fold's takes (one contiguous run per take, run lengths drawn from
# the test take sizes): accuracy with and without take decoding, overall and per subject.
# Used by: training/47_matched_continuation.py (evaluate_fold, summarize) and
# training/101_matched_replay.py (evaluate_fold); run as a script on a results directory, it calls
# summarize; training only, not used by the delivered run.
# Inputs (not distributed): folds.parquet, takes_train.parquet, takes_test.parquet and the
# reference run's out-of-fold file oof.npz.

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cuhkx package (src/) importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuhkx.paths import p  # noqa: E402
from cuhkx.takes import decode_all, transition_matrix  # noqa: E402

# The contiguous thinner of 95_fusion_shape_audit.py, not the random `subsample` of
# 52_simulate_incomplete_takes.py. The digit-named module is found because training/ is on
# sys.path (the script's own directory, or added by the importing entry point).
subsample = import_module("95_fusion_shape_audit").subsample_contiguous


# `names` must be exactly the clips of `fold`, and `arms` maps each variant name to its
# (len(names), 40) probabilities, e.g. "before" (the starting model) and "A0", "A1" (two
# continued-training variants). Returns a JSON-ready dict of per-thinning and mean accuracies.
def evaluate_fold(fold, names, arms, reference_run="ig65m-t32-det-60ep"):
    # Step 1: the clip universe, folds and labels of the reference run's out-of-fold file,
    # checked against the fold table.
    table = pd.read_parquet(p("processed_root") / "folds.parquet").set_index("clip_id")
    reference = np.load(p("models_root") / reference_run / "oof.npz")
    universe = reference["clip_ids"]
    assert np.array_equal(table.loc[universe, "fold"].to_numpy(), reference["folds"])
    assert np.array_equal(table.loc[universe, "action_id"].to_numpy(), reference["labels"])
    table = table.loc[universe]
    # Training takes within that universe (clip ids in recording order); a take's fold is the fold
    # of its first clip. `target` holds the clip count of every test take.
    take_table = pd.read_parquet(p("processed_root") / "takes_train.parquet")
    take_table = take_table[take_table.clip_id.isin(universe)]
    takes = {
        t: g.sort_values("position")["clip_id"].tolist() for t, g in take_table.groupby("take_id")
    }
    take_folds = {t: int(table.loc[m[0], "fold"]) for t, m in takes.items()}
    truth = table.action_id.to_dict()
    target = (
        pd.read_parquet(p("processed_root") / "takes_test.parquet")
        .groupby("take_id")
        .size()
        .to_numpy()
    )
    # Step 2: validate the inputs: the complete fold, and finite probability rows that sum to 1.
    expected = set(table[table.fold == fold].index)
    if set(names) != expected or len(set(names)) != len(names):
        raise ValueError("evaluation requires the complete held-out fold without duplicates")
    probabilities = {}
    for arm, probs in arms.items():
        if probs.shape != (len(names), 40) or not np.isfinite(probs).all():
            raise ValueError(f"invalid probabilities: {arm}")
        if not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError(f"unnormalized probabilities: {arm}")
        probabilities[arm] = dict(zip(names, probs, strict=True))
    # Advance the RNG through every preceding fold exactly as 70 does. Calling one fold
    # alone or after another fold must produce the same paired thinning.
    # Folds are visited in ascending order up to the requested one, each drawing 20 thinnings
    # from this single generator seeded 0.
    rng = np.random.default_rng(0)
    thinnings = []
    for f in sorted(set(take_folds.values())):
        mine = {t: m for t, m in takes.items() if take_folds[t] == f}
        samples = [subsample(mine, target, rng) for _ in range(20)]
        if f == fold:
            thinnings = samples
            break
    # Transition prior counted on the other folds' complete takes only.
    transitions = transition_matrix(
        {t: m for t, m in takes.items() if take_folds[t] != fold}, truth
    )
    # Step 3: on each thinning, score every variant by per-clip argmax ("raw") and after take
    # decoding with transition weight 0.75 ("decoded"), overall and per subject.
    subjects = sorted(set(table.loc[list(names), "user"]))
    rows = {a: [] for a in arms}
    user_rows = {a: {u: [] for u in subjects} for a in arms}
    histograms = []
    for thinned in thinnings:
        kept = [c for m in thinned.values() for c in m]
        # Take-size histogram of this thinning: {kept clips per take: number of takes}.
        sizes, counts = np.unique([len(m) for m in thinned.values()], return_counts=True)
        histograms.append(dict(zip(sizes.tolist(), counts.tolist(), strict=True)))
        y = np.array([truth[c] for c in kept])
        users = table.loc[kept, "user"].to_numpy()
        for arm, probs in probabilities.items():
            raw = np.array([probs[c].argmax() for c in kept])
            decoded_map = decode_all(thinned, probs, transitions, 0.75)
            decoded = np.array([decoded_map[c] for c in kept])
            rows[arm].append(
                {
                    "n": len(kept),
                    "raw": float((raw == y).mean()),
                    "decoded": float((decoded == y).mean()),
                    "raw_correct": int((raw == y).sum()),
                    "decoded_correct": int((decoded == y).sum()),
                }
            )
            for user in subjects:
                mask = users == user
                user_rows[arm][user].append(
                    {
                        "n": int(mask.sum()),
                        "raw": float((raw[mask] == y[mask]).mean()),
                        "decoded": float((decoded[mask] == y[mask]).mean()),
                    }
                )
    # Step 4: means over the 20 thinnings; full_raw is the argmax accuracy on the complete fold,
    # without thinning or decoding.
    result = {
        "fold": int(fold),
        "n_full": len(names),
        "transductive": False,
        "estimator": {"thinning": "contiguous", "tw": 0.75, "seed": 0, "repeats": 20},
        "take_histograms": histograms,
        "arms": {},
    }
    for arm in arms:
        result["arms"][arm] = {
            "raw": float(np.mean([r["raw"] for r in rows[arm]])),
            "decoded": float(np.mean([r["decoded"] for r in rows[arm]])),
            "full_raw": float(
                np.mean(arms[arm].argmax(1) == table.loc[names, "action_id"].to_numpy())
            ),
            "per_repeat": rows[arm],
            "subjects": {
                u: {
                    metric: float(np.mean([r[metric] for r in user_rows[arm][u]]))
                    for metric in ("raw", "decoded", "n")
                }
                for u in subjects
            },
        }
    # Paired differences in percentage points, only between the variants named A1, A0 and before.
    result["contrasts_pp"] = {
        f"{a}-{b}": {
            metric: 100 * (result["arms"][a][metric] - result["arms"][b][metric])
            for metric in ("raw", "decoded")
        }
        for a, b in (("A1", "A0"), ("A1", "before"), ("A0", "before"))
        if a in arms and b in arms
    }
    return result


# Evaluate every fold*/ directory under `root` that holds before.npz (probabilities before further
# training) and A0/after.npz and/or A1/after.npz, aggregate the paired differences over folds and
# write them to paired-evaluation.json in `root`.
def summarize(root):
    root = Path(root)
    results = []
    for directory in sorted(root.glob("fold*")):
        before_path = directory / "before.npz"
        if not before_path.exists():
            continue
        blob = np.load(before_path)
        arms = {"before": blob["probs"]}
        for arm in ("A0", "A1"):
            path = directory / arm / "after.npz"
            if path.exists():
                after = np.load(path)
                assert np.array_equal(blob["clip_ids"], after["clip_ids"])
                assert np.array_equal(blob["labels"], after["labels"])
                arms[arm] = after["probs"]
        # Only folds with at least one variant after training are scored; the fold index is read
        # from the directory name fold<k>.
        if len(arms) > 1:
            results.append(evaluate_fold(int(directory.name[4:]), blob["clip_ids"], arms))
    # Per contrast over folds: the mean, the per-fold values, the number of positive folds, and
    # the standard error of the mean over folds (None with a single fold).
    contrasts = {}
    for name in ("A1-A0", "A1-before", "A0-before"):
        pairs = [r["contrasts_pp"][name] for r in results if name in r["contrasts_pp"]]
        if pairs:
            contrasts[name] = {
                metric: {
                    "mean_pp": float(np.mean([r[metric] for r in pairs])),
                    "fold_deltas_pp": [r[metric] for r in pairs],
                    "folds_positive": sum(r[metric] > 0 for r in pairs),
                    "n_folds": len(pairs),
                    "paired_se_pp": float(
                        np.std([r[metric] for r in pairs], ddof=1) / np.sqrt(len(pairs))
                    )
                    if len(pairs) > 1
                    else None,
                }
                for metric in ("raw", "decoded")
            }
    summary = {"folds": results, "contrasts": contrasts}
    (root / "paired-evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(contrasts, indent=2), flush=True)
    return summary


# Command line: python training/48_evaluate_matched.py <results directory>.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
