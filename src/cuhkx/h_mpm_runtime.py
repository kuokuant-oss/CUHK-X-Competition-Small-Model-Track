"""Fixed H-MPM deployment adapter; original H and BNcal entrypoints are unchanged."""
# Role: the decoder driver. It checks the probabilities, the take table and the transition prior,
# decodes every take jointly and gives each remaining clip its argmax (class 0 if it has no
# probabilities); submission_frame returns the filled sample-submission table and a report.
# Used by: scripts/el25r_p1_repeat_runtime.py, which always calls submission_frame with
# mode='original' (the takes.decode_take beam search); prediction.csv is decoded from the
# probabilities after repeat-chain borrowing, prediction-original.csv from those before; inference.
# mode='mpm' (h_mpm_prototype.mpm) and decode_to_csv are not used by the delivered run.

import numpy as np
import pandas as pd

# mpm is called only when mode='mpm'.
from cuhkx.h_mpm_prototype import mpm
# take_frame is imported but not used here; the take table comes from el22_clock_adapter.from_raw.
from cuhkx.take_runtime_v2 import take_frame  # noqa: F401
from cuhkx.takes import decode_take


def decode_predictions(clip_ids, takes, probs, transitions, *, mode="mpm"):
    """Retain the existing coverage/raw/class-0 fallback, with explicit missing counts."""
    # Inputs: clip_ids are the ids that need a label, probs maps a clip id to its 40-class
    # probability vector (clips may be missing), takes maps a take id to its clip ids in take
    # order, and transitions is the 40x40 transition prior.
    # Step 1: validate the inputs. Ids must be unique and every probability id must be in clip_ids.
    names = list(map(str, clip_ids))
    known = set(names)
    if len(names) != len(known) or not set(probs) <= known:
        raise ValueError("duplicate clip IDs or probabilities outside sample")
    if mode not in ("mpm", "original"):
        raise ValueError("only fixed MPM and original MAP are supported")
    # Every vector must be a finite, non-negative distribution over the 40 classes.
    for vector in probs.values():
        if (
            vector.shape != (40,)
            or not np.isfinite(vector).all()
            or (vector < 0).any()
            or not np.isclose(vector.sum(), 1, atol=1e-5)
        ):
            raise ValueError("invalid 40-class probability vector")
    # The prior must be a finite, non-negative 40x40 matrix whose rows sum to one.
    if (
        transitions.shape != (40, 40)
        or not np.isfinite(transitions).all()
        or (transitions < 0).any()
        or not np.allclose(transitions.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("invalid packed transition prior")
    # Step 2: restrict each take to ids in clip_ids; no clip may be in two takes.
    selected = {t: [c for c in members if c in known] for t, members in takes.items()}
    covered = [c for members in selected.values() for c in members]
    if len(covered) != len(set(covered)):
        raise ValueError("clip appears more than once in takes")
    # Step 3: coverage = share of clip_ids that lie in a take and have probabilities.
    coverage = sum(c in probs for c in covered) / max(len(names), 1)
    out = {}
    oversize = 0
    # Step 4: joint decoding per take, only if at least half of the clips are covered.
    if coverage >= 0.5:
        for members in selected.values():
            present = [c for c in members if c in probs]
            if not present:
                continue
            # More than 40 clips cannot get pairwise distinct labels from 40 classes; such a take
            # is skipped, and its clips get their argmax in step 5.
            if len(present) > 40:
                oversize += 1
                continue
            # A (clips in take, 40) matrix in take order.
            matrix = np.stack([probs[c] for c in present])
            # mode='original' (the delivered run) is the beam search with prior weight 0.75 and
            # the defaults of 64 beams and 16 candidates per clip; a single-clip take keeps its
            # argmax.
            labels = (
                mpm(matrix, transitions)
                if mode == "mpm"
                else decode_take(matrix, transitions, 0.75)
            )
            out.update((c, int(y)) for c, y in zip(present, labels, strict=True))
    # Step 5: every clip not decoded above gets its argmax, or class 0 if it has no probabilities.
    raw_fills = missing = 0
    for c in names:
        if c not in out:
            if c in probs:
                out[c] = int(probs[c].argmax())
                raw_fills += 1
            else:
                out[c] = 0  # Existing explicit last-resort contract, not a model prediction.
                missing += 1
    # Every id in clip_ids, and no other id, now has a label in 0..39.
    assert set(out) == known and all(0 <= y < 40 for y in out.values())
    # Step 6: the report. fell_back is set when coverage was too low or any clip used a fallback.
    reasons = []
    if coverage < 0.5:
        reasons.append("take coverage below .5; use raw argmax where probabilities exist")
    elif raw_fills:
        reasons.append("uncovered or impossible >40-clip takes use raw argmax")
    if missing:
        reasons.append("missing probabilities use explicit class 0 fallback")
    return out, dict(
        mode=mode,
        clips=len(names),
        takes=len(takes),
        coverage=coverage,
        fell_back=bool(coverage < 0.5 or raw_fills or missing),
        reason="; ".join(reasons),
        filled_by_argmax=raw_fills,
        missing_probability_class0=missing,
        oversize_takes=oversize,
    )


def submission_frame(clip_ids, fused, table, transitions, template, *, mode="mpm"):
    # template is the sample submission (columns path, prediction); the last component of each
    # path is the clip id. fused is (n_clips, 40) with row i for clip_ids[i]; table is the take
    # table from el22_clock_adapter.from_raw.
    if list(template.columns) != ["path", "prediction"]:
        raise ValueError("unexpected sample submission columns")
    expected = [v.rstrip("/").split("/")[-1] for v in template.path]
    names = list(map(str, clip_ids))
    # Probability ids must be unique and must all appear in the template.
    if len(names) != len(set(names)) or not set(names) <= set(expected):
        raise ValueError("duplicate or unknown probability IDs")
    if fused.shape != (len(names), 40):
        raise ValueError("invalid probability array shape")
    # Take id -> clip ids ordered by position; there are no takes if the table is empty.
    takes = (
        {}
        if table.empty
        else {t: g.sort_values("position").clip_id.tolist() for t, g in table.groupby("take_id")}
    )
    # The tensordot with a single weight of 1.0 yields fused itself as float64, shape (n_clips, 40).
    # Match the original single-member deployment's float64 conversion exactly.
    vectors = np.tensordot(np.ones(1, dtype=np.float64), np.stack([fused]), axes=(0, 0))
    # Decoding runs over all template ids, so a template clip without probabilities gets class 0.
    predictions, report = decode_predictions(
        expected, takes, dict(zip(names, vectors, strict=True)), transitions, mode=mode
    )
    # The predictions are filled in template row order.
    result = template.copy()
    result["prediction"] = [predictions[c] for c in expected]
    return result, report


# The delivered run does not call this function (its entry point writes the CSV files itself);
# note that it decodes with submission_frame's default mode='mpm'.
def decode_to_csv(clip_ids, fused, table, transitions, template_path, out_path):
    template = pd.read_csv(template_path, encoding="utf-8-sig")
    result, report = submission_frame(clip_ids, fused, table, transitions, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Mode "x" refuses to overwrite an existing file.
    with out_path.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    # The file is read back to check that paths and predictions survived the round trip.
    actual = pd.read_csv(out_path)
    assert actual.path.tolist() == template.path.tolist()
    assert actual.prediction.tolist() == result.prediction.tolist()
    return report
