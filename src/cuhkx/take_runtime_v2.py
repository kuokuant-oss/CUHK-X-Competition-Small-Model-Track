"""Versioned deployment adapter: complete sample IDs even when take timestamps are missing."""
# Role: an alternative decoder driver: take tables from the gap-based takes.group_takes rule
# (through scripts/12_build_take_index.py), and decoding through takes.decode_with_fallback with
# prior weight 0.75 and a 50% coverage threshold, written to a CSV file.
# Used by: h_mpm_runtime imports take_frame without using it; none of the functions in this file
# is called by the delivered run.

from importlib import import_module

import numpy as np
import pandas as pd

from cuhkx.takes import decode_with_fallback

TAKE_COLUMNS = ["clip_id", "take_id", "position", "take_size", "date", "seconds", "frame"]


def take_frame(spans):
    # This needs scripts/ on sys.path. The script's group_takes is takes.group_takes, and its
    # to_frame builds the take table with the columns above.
    helper = import_module("12_build_take_index")
    if not spans:
        return pd.DataFrame(columns=TAKE_COLUMNS)
    return helper.to_frame(helper.group_takes(spans), spans)


def decode_to_csv(clip_ids, fused, table, transitions, template_path, out_path):
    # Step 1: read the sample submission and check that its clip ids and the probability rows
    # match one to one; the last component of each template path is the clip id.
    template = pd.read_csv(template_path, encoding="utf-8-sig")
    if list(template.columns) != ["path", "prediction"]:
        raise ValueError("unexpected sample submission columns")
    expected = [v.rstrip("/").split("/")[-1] for v in template.path]
    names = list(map(str, clip_ids))
    if len(expected) != len(set(expected)) or len(names) != len(set(names)):
        raise ValueError("duplicate sample or probability IDs")
    if set(expected) != set(names) or len(fused) != len(names):
        raise ValueError("sample and primary IDs must match exactly")
    # Step 2: fused must be a finite 2-D array whose rows sum to one.
    if fused.ndim != 2 or not np.isfinite(fused).all():
        raise ValueError("invalid probabilities")
    if not np.allclose(fused.sum(1), 1, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    # Step 3: take id -> known clip ids in position order; no clip may be in two takes.
    known = set(names)
    takes = (
        {}
        if table.empty
        else {
            key: [c for c in group.sort_values("position").clip_id.tolist() if c in known]
            for key, group in table.groupby("take_id")
        }
    )
    covered = [c for members in takes.values() for c in members]
    if len(covered) != len(set(covered)):
        raise ValueError("clip appears in more than one take")
    # Step 4: decode with takes.decode_with_fallback (prior weight 0.75, 50% coverage threshold);
    # the tensordot with a single weight of 1.0 yields fused itself as float64.
    vectors = np.tensordot(np.ones(1, dtype=np.float64), np.stack([fused]), axes=(0, 0))
    predictions, report = decode_with_fallback(
        clip_ids=names,
        takes=takes,
        probs=dict(zip(names, vectors, strict=True)),
        transitions=transitions,
        transition_weight=0.75,
        min_coverage=0.5,
    )
    if set(predictions) != known:
        raise ValueError("decoder returned extra or missing IDs")
    # A partial fallback (some clips outside every take) also gets a reason in the report.
    if report["filled_by_argmax"] and not report["reason"]:
        report["reason"] = "partial timestamp coverage; uncovered clips use raw argmax"
    # Step 5: fill the template in row order, check the class range, write the CSV without
    # overwriting an earlier one, then read it back and compare.
    result = template.copy()
    result["prediction"] = [int(predictions[c]) for c in expected]
    if not result.prediction.between(0, fused.shape[1] - 1).all():
        raise ValueError("class index outside model class order")
    if out_path.exists():
        raise FileExistsError("preserve prior CSV")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    actual = pd.read_csv(out_path)
    assert actual.path.tolist() == template.path.tolist()
    assert actual.prediction.tolist() == result.prediction.tolist()
    return report
