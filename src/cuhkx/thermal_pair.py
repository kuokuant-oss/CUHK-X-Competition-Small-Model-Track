"""Fixed H fusion: primary coverage is mandatory; thermal absence is primary-only."""
# Role: clip-wise average of two probability tables. Despite the names, in the delivered run
#   `primary` is member C on the det view and `thermal` is member C on the miw view (see
#   fd13_inference.infer_f0), so fixed_pair() forms member C's det/miw average; the delivered
#   system uses no thermal data.
# Used by: fd13_inference.infer_f0 (fixed_pair); inference. h_decision is not used by the
#   delivered run.

from __future__ import annotations

import numpy as np


# Checks both tables (unique IDs, shape (clips, 40), finite, non-negative, rows summing to 1
# within 1e-5) and that every `thermal` ID is also a `primary` ID. A clip with a `thermal` row gets
# (primary + thermal) / 2; the others keep their primary row. Returns (fused, present) in
# primary order, where `present` marks the averaged clips.
def fixed_pair(primary_ids, primary, thermal_ids, thermal):
    primary_ids = [str(c) for c in primary_ids]
    thermal_ids = [str(c) for c in thermal_ids]
    for names, probs in ((primary_ids, primary), (thermal_ids, thermal)):
        if len(names) != len(set(names)) or probs.shape != (len(names), 40):
            raise ValueError("duplicate IDs or wrong probability shape")
        if not np.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("invalid probabilities")
        if not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError("probabilities not normalized")
    if not set(thermal_ids) <= set(primary_ids):
        raise ValueError("thermal IDs outside primary coverage")
    lookup = {c: i for i, c in enumerate(thermal_ids)}
    fused = primary.copy()
    present = np.array([c in lookup for c in primary_ids])
    for i, c in enumerate(primary_ids):
        if present[i]:
            fused[i] = (primary[i] + thermal[lookup[c]]) * 0.5
    assert np.array_equal(fused[~present], primary[~present])
    return fused, present


# Not used by the delivered run. A decision rule over four validation folds: it compares the
# 'H-int7' results with 'B2-BNcal' (accuracy change) and with 'H-fp32' (quantisation cost), in
# percentage points, for raw and decoded accuracy, and reports whether every criterion holds.
def h_decision(folds):
    """Predeclared two-estimator accuracy and matched quantization gates."""
    delta = {}
    quant = {}
    positive = {}
    for setting in ("primary", "sensitivity"):
        delta[setting] = {}
        quant[setting] = {}
        for metric in ("raw", "decoded"):
            delta[setting][metric] = [
                100
                * (r[setting]["arms"]["H-int7"][metric] - r[setting]["arms"]["B2-BNcal"][metric])
                for r in folds
            ]
            quant[setting][metric] = [
                100 * (r[setting]["arms"]["H-int7"][metric] - r[setting]["arms"]["H-fp32"][metric])
                for r in folds
            ]
        positive[setting] = sum(v > 0 for v in delta[setting]["decoded"])
    passed = (
        len(folds) == 4
        and all(np.mean(v) > 0 for settings in delta.values() for v in settings.values())
        and all(n >= 3 for n in positive.values())
        and all(
            np.mean(v) >= -0.5 and min(v) >= -1.0
            for settings in quant.values()
            for v in settings.values()
        )
    )
    return dict(
        eligible=bool(passed),
        deltas_pp=delta,
        quantization_pp=quant,
        positive_decoded_folds=positive,
    )
