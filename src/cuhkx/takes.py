"""Recover recording *takes* and decode a take's clips jointly.

Clips were cut from continuous recordings, so they are not independent samples (ADR 0007).
Two properties of a take, measured over 791 training takes, are worth more than any model
improvement available in the time we have:

* **Distinctness** — no take ever repeats an action. Zero exceptions in 791 takes.
* **Transition prior** — ``H(A)`` is 5.04 bits but ``H(A|previous)`` is 2.72, so the order
  carries 2.32 bits. Guessing the next action from the previous one alone scores 0.3645
  against a 0.120 majority baseline.

Both are recoverable at test time: every test clip's ``Depth_Color`` filename carries
``<date>_<hh-mm-ss.mmm>_<8-digit global frame index>``, which is enough to group clips into
takes and order them within one.

The decoder deliberately stays *soft*. Test takes are shorter than training ones (median 2
against 3) because the 405 clips are a sample of the recordings, so a take is usually
incomplete and matching whole sequences would be wrong. Distinctness plus a bigram prior
survives incompleteness; whole-sequence matching does not.
"""
# Role: reads (date, time, frame counter) from frame file names, groups clips into takes, counts
# the transition prior and decodes the clips of one take jointly (decode_take, a beam search over
# pairwise distinct labels).
# Used by: inference calls clip_span (through el22_clock_adapter.from_raw) and decode_take (through
# h_mpm_runtime.decode_predictions); scripts/12_build_take_index.py uses group_takes and
# group_train_takes, and training/ uses transition_matrix, decode_all and decode_take; both.
# At inference, takes are grouped by recorder start time (el21_take_geometry.group_clock), not by
# group_takes; decode_with_fallback and describe_fallback are not called by the delivered run.

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# This pattern finds the date, the time of day (with milliseconds) and the recorder's 8-digit frame
# counter in a frame file name such as "Depth_2025-06-11_13-21-46.616_00000067_Color.png".
FILENAME_STAMP = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})\.(?P<ms>\d{3})"
    r"_(?P<frame>\d{8})"
)

# A new take starts when the frame counter restarts, or after a gap this long.
MAX_GAP_SECONDS = 60.0
N_CLASSES = 40


def parse_stamp(filename: str) -> tuple[str, float, int] | None:
    """``(date, seconds_since_midnight, global_frame_index)`` from a frame filename."""
    match = FILENAME_STAMP.search(filename)
    if not match:
        return None
    parts = match.groupdict()
    seconds = (
        int(parts["h"]) * 3600 + int(parts["m"]) * 60 + int(parts["s"]) + int(parts["ms"]) / 1000
    )
    return parts["date"], seconds, int(parts["frame"])


def clip_span(clip_dir: Path) -> tuple[str, float, int, float, int] | None:
    """``(date, start_seconds, start_frame, end_seconds, end_frame)`` for one clip.

    Both ends are needed: a clip that belongs to the same take as its predecessor starts
    *after* that predecessor ends, so the gap and the frame ordering must both be judged
    against the previous clip's end, not its start. Judging against the start merges
    distinct takes (measured: 144 takes instead of 168 on the test set).
    """
    stamps = []
    # Use the first source that yields any stamp: Depth_Color names, then IR names, then skeleton
    # JSON names, then files directly in the clip folder.
    for pattern in (
        "Depth_Color/*.png",
        "IR/*.png",
        "Skeleton/predictions/*.json",
        "*.png",
        "*.json",
    ):
        for path in clip_dir.glob(pattern):
            stamp = parse_stamp(path.name)
            if stamp:
                stamps.append(stamp)
        if stamps:
            break
    if not stamps:
        return None
    # Stamps compare as (date, seconds, frame) tuples, so these are the earliest and latest stamps.
    first, last = min(stamps), max(stamps)
    return first[0], first[1], first[2], last[1], last[2]


def group_takes(spans: dict[str, tuple[str, float, int, float, int]]) -> dict[str, list[str]]:
    """Group clips into takes by their position in the original recording, ordered within.

    A take breaks on a new date, a frame counter that goes backwards (the recorder restarts
    per take), or a gap over ``MAX_GAP_SECONDS`` since the previous clip ended.

    This runs identically on training clips, where the true takes are known from
    ``user/trial``, which is how the reconstruction is validated rather than assumed —
    see ``scripts/12_build_take_index.py``.
    """
    ordered = sorted(spans.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    takes: dict[str, list[str]] = {}
    current: list[str] = []
    previous: tuple[str, float, int, float, int] | None = None

    for clip_id, span in ordered:
        date, start_seconds, start_frame = span[0], span[1], span[2]
        is_new = (
            previous is None
            or date != previous[0]
            or start_frame < previous[4]
            or start_seconds - previous[3] > MAX_GAP_SECONDS
        )
        if is_new and current:
            takes[f"take_{len(takes):04d}"] = current
            current = []
        current.append(clip_id)
        previous = span
    if current:
        takes[f"take_{len(takes):04d}"] = current
    return takes


def group_train_takes(clip_ids: list[str], frame_index: dict[str, int]) -> dict[str, list[str]]:
    """Training takes are explicit: a clip id is ``action/user/trial`` and one ``user/trial``
    *is* one continuous recording. Ordering still comes from the frame counter."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for clip_id in clip_ids:
        _, user, trial = clip_id.split("/")
        grouped[f"{user}/{trial}"].append(clip_id)
    return {
        take_id: sorted(members, key=lambda c: frame_index.get(c, 0))
        for take_id, members in grouped.items()
    }


def transition_matrix(
    takes: dict[str, list[str]],
    labels: dict[str, int],
    smoothing: float = 1.0,
) -> np.ndarray:
    """Row-stochastic ``P(next | previous)`` learned from training takes.

    Laplace smoothing keeps unseen transitions possible — test takes are incomplete, so a
    pair that never occurred adjacently in training may well be adjacent here.
    """
    # counts[a, b] = smoothing + the number of times action b directly follows action a in a take;
    # clips without a label are skipped, so their neighbours count as adjacent.
    counts = np.full((N_CLASSES, N_CLASSES), smoothing, dtype=np.float64)
    for members in takes.values():
        sequence = [labels[c] for c in members if c in labels]
        for a, b in zip(sequence, sequence[1:], strict=False):
            counts[a, b] += 1.0
    # Each row is normalised to P(next = b | previous = a).
    return counts / counts.sum(axis=1, keepdims=True)


def decode_take(
    probs: np.ndarray,
    transitions: np.ndarray | None = None,
    transition_weight: float = 1.0,
    beam_width: int = 64,
    candidates_per_step: int = 16,
) -> np.ndarray:
    """Assign one distinct class to each clip of a take, maximising total log-probability.

    ``probs`` is ``(n_clips, 40)`` in take order.

    With no transition prior this is a pure assignment problem and Hungarian solves it
    exactly. With one it is not: the objective couples *adjacent* choices while distinctness
    couples *all* of them, so it is solved by beam search over positions, each partial
    sequence carrying the set of classes it has already used.

    The obvious cheap approximation — take each clip's independent argmax as the condition
    for the next — was measured and is much weaker: at a per-clip accuracy of 0.42 it
    conditions on a wrong predecessor most of the time, so the prior misleads more than it
    helps and larger weights actively hurt (+0.0246 at w=0.5, -0.0464 at w=2.0). Beam
    search keeps the alternatives alive instead of committing to a likely-wrong one.
    """
    n_clips = len(probs)
    # Probabilities are clipped at 1e-12 so that every log is finite.
    log_probs = np.log(np.clip(probs, 1e-12, None))
    # Case 1: a single-clip take keeps its argmax.
    if n_clips == 1:
        return np.array([int(log_probs[0].argmax())])

    # Case 2: without a transition prior, the best set of distinct labels is an assignment problem,
    # solved exactly on the negative log-probabilities; rows are clip indices, cols their classes.
    if transitions is None or transition_weight <= 0:
        rows, cols = linear_sum_assignment(-log_probs)
        assignment = np.empty(n_clips, dtype=int)
        assignment[rows] = cols
        return assignment

    # Case 3: beam search under the transition prior. A label sequence y scores
    # sum_i log p_i(y_i) + transition_weight * sum_{i>0} log T(y_{i-1}, y_i). The delivered run
    # uses transition_weight 0.75 with the defaults of 64 beams and 16 candidates per clip.
    log_trans = np.log(np.clip(transitions, 1e-12, None))
    # Only the most plausible classes per clip are worth expanding; the tail carries no mass.
    shortlist = [
        np.argsort(-log_probs[i])[: min(candidates_per_step, log_probs.shape[1])]
        for i in range(n_clips)
    ]

    # Each beam is (score, chosen classes so far, bitmask of used classes).
    beams: list[tuple[float, list[int], int]] = [(0.0, [], 0)]
    for i in range(n_clips):
        # Step 1: extend every beam by each shortlisted class of clip i that it has not used yet;
        # the step adds the clip's log-probability and, after the first clip, the weighted log
        # transition from the beam's previous label.
        expanded: list[tuple[float, list[int], int]] = []
        for score, chosen, used in beams:
            for action in shortlist[i]:
                bit = 1 << int(action)
                if used & bit:
                    continue  # distinctness: no take repeats an action
                step = log_probs[i, action]
                if chosen:
                    step += transition_weight * log_trans[chosen[-1], action]
                expanded.append((score + step, [*chosen, int(action)], used | bit))
        # Step 2: every beam has used exactly i classes here, so this fallback can only trigger
        # once i reaches the shortlist length (16 by default; test takes have at most 8 clips).
        if not expanded:  # every shortlisted class already used; fall back to all classes
            for score, chosen, used in beams:
                for action in range(log_probs.shape[1]):
                    bit = 1 << action
                    if used & bit:
                        continue
                    step = log_probs[i, action]
                    if chosen:
                        step += transition_weight * log_trans[chosen[-1], action]
                    expanded.append((score + step, [*chosen, action], used | bit))
        # Step 3: keep the beam_width best partial sequences; the sort is stable, so equal scores
        # keep the order in which they were generated.
        expanded.sort(key=lambda b: -b[0])
        beams = expanded[:beam_width]

    # beams[0] is the best complete sequence: one class per clip, in take order.
    return np.array(beams[0][1])


# Below this share of clips recovered into takes, the structure is too broken to trust and
# the decoder is skipped entirely rather than applied to a fragment. Half is a judgement,
# not a measurement: the real test set recovers 100%, so anything far below that means the
# filenames are not the ones this code was written for.
MIN_TAKE_COVERAGE = 0.5


# The delivered run does not call this function: h_mpm_runtime.decode_predictions applies its own
# version of the same coverage threshold and fallbacks.
def decode_with_fallback(
    clip_ids: list[str],
    takes: dict[str, list[str]],
    probs: dict[str, np.ndarray],
    transitions: np.ndarray | None = None,
    transition_weight: float = 1.0,
    min_coverage: float = MIN_TAKE_COVERAGE,
) -> tuple[dict[str, int], dict[str, float | int | bool | str]]:
    """Decode by take where the structure is recoverable, per-clip argmax where it is not.

    The finals run on a fresh on-site test set, and nothing guarantees its filenames carry
    the timestamps and global frame counters this module reads. If they do not,
    :func:`clip_span` returns ``None`` for every clip, ``takes`` comes back empty, and
    :func:`decode_all` returns predictions for nobody — at which point ``build_submission``
    raises and the entire run produces no answer at all. Losing the decoder's few points is
    survivable; producing no submission is not.

    So this is the entry point every caller should use. It guarantees one prediction per
    clip in ``clip_ids`` under three separate failures: no takes recovered, some clips left
    out of the takes that were, and a clip with no probabilities at all.

    Returns the predictions and a report worth printing, since a silent fallback is its own
    kind of failure — the run would look fine while quietly scoring several points lower.
    """
    # Step 1: coverage = (clips that lie in some take and have probabilities) / len(clip_ids).
    covered = {c for members in takes.values() for c in members if c in probs}
    coverage = len(covered) / max(len(clip_ids), 1)
    report: dict[str, float | int | bool | str] = {
        "clips": len(clip_ids),
        "takes": len(takes),
        "coverage": coverage,
        "fell_back": False,
        "reason": "",
    }

    # Step 2: decode take by take, unless too few clips were grouped into takes.
    if coverage < min_coverage:
        report["fell_back"] = True
        report["reason"] = (
            f"only {coverage:.1%} of clips grouped into takes (need {min_coverage:.0%}); "
            "take structure is not recoverable from these filenames"
        )
        decoded: dict[str, int] = {}
    else:
        decoded = decode_all(takes, probs, transitions, transition_weight)

    # Step 3: each clip the decoder did not label gets its argmax (class 0 without probabilities).
    # Whatever the decoder did or did not cover, every clip leaves here with an answer.
    predictions = dict(decoded)
    filled = 0
    for clip_id in clip_ids:
        if clip_id in predictions:
            continue
        vector = probs.get(clip_id)
        # A clip with no probabilities cannot be argmaxed; class 0 is arbitrary but a
        # submission with a wrong row beats a submission that was never written.
        predictions[clip_id] = int(vector.argmax()) if vector is not None else 0
        filled += 1
    report["filled_by_argmax"] = filled
    return predictions, report


def describe_fallback(report: dict) -> str:
    """One line for the log, loud when the decoder was skipped."""
    head = (
        f"take decoding: {report['takes']} takes covering {report['coverage']:.1%} "
        f"of {report['clips']} clips"
    )
    if report["fell_back"]:
        return f"!! FALLBACK to per-clip argmax ({report['reason']}); {head}"
    if report["filled_by_argmax"]:
        return f"{head}; {report['filled_by_argmax']} clip(s) filled by per-clip argmax"
    return head


def decode_all(
    takes: dict[str, list[str]],
    probs: dict[str, np.ndarray],
    transitions: np.ndarray | None = None,
    transition_weight: float = 1.0,
) -> dict[str, int]:
    """Decode every take, returning clip id -> predicted action id."""
    predictions: dict[str, int] = {}
    for members in takes.values():
        # Clips without probabilities are dropped; the others keep their take order.
        present = [c for c in members if c in probs]
        if not present:
            continue
        # Each take becomes one (n_present, 40) matrix and is decoded jointly.
        matrix = np.stack([probs[c] for c in present])
        for clip_id, action in zip(
            present, decode_take(matrix, transitions, transition_weight), strict=True
        ):
            predictions[clip_id] = int(action)
    return predictions
