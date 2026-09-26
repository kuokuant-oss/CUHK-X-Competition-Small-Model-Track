"""Weight averaging across folds, plus the greedy variant.

The four fold models are the only thing standing between a leaderboard score we cannot ship
and a deliverable we can. `wgt_imgonly_tw075` scores 0.79104 as a 4-fold 4-pass ensemble,
which is 4 x 63.5M parameters and about 254 MB at int8 -- well over the 100 MB single-file
limit (organiser C-4). The verified single-model fallback is `CAND_A_fulldata` at 0.75621.
Collapsing four folds into one set of weights is what closes that 3.48 pp gap, and 3.48 pp
happens to be exactly the distance to the Top-15 cutoff.

Why this should work at all: the four folds start from the same IG65M initialisation and
train on 75%-overlapping data, so they are very likely in the same loss basin and linearly
mode-connected. Where it usually goes wrong is not the basin but BatchNorm -- averaging
weights does not give you the averaged model's activation statistics, so the running buffers
have to be recomputed afterwards (`cuhkx.train._update_bn`). That is why `average_states`
refuses to touch integer buffers and why the CLI always re-estimates.

Everything here is pure tensor arithmetic on CPU: it is testable, and is tested, without a
GPU or a checkpoint.
"""
# Role: weight averaging of fold models (the uniform "soup"): compatibility checks, a float64 mean
# of every floating-point tensor, a greedy variant, and reading a checkpoint's held-out fold from
# its path. BatchNorm statistics have to be re-estimated after averaging; that is not done here.
# Used by: training/44_self_training.py and training/47_matched_continuation.py (held_out_fold)
# and, for the averaging, an earlier pipeline stage whose script is not included; training.

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch


class IncompatibleStates(ValueError):
    """Raised when the state dicts cannot be averaged, with the reason spelled out."""


def held_out_fold(path) -> int:
    """Which fold the checkpoint at ``path`` did NOT train on, read off its own filename.

    ``train.py`` sets ``is_val = folds == fold`` and trains on the complement, so
    ``<run>/folds/foldN.pt`` is the model that held fold N OUT. That inversion is what the
    2026-09-07 measurement got backwards, producing 0.9737 against a 0.7289 anchor, so the
    fact is read from the artefact's own path rather than passed in by a caller.

    Plain string parsing rather than a pattern: the single fact that decides whether a soup's
    score is honest should be obvious to read.
    """
    path = Path(path)
    stem = path.stem
    if path.parent.name != "folds" or not stem.startswith("fold") or not stem[4:].isdigit():
        raise ValueError(
            f"{path}: expected <run>/folds/foldN.pt; the held-out fold is read from the "
            f"filename and must not be guessed"
        )
    return int(stem[4:])


def clean_holdout(paths, holdout: int | None) -> bool:
    """Can a soup of these members be scored honestly on fold ``holdout``?

    Only if EVERY member held that fold out, i.e. the fold lies in the intersection of the
    members' held-out folds. For members from one run that intersection is empty as soon as
    there are two of them -- which is the structural impossibility ``67_fold_soup.py``
    refuses on. Across runs it is not: ``runA/folds/foldN.pt`` and ``runB/folds/foldN.pt``
    both held out fold N, so fold N's subjects are unseen by both and the score is real.
    """
    if holdout is None or not paths:
        return False
    return all(held_out_fold(q) == holdout for q in paths)


def is_bn_counter(key: str) -> bool:
    """``num_batches_tracked`` is an int64 counter, not a weight."""
    return key.endswith("num_batches_tracked")


def check_compatible(states: Sequence[dict[str, Any]]) -> None:
    """Fail closed on any structural difference, naming it.

    A silent mismatch here would produce a model that loads and runs and is quietly wrong,
    which is the single worst outcome available: it would be measured, believed, and shipped.
    """
    if len(states) < 2:
        raise IncompatibleStates(f"need at least two state dicts to average, got {len(states)}")
    # Every state must have the reference's key set (an error names up to five missing and five
    # unexpected keys), equal non-tensor entries, and equal tensor shapes.
    reference = states[0]
    ref_keys = set(reference)
    for i, state in enumerate(states[1:], start=1):
        keys = set(state)
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)[:5]
            extra = sorted(keys - ref_keys)[:5]
            raise IncompatibleStates(
                f"state {i} has different keys; missing {missing}, unexpected {extra}"
            )
    for key, ref in reference.items():
        if not torch.is_tensor(ref):
            for i, state in enumerate(states[1:], start=1):
                if state[key] != ref:
                    raise IncompatibleStates(f"non-tensor entry {key!r} differs at state {i}")
            continue
        for i, state in enumerate(states[1:], start=1):
            if state[key].shape != ref.shape:
                raise IncompatibleStates(
                    f"{key!r} shape {tuple(state[key].shape)} at state {i} "
                    f"!= {tuple(ref.shape)} at state 0"
                )


def average_states(
    states: Sequence[dict[str, Any]],
    weights: Sequence[float] | None = None,
) -> OrderedDict:
    """Weighted mean of floating tensors; integer buffers taken from the first state.

    Integer entries -- in practice BatchNorm's ``num_batches_tracked`` -- are counters, and a
    mean of counters is meaningless. They are copied from the first state and are overwritten
    anyway by the BatchNorm re-estimation that has to follow, so the choice does not matter;
    what matters is not silently casting them to float and changing the dtype of the
    checkpoint.
    """
    check_compatible(states)
    # weights=None gives the uniform mean; otherwise the weights are normalised to sum to 1.
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise IncompatibleStates(f"{len(weights)} weights for {len(states)} states")
    total = float(sum(weights))
    if total <= 0:
        raise IncompatibleStates(f"weights must sum to a positive number, got {total}")
    norm = [w / total for w in weights]

    out: OrderedDict = OrderedDict()
    for key, ref in states[0].items():
        if not torch.is_tensor(ref):
            out[key] = ref
            continue
        if not ref.is_floating_point() or is_bn_counter(key):
            out[key] = ref.clone()
            continue
        # Floating-point tensors, including BatchNorm running_mean and running_var (which the
        # re-estimation then replaces): accumulate in float64, cast back to the tensor's dtype.
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for w, state in zip(norm, states, strict=True):
            acc += state[key].to(torch.float64) * w
        out[key] = acc.to(ref.dtype)
    return out


def greedy_soup(
    states: Sequence[dict[str, Any]],
    score: Callable[[dict[str, Any]], float],
    labels: Sequence[str] | None = None,
) -> tuple[OrderedDict, list[int], list[float]]:
    """Add folds one at a time, keeping one only if it improves the score.

    The uniform soup is the first thing to try because it is free, but it is not the thing to
    fall back on when it disappoints: a single fold that landed in a different basin drags the
    average down, and dropping it recovers most of the loss. Ordering is by each fold's own
    solo score, best first, which is the standard greedy construction.

    ``score`` is called once per candidate soup and is expected to be the mock public test --
    the estimator that thins takes to the real test size distribution. It is the caller's job
    to make that a *validation* measurement; nothing here can check that for you.

    Returns the winning soup, the indices kept in the order they were added, and the score
    after each accepted addition.
    """
    check_compatible(states)
    # Rank the models by their own score, start from the best one, and add each next one only
    # if the uniform average of the kept models then scores higher.
    solo = sorted(range(len(states)), key=lambda i: score(states[i]), reverse=True)
    kept: list[int] = [solo[0]]
    best = score(states[solo[0]])
    history = [best]
    for idx in solo[1:]:
        trial = kept + [idx]
        candidate = average_states([states[i] for i in trial])
        value = score(candidate)
        if value > best:
            kept, best = trial, value
            history.append(value)
    if labels is not None:
        names = ", ".join(labels[i] for i in kept)
        print(f"greedy soup kept {len(kept)}/{len(states)}: {names} -> {best:.5f}")
    return average_states([states[i] for i in kept]), kept, history
