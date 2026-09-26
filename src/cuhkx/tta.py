"""The 4-pass test-time augmentation, in one place because two paths must not drift.

Every submitted csv since 2026-09-06 was produced by ``scripts/90_infer_tta.py``, which
predicts from *training* artifacts. The graded path is different: the finals hand over a
test set and run ``inference.sh``, which must rebuild the model from the shipped
deliverable alone. Those two paths have to compute the same function, and the S6 gate
states that as ``inference.sh`` reproducing the submission with **D = 0** discordant clips.

Duplicating the averaging in both scripts is how that gate quietly stops holding: a change
to one side is invisible until a leaderboard score disagrees with a local one, which is the
one comparison nobody can run before the deadline. So the passes and the averaging live
here and both callers import them.

Moved verbatim out of ``scripts/90_infer_tta.py``; the extraction was verified by re-running
that script and checking the probabilities against the file it had already written.
"""
# Role: the four test-time passes of members B and C (plain, horizontal flip, frame sequence
#   rolled by +1 and by -1) and the averaging of their softmax outputs.
# Used by: fd13_inference.infer_f0 (inference) and five scripts in training/ (44, 47, 69, 101,
#   103); both.

from __future__ import annotations

import numpy as np
import torch

from cuhkx.train import _collate


def tta_passes(batch: torch.Tensor):
    """``(B, T, C, H, W)`` -> the four views, as (name, tensor).

    The roll is along the time axis, so it is a cheap stand-in for sampling a different
    temporal window; the flip is along width. Both are geometric, so ADR 0002's ban on colour
    augmentation over a colourised depth map does not reach them.
    """
    return (
        ("plain", batch),
        ("hflip", batch.flip(-1)),
        # torch.roll is circular: +1 moves the last frame to the front, -1 the first to the end.
        ("roll+1", batch.roll(1, dims=1)),
        ("roll-1", batch.roll(-1, dims=1)),
    )


# Class probabilities (clips, classes) as a NumPy array in dataset order: each clip's softmax
# averaged over the four passes, or the plain pass alone when tta is False.
def predict(model, dataset, device, batch_size, tta: bool):
    from torch.utils.data import DataLoader

    # No shuffling, so the output rows follow the dataset order.
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)
    out = []
    with torch.no_grad():
        for inputs, _ in loader:
            # The first input tensor is the (B, T, C, H, W) frame batch; the labels are unused.
            batch = inputs[0].to(device)
            views = tta_passes(batch) if tta else (("plain", batch),)
            total = None
            for _name, view in views:
                # Average softmax, not logits: the passes are different *observations* of the
                # same clip, and averaging logits would let one confident-but-wrong view
                # dominate in a way averaging probabilities does not.
                probs = model(view).softmax(dim=1)
                total = probs if total is None else total + probs
            out.append((total / len(views)).cpu().numpy())
    return np.concatenate(out)
