"""Mass-normalized soft replay: weights travel with targets through mixup."""
# Role: soft-target cross-entropy with one loss weight per sample. Member B is fine-tuned on
# labelled clips (weight 1, label-smoothed one-hot target) plus pseudo-labelled test clips (weight
# 0.5, the two teachers' mean four-pass softmax); the loss is normalised per epoch, not per batch.
# Used by: training/101_matched_replay.py (fit) and training/103_b_final_student.py; training.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset


class ReplayDataset(Dataset):
    """Images keep their dataset RNG; supervised/pseudo targets have explicit mass."""

    def __init__(self, base, targets, weights):
        # base: an image dataset (fused.FusedFrameDataset) yielding ((frames,), label);
        # targets: (N, n_classes) probability rows; weights: (N,) loss weight of each sample.
        if len(base) != len(targets) or len(base) != len(weights):
            raise ValueError("base, targets and weights must have matching lengths")
        self.base, self.targets, self.weights = base, targets, weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        # The base dataset's integer label is ignored (it is -1 for the test clips); the stored
        # soft target and weight replace it.
        inputs, _ = self.base[index]
        return inputs, self.targets[index], self.weights[index]

    # Epoch handling is forwarded to the image dataset, whose augmentation RNG is keyed on
    # (seed, epoch, clip).
    def enable_epoch_sync(self):
        self.base.enable_epoch_sync()

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)


# ConcatDataset of the labelled and the pseudo-labelled parts that also forwards the epoch
# handling to every part.
class ReplayConcat(ConcatDataset):
    def enable_epoch_sync(self):
        for dataset in self.datasets:
            dataset.enable_epoch_sync()

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


def weighted_targets(targets, weights, permutation=None, lam=1.0):
    """Return target mass, preserving each sample's weight under permutation mixup."""
    if targets.ndim != 2 or weights.shape != targets.shape[:1]:
        raise ValueError("one scalar weight per target distribution is required")
    # Row i is weight[i] * target[i]; for probability targets it sums to weight[i].
    mass = targets * weights[:, None]
    if permutation is None:
        return mass
    if not 0 <= lam <= 1:
        raise ValueError("mixup lambda must be in [0, 1]")
    # Mixup blends clip i with clip permutation[i]; mixing the weighted rows keeps each clip's
    # weight attached to its own target, and the batch's total mass is unchanged.
    return lam * mass + (1 - lam) * mass[permutation]


def epoch_mass_loss(logits, target_mass, *, epoch_weight, steps_per_epoch):
    """Mean over steps equals total weighted CE / epoch weight, including a short batch.

    Unlike a per-batch mean, the last partial batch does not increase any sample's
    coefficient. Under mixup a permutation preserves total target mass exactly.
    """
    if epoch_weight <= 0 or steps_per_epoch <= 0:
        raise ValueError("positive epoch mass and step count required")
    if logits.shape != target_mass.shape:
        raise ValueError("logits and target mass must have identical shapes")
    # Weighted cross-entropy summed over the batch (log-softmax in float32 even under autocast),
    # scaled by steps_per_epoch / epoch_weight. epoch_weight is the summed weight of every sample
    # in the dataset; for member B 2,931 + 0.5 x 405 = 3,133.5, with 417 steps of 8 clips.
    return -(target_mass * F.log_softmax(logits.float(), dim=-1)).sum() * (
        steps_per_epoch / epoch_weight
    )


def epoch_batches(loader, dataset, epoch):
    """Set the epoch before obtaining the iterator, including persistent-worker prefetch."""
    # iter() already dispatches the first prefetch requests to the workers, which read the
    # shared epoch when they build a sample.
    dataset.set_epoch(epoch)
    return iter(loader)


# Not called by the training scripts in this package.
def actual_mass(target_mass):
    """Detached telemetry only; do not use a random batch's mass to normalize the loss."""
    return float(torch.sum(target_mass.detach()))
