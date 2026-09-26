"""C pilot: one clip exposure, one deterministic independent view draw per epoch."""
# Role: training dataset for member C. Every clip is served once per epoch, from either the det or
# the miw ("local") view, drawn per clip and epoch from a hash of (seed, epoch, clip ID); a clip
# that lacks one view always gets the other. An optional deadline stops a run that is out of time.
# Used by: training/124_c_full_student.py (member C training and BatchNorm re-estimation); training.

from __future__ import annotations

import hashlib
import multiprocessing
import time

import numpy as np


# True selects the miw view, False the det view. The draw is the lowest bit of the first byte of
# the key's SHA-256 digest, so it depends only on (seed, epoch, clip ID), not on worker count or
# data order, and splits the clips about evenly. The fixed prefix names this version of the draw.
def choose_local(seed: int, epoch: int, clip_id: str) -> bool:
    key = f"cuhkx-C-view-v1|{seed}|{epoch}|{clip_id}".encode()
    return bool(hashlib.sha256(key).digest()[0] & 1)


class SharedViewDataset:
    def __init__(self, det, local, clip_ids, seed=42):
        # det, local: fused.FusedFrameDataset objects over the det and miw caches, expected to
        # hold the same clips in the same order; clip_ids names them and keys the view draw.
        self.det, self.local = det, local
        self.clip_ids = np.asarray(clip_ids).astype(str)
        self.seed = seed
        self.epoch = 0
        # Shared-memory copy of the epoch for DataLoader workers, created by enable_epoch_sync().
        self._shared_epoch = None
        # Optional time.monotonic() limit set by the caller; see set_epoch and __getitem__.
        self.deadline = None
        # Both views must list the same number of unique clips with equal labels.
        if not len(det) == len(local) == len(self.clip_ids):
            raise ValueError("view lengths differ")
        if len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("duplicate clip IDs")
        if not np.array_equal(det.labels[det.indices], local.labels[local.indices]):
            raise ValueError("view labels differ")
        # A view is absent for a clip when its cache stored no frames for it (empty offset range).
        self.det_present = det.offsets[det.indices + 1] > det.offsets[det.indices]
        self.local_present = local.offsets[local.indices + 1] > local.offsets[local.indices]
        if not (self.det_present | self.local_present).all():
            raise ValueError("both views absent")

    def __len__(self):
        return len(self.clip_ids)

    # Called before the DataLoader starts its workers, so that later set_epoch calls reach them.
    def enable_epoch_sync(self):
        self.det.enable_epoch_sync()
        self.local.enable_epoch_sync()
        if self._shared_epoch is None:
            self._shared_epoch = multiprocessing.get_context("spawn").Value("q", self.epoch)

    def set_epoch(self, epoch):
        # Do not start an epoch when less than 240 s remain before the deadline.
        if self.deadline is not None and time.monotonic() + 240 > self.deadline:
            raise RuntimeError("C budget insufficient for next complete epoch; checkpoint retained")
        # The epoch keys the view draw and both views' per-sample RNG (frame picks, augmentation).
        self.epoch = int(epoch)
        self.det.set_epoch(epoch)
        self.local.set_epoch(epoch)
        if self._shared_epoch is not None:
            self._shared_epoch.value = epoch

    def local_at(self, index, epoch):
        # miw missing -> det; det missing -> miw; both present -> the hash draw.
        if not self.local_present[index]:
            return False
        if not self.det_present[index]:
            return True
        return choose_local(self.seed, epoch, self.clip_ids[index])

    # Number of clips served from each view in one epoch, and how many clips lack each view;
    # training/124_c_full_student.py records these in the run's identity file.
    def view_counts(self, epoch):
        n_local = sum(self.local_at(i, epoch) for i in range(len(self)))
        return dict(
            epoch=epoch,
            det=len(self) - n_local,
            miw=n_local,
            det_missing=int((~self.det_present).sum()),
            miw_missing=int((~self.local_present).sum()),
        )

    def __getitem__(self, index):
        # Past the deadline, fail mid-epoch; the checkpoint of the last finished epoch remains.
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise RuntimeError("C pilot budget exhausted; preserve previous epoch checkpoint")
        # Worker processes read the epoch from shared memory once epoch sync is enabled.
        epoch = self._shared_epoch.value if self._shared_epoch is not None else self.epoch
        # The sample of the chosen view, unchanged: ((frames,), label).
        return (self.local if self.local_at(index, epoch) else self.det)[index]

    # Fixed, evenly spaced subsample without augmentation for the clean-train accuracy that
    # train.train_one_fold logs. It picks the same positions as FusedFrameDataset.clean_view, so
    # the clip IDs stay aligned, and its epoch stays 0, so each clip's view is fixed.
    def clean_view(self, limit):
        if limit <= 0:
            return None
        picked = (
            np.arange(len(self))
            if len(self) <= limit
            else np.linspace(0, len(self) - 1, limit).astype(int)
        )
        return SharedViewDataset(
            self.det.clean_view(limit),
            self.local.clean_view(limit),
            self.clip_ids[picked],
            self.seed,
        )
