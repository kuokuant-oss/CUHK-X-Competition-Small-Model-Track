"""Load per-frame skeleton JSON into a ``(T, 17, 3)`` tensor.

The format is not documented by the organizers, so it was reverse-engineered from the
data (see ADR 0005). What the checks established:

* Joints follow the **Human3.6M 17-joint order**, not COCO-17. Decisive evidence: joint 0
  has ``x == y == 0`` in 100% of frames (it is the root hip), joints 3 and 6 are almost
  always the lowest (the feet) and joint 10 almost always the highest (the head), and the
  resulting bone lengths are anatomically real — thigh 0.28-0.32 m, upper arm 0.35-0.38 m,
  forearm 0.25-0.29 m, with left/right symmetric to within 7-13%.
* **x and y are root-relative**; ``z`` is shifted so the lowest joint of each frame sits at
  exactly 0, which holds in 100% of frames. So ``z`` is height above the ground plane and
  the units are metres: the topmost joint sits at a median 1.18 m.
* ``keypoint_scores`` is **always exactly 1.0** and carries no information — there is no
  confidence signal to filter on.
* Tracking is stable: adjacent frames move a joint ~0.02-0.04 m, far less than a bone
  length, so the sequences are usable as trajectories rather than independent detections.

``z`` being ground-relative already removes camera placement. Removing *body size* on top
of that is what cross-subject generalisation needs (ADR 0003) — but the natural scale to
divide by turns out not to be the apparent height; see :func:`normalize`.
"""
# Role: loads the per-frame skeleton JSON files into (T, 17, 3) arrays (Human3.6M joint order,
#   metres) and provides normalisation, resampling and geometric augmentations for pose models.
# Used by: data.build_skeleton_cache / data.skeleton_tensor and models.SkeletonEncoder (BONES,
#   N_JOINTS); not used by the delivered run: the pipeline imports this module through data.py
#   and models.py, but no function in it runs. The take clock lists Skeleton file names through
#   takes.clip_span, not through this module.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Human3.6M 17-joint order, confirmed against this dataset.
JOINTS = [
    "hip",
    "r_hip",
    "r_knee",
    "r_foot",
    "l_hip",
    "l_knee",
    "l_foot",
    "spine",
    "thorax",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
]
N_JOINTS = len(JOINTS)
ROOT = 0

# Bones, for graph-based models and for the anatomy checks.
BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

# Left/right joint pairs — a horizontal flip must swap these, or the augmented pose is
# anatomically wrong and will not match a flipped Depth/IR frame (ADR 0002).
LR_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]

# Summed bone length of a real adult is ~4 m; below this the pose has collapsed.
_MIN_SCALE_M = 1.0


def _frame_files(clip_dir: Path) -> list[Path]:
    """Frames live in ``<clip>/predictions/*.json``; sorting by name sorts by timestamp."""
    pred = clip_dir / "predictions"
    return sorted(pred.glob("*.json")) if pred.is_dir() else sorted(clip_dir.glob("*.json"))


def load_clip_skeleton(clip_dir: Path) -> np.ndarray:
    """Return ``(T, 17, 3)`` float32 for one clip; ``(0, 17, 3)`` if it has no usable frames.

    Some frames contain more than one detected person. The dataset is single-subject, so
    the first entry is taken — consistent with how the organizers' own visualisations index
    it, and multi-person frames are rare (~3%).
    """
    poses = []
    for path in _frame_files(clip_dir):
        try:
            people = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not people:
            continue
        joints = np.asarray(people[0]["keypoints"], dtype=np.float32)
        if joints.shape != (N_JOINTS, 3) or not np.isfinite(joints).all():
            continue
        poses.append(joints)
    if not poses:
        return np.zeros((0, N_JOINTS, 3), dtype=np.float32)
    return np.stack(poses)


def normalize(poses: np.ndarray) -> np.ndarray:
    """Make a clip invariant to where the person stands and how big they are.

    ``x``/``y`` are re-centred on the root every frame (the dataset already stores them
    that way, but a clip from a different pipeline version might not be), then every axis
    is divided by a body-size estimate taken once per clip, so a single bad frame cannot
    rescale the whole sequence.

    The scale is the **summed length of all 16 bones**, which is posture-invariant. The
    obvious alternative — the frame's z extent, i.e. the person's apparent height — was
    measured and is *worse than doing nothing*, because a seated person's z extent is a
    fraction of a standing one's, so it encodes posture rather than body size. Measured
    cross-clip CV of bone lengths over a 400-clip sample:

    ==========================  =====
    no normalisation             0.323
    frame z extent               0.350   <- worse than baseline
    **sum of bone lengths**      0.268
    spine chain hip->neck        0.304
    torso hip->thorax            0.339
    ==========================  =====

    The residual 0.268 is pose-estimator noise, not body size, so do not expect
    normalisation alone to close the cross-subject gap.
    """
    if len(poses) == 0:
        return poses
    out = poses.copy()
    # Re-centre x and y on the root joint in every frame; z (height above ground) is kept.
    out[:, :, :2] -= out[:, ROOT : ROOT + 1, :2]

    # One scale per clip, the median over frames of the summed bone length; it is applied only
    # when it reaches _MIN_SCALE_M.
    scale = float(np.median(bone_lengths(out).sum(axis=1)))
    if scale >= _MIN_SCALE_M:
        out /= scale
    return out


def resample(poses: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a clip to exactly ``n_frames`` along time.

    Clips run from 1 to 236 frames, so both padding and subsampling have to work; linear
    interpolation over the frame index handles the whole range, and a single-frame clip is
    repeated rather than dropped.
    """
    if len(poses) == 0:
        return np.zeros((n_frames, N_JOINTS, 3), dtype=np.float32)
    if len(poses) == 1:
        return np.repeat(poses, n_frames, axis=0)

    src = np.linspace(0.0, 1.0, len(poses))
    dst = np.linspace(0.0, 1.0, n_frames)
    flat = poses.reshape(len(poses), -1)
    out = np.empty((n_frames, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape(n_frames, N_JOINTS, 3)


def flip_lr(poses: np.ndarray) -> np.ndarray:
    """Mirror a pose across the body's left/right axis, swapping paired joints."""
    out = poses.copy()
    # Negate x, then swap every left/right joint pair.
    out[..., 0] *= -1.0
    for left, right in LR_PAIRS:
        out[..., [left, right], :] = out[..., [right, left], :]
    return out


def to_bones(poses: np.ndarray) -> np.ndarray:
    """Joint coordinates -> bone vectors: each joint minus its parent in the skeleton tree.

    The standard second stream in this literature, and it is not a cosmetic reparameterisation.
    Joint coordinates say *where the body is*; bone vectors say *how it is folded*, and the
    latter is far less sensitive to who is performing — a tall and a short person bending an
    elbow the same way produce nearly the same bone direction but quite different joint
    positions. On a cross-subject split that is the difference that matters.

    ``BONES`` is a tree rooted at the hip and every joint but the root appears exactly once
    as a child, so this is a complete reparameterisation, not a lossy summary. The root keeps
    a zero vector: it has no parent, and its absolute position was normalised away anyway.
    """
    out = np.zeros_like(poses)
    for parent, child in BONES:
        out[..., child, :] = poses[..., child, :] - poses[..., parent, :]
    return out


def rotate_z(poses: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a clip about the vertical axis by ``angle`` radians.

    ``z`` is height above the ground (ADR 0005), so rotating in the x-y plane is exactly a
    change of camera yaw — the person and the action are untouched, only the direction they
    are viewed from. ``normalize`` re-centres and rescales but never canonicalises heading,
    so absolute yaw really is still in the data and really does differ between recording
    sessions; that is what this is for.

    One angle for the whole clip, not one per frame: a camera does not swivel mid-clip, and
    a per-frame angle would fabricate rotation that reads as motion.
    """
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    out = poses.copy()
    x, y = poses[..., 0], poses[..., 1]
    out[..., 0] = cos * x - sin * y
    out[..., 1] = sin * x + cos * y
    return out


def time_crop(poses: np.ndarray, min_fraction: float, rng=np.random) -> np.ndarray:
    """Take a random contiguous slice of the clip and stretch it back to the same length.

    Two augmentations for the price of one: *when* the action was observed (a random window)
    and *how fast* it appears to run (the stretch). It is the default spatial-temporal
    augmentation in the CTR-GCN family's NTU configuration, and we had nothing like it —
    a fixed resample shows the model the identical frames every epoch.

    Note the frame counts here are already small (median 20, shortest 2), so the fraction
    should stay well above the 0.5 that NTU uses; there is no 64-frame floor to lean on.
    """
    frames = len(poses)
    if min_fraction <= 0.0 or min_fraction >= 1.0 or frames < 4:
        return poses
    keep = max(2, int(frames * rng.uniform(min_fraction, 1.0)))
    start = min(int(rng.uniform(0.0, frames - keep + 1)), frames - keep)
    window = poses[start : start + keep]

    source = np.linspace(0.0, 1.0, keep)
    target = np.linspace(0.0, 1.0, frames)
    flat = window.reshape(keep, -1)
    out = np.empty((frames, flat.shape[1]), dtype=np.float32)
    for channel in range(flat.shape[1]):
        out[:, channel] = np.interp(target, source, flat[:, channel])
    return out.reshape(poses.shape)


def augment(
    poses: np.ndarray,
    rng=np.random,
    max_yaw: float = 0.0,
    flip_prob: float = 0.0,
    noise: float = 0.0,
    min_time_fraction: float = 0.0,
) -> np.ndarray:
    """Apply the enabled geometric augmentations to one clip. All default to off.

    Defaults are off so that turning each one on is an explicit, separately measured
    decision rather than something that arrives bundled with a refactor.
    """
    if min_time_fraction > 0.0:
        poses = time_crop(poses, min_time_fraction, rng)
    if max_yaw > 0.0:
        poses = rotate_z(poses, rng.uniform(-max_yaw, max_yaw))
    if flip_prob > 0.0 and rng.random() < flip_prob:
        poses = flip_lr(poses)
    if noise > 0.0:
        poses = poses + rng.normal(0.0, noise, poses.shape).astype(np.float32)
    return poses


def bone_lengths(poses: np.ndarray) -> np.ndarray:
    """``(T, len(BONES))`` bone lengths — useful as features and as a sanity check."""
    i = np.array([a for a, _ in BONES])
    j = np.array([b for _, b in BONES])
    return np.linalg.norm(poses[:, i] - poses[:, j], axis=-1)
