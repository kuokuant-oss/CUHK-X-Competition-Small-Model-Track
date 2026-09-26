"""Augmentation for the image modalities, and the matching deterministic test-time view.

Measured before writing any of this: a depth model reaches **0.9903 training accuracy
against 0.2289 validation** — a gap of 0.76. It memorises all eighteen training subjects.
That single number decides the direction: the encoder has capacity to spare, so adding more
would make things worse, and what is missing is anything at all standing between the model
and memorisation. The image pipeline had no augmentation whatsoever.

Each transform is separately switchable because each has to earn its place on the mock
public test, and a bundle that arrives together can only be judged together.

What is deliberately **absent**:

* **Horizontal flip is available but off by default.** It was originally excluded by
  extrapolating from skeleton, where flipping was worth +0.08 pp — but a skeleton flip
  swaps paired *joint indices*, a structural edit, while an image flip only mirrors pixels,
  and the skeleton already had rotation augmentation covering similar variation while the
  image path had none. Both public notebooks use it. The real risk is different: several
  classes are handed (brushing teeth, writing, combing hair), and if all eighteen training
  subjects happen to be right-handed, flipping invents a distribution the data never had.
  That is why it is a flag to measure, not a default to assume either way.
* **No colour jitter.** After decoding, the channel is a depth in metres wearing no
  disguise; perturbing it as if it were brightness would move the person in space.
"""
# Role: clip-level spatial augmentations on (T, C, H, W) uint8 arrays (random crop, horizontal
#   flip, erasing) and the deterministic centre crop used at validation and inference.
# Used by: fused.FusedTransform (center_crop at inference; random_crop, horizontal_flip and erase
#   in training) and tools/make_preprocessing_figure.py; both. jitter_depth, time_crop and
#   make_transform belong to the earlier single-modality pipeline and are not used by the
#   delivered run.

from __future__ import annotations

import numpy as np

# Value written into erased pixels. On the fused frames it is written to every channel, giving
# black Depth_Color (the "no return" colour) and zero IR.
INVALID_FILL = 0


def _below(rng, high: float) -> int:
    """A uniform integer in ``[0, high)``.

    Via ``uniform`` because that is the one spelling shared by ``np.random`` (whose global
    seed ``seed_everything`` sets, so runs stay reproducible) and by a ``Generator`` (which
    the tests pass in for determinism). ``integers`` exists only on the latter.
    """
    return int(rng.uniform(0.0, max(high, 1e-9)))


def _window(size: int, crop: int, rng) -> tuple[int, int]:
    # (start, end) of a random crop-long span of one axis; the whole axis if the crop is larger.
    if crop >= size:
        return 0, size
    # min() guards against uniform() returning its upper bound through floating-point rounding.
    start = min(_below(rng, size - crop + 1), size - crop)
    return start, start + crop


def random_crop(frames: np.ndarray, crop: int, rng) -> np.ndarray:
    """Crop ``(T, C, H, W)`` to ``crop`` square, at one position for the whole clip.

    One window per clip, not per frame. A window that wandered would add camera motion the
    recording never had, and the model would have to learn to undo it.
    """
    top, bottom = _window(frames.shape[2], crop, rng)
    left, right = _window(frames.shape[3], crop, rng)
    return frames[:, :, top:bottom, left:right]


def center_crop(frames: np.ndarray, crop: int) -> np.ndarray:
    """The deterministic view used for validation and test, matching the training crop size."""
    # For the members: 128 x 128 out of the 144 x 144 cache, starting at row 8, column 8.
    height, width = frames.shape[2], frames.shape[3]
    top = max(0, (height - crop) // 2)
    left = max(0, (width - crop) // 2)
    return frames[:, :, top : top + crop, left : left + crop]


def jitter_depth(frames: np.ndarray, amount: int, rng) -> np.ndarray:
    """Shift every valid depth by one random offset, clipped to the scale.

    This is the augmentation the modality asks for: absolute depth encodes how far the
    person stood from the camera, which varies by subject and by session and has nothing to
    do with the action. Shifting it teaches the model to read posture rather than distance.
    Invalid pixels keep their zero — moving them would invent returns the sensor never got.
    """
    # Two channels means depth-plus-validity. Anything else is a colour-mapped modality
    # whose channels are not a depth and a mask, and shifting one of them would be
    # meaningless — for thermal it would read the green channel as a validity mask.
    if amount <= 0 or frames.shape[1] != 2:
        return frames
    offset = _below(rng, 2 * amount + 1) - amount
    out = frames.copy()
    valid = out[:, 1] > 0
    depth = out[:, 0].astype(np.int16)
    depth[valid] = np.clip(depth[valid] + offset, 0, 254)
    out[:, 0] = depth.astype(np.uint8)
    return out


def erase(frames: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Blank a rectangle, in the same place across the clip, and mark it invalid.

    Occlusion happens — a limb behind furniture, a body out of the sensor's range — and the
    validity channel already carries a vocabulary for "nothing was seen here", so this
    perturbs the model with a pattern the real data also produces.
    """
    if fraction <= 0.0:
        return frames
    height, width = frames.shape[2], frames.shape[3]
    # Box sides are fraction x U(0.5, 1) of the frame height and width, drawn independently; the
    # same box is blanked in every frame and channel.
    box_h = max(1, int(height * fraction * rng.uniform(0.5, 1.0)))
    box_w = max(1, int(width * fraction * rng.uniform(0.5, 1.0)))
    top = min(_below(rng, height - box_h + 1), height - box_h)
    left = min(_below(rng, width - box_w + 1), width - box_w)
    out = frames.copy()
    out[:, :, top : top + box_h, left : left + box_w] = INVALID_FILL
    return out


def time_crop(frames: np.ndarray, min_fraction: float, rng) -> np.ndarray:
    """A random contiguous slice of the clip, resampled back to the same number of frames.

    The counterpart of the skeleton version, but by nearest index rather than interpolation:
    blending two frames of a moving person makes a ghost that occurs in no real frame, and
    the network would learn to expect it.
    """
    count = frames.shape[0]
    if min_fraction <= 0.0 or min_fraction >= 1.0 or count < 4:
        return frames
    keep = max(2, int(count * rng.uniform(min_fraction, 1.0)))
    start = min(_below(rng, count - keep + 1), count - keep)
    picks = start + np.linspace(0, keep - 1, count).round().astype(np.int64)
    return frames[picks]


def horizontal_flip(frames: np.ndarray) -> np.ndarray:
    """Mirror ``(T, C, H, W)`` left to right, identically across the clip."""
    # A reversed view, not a copy; later steps make the array contiguous.
    return frames[..., ::-1]


def make_transform(
    training: bool,
    crop: int = 96,
    depth_jitter: int = 0,
    erase_fraction: float = 0.0,
    erase_prob: float = 0.0,
    min_time_fraction: float = 0.0,
    flip_prob: float = 0.0,
    rng=np.random,
):
    """Build the per-sample transform for one split, or ``None`` when it is a no-op.

    Validation and test get the same crop *size* as training but the centre window, so the
    two splits see the same field of view and the comparison stays honest.
    """
    if not training:
        if crop <= 0:
            return None

        def evaluate(sample: dict) -> dict:
            return {**sample, "frames": np.ascontiguousarray(center_crop(sample["frames"], crop))}

        return evaluate

    if (
        crop <= 0
        and depth_jitter <= 0
        and erase_prob <= 0.0
        and min_time_fraction <= 0.0
        and flip_prob <= 0.0
    ):
        return None

    def apply(sample: dict) -> dict:
        frames = sample["frames"]
        if min_time_fraction > 0.0:
            frames = time_crop(frames, min_time_fraction, rng)
        if crop > 0:
            frames = random_crop(frames, crop, rng)
        if depth_jitter > 0:
            frames = jitter_depth(frames, depth_jitter, rng)
        if flip_prob > 0.0 and rng.random() < flip_prob:
            frames = horizontal_flip(frames)
        if erase_prob > 0.0 and rng.random() < erase_prob:
            frames = erase(frames, erase_fraction, rng)
        return {**sample, "frames": np.ascontiguousarray(frames)}

    return apply
