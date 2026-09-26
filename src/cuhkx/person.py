"""Find the person in a clip with a clean-licence detector, on the modality that works.

The fused cache has always cropped to a **motion** box (``depth.motion_bbox``), and that
box degenerates to the whole frame for **12.1% of train and 15.6% of test clips**. The
failure is not random: motion is the cue, so the clips it loses are the ones where the
actor barely moves -- seated, holding a small object -- which is precisely the 15-class,
27.1%-of-clips band this project has been unable to lift. The crop fails hardest exactly
where we need it.

Three candidate localisers were considered and two were measured:

* **Skeleton joints.** Dead on arrival, and worth writing down so it is not re-proposed:
  the released skeletons are **root-relative**. Joint 0 is exactly ``x == y == 0`` in every
  frame of every clip (verified over all 86,050 cached frames), so the representation
  carries body *pose* and nothing about where in the image the body is. No affine fit can
  recover a translation that was subtracted before we ever saw the data.
* **COCO detector on ``Depth_Color``.** Measured on 60 random test clips, one mid-clip
  frame each: a person is found at score > 0.3 in **16.7%**. Depth_Color is a JET-family
  colourisation, not a photograph -- an object detector trained on natural images has no
  prior for it, which is the same reason ``depth.py`` exists at all.
* **COCO detector on ``IR``.** Same 60 clips, same threshold: **96.7%**, median score 0.87.
  IR is a real grayscale image of the scene, so the prior transfers.

IR is the right input for a second reason: it comes off the *same* Vzense NYX 650 at the
same 640x480 as Depth_Color, so a box found in IR is valid in Depth_Color pixel-for-pixel.
That is the identical premise the fused cache is built on (see ``cuhkx.fused``), so using
IR to place a window shared by both channels adds no new assumption.

**Licence, deliberately.** The external 0.716 solution crops with YOLO11n, which is
**AGPL-3.0**; the finals require the top entries to open-source under **Apache-2.0** within
30 days, so a YOLO in the inference path is a compliance problem, not just a dependency.
``ssdlite320_mobilenet_v3_large`` ships with torchvision under **BSD-3-Clause** at 3.44M
parameters (13.8 MB fp32, 6.9 MB fp16), which fits the 100 MB delivery budget with room to
spare and can be released.

The window geometry deliberately copies the external solution's, because that is the one
configuration with a measured leaderboard score on these exact clips: probe frames spread
over the clip, the **median** of their box centres, a side taken from the **largest** box,
a 1.40 margin, and a floor under the side so a distant actor still yields a usable crop.
"""
# Role: steps of the per-clip person-window cascade: detector loading, the best person box per
#   IR probe frame, the clip-level accept rule, the four-tile rescue pass and the window geometry
#   (median box centre, side max(1.4 x largest box side, 0.35 x frame width)).
# Used by: scripts/22_person_windows.py, which runs the cascade (inference.sh reaches it through
#   scripts/el25r_p1_repeat_runtime.py --detector-only), and tools/make_preprocessing_figure.py;
#   both (the training caches were cut with windows from the same script).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Grow the detected box about its centre. 1.40 is the external solution's ``CROP_MARGIN``;
#: it exists because a tight person box cuts off the objects the action is defined by --
#: the television in ``25_Watch_TV``, the desk in ``18_Write``.
CROP_MARGIN = 1.40

#: Never crop tighter than this fraction of the frame width. A confident but tiny box (a
#: seated actor far from the camera) would otherwise be upsampled into a blur.
MIN_SIDE_FRACTION = 0.35

#: How many frames to run the detector on. One frame is a coin flip on a bad pose; eight
#: spread over the clip and combined by median is not.
DETECTION_FRAMES = 8

#: torchvision detection heads use COCO category ids, where 1 is ``person``.
PERSON_LABEL = 1

#: A single frame is believed on its own at this score.
STRONG_SCORE = 0.30

#: ...and below it, only with agreement. IR is out of distribution for a COCO detector, so
#: its scores are systematically depressed: measured over 60 test clips, four came back
#: with a top-person score of 0.13-0.25 on **every** one of eight probe frames, with the box
#: in the same place each time. A confident detection is not the only evidence available --
#: a weak box that does not move across eight frames spread over the clip is a person, and a
#: weak box that jumps around the frame is not. Rejecting the former sends the clip to the
#: motion fallback, which is exactly the rung that fails on the seated classes.
WEAK_SCORE = 0.10

#: Fraction of probe frames that must carry a weak box for the agreement rule to fire.
WEAK_QUORUM = 0.5

#: ...and how far their centres may scatter, as a fraction of frame width.
WEAK_SPREAD = 0.10

# Per-frame floor in detect_boxes and detect_boxes_tiled: a frame whose best person box scores
# lower contributes no box at all.
MIN_SCORE = WEAK_SCORE


# Typed row of the person-window table. Only read_windows builds it, and nothing in this package
# calls read_windows: scripts/21_build_fused_cache.py reads the parquet directly.
@dataclass(frozen=True)
class PersonWindow:
    """One square crop window per clip, in source-image pixels. May exceed the frame."""

    top: int
    left: int
    bottom: int
    right: int
    n_detected: int
    # scripts/22_person_windows.py writes source as strong, agree, tiled, motion or whole.
    source: str  # "detector" | "motion" | "whole"

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.top, self.left, self.bottom, self.right


def probe_indices(n_frames: int, n_probe: int = DETECTION_FRAMES) -> np.ndarray:
    """Evenly spaced frame indices to run the detector on."""
    if n_frames <= 0:
        return np.zeros(0, dtype=int)
    n = min(n_probe, n_frames)
    return np.linspace(0, n_frames - 1, n).round().astype(int)


def window_from_boxes(
    boxes: np.ndarray,
    height: int,
    width: int,
    margin: float = CROP_MARGIN,
    min_side_fraction: float = MIN_SIDE_FRACTION,
) -> tuple[int, int, int, int]:
    """``(N, 4)`` xyxy boxes from several frames -> one square window for the whole clip.

    Median centre, max side. The median is over *centres* rather than over box edges
    because a single frame where the detector grabs a chair as well as the actor moves an
    edge a long way and the centre hardly at all. The side takes the maximum instead, since
    the window has to contain the actor in every frame, not the typical one.

    Returned coordinates are **unclamped** on purpose: the caller pads (``fused.crop_pad``).
    Clamping a square box that pokes outside the frame makes it non-square again, which
    reintroduces the per-clip aspect distortion the square was for.
    """
    # boxes columns are (x1, y1, x2, y2) in full-frame IR pixels, one row per accepted frame.
    centres_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centres_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centre_x, centre_y = float(np.median(centres_x)), float(np.median(centres_y))
    # Side: the largest box width or height in the clip, times the margin, and never less than
    # min_side_fraction of the frame width.
    side = float(np.max(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
    side = max(side * margin, min_side_fraction * width)
    half = side / 2.0
    del height  # kept in the signature so callers pass the frame shape, not just a width
    # (top, left, bottom, right) of a square window; it may extend past the frame edges.
    return (
        int(round(centre_y - half)),
        int(round(centre_x - half)),
        int(round(centre_y + half)),
        int(round(centre_x + half)),
    )


def load_detector(device: str = "cuda", weights_path: Path | str | None = None):
    """SSDLite320 MobileNetV3-Large, COCO weights, eval mode. BSD-3, 3.44M parameters.

    ``weights_path`` loads the detector from a local checkpoint instead of torchvision's
    download cache, and is how the graded run gets its detector. Two reasons it has to exist:
    the organisers count the preprocessing model inside the 100 MB budget (C-9, C-10), which
    only means anything if the packaged copy is the one that runs; and the finals are a
    45-minute recorded session on a two-hour clock, where reaching for the network is a
    failure mode we can simply not have. ``weights=None, weights_backbone=None`` because the
    default would still fetch the ImageNet backbone before the state dict overwrote it.

    Accepts a plain state dict or one packed by ``budget.quantize_state``.
    """
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    # Without a path, torchvision's COCO weights come from its download cache. The delivered run
    # always passes one (--detector-weights weights/model.pt).
    if weights_path is None:
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    else:
        from cuhkx.budget import dequantize_state

        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        blob = torch.load(weights_path, map_location="cpu", weights_only=True)
        # A packed checkpoint keeps the detector under components/detector. The delivered run
        # wraps torch.load so that weights/model.pt comes back in this form (see
        # scripts/el25r_p1_repeat_runtime.py, --detector-only).
        if "components" in blob:
            if "detector" not in blob["components"]:
                raise ValueError("packed deliverable has no detector component")
            blob = blob["components"]["detector"]
        # dequantize_state expands quantised entries and keeps fp32 tensors as they are; the
        # delivered detector is stored in fp32.
        model.load_state_dict(dequantize_state(blob))
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return model.to(device), device


def detect_boxes(model, device: str, frames: np.ndarray, min_score: float = MIN_SCORE):
    """``(n, H, W)`` uint8 IR frames -> ``((m, 4)`` xyxy person boxes, ``(m,)`` scores).

    Only the highest-scoring person per frame is kept. The dataset is single-subject, so a
    second person is a false positive (commonly a chair or a coat rack), and keeping it
    would drag the median centre off the actor.

    Scores come back with the boxes rather than being thresholded away here, because the
    accept/reject decision is made per **clip** by :func:`accept_boxes`, not per frame.
    """
    import torch

    if len(frames) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    # The detector takes a list of 3-channel float images in [0, 1], so the IR channel is
    # repeated three times. torchvision resizes each image to 320 x 320 internally and returns
    # boxes in the input image's pixels.
    batch = [
        torch.from_numpy(np.repeat(f[None], 3, axis=0)).float().div_(255.0).to(device)
        for f in frames
    ]
    with torch.no_grad():
        outputs = model(batch)
    boxes, scores = [], []
    for out in outputs:
        # Detections come sorted by descending score, so index 0 is the best person box.
        mask = out["labels"] == PERSON_LABEL
        s, b = out["scores"][mask], out["boxes"][mask]
        # A frame whose best person is below min_score adds nothing, so the result can have
        # fewer rows than there are probe frames.
        if len(s) and float(s[0]) >= min_score:
            boxes.append(b[0].detach().cpu().numpy())
            scores.append(float(s[0]))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(boxes).astype(np.float32), np.asarray(scores, dtype=np.float32)


def accept_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    n_probe: int = DETECTION_FRAMES,
) -> tuple[np.ndarray, str]:
    """Decide, for one clip, whether the detector found the actor. ``(boxes, reason)``.

    Two ways to pass, and the second is the one that matters here:

    * **strong** -- any probe frame at or above :data:`STRONG_SCORE`. This is the ordinary
      case: 90% of test clips, median top score 0.87.
    * **agree** -- at least half the probe frames carry a box above :data:`WEAK_SCORE`
      *and* those boxes sit in the same place, within :data:`WEAK_SPREAD` of the frame
      width. Consistency across eight frames spread over the whole clip is independent
      evidence that a low-confidence box is real, and it is evidence a per-frame threshold
      throws away.

    An empty return means fall through to the motion box.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), "none"
    # Strong: one confident frame is enough; every box at or above the weak threshold is kept.
    if float(scores.max()) >= STRONG_SCORE:
        return boxes[scores >= WEAK_SCORE], "strong"
    # Agreement needs max(2, round(0.5 x n_probe)) weak boxes, i.e. 4 with the default 8 probe
    # frames. n_probe is the requested count, so a clip with fewer frames needs the same number.
    weak = scores >= WEAK_SCORE
    if weak.sum() < max(2, int(round(WEAK_QUORUM * n_probe))):
        return np.zeros((0, 4), dtype=np.float32), "sparse"
    kept = boxes[weak]
    # Scatter: the larger of the x and y standard deviations of the box centres, in pixels.
    centres = np.stack([(kept[:, 0] + kept[:, 2]) / 2.0, (kept[:, 1] + kept[:, 3]) / 2.0], axis=1)
    if float(centres.std(axis=0).max()) > WEAK_SPREAD * width:
        return np.zeros((0, 4), dtype=np.float32), "scattered"
    return kept, "agree"


# Second rung of the cascade: the same detector on four overlapping corner tiles after a
# contrast stretch, for clips that fail the whole-frame pass.
#: Side of each tile as a fraction of the frame, in the small-actor rescue pass.
TILE_FRACTION = 0.6

#: Percentiles the IR frame is stretched to before the rescue pass.
STRETCH_PERCENTILES = (1.0, 99.5)


def stretch_contrast(
    frames: np.ndarray, percentiles: tuple[float, float] = STRETCH_PERCENTILES
) -> np.ndarray:
    """Per-frame percentile stretch to full range. Cheap, and the rescue scenes are dark."""
    out = np.empty_like(frames)
    low, high = percentiles
    for i, frame in enumerate(frames):
        lo, hi = np.percentile(frame, [low, high])
        # A flat frame (for example all zeros) has no range to stretch and is copied unchanged.
        if hi <= lo:
            out[i] = frame
        else:
            out[i] = np.clip((frame.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(
                np.uint8
            )
    return out


def tile_offsets(height: int, width: int, fraction: float = TILE_FRACTION):
    """Four overlapping tiles, one per corner. ``(oy, ox, tile_h, tile_w)``."""
    # Order: top-left, top-right, bottom-left, bottom-right. With fraction 0.6, neighbouring
    # tiles overlap by 20% of the frame height or width.
    tile_h, tile_w = int(height * fraction), int(width * fraction)
    return [
        (0, 0, tile_h, tile_w),
        (0, width - tile_w, tile_h, tile_w),
        (height - tile_h, 0, tile_h, tile_w),
        (height - tile_h, width - tile_w, tile_h, tile_w),
    ]


def detect_boxes_tiled(model, device: str, frames: np.ndarray, min_score: float = MIN_SCORE):
    """Rescue pass for actors too small for the detector's input resolution.

    ``ssdlite320`` resizes whatever it is given to 320x320. Measured on the 123 train clips
    the plain pass missed: every one has readable IR, and the failure mode is a single scene
    type -- a dim living room with the actor roughly 50 px tall in a 640x480 frame, so about
    25 px after the resize, below the scale the model fires at. The Depth_Color channel shows
    the person plainly, so the information is present and the detector's *input* is what is
    wrong.

    Running the same detector on four overlapping 60% tiles makes the actor ~1.7x larger
    relative to the 320 px input, and stretching the contrast first fixes the dark scene.
    Measured recovery over those 123 clips:

    ==========================  ======
    percentile stretch only      21.1%
    LUT-decoded depth, inverted  22.0%
    **four tiles**               87.0%
    **four tiles + stretch**     89.4%
    ==========================  ======

    Boxes are mapped back to full-frame coordinates, and the best-scoring tile wins per
    frame, so the output is interchangeable with :func:`detect_boxes`.

    This is a *cascade* rung, not the default: it costs four extra forward passes, and only
    about 4% of clips reach it.
    """
    if len(frames) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    # Stretch the contrast, run the detector on each tile of every frame, and keep per frame the
    # best-scoring person box over the four tiles.
    stretched = stretch_contrast(frames)
    height, width = frames.shape[1], frames.shape[2]
    best_score = np.zeros(len(frames), dtype=np.float32)
    best_box = np.zeros((len(frames), 4), dtype=np.float32)
    for oy, ox, tile_h, tile_w in tile_offsets(height, width):
        tile = np.ascontiguousarray(stretched[:, oy : oy + tile_h, ox : ox + tile_w])
        boxes, scores = _detect_raw(model, device, tile)
        # Shift tile coordinates back to the full frame (x1, y1, x2, y2 get ox, oy, ox, oy).
        boxes = boxes + np.array([ox, oy, ox, oy], dtype=np.float32)
        # A frame with no person in this tile has score 0 and never replaces an earlier box.
        better = scores > best_score
        best_score[better], best_box[better] = scores[better], boxes[better]
    keep = best_score >= min_score
    return best_box[keep], best_score[keep]


def _detect_raw(model, device: str, frames: np.ndarray):
    """One forward pass; the top person box and score per frame, zeros where there is none."""
    import torch

    batch = [
        torch.from_numpy(np.repeat(f[None], 3, axis=0)).float().div_(255.0).to(device)
        for f in frames
    ]
    with torch.no_grad():
        outputs = model(batch)
    # Unlike detect_boxes, every frame keeps a row (zeros when there is no person), so the four
    # tiles can be compared frame by frame.
    boxes = np.zeros((len(frames), 4), dtype=np.float32)
    scores = np.zeros(len(frames), dtype=np.float32)
    for i, out in enumerate(outputs):
        mask = out["labels"] == PERSON_LABEL
        s, b = out["scores"][mask], out["boxes"][mask]
        if len(s):
            boxes[i] = b[0].detach().cpu().numpy()
            scores[i] = float(s[0])
    return boxes, scores


def read_windows(path: Path) -> dict[str, PersonWindow]:
    """Load the table written by ``scripts/22_person_windows.py``."""
    import pandas as pd

    table = pd.read_parquet(path)
    return {
        row.clip_id: PersonWindow(
            int(row.top),
            int(row.left),
            int(row.bottom),
            int(row.right),
            int(row.n_detected),
            str(row.source),
        )
        for row in table.itertuples()
    }
