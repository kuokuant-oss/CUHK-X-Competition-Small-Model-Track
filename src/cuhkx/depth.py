"""Undo the colormap on ``Depth_Color``, and find the person in the frame.

``Depth_Color`` is not a photograph. It is a scalar depth map pushed through a JET-family
colormap, and treating it as RGB costs us twice:

* **Resizing blends colours that mean nothing.** Averaging two LUT entries lands off the
  colormap curve entirely, and decodes to a depth that neither neighbour had. The error is
  worst exactly at depth discontinuities — which is the outline of the person.
* **17.98% of pixels are pure black**, the reserved "no return" colour. As RGB that reads
  as a very dark blue, i.e. a plausible depth, so the network cannot tell missing data
  from near data.

Measured, not assumed: 100 clips / 30.7M pixels contain exactly **255 distinct colours**
plus black. Against matplotlib's ``jet`` the median distance is 6.55 (p95 10.93) and 233 of
256 indices are hit, so it is a jet *variant* — close enough to order by, not close enough
to decode by. So the lookup table is built **from the data**, using nearest-jet-index only
as a sort key; the resulting depth scale is monotone and evenly spaced, which is all a
network needs since the true metric range is not published anyway.
"""
# Role: decodes Depth_Color to a scalar depth rank with a palette lookup table (0-254; 255 = no
#   return or unknown colour) and finds one motion box per clip on the decoded depth.
# Used by: fused.load_clip_fused (miw box, motion fallback), scripts/22_person_windows.py (motion
#   fallback of the person window) and data.py; both. At inference the palette is the copy in
#   weights/model.pt: scripts/el25r_p1_repeat_runtime.py checks its SHA256 and writes it to
#   depth_lut.npz (or compares it with an existing copy). scan_colours, build_lut, save_lut and
#   smoothness make and check a palette offline; nothing in this package calls them.

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

#: Packed-RGB space, so decoding is one fancy-index instead of a nearest-neighbour search.
_TABLE_SIZE = 1 << 24  # One uint8 depth code per 24-bit RGB colour: a 16 MiB table.
INVALID = 255  # Code of black ("no return") and of every colour outside the palette.


def pack(rgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` uint8 -> ``(...)`` uint32, one integer per colour."""
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def scan_colours(image_paths: list[Path], stride: int = 1) -> Counter:
    """Count every distinct colour over a sample of frames."""
    from PIL import Image

    counts: Counter = Counter()
    for path in image_paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)[::stride]
        values, seen = np.unique(pack(pixels), return_counts=True)
        counts.update(dict(zip(values.tolist(), seen.tolist(), strict=True)))
    return counts


def build_lut(counts: Counter, min_pixels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(colours, depths)`` — the observed palette, ordered along the colormap.

    Black is dropped here rather than given a depth: it is the sensor's "no return" code,
    and giving it a value would place missing data somewhere on the depth scale.
    """
    import matplotlib

    # Colours seen at least min_pixels times, as packed integers, leaving out pure black (0).
    packed = np.array(
        [value for value, seen in counts.items() if value != 0 and seen >= min_pixels],
        dtype=np.uint32,
    )
    colours = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(
        np.uint8
    )

    # matplotlib's 256-entry jet in 0-255 RGB, used only as a sort key.
    reference = np.array([matplotlib.colormaps["jet"](i / 255.0)[:3] for i in range(256)]) * 255
    distance = np.linalg.norm(colours[:, None, :].astype(float) - reference[None, :, :], axis=2)
    # Sort by nearest jet index, ties broken by the distance to that entry (np.lexsort sorts by
    # its last key first).
    order = np.lexsort((distance.min(axis=1), distance.argmin(axis=1)))

    colours = colours[order]
    # Evenly spaced ranks 0..254 in palette order; 255 stays reserved for INVALID. The palette
    # stored in weights/model.pt has 254 colours, so every colour gets its own rank.
    depths = np.linspace(0, 254, len(colours)).round().astype(np.uint8)
    return colours, depths


def lookup_table(colours: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Expand the palette into a packed-RGB -> depth array; unknown colours map to INVALID."""
    # Every colour not in the palette, black included, keeps INVALID.
    table = np.full(_TABLE_SIZE, INVALID, dtype=np.uint8)
    table[pack(colours)] = depths
    return table


def decode(rgb: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(H, W, 3)`` colourised depth -> ``(depth uint8, valid bool)``.

    ``depth`` is 0..254 with 255 meaning invalid, so the pair round-trips through a single
    uint8 array and a caller that ignores the mask still sees a legal image.
    """
    depth = table[pack(rgb)]
    return depth, depth != INVALID


def save_lut(path: Path, colours: np.ndarray, depths: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, colours=colours, depths=depths)


def load_lut(path: Path) -> np.ndarray:
    """Load a saved palette and return the expanded lookup table."""
    blob = np.load(Path(path), allow_pickle=False)
    return lookup_table(blob["colours"], blob["depths"])


def smoothness(depth: np.ndarray, valid: np.ndarray) -> float:
    """Mean absolute horizontal difference over valid neighbours.

    This is the falsification test for the palette ordering. A real depth map is piecewise
    smooth, so a correct ordering scores low; a wrong one scrambles neighbouring depths and
    scores near the value of a random permutation. Comparing the two is what turns "the
    ordering looks right" into a number.
    """
    left, right = depth[:, :-1].astype(np.int16), depth[:, 1:].astype(np.int16)
    both = valid[:, :-1] & valid[:, 1:]
    if not both.any():
        return float("nan")
    return float(np.abs(left - right)[both].mean())


def motion_bbox(
    depths: np.ndarray,
    valid: np.ndarray,
    quantile: float = 0.98,
    margin: float = 0.25,
    min_side: int = 48,
) -> tuple[int, int, int, int]:
    """``(top, left, bottom, right)`` around the part of the frame that moves.

    The person is found by *motion*, not by depth value: the background is static across a
    clip while the actor is not, so the deviation from the per-pixel temporal median
    localises them without needing to know which end of the colormap is near — a polarity
    the dataset does not document.

    One box for the whole clip, never per frame. A box that tracked the person would hold
    them still and turn their movement into background movement, destroying the very cue
    the action is made of.
    """
    # depths and valid are (T, H, W). The callers pass every 2nd or 4th pixel, so the box and
    # min_side are in the pixels of the array given, not of the full frame.
    height, width = depths.shape[1:]
    # Step 1: per-pixel motion, the mean absolute deviation from the per-pixel temporal median,
    # computed over the frames in which that pixel is valid.
    usable = depths.astype(np.float32)
    usable[~valid] = np.nan
    # A pixel the sensor never saw in any frame has no motion by definition. Saying so up
    # front is better than letting it become an all-NaN slice and silencing the warning:
    # warnings filters are not thread-safe, and this runs inside a thread pool.
    usable[:, ~valid.any(axis=0)] = 0.0
    background = np.nanmedian(usable, axis=0)
    motion = np.nan_to_num(np.nanmean(np.abs(usable - background), axis=0), nan=0.0)

    # Step 2: keep the pixels at or above the given quantile of the non-zero motion values (the
    # top 2% by default) and take their bounding box. No motion at all gives the whole frame.
    if not np.any(motion > 0):
        return 0, 0, height, width
    threshold = np.quantile(motion[motion > 0], quantile)
    rows, cols = np.nonzero(motion >= threshold)
    if len(rows) == 0:
        return 0, 0, height, width

    # Step 3: pad each side by margin times the box height or width, clamped to the frame.
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    pad_y = int((bottom - top) * margin)
    pad_x = int((right - left) * margin)
    top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)
    left, right = max(0, left - pad_x), min(width, right + pad_x)

    # A degenerate box means the motion cue failed; the whole frame is the honest fallback.
    if bottom - top < min_side or right - left < min_side:
        return 0, 0, height, width

    # Grow the short side to square before cropping. The output is square, so a box that is
    # not would be squashed — and by a different amount per clip, which turns a person's
    # build into a per-clip distortion the model would have to learn to ignore.
    side = max(bottom - top, right - left)
    centre_y, centre_x = (top + bottom) // 2, (left + right) // 2
    half = side // 2
    top, bottom = centre_y - half, centre_y + (side - half)
    left, right = centre_x - half, centre_x + (side - half)
    # The square is clamped to the frame again, so near an edge it can come back non-square; the
    # callers in fused.py and scripts/22_person_windows.py re-square it with fused.square_window.
    top, bottom = max(0, top), min(height, bottom)
    left, right = max(0, left), min(width, right)
    return top, left, bottom, right
