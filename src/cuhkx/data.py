"""Compact caches for the light modalities, and the Dataset that reads them.

Iteration speed decides this competition (plan §2), and the cost is not compute — it is
opening 86,050 skeleton JSONs and 5,806 IMU CSVs per epoch. So each modality is parsed
**once** into a single ragged array that fits in RAM:

* skeleton — 86,050 frames x 17 joints x 3 = about 18 MB as float32
* IMU — roughly 273,000 rows x 16 channels = about 17 MB

Sequences are cached **raw and variable-length**, not resampled to a fixed T. Resampling is
cheap at load time, while re-parsing to try a different T is not, so this keeps the frame
count a free hyperparameter instead of baking it into the cache.

Ragged layout: one ``data`` array of every clip's rows concatenated, plus ``offsets`` where
clip ``i`` occupies ``data[offsets[i]:offsets[i+1]]``. Empty clips are legal and give an
empty slice — some clips genuinely have no IMU at all.
"""
# Role: the ragged per-clip cache layout (RaggedCache, _pack) and a frame decoder that skips
#   unreadable files (_decode_frames), both used by the fused caches; plus per-modality caches and
#   readers from development (skeleton, IMU, single-modality images, decoded depth).
# Used by: fused.py and scripts/22_person_windows.py (IMAGE_SUFFIXES, RaggedCache, _pack,
#   _decode_frames); both. The per-modality cache builders and the *_tensor readers are not used
#   by the delivered run.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cuhkx import depth as depth_codec
from cuhkx.imu import N_CHANNELS, N_DEVICES, parse_imu_file
from cuhkx.skeleton import N_JOINTS, load_clip_skeleton, normalize, resample

MODALITY_DIRNAME = {
    "skeleton": "Skeleton",
    "imu": "IMU",
    "depth": "Depth_Color",
    "ir": "IR",
    "thermal": "Thermal",
}

IMAGE_MODALITIES = ("depth", "ir", "thermal")

#: Frames are 640x480; an exact 1/8 gives (80, 60) as PIL ``(width, height)``. Keeping the
#: 4:3 ratio matters more than reaching a round square size — squashing to a square would
#: change every aspect-dependent cue (a person's height-to-width) into a fixed distortion.
IMAGE_SIZE = (80, 60)

#: Depth is cropped to the actor first, so it can afford square and larger: the crop is
#: roughly 2x downscale instead of the 8x a full frame needs.
DEPTH_SIZE = (112, 112)
# File extensions treated as frames; callers compare them with the lower-cased suffix.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class RaggedCache:
    """Variable-length per-clip sequences stored as one array plus offsets."""

    data: np.ndarray
    offsets: np.ndarray
    clip_ids: list[str]
    extra: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.clip_ids)

    @property
    def index(self) -> dict[str, int]:
        return {clip_id: i for i, clip_id in enumerate(self.clip_ids)}

    def get(self, i: int) -> np.ndarray:
        return self.data[self.offsets[i] : self.offsets[i + 1]]

    def get_extra(self, name: str, i: int) -> np.ndarray:
        return self.extra[name][self.offsets[i] : self.offsets[i + 1]]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=self.data,
            offsets=self.offsets,
            clip_ids=np.asarray(self.clip_ids),
            **self.extra,
        )

    @classmethod
    def load(cls, path: Path) -> RaggedCache:
        blob = np.load(path, allow_pickle=False)
        reserved = {"data", "offsets", "clip_ids"}
        return cls(
            data=blob["data"],
            offsets=blob["offsets"],
            clip_ids=[str(c) for c in blob["clip_ids"]],
            extra={k: blob[k] for k in blob.files if k not in reserved},
        )


def _pack(
    sequences: list[np.ndarray],
    clip_ids: list[str],
    row_shape: tuple[int, ...],
    dtype: type = np.float32,
):
    """``dtype`` is a parameter because images must stay ``uint8``: as float32 the training
    frames would be 5 GB instead of 1.2 GB, and the conversion is one cheap op on the GPU."""
    # offsets[i]:offsets[i + 1] are clip i's rows; offsets[-1] is the total number of rows.
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(seq)

    # Allocate once and fill in place, releasing each sequence as it is copied. The obvious
    # `np.concatenate(...).astype(dtype)` needs three copies of the whole set alive at the
    # same time — the list, the concatenation, and astype's output — which is 9 GB for the
    # thermal cache and fails outright.
    data = np.empty((int(offsets[-1]), *row_shape), dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq):
            data[offsets[i] : offsets[i + 1]] = seq
        sequences[i] = None
    return data, offsets, clip_ids


# Development caches for the skeleton and IMU modalities; not used by the delivered run.
def build_skeleton_cache(clip_dirs: dict[str, Path]) -> RaggedCache:
    """Parse every clip's pose JSONs once. Poses are stored *unnormalised*, so the
    normalisation choice stays revisable without re-parsing (ADR 0005 measured one already,
    but that decision should not be frozen into the cache)."""
    clip_ids = list(clip_dirs)
    sequences = [load_clip_skeleton(clip_dirs[c]) for c in clip_ids]
    data, offsets, ids = _pack(sequences, clip_ids, (N_JOINTS, 3))
    return RaggedCache(data=data, offsets=offsets, clip_ids=ids, extra={})


def build_imu_cache(clip_dirs: dict[str, Path]) -> RaggedCache:
    """Parse the two CSVs per clip into rows tagged with which of the 5 devices they came
    from, so each device's own irregular clock can be resampled separately at load time."""
    clip_ids = list(clip_dirs)
    sequences, device_ids = [], []
    for clip_id in clip_ids:
        per_device: dict[str, np.ndarray] = {}
        clip_dir = clip_dirs[clip_id]
        if clip_dir.is_dir():
            for csv_path in sorted(clip_dir.glob("*.csv")):
                per_device.update(parse_imu_file(csv_path))
        rows, devs = [], []
        for name, series in per_device.items():
            from cuhkx.imu import DEVICE_INDEX

            rows.append(series)
            devs.append(np.full(len(series), DEVICE_INDEX[name], dtype=np.int8))
        if rows:
            sequences.append(np.concatenate(rows, axis=0))
            device_ids.append(np.concatenate(devs))
        else:
            sequences.append(np.zeros((0, N_CHANNELS), dtype=np.float32))
            device_ids.append(np.zeros(0, dtype=np.int8))

    data, offsets, ids = _pack(sequences, clip_ids, (N_CHANNELS,))
    devices = (
        np.concatenate([d for d in device_ids if len(d)])
        if any(len(d) for d in device_ids)
        else np.zeros(0, dtype=np.int8)
    )
    return RaggedCache(data=data, offsets=offsets, clip_ids=ids, extra={"device": devices})


# Frame decoding shared by every image cache, the fused one included.
def _decode_frames(files, convert: str = "RGB"):
    """Decode frames, skipping any the decoder cannot read, and say how many were skipped.

    A single unreadable file used to abort the whole cache build. That is the wrong trade
    at any time and an unacceptable one at the finals, where the test set is handed over on
    the day and a cache build is the first thing that runs: losing one clip's frames costs
    one prediction, while raising costs the entire submission.

    It is not hypothetical. Four of the 405 Kaggle test clips ship IR frames whose bytes are
    all zero -- correct size, correct name, no PNG header -- and PIL raises
    ``UnidentifiedImageError`` on them. Nothing about the data promises the finals set is
    cleaner.
    """
    from PIL import Image, UnidentifiedImageError

    # convert is a PIL mode: "RGB" for Depth_Color, "L" (8-bit grey) for IR. Frames keep file
    # order; unreadable files are counted and left out.
    frames, skipped = [], 0
    for path in files:
        try:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert(convert), dtype=np.uint8))
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
    return frames, skipped


# Single-modality image and decoded-depth caches from development; not used by the delivered run.
def _load_clip_images(clip_dir: Path, size: tuple[int, int], max_frames: int = 0) -> np.ndarray:
    """Decode one clip's frames to ``(n, H, W, 3)`` uint8, or an empty array if it has none.

    Downsampling is bilinear, which does blend neighbouring pixels. On a colormapped depth
    image (ADR 0002) that is not strictly meaning-preserving — the average of two colormap
    colours is not the colour of the average depth — but the error is confined to depth
    discontinuities, since smooth regions have near-identical neighbours. Nearest-neighbour
    avoids inventing colours at the cost of throwing away 63 of every 64 pixels. Which is
    better is an ablation, not something to assume; see the Phase 4 plan.
    """
    from PIL import Image

    if not clip_dir.is_dir():
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)
    files = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames and len(files) > max_frames:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_frames).round().astype(int)]
    decoded, skipped = _decode_frames(files)
    if skipped:
        print(f"  !! {clip_dir}: skipped {skipped}/{len(files)} unreadable frame(s)")
    frames = [
        np.asarray(Image.fromarray(frame).resize(size, Image.BILINEAR), dtype=np.uint8)
        for frame in decoded
    ]
    if not frames:
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)
    return np.stack(frames)


def _load_clip_cropped(clip_dir: Path, size: tuple[int, int], max_frames: int = 0) -> np.ndarray:
    """One clip's frames, cropped to the actor and resized to ``size``, as ``(n, H, W, 3)``.

    The same motion cue the depth path uses — deviation from the per-pixel temporal median —
    but computed on luminance, so it works for a modality whose colours we cannot decode.
    Thermal is colour-mapped by the camera vendor and then JPEG-compressed: no standard
    colormap fits it (the closest, plasma, lands a median 26.3 RGB units away with 4% of
    pixels within 12), so there is no scalar to recover and the crop has to work on what is
    actually there.
    """
    from PIL import Image

    if not clip_dir.is_dir():
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)
    files = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    # Thermal runs at ~25 fps, so a two-second clip is ~50 frames while training never asks
    # for more than 32. Caching the rest would cost 7.6 GB of RAM to hold frames no epoch
    # ever reads, so the cap is applied here rather than regretted later.
    if max_frames and len(files) > max_frames:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_frames).round().astype(int)]
    frames, skipped = _decode_frames(files)
    if skipped:
        print(f"  !! {clip_dir}: skipped {skipped}/{len(files)} unreadable frame(s)")
    if not frames:
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)

    stack = np.stack(frames)
    # ITU-R BT.601 luma weights.
    luminance = stack.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    step = 2
    small = luminance[:, ::step, ::step].astype(np.uint8)
    top, left, bottom, right = depth_codec.motion_bbox(
        small, np.ones_like(small, dtype=bool), min_side=32 // step
    )
    top, left, bottom, right = top * step, left * step, bottom * step, right * step
    bottom, right = min(bottom, stack.shape[1]), min(right, stack.shape[2])

    out = np.empty((len(stack), size[1], size[0], 3), dtype=np.uint8)
    for i, frame in enumerate(stack):
        window = Image.fromarray(frame[top:bottom, left:right])
        out[i] = np.asarray(window.resize(size, Image.BILINEAR), dtype=np.uint8)
    return out


def build_image_cache(
    clip_dirs: dict[str, Path],
    size: tuple[int, int] = DEPTH_SIZE,
    workers: int = 8,
    crop_to_actor: bool = True,
    max_frames: int = 32,
) -> RaggedCache:
    """Decode an image modality once into a RAM-sized uint8 cache.

    Threads rather than processes: the cost here is image decoding inside Pillow, which
    releases the GIL, so threads get the parallelism without paying Windows' process-spawn
    cost or pickling a gigabyte of pixels back to the parent.
    """
    clip_ids = list(clip_dirs)
    if crop_to_actor:

        def loader(clip_id: str) -> np.ndarray:
            return _load_clip_cropped(clip_dirs[clip_id], size, max_frames)
    else:

        def loader(clip_id: str) -> np.ndarray:
            return _load_clip_images(clip_dirs[clip_id], size, max_frames)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sequences = list(pool.map(loader, clip_ids))
    data, offsets, ids = _pack(sequences, clip_ids, (size[1], size[0], 3), dtype=np.uint8)
    return RaggedCache(data=data, offsets=offsets, clip_ids=ids, extra={})


def _load_clip_depth(clip_dir: Path, table: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """One clip's frames as ``(n, H, W)`` uint8 depth, 255 meaning no return.

    The colormap is undone *before* anything else, so every later step — cropping,
    resizing, augmentation — operates on a scalar depth where averaging is meaningful.
    Doing it in the other order is the defect this replaces.
    """
    from PIL import Image

    if not clip_dir.is_dir():
        return np.zeros((0, size[1], size[0]), dtype=np.uint8)
    files = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    decoded, skipped = _decode_frames(files)
    if skipped:
        print(f"  !! {clip_dir}: skipped {skipped}/{len(files)} unreadable frame(s)")
    frames = [depth_codec.decode(frame, table)[0] for frame in decoded]
    if not frames:
        return np.zeros((0, size[1], size[0]), dtype=np.uint8)

    depths = np.stack(frames)
    valid = depths != depth_codec.INVALID

    # The box is found on a 1/4-scale copy. A bounding box does not need full resolution,
    # and the temporal median behind it is the most expensive step in the whole build.
    step = 4
    top, left, bottom, right = depth_codec.motion_bbox(
        depths[:, ::step, ::step], valid[:, ::step, ::step], min_side=48 // step
    )
    top, left, bottom, right = top * step, left * step, bottom * step, right * step
    bottom, right = min(bottom, depths.shape[1]), min(right, depths.shape[2])

    out = np.empty((len(depths), size[1], size[0]), dtype=np.uint8)
    for i, (frame, keep) in enumerate(zip(depths, valid, strict=True)):
        window, window_valid = frame[top:bottom, left:right], keep[top:bottom, left:right]
        # Fill the holes with a plausible depth before resampling, so a missing return does
        # not drag its neighbours toward zero and invent an edge that is not there.
        filled = np.where(
            window_valid,
            window,
            np.uint8(np.median(window[window_valid])) if window_valid.any() else 0,
        )
        small = np.asarray(Image.fromarray(filled).resize(size, Image.BILINEAR))
        mask = np.asarray(
            Image.fromarray(window_valid.astype(np.uint8)).resize(size, Image.NEAREST)
        ).astype(bool)
        out[i] = np.where(mask, np.minimum(small, depth_codec.INVALID - 1), depth_codec.INVALID)
    return out


def build_depth_cache(
    clip_dirs: dict[str, Path],
    lut_path: Path,
    size: tuple[int, int] = DEPTH_SIZE,
    workers: int = 8,
) -> RaggedCache:
    """Decode, crop to the actor, and cache Depth_Color as scalar depth."""
    table = depth_codec.load_lut(lut_path)
    clip_ids = list(clip_dirs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sequences = list(pool.map(lambda c: _load_clip_depth(clip_dirs[c], table, size), clip_ids))
    data, offsets, ids = _pack(sequences, clip_ids, (size[1], size[0]), dtype=np.uint8)
    return RaggedCache(data=data, offsets=offsets, clip_ids=ids, extra={})


# Readers that turn one cached clip into a fixed-length model input (development pipeline).
def depth_tensor(cache: RaggedCache, i: int, n_frames: int) -> np.ndarray:
    """``(n_frames, 2, H, W)`` uint8: channel 0 is depth, channel 1 is the validity mask.

    The mask is a channel rather than a convention because 17.98% of pixels are "no return"
    and that is information — where the sensor saw nothing is itself a cue — not a value to
    be imputed and forgotten.
    """
    frames = cache.get(i)
    height, width = cache.data.shape[1], cache.data.shape[2]
    if len(frames) == 0:
        return np.zeros((n_frames, 2, height, width), dtype=np.uint8)
    picks = np.linspace(0, len(frames) - 1, n_frames).round().astype(np.int64)
    chosen = frames[picks]
    valid = chosen != depth_codec.INVALID
    return np.stack([np.where(valid, chosen, 0), valid.astype(np.uint8) * 255], axis=1)


def image_tensor(cache: RaggedCache, i: int, n_frames: int) -> np.ndarray:
    """``(n_frames, 3, H, W)`` uint8, channels-first and ready to stack.

    Frames are picked by nearest index rather than interpolated. Blending two frames of a
    moving person produces a ghost that occurs in no real frame, and at these clip lengths
    (median 20 frames) there is no shortage of real frames to choose from.
    """
    frames = cache.get(i)
    height, width = cache.data.shape[1], cache.data.shape[2]
    if len(frames) == 0:
        return np.zeros((n_frames, 3, height, width), dtype=np.uint8)
    picks = np.linspace(0, len(frames) - 1, n_frames).round().astype(np.int64)
    return np.ascontiguousarray(frames[picks].transpose(0, 3, 1, 2))


def skeleton_tensor(cache: RaggedCache, i: int, n_frames: int, do_normalize: bool = True):
    poses = cache.get(i)
    if do_normalize:
        poses = normalize(poses)
    return resample(poses, n_frames)


def imu_tensor(cache: RaggedCache, i: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N_DEVICES, n_frames, N_CHANNELS)`` plus a presence mask, as in ``imu.load_clip_imu``
    but reading from the cache instead of the filesystem."""
    rows = cache.get(i)
    devices = cache.get_extra("device", i)
    out = np.zeros((N_DEVICES, n_frames, N_CHANNELS), dtype=np.float32)
    mask = np.zeros(N_DEVICES, dtype=bool)
    for device in range(N_DEVICES):
        series = rows[devices == device]
        if len(series) == 0:
            continue
        mask[device] = True
        if len(series) == 1:
            out[device] = series[0]
            continue
        src = np.linspace(0.0, 1.0, len(series))
        dst = np.linspace(0.0, 1.0, n_frames)
        for channel in range(N_CHANNELS):
            out[device, :, channel] = np.interp(dst, src, series[:, channel])
    return out, mask
