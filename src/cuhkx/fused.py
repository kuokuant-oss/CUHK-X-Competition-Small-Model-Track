"""Pixel-aligned Depth+IR cache: the one thing early fusion needs and we never had.

Late fusion averages two *predictions*. Early fusion hands the network two *signals*
registered pixel to pixel, which lets it compute quantities neither branch can compute
alone -- "IR reflectance at a known depth" is roughly surface material, and that is exactly
the "what is in their hand" cue that the skeleton line is physically incapable of carrying
(15 of 40 classes sit below 0.35 accuracy and they are all seated-with-a-small-object).

The physical basis is that Depth_Color and IR come off the *same* Vzense NYX 650 at the
same 10 fps and the same 640x480, so a pixel means the same place in both. Thermal is a
separate camera and must not be stacked this way.

Until now that basis did not hold in code: the two caches used different crop windows
(depth cropped on scalar depth with step 4 / min_side 12 / no frame cap, IR on luminance
with step 2 / min_side 16 / max_frames 32), so channel i of one was not channel i of the
other. This module computes **one** window and **one** set of frame picks and applies both
to both modalities.

Three deliberate departures from ``data.build_image_cache``:

* **Four channels, Depth_Color RGB + IR**, matching the only external result measured on
  our exact leaderboard (0.716). Keeping Depth_Color as RGB rather than LUT-decoding it to
  a scalar costs some information but means channels 0-2 are a genuine RGB image, so an
  ImageNet stem transfers into them unmodified. Decoded-depth-plus-validity is the obvious
  ablation and is left for one.
* **Every frame up to the cap is stored**, and the choice of which to feed the model moves
  to training time. The old caches froze the pick at build time via ``np.linspace``, which
  meant every epoch saw the same 8 frames of each clip forever -- see ``TSNSampler``.
* **Out-of-bounds crops are padded, not clipped.** ``depth.motion_bbox`` grows its box to a
  square for a documented reason (an unsquare box gets squashed by a per-clip amount, which
  turns build into a distortion the model must learn to ignore) and then clips it back to
  the frame, undoing that for any box near an edge. Measured: about a quarter of clips came
  back unsquare, one at 1.33x. Padding keeps the square.
"""
# Role: builds the four-channel view caches (Depth_Color R, G, B + IR, uint8) with one crop window
#   and one set of frame indices shared by both modalities, and serves them to the networks:
#   TSN frame picks, the dataset, crop transforms and memory-mapped loading.
# Used by: scripts/21_build_fused_cache.py (det, miw and det248 caches), fd13_inference and
#   el22_support_inference (members and transformer), scripts/el25r_p1_repeat_runtime.py and
#   the training entry points in training/; both.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cuhkx import depth as depth_codec
from cuhkx.data import IMAGE_SUFFIXES, RaggedCache, _decode_frames, _pack

#: Depth_Color R,G,B then IR. IR's PNGs are mode ``L``; the old cache ran ``convert("RGB")``
#: on them and stored three identical channels, tripling the cache and the first-layer
#: FLOPs for no information.
N_CHANNELS = 4

# Fallback cache size (PIL width, height). The delivered caches are built with --size 144 (det,
# miw) and 248 (det248).
FUSED_SIZE = (128, 128)
MAX_FRAMES = 32  # Longer clips keep 32 evenly spaced frames (load_clip_fused).


def square_window(
    top: int, left: int, bottom: int, right: int, scale: float
) -> tuple[int, int, int, int]:
    """Scale a box about its centre and force it square. May fall outside the frame.

    Returned coordinates are deliberately unclamped: the caller pads. Clamping here is what
    reintroduces the aspect distortion the square was for.
    """
    centre_y, centre_x = (top + bottom) / 2.0, (left + right) / 2.0
    side = max(bottom - top, right - left) * scale
    half = side / 2.0
    return (
        int(round(centre_y - half)),
        int(round(centre_x - half)),
        int(round(centre_y + half)),
        int(round(centre_x + half)),
    )


def crop_pad(frame: np.ndarray, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    """Crop, replicating the edge for any part of the window outside the frame.

    Edge replication rather than zeros: a black border is a hard edge that occurs in no real
    frame and that a convolution will happily key on, and it would appear only for clips
    whose actor stands near the frame boundary -- i.e. correlated with the subject.
    """
    height, width = frame.shape[:2]
    # How far the window reaches past each frame edge; the in-frame part is cut first.
    pad_top, pad_left = max(0, -top), max(0, -left)
    pad_bottom, pad_right = max(0, bottom - height), max(0, right - width)
    window = frame[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if pad_top or pad_left or pad_bottom or pad_right:
        # (before, after) for rows and columns; no padding on a channel axis, if there is one.
        pads = [(pad_top, pad_bottom), (pad_left, pad_right)] + [(0, 0)] * (frame.ndim - 2)
        window = np.pad(window, pads, mode="edge")
    return window


def _frame_files(clip_dir: Path) -> list[Path]:
    # Frame images of one modality folder in file-name order; the names carry the timestamp.
    if not clip_dir.is_dir():
        return []
    return sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


def load_clip_fused(
    depth_dir: Path,
    ir_dir: Path,
    table: np.ndarray,
    size: tuple[int, int] = FUSED_SIZE,
    max_frames: int = MAX_FRAMES,
    bbox_scale: float = 1.0,
    whole_frame: bool = False,
    window_override: tuple[int, int, int, int] | None = None,
    motion_margin: float = 0.25,
    motion_quantile: float = 0.98,
    motion_in_window: bool = False,
) -> tuple[np.ndarray, dict]:
    """``((n, H, W, 4) uint8, stats)`` -- Depth_Color RGB plus IR, one shared window.

    ``motion_in_window`` (2026-09-05): with a detector ``window_override``, find the moving
    region *inside* that person box and crop to it instead. Measured on the whole frame the
    motion box is useless for seated classes -- depth flicker across the room outranks a
    pair of hands, and the box comes back near-whole-frame -- but inside the person box the
    98th percentile of motion is the arms, hands and the object in them. When the in-box
    motion degenerates to the whole box, the person box itself is used, so this view is never
    worse-framed than the detector crop it derives from.

    ``motion_margin`` / ``motion_quantile`` reach ``depth.motion_bbox`` unchanged; the
    defaults are its defaults, so every existing cache is reproduced. A small margin gives
    the *tight* "what moves" view (2026-09-05): for seated classes the moving pixels are the
    hands and the object in them, which the full-body detector box shrinks to a few pixels.

    ``bbox_scale`` multiplies the motion box about its centre. 1.0 reproduces the existing
    crop; larger values pull scene context back in. That matters because the box is found by
    *motion*, so a static object -- the television in ``25_Watch_TV``, which scores 0.000 --
    is outside it by construction, and the class is defined by that object.

    ``window_override`` supplies a window computed elsewhere -- in practice the detector box
    from ``scripts/22_person_windows.py``, which finds the actor by appearance rather than
    by motion and so does not fail on the seated classes. It is applied verbatim: it is
    already square, already margined, and deliberately unclamped so ``crop_pad`` pads.
    ``bbox_scale`` does not apply to it, because the margin is baked in at detection time
    (``person.CROP_MARGIN``) and applying both would compound two margins silently.
    """
    # Step 1: list both modalities; a clip with either folder empty yields no frames. size is
    # PIL's (width, height), so the output is (n, height, width, 4).
    depth_files, ir_files = _frame_files(depth_dir), _frame_files(ir_dir)
    stats = {"n_depth": len(depth_files), "n_ir": len(ir_files), "skipped": 0, "aligned": True}
    empty = np.zeros((0, size[1], size[0], N_CHANNELS), dtype=np.uint8)
    if not depth_files or not ir_files:
        stats["aligned"] = False
        return empty, stats

    # Step 2: give both modalities the same frame count.
    # Same sensor, same rate: the frame lists should be the same length. When they are not,
    # index the longer one onto the shorter rather than dropping the clip -- but say so, and
    # count it, because a systematic misalignment would silently destroy the whole premise.
    n = min(len(depth_files), len(ir_files))
    if len(depth_files) != len(ir_files):
        stats["aligned"] = False
        onto = np.linspace(0, len(depth_files) - 1, n).round().astype(int)
        depth_files = [depth_files[i] for i in onto]
        onto = np.linspace(0, len(ir_files) - 1, n).round().astype(int)
        ir_files = [ir_files[i] for i in onto]

    # Step 3: keep at most max_frames evenly spaced frames, the same indices for both.
    if max_frames and n > max_frames:
        picks = np.linspace(0, n - 1, max_frames).round().astype(int)
        depth_files = [depth_files[i] for i in picks]
        ir_files = [ir_files[i] for i in picks]

    # Step 4: decode Depth_Color as RGB and IR as 8-bit grey; unreadable files are skipped.
    depth_frames, skipped_d = _decode_frames(depth_files, convert="RGB")
    ir_frames, skipped_i = _decode_frames(ir_files, convert="L")
    stats["skipped"] = skipped_d + skipped_i
    if skipped_d or not depth_frames:
        # Depth is the modality the window is derived from; without it there is nothing.
        stats["aligned"] = False
        return empty, stats

    depth_stack = np.stack(depth_frames)
    height, width = depth_stack.shape[1], depth_stack.shape[2]

    # A skip on the IR side breaks the index correspondence, so IR goes entirely -- but the
    # clip does NOT. Four of the 405 test clips ship IR that is all zero bytes while their
    # Depth is fine; dropping them would forfeit four predictions, which on a 402-clip
    # leaderboard is a full percentage point. Zeroing one channel keeps three good ones.
    if skipped_i or not ir_frames or len(ir_frames) != len(depth_frames):
        stats["aligned"] = False
        stats["ir_dead"] = True
        ir_stack = np.zeros((len(depth_stack), height, width), dtype=np.uint8)
    else:
        ir_stack = np.stack(ir_frames)
    if ir_stack.shape[1] != height or ir_stack.shape[2] != width:
        # Not expected for this sensor pair; resize IR onto depth's grid and record it.
        from PIL import Image

        ir_stack = np.stack(
            [
                np.asarray(
                    Image.fromarray(f).resize((width, height), Image.BILINEAR), dtype=np.uint8
                )
                for f in ir_stack
            ]
        )
        stats["ir_resized"] = True

    # Step 5: one crop window (top, left, bottom, right) in full-frame pixels, taken from, in
    # order: the motion box inside the person window (miw view), the person window as given
    # (det and det248 views), the whole frame, or the motion box of the whole frame (a clip
    # with no person window).
    if window_override is not None and motion_in_window:
        # Motion is searched on every second pixel; min_side 32 // step is 32 full-size pixels.
        step = 2
        decoded = np.stack([depth_codec.decode(f, table)[0] for f in depth_stack])
        valid = decoded != depth_codec.INVALID
        w_top, w_left, w_bottom, w_right = (int(v) for v in window_override)
        # The override is deliberately unclamped (crop_pad pads); the motion search has to
        # stay inside the frame.
        c_top, c_left = max(0, w_top), max(0, w_left)
        c_bottom, c_right = min(height, w_bottom), min(width, w_right)
        sub_d = decoded[:, c_top:c_bottom:step, c_left:c_right:step]
        sub_v = valid[:, c_top:c_bottom:step, c_left:c_right:step]
        # Degenerate: the in-frame part of the window is under 8 strided pixels high or wide, or
        # the motion box is the whole searched region; the person window is then kept.
        if sub_d.shape[1] >= 8 and sub_d.shape[2] >= 8:
            # The delivered miw cache uses quantile 0.98 and margin 0.1.
            top, left, bottom, right = depth_codec.motion_bbox(
                sub_d,
                sub_v,
                quantile=motion_quantile,
                margin=motion_margin,
                min_side=32 // step,
            )
            whole = (0, 0, sub_d.shape[1], sub_d.shape[2])
            degenerate = (top, left, bottom, right) == whole
        else:
            degenerate = True
        if degenerate:
            window = (w_top, w_left, w_bottom, w_right)
            stats["window_source"] = "override(motion degenerate)"
        else:
            # Back to full-frame pixels and squared about the centre (bbox_scale is 1.0 in the
            # delivered caches).
            window = square_window(
                c_top + top * step,
                c_left + left * step,
                c_top + bottom * step,
                c_left + right * step,
                bbox_scale,
            )
            stats["window_source"] = "motion-in-override"
        stats["degenerate"] = bool(degenerate)
    elif window_override is not None:
        # det and det248 views: the person window exactly as given.
        window = tuple(int(v) for v in window_override)
        stats["degenerate"] = False
        stats["window_source"] = "override"
    elif whole_frame:
        # No crop. The delivered run uses this only to check which clips decode
        # (availability() in scripts/el25r_p1_repeat_runtime.py).
        window = (0, 0, height, width)
    else:
        # The box is found on LUT-decoded **scalar depth**, not on Depth_Color's luminance.
        # Measured: luminance is a dead cue here. JET's brightness is non-monotonic in depth
        # (blue -> cyan -> green -> yellow -> red rises then falls), so a small depth change
        # anywhere produces a large luminance swing, the motion map has no concentrated peak,
        # and the 98th-percentile threshold selects scattered pixels across the whole frame.
        # 10 of the first 16 test clips came back as the degenerate whole-frame fallback and
        # the rest were near-whole. On scalar depth the same function gives 244-480 px boxes.
        # This is exactly why build_depth_cache decodes before cropping.
        step = 2
        decoded = np.stack([depth_codec.decode(f, table)[0] for f in depth_stack])
        valid = decoded != depth_codec.INVALID
        top, left, bottom, right = depth_codec.motion_bbox(
            decoded[:, ::step, ::step],
            valid[:, ::step, ::step],
            quantile=motion_quantile,
            margin=motion_margin,
            min_side=32 // step,
        )
        whole = (0, 0, decoded.shape[1] // step, decoded.shape[2] // step)
        degenerate = (top, left, bottom, right) == whole
        stats["degenerate"] = bool(degenerate)
        window = square_window(top * step, left * step, bottom * step, right * step, bbox_scale)

    # Step 6: cut both modalities with the same window (edge-padded outside the frame) and resize
    # each frame bilinearly to size: channels 0-2 are Depth_Color R, G, B, channel 3 is IR.
    from PIL import Image

    out = np.empty((len(depth_stack), size[1], size[0], N_CHANNELS), dtype=np.uint8)
    for i in range(len(depth_stack)):
        d = crop_pad(depth_stack[i], *window)
        r = crop_pad(ir_stack[i], *window)
        out[i, :, :, :3] = np.asarray(Image.fromarray(d).resize(size, Image.BILINEAR))
        out[i, :, :, 3] = np.asarray(Image.fromarray(r).resize(size, Image.BILINEAR))
    # Window height before resizing (the side of a square window), reported by the cache builder.
    stats["side"] = window[2] - window[0]
    return out, stats


def build_fused_cache(
    depth_dirs: dict[str, Path],
    ir_dirs: dict[str, Path],
    lut_path: Path,
    size: tuple[int, int] = FUSED_SIZE,
    workers: int = 8,
    max_frames: int = MAX_FRAMES,
    bbox_scale: float = 1.0,
    whole_frame: bool = False,
    windows: dict[str, tuple[int, int, int, int]] | None = None,
    motion_margin: float = 0.25,
    motion_quantile: float = 0.98,
    motion_in_window: bool = False,
) -> tuple[RaggedCache, dict]:
    """Build the aligned Depth+IR cache over every clip present in both modalities."""
    table = depth_codec.load_lut(lut_path)
    clip_ids = list(depth_dirs)
    report: dict[str, list] = {
        "unaligned": [],
        "empty": [],
        "sides": [],
        "native": [],
        "degenerate": [],
        "ir_dead": [],
    }

    # A clip with no row in windows gets None and is cropped as if no windows were given (the
    # motion box, unless whole_frame).
    def loader(clip_id: str):
        frames, stats = load_clip_fused(
            depth_dirs[clip_id],
            ir_dirs.get(clip_id, Path("__missing__")),
            table,
            size,
            max_frames,
            bbox_scale,
            whole_frame,
            None if windows is None else windows.get(clip_id),
            motion_margin=motion_margin,
            motion_quantile=motion_quantile,
            motion_in_window=motion_in_window,
        )
        return clip_id, frames, stats

    # Decode the clips on a thread pool; map() returns the results in clip order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(loader, clip_ids))

    # Collect the frames and the per-clip diagnostics that the cache builder prints.
    sequences = []
    for clip_id, frames, stats in results:
        sequences.append(frames)
        report["native"].append(max(stats["n_depth"], stats["n_ir"]))
        if not stats["aligned"]:
            report["unaligned"].append((clip_id, stats["n_depth"], stats["n_ir"], stats["skipped"]))
        if len(frames) == 0:
            report["empty"].append(clip_id)
        else:
            report["sides"].append(stats.get("side", 0))
        if stats.get("degenerate"):
            report["degenerate"].append(clip_id)
        if stats.get("ir_dead"):
            report["ir_dead"].append(clip_id)

    # All clips in one (total_frames, H, W, 4) uint8 array; clip i is rows offsets[i]:offsets[i+1].
    row = (size[1], size[0], N_CHANNELS)
    data, offsets, ids = _pack(sequences, clip_ids, row, dtype=np.uint8)
    return RaggedCache(data=data, offsets=offsets, clip_ids=ids, extra={}), report


def tsn_picks(
    n_available: int, n_frames: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Segment-based frame indices: split the clip into ``n_frames`` spans, take one from each.

    With ``rng`` the pick inside each span is random, so a clip is a *different* sequence
    every epoch; without it the span centre is taken, which is the deterministic view for
    validation and inference.

    This replaces the single ``np.linspace`` that ``data.depth_tensor`` and
    ``data.image_tensor`` apply once, at cache-build time, before the fold loop. The
    consequence of doing it there was that every epoch saw the same 8 frames of each clip,
    for the entire run -- with median clips of ~24 native frames and a 32-frame cap, roughly
    a third to a seventh of the recording, always the same third. ``images.time_crop`` is
    not a substitute: it operates on the already-frozen tensor and can only drop or
    duplicate frames that linspace already chose, never surface one it did not.

    Segments rather than a uniform random draw over the whole clip (TSN, Wang et al. ECCV
    2016): uniform draws clump, and a batch that happens to sample six frames from the first
    half sees an action fragment rather than an action.
    """
    # An empty clip gets zeros; FusedFrameDataset then feeds empty_clip() instead.
    if n_available <= 0:
        return np.zeros(n_frames, dtype=np.int64)
    # n_frames equal segments of [0, n_available); n_frames is 32 for the members, 16 for the
    # transformer.
    bounds = np.linspace(0, n_available, n_frames + 1)
    low, high = bounds[:-1], bounds[1:]
    # Position inside each segment: uniform in training, the centre (0.5) otherwise.
    offsets = rng.random(n_frames) if rng is not None else np.full(n_frames, 0.5)
    picks = low + offsets * (high - low)
    # Truncate to frame indices. With more segments than frames, neighbouring segments share a
    # frame, so short clips repeat frames.
    return np.clip(picks.astype(np.int64), 0, n_available - 1)


class FusedFrameDataset:
    """Ragged fused cache -> ``((T, C, H, W) uint8,)`` plus a label, sampled per access.

    Deliberately not ``train.ArrayDataset``: that one indexes a dense ``(N, T, C, H, W)``
    array built by ``materialize()`` before the fold loop, which is exactly what freezes the
    temporal axis. Holding the ragged cache instead also avoids a second dense copy -- at
    144px with 32 frames the dense form would be about 5.5 GB per split.

    ``labels`` is indexed by **cache row**, exactly like ``offsets`` and ``indices``, not by
    position within the selection. Getting that wrong pairs every clip with some other
    clip's label, and the symptom is a plausible-looking bad accuracy that reads as "the
    idea does not work" -- so :func:`labels_by_row` builds it and the constructor checks
    the length rather than trusting the caller.
    """

    def __init__(
        self,
        data: np.ndarray,
        offsets: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        n_frames: int,
        train: bool,
        transform=None,
        seed: int = 0,
        freeze_temporal: bool = False,
        fixed_epoch: int | None = None,
    ):
        self.data, self.offsets, self.labels = data, offsets, np.asarray(labels)
        self.indices = np.asarray(indices)
        n_clips = len(offsets) - 1
        if len(self.labels) != n_clips:
            raise ValueError(
                f"labels must be indexed by cache row: got {len(self.labels)} for "
                f"{n_clips} clips. Use fused.labels_by_row() to build it."
            )
        self.n_frames, self.train, self.transform = n_frames, train, transform
        self.seed = seed
        # The two randomnesses are separate switches on purpose. `train=False` turns both
        # off, which is right for validation and inference. `freeze_temporal` turns off only
        # the frame sampling, which is what the old pipeline did -- it froze the frames in
        # materialize() but still applied random crops and erasing. Collapsing them into one
        # flag would make the ablation arm that isolates TSN sampling secretly also remove
        # spatial augmentation, and the measured delta would be attributable to neither.
        self.freeze_temporal = freeze_temporal
        self.epoch = 0
        self.fixed_epoch = fixed_epoch
        self._shared_epoch = None
        # Where `data` came from, so worker processes can reopen it instead of receiving a
        # copy. See __getstate__.
        self.data_path = Path(data.filename) if isinstance(data, np.memmap) else None

    # DataLoader workers on Windows (and anywhere `spawn` is used) are given the dataset by
    # pickle. `np.memmap` pickles as an ordinary ndarray -- it materialises every byte -- so
    # a 5.5 GB cache would be copied once per worker, and the fix for a slow input pipeline
    # would instead be an out-of-memory kill. Send the path and remap on the far side; the
    # pages are then shared through the OS page cache, which is the whole point of a memmap.
    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("data_path") is not None:
            state["data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.data is None and self.data_path is not None:
            self.data = np.load(self.data_path, mmap_mode="r")

    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-sample RNG so epochs differ but a run stays reproducible.

        The RNG is derived from (seed, epoch, clip index) rather than drawn from a shared
        generator: DataLoader workers would otherwise each fork the same stream, and the
        determinism pin this project spent a day establishing would silently stop holding.
        """
        epoch = self.fixed_epoch if self.fixed_epoch is not None else epoch
        self.epoch = epoch
        if self._shared_epoch is not None:
            self._shared_epoch.value = epoch

    def enable_epoch_sync(self) -> None:
        """Call before spawning workers; share only the epoch, never the image cache.

        Spawn-context synchronization also works with fork workers. An ordinary epoch
        attribute remains useful for direct access, but persistent workers keep copies.
        Epoch changes must occur between fully consumed DataLoader iterators.
        """
        import multiprocessing

        if self._shared_epoch is None:
            self._shared_epoch = multiprocessing.get_context("spawn").Value("q", self.epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def clean_view(self, limit: int) -> FusedFrameDataset:
        """A fixed subsample of this split with sampling and augmentation made deterministic.

        ``train.train_one_fold`` logs accuracy on this every few epochs. It is the number
        that separates "cannot fit" from "fits a harder, augmented target" -- opposite
        problems with opposite fixes -- and this project already spent a day reading an
        augmented 0.50 as underfitting for want of it.
        """
        size = len(self)
        picked = np.arange(size) if size <= limit else np.linspace(0, size - 1, limit).astype(int)
        return FusedFrameDataset(
            self.data,
            self.offsets,
            self.labels,
            self.indices[picked],
            self.n_frames,
            train=False,
            transform=self.transform,
            seed=self.seed,
        )

    def __getitem__(self, i: int):
        import torch

        # j is the cache row of the i-th selected clip; its frames are (n_j, H, W, C) uint8, with
        # C = 4 for the fused caches.
        j = int(self.indices[i])
        frames = self.data[self.offsets[j] : self.offsets[j + 1]]
        # Worker processes read the epoch from shared memory after enable_epoch_sync().
        epoch = self._shared_epoch.value if self._shared_epoch is not None else self.epoch
        # Training: a generator seeded by (seed, epoch, row). Validation and inference: None, so
        # the frame picks are segment centres and the transform takes the centre crop.
        rng = np.random.default_rng((self.seed, epoch, j)) if self.train else None
        picks = tsn_picks(len(frames), self.n_frames, None if self.freeze_temporal else rng)
        if len(frames) == 0:
            chosen = empty_clip(self.data, self.n_frames)
        else:
            # (T, H, W, C) -> (T, C, H, W), the layout the transforms and the models expect.
            chosen = np.ascontiguousarray(frames[picks].transpose(0, 3, 1, 2))
        if self.transform is not None:
            chosen = self.transform(chosen, rng)
        # A 1-tuple of inputs plus the label: the format that train._collate stacks into batches.
        return (torch.from_numpy(np.ascontiguousarray(chosen)),), int(self.labels[j])


def empty_clip(data: np.ndarray, n_frames: int) -> np.ndarray:
    """``(n_frames, C, H, W)`` of zeros for a clip that cached no frames, ``C`` from ``data``.

    The channel count is read off the cache rather than taken from this module's
    ``N_CHANNELS``. That constant is 4 because Depth+IR is 4, and every other consumer
    already derives the width from the data -- ``31_train_fused.py`` builds the model from
    ``data.shape[-1]``, ``69_infer_deliverable.py`` refuses to run when the two disagree --
    so hardcoding it made this branch the single place that could contradict the model it
    was feeding. On a 3-channel thermal cache it would have produced a 4-channel tensor for
    precisely the clips with no frames, which is the 10 test clips that ship no Thermal
    directory at all: a shape error at inference, on the clips least able to afford one.

    A function rather than three lines inline so the empty-clip contract can be tested
    without a torch import -- ``__getitem__`` needs torch only to wrap its result.
    """
    height, width, channels = data.shape[1], data.shape[2], data.shape[3]
    return np.zeros((n_frames, channels, height, width), dtype=np.uint8)


def labels_by_row(n_clips: int, rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Scatter ``labels`` (aligned to ``rows``) into a full-length, cache-row-indexed array.

    Cached clips outnumber labelled ones -- the cache holds every clip with the modality
    present, the fold table only those in the training split -- so the two orderings differ
    and pairing them by position silently mislabels everything.
    """
    # Cache rows without a label keep -1.
    out = np.full(n_clips, -1, dtype=np.int64)
    out[np.asarray(rows)] = np.asarray(labels)
    return out


class FusedTransform:
    """Spatial augmentation for the fused cache, as an object rather than a closure.

    A closure cannot be pickled, and DataLoader workers are created by ``spawn`` on Windows,
    so ``num_workers > 0`` died with ``Can't pickle local object
    'make_fused_transform.<locals>.transform'``. On Linux the default start method is
    ``fork`` and the same code works, which is exactly why this is worth fixing rather than
    working around: the bug is invisible on the machine the training will run on and fatal on
    the machine the measurement runs on.

    **The crop is the augmentation.** ``random_crop`` takes ``crop`` pixels out of whatever
    the cache stores, so the slack between the two is the entire spatial jitter: the deployed
    cache is 144 px and training passes ``crop=128``. Building the cache at 128 makes this a
    no-op and silently removes the augmentation, which is why
    ``scripts/21_build_fused_cache.py`` now defaults ``--size`` to the existing cache's.

    ``flip_prob`` is off by default and is a **registered variable**, not a default change.
    The -1.47 pp measured for horizontal flip was on **thermal**, and the project rules forbid
    carrying an augmentation result across modalities for exactly this reason. ADR 0002 bans
    *colour* augmentation because Depth_Color encodes distance as colour; a horizontal flip is
    geometric and touches no colour. What is genuinely unknown is whether the IR channel is
    left-right symmetric under this sensor's illuminator, so it stays a measured arm.

    No colour jitter on channels 0-2: ADR 0002, unchanged.
    """

    def __init__(
        self,
        crop: int,
        erase_prob: float = 0.0,
        erase_fraction: float = 0.25,
        flip_prob: float = 0.0,
        scale_range: tuple[float, float] | None = None,
    ):
        self.crop = crop
        self.erase_prob = erase_prob
        self.erase_fraction = erase_fraction
        self.flip_prob = flip_prob
        # Scale jitter (2026-09-05, lever L1): the training window side is drawn as
        # ``crop * U(lo, hi)`` and the window is resized back to ``crop``, so the actor's
        # apparent size varies per clip. Subject distance from the camera is one of the
        # cross-subject shifts this dataset has and the fixed 144 -> 128 crop never varied
        # it. None reproduces every earlier run exactly.
        self.scale_range = scale_range

    def _scaled_crop(self, frames: np.ndarray, rng) -> np.ndarray:
        import torch

        from cuhkx.images import random_crop

        lo, hi = self.scale_range
        height, width = frames.shape[2], frames.shape[3]
        # Window side crop x U(lo, hi), limited to [16, frame size], at one random position.
        side = int(round(self.crop * rng.uniform(lo, hi)))
        side = max(16, min(side, height, width))
        window = random_crop(frames, side, rng)
        if side == self.crop:
            return window
        # Resize (T, C, side, side) back to crop x crop, antialiased when shrinking, as uint8.
        x = torch.from_numpy(np.ascontiguousarray(window)).float()
        x = torch.nn.functional.interpolate(
            x,
            size=(self.crop, self.crop),
            mode="bilinear",
            align_corners=False,
            antialias=side > self.crop,
        )
        return x.round().clamp_(0, 255).to(torch.uint8).numpy()

    def __call__(self, frames: np.ndarray, rng) -> np.ndarray:
        from cuhkx.images import center_crop, erase, horizontal_flip, random_crop

        # Validation and inference (rng is None): the centre crop only, 128 of 144 pixels for the
        # members.
        if rng is None:
            return center_crop(frames, self.crop)
        # Training: one random crop per clip (scale-jittered when scale_range is set), then an
        # optional horizontal flip and optional random erasing.
        if self.scale_range is not None:
            frames = self._scaled_crop(frames, rng)
        else:
            frames = random_crop(frames, self.crop, rng)
        if self.flip_prob > 0.0 and rng.random() < self.flip_prob:
            frames = horizontal_flip(frames)
        if self.erase_prob > 0.0 and rng.random() < self.erase_prob:
            frames = erase(frames, self.erase_fraction, rng)
        return frames


def make_fused_transform(
    crop: int,
    erase_prob: float = 0.0,
    erase_fraction: float = 0.25,
    flip_prob: float = 0.0,
    scale_range: tuple[float, float] | None = None,
) -> FusedTransform:
    """Kept as a function so every existing call site is unchanged."""
    return FusedTransform(crop, erase_prob, erase_fraction, flip_prob, scale_range)


# channel_stats and cached_channel_stats have no caller in this package: the delivered members
# normalise with the Kinetics constants in models.PretrainedVideo3D.
def channel_stats(
    data: np.ndarray, max_frames: int = 4_000, pixel_stride: int = 4
) -> tuple[tuple, tuple]:
    """Per-channel mean/std in 0-1 units -- ADR 0002's "use this dataset's own statistics".

    Subsamples on both axes and accumulates float64 sums rather than materialising anything.
    ``data`` is (frames, H, W, C), so one row is a whole 82,944-byte frame, not a pixel: a
    slice that looks small is four times its size again as float32. The first version of
    this function read ``len(data)`` as a pixel count, decided 65,866 was under its sample
    budget, and asked for the entire train cache as float32 -- 21.8 GB on a 34 GB machine,
    which does not raise, it just pages. The symptom was a training run sitting at 23 GB
    resident with the GPU at 0%.

    A few thousand frames spread across every clip is far more precision than two constants
    that the first BatchNorm immediately absorbs.
    """
    step = max(1, len(data) // max_frames)
    channels = data.shape[-1]
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(data), step):
        frame = np.asarray(data[start])[::pixel_stride, ::pixel_stride].astype(np.float64) / 255.0
        flat = frame.reshape(-1, channels)
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        count += len(flat)
    if count == 0:
        return (0.5,) * channels, (0.25,) * channels
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    return tuple(mean.tolist()), tuple(np.sqrt(var).tolist())


def export_memmap(npz_path: Path) -> Path:
    """Rewrite a saved fused cache as a plain ``.npy`` beside it, plus a small sidecar.

    The ablation runs four arms over the same cache and the full run adds four folds on top.
    ``np.load`` on a 5.5 GB npz decompresses the whole thing into the process every single
    time; a ``.npy`` can be memory-mapped, so the second run onward reads it out of the OS
    page cache and the arms start in seconds rather than minutes. With 34 GB of RAM the
    whole cache stays resident after the first pass.
    """
    npz_path = Path(npz_path)
    # <stem>_data.npy holds the frames; <stem>_meta.npz the offsets, clip ids and build record.
    data_path = npz_path.with_name(npz_path.stem + "_data.npy")
    side_path = npz_path.with_name(npz_path.stem + "_meta.npz")
    if data_path.exists() and side_path.exists():
        return data_path
    blob = np.load(npz_path, allow_pickle=False)
    np.save(data_path, blob["data"])
    np.savez(
        side_path,
        offsets=blob["offsets"],
        clip_ids=blob["clip_ids"],
        build=blob["build"] if "build" in blob else np.array([]),
    )
    return data_path


def load_fused(npz_path: Path, mmap: bool = True):
    """``(data, offsets, clip_ids, build)``, memory-mapped when the sidecar exists."""
    npz_path = Path(npz_path)
    data_path = npz_path.with_name(npz_path.stem + "_data.npy")
    side_path = npz_path.with_name(npz_path.stem + "_meta.npz")
    # Memory-map the frames when export_memmap has run (fd13_inference asserts a memmap);
    # otherwise the whole npz is read into memory.
    if mmap and data_path.exists() and side_path.exists():
        side = np.load(side_path, allow_pickle=False)
        data = np.load(data_path, mmap_mode="r")
        build = [str(x) for x in side["build"]]
        return data, side["offsets"], [str(c) for c in side["clip_ids"]], build
    blob = np.load(npz_path, allow_pickle=False)
    build = [str(x) for x in blob["build"]] if "build" in blob else []
    return blob["data"], blob["offsets"], [str(c) for c in blob["clip_ids"]], build


def cached_channel_stats(npz_path: Path, data: np.ndarray) -> tuple[tuple, tuple]:
    """``channel_stats`` memoised beside the cache it describes.

    The statistics are a property of the cache, and the cache is immutable once built, so
    recomputing them costs two minutes of random reads across 5.5 GB for every run that
    touches it -- four ablation arms and four folds pay it eight times for one answer.
    Keyed on the cache filename, so a differently-built cache gets its own file.
    """
    npz_path = Path(npz_path)
    stats_path = npz_path.with_name(npz_path.stem + "_stats.npz")
    if stats_path.exists():
        blob = np.load(stats_path, allow_pickle=False)
        return tuple(blob["mean"].tolist()), tuple(blob["std"].tolist())
    mean, std = channel_stats(data)
    np.savez(stats_path, mean=np.array(mean), std=np.array(std))
    return mean, std
