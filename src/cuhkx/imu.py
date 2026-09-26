"""Parse the IMU CSVs into a fixed ``(device, time, channel)`` tensor.

These files are the messiest part of the dataset, so the parsing rules are deliberate:

* **Two files per clip.** ``down(LL+RL).csv`` carries the two leg sensors, ``up(LA+RA+C).csv``
  the two arm sensors and the chest one. Five devices in total.
* **Headers are not trustworthy.** Training files use UTF-8 Chinese headers; some test files
  use English headers in a legacy Chinese codepage. So columns are read **by position**,
  never by name, and the encoding is sniffed.
* **Column counts differ between files.** Some carry extra trailing Height/Pressure columns.
  Only positions 0–17 are stable, which is exactly the 16 numeric channels we want.
* **Rows of different devices are interleaved** in one file and each device has its own
  irregular timestamps, so rows are grouped by device prefix — the bracketed MAC address
  varies per session and must be ignored.
* Sampling is ~10 Hz over a ~2 s clip, so a device contributes on the order of 20 rows.

Layout of every row, by position::

    0  timestamp
    1  device name, e.g. "WTRL(E5:9E:9B:1F:CE:48)"
    2..4    acceleration  X Y Z   (g)
    5..7    angular rate  X Y Z   (deg/s)
    8..10   angle         X Y Z   (deg)
    11..13  magnetic      X Y Z   (uT)
    14..17  quaternion    0 1 2 3
    18      temperature            <- last stable column; anything after varies
"""
# Role: parses the IMU CSV files (five body-worn sensors, 16 numeric channels each) into
#   per-device series or a fixed (device, time, channel) tensor, for the IMU models.
# Used by: data.build_imu_cache / data.imu_tensor and models.ImuEncoder (constants only); not used
#   by the delivered run: the pipeline imports this module through data.py and models.py, but no
#   function in it runs, and the submitted system reads no IMU data.

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed device order, so channel c of the output always means the same sensor.
DEVICES = ["WTLL", "WTRL", "WTLA", "WTRA", "WTC"]
DEVICE_INDEX = {name: i for i, name in enumerate(DEVICES)}

FIRST_CHANNEL, LAST_CHANNEL = 2, 17  # inclusive; 16 numeric channels
N_CHANNELS = LAST_CHANNEL - FIRST_CHANNEL + 1
N_DEVICES = len(DEVICES)

# Tried in order by _read_text; if none decodes the file, it falls back to UTF-8 with
# replacement characters.
_ENCODINGS = ("utf-8", "gbk", "cp1252")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _device_prefix(field: str) -> str | None:
    """``"WTRL(E5:9E:...)"`` -> ``"WTRL"``; the MAC differs per session, so drop it."""
    name = field.split("(", 1)[0].strip()
    return name if name in DEVICE_INDEX else None


def parse_imu_file(path: Path) -> dict[str, np.ndarray]:
    """Return ``{device_prefix: (n_rows, N_CHANNELS) float32}`` for one CSV."""
    per_device: dict[str, list[list[float]]] = {}
    for line_no, line in enumerate(_read_text(path).splitlines()):
        if line_no == 0 or not line.strip():
            continue  # header, whatever language it is in
        # Columns by position: 1 is the device name, 2..17 are the 16 numeric channels.
        fields = line.split(",")
        if len(fields) <= LAST_CHANNEL:
            continue  # truncated row
        device = _device_prefix(fields[1])
        if device is None:
            continue
        try:
            values = [float(v) for v in fields[FIRST_CHANNEL : LAST_CHANNEL + 1]]
        except ValueError:
            continue  # a malformed row must not take the whole clip down
        per_device.setdefault(device, []).append(values)
    return {d: np.asarray(v, dtype=np.float32) for d, v in per_device.items()}


def load_clip_imu(clip_dir: Path, n_steps: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Load one clip's IMU into a dense array plus a per-device presence mask.

    Returns ``(tensor, mask)`` where tensor is ``(N_DEVICES, n_steps, N_CHANNELS)`` and
    mask is ``(N_DEVICES,)`` booleans. Each device's own irregular series is resampled
    onto ``n_steps`` points by linear interpolation over its row index, which sidesteps
    the fact that devices neither share a clock nor a sample count. Absent devices are
    left as zeros and flagged in the mask, so a model can learn to ignore them.
    """
    per_device: dict[str, np.ndarray] = {}
    for csv_path in sorted(clip_dir.glob("*.csv")):
        # A file that will not parse costs one device, not the whole run. The image
        # loaders learned this the hard way (see cuhkx.data._decode_frames): the finals
        # build their caches from data nobody has inspected, and four of the 405 Kaggle
        # test clips already ship frames that are the right size and entirely zero bytes.
        # Skeleton has guarded its JSON reads since the start; this closes the last
        # modality that could still take the whole build down with it.
        try:
            per_device.update(parse_imu_file(csv_path))
        except (OSError, ValueError, UnicodeDecodeError) as error:
            print(f"  !! {csv_path}: unreadable IMU file, skipped ({type(error).__name__})")

    tensor = np.zeros((N_DEVICES, n_steps, N_CHANNELS), dtype=np.float32)
    mask = np.zeros(N_DEVICES, dtype=bool)

    for device, series in per_device.items():
        if len(series) == 0:
            continue
        idx = DEVICE_INDEX[device]
        mask[idx] = True
        if len(series) == 1:
            tensor[idx] = series[0]  # broadcast the single sample across time
            continue
        src = np.linspace(0.0, 1.0, len(series))
        dst = np.linspace(0.0, 1.0, n_steps)
        for channel in range(N_CHANNELS):
            tensor[idx, :, channel] = np.interp(dst, src, series[:, channel])

    return tensor, mask
