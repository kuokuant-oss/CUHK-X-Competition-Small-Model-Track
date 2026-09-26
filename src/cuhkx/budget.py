"""The 100 MB deliverable budget, as a gate rather than a reminder.

Small Model Track caps the **delivered model**, so the number that matters is the actual
bytes on disk of everything inference needs to load — every modality's weights, the fusion
head, and any preprocessing artifact such as normalisation statistics or a label map. Not a
parameter count, not an estimate.

Two conventions, both deliberately conservative:

* ``MB`` is 10^6 bytes, not 2^20. The organizers do not say which they mean, and 10^6 is the
  smaller of the two, so a deliverable that passes here passes under either reading.
* The budget covers **all** components summed. Whether the official rule means per-model or
  the total is unresolved (U-06); until it is answered, the total is what gets checked.

Use :func:`save_deliverable` to write checkpoints — it refuses to save optimizer state,
which is the usual reason a checkpoint silently comes out two or three times too large.
"""
# Role: the 100 MB size budget (MB = 10^6 bytes) measured on disk, per-output-channel int8
# quantisation of state dicts (quantize_state, dequantize_state) and the deliverable writers.
# Used by: training (train.py; training/103_b_final_student.py, 108_finalize_calibrated.py and
# 124_c_full_student.py write members B and C as int8) and inference (dequantize_state, via
# person.load_detector and fd_codec.fp32); both. The runtime's check that weights/model.pt is
# below 100,000,000 bytes is in the inference entry point, not here.

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

LIMIT_BYTES = 100 * 10**6
MB = 10**6


class BudgetExceeded(RuntimeError):
    """Raised when a deliverable is over the size cap. Never catch this to proceed."""


# Measuring a deliverable: sizes in bytes on disk of every file under the given paths.
def _iter_files(paths: Iterable[Path] | Path) -> list[Path]:
    """Expand a path or paths into every file underneath, following directories."""
    if isinstance(paths, str | Path):
        paths = [Path(paths)]
    found: list[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            found.extend(sorted(f for f in entry.rglob("*") if f.is_file()))
        elif entry.is_file():
            found.append(entry)
        else:
            raise FileNotFoundError(f"deliverable component does not exist: {entry}")
    return found


def breakdown(paths: Iterable[Path] | Path) -> list[tuple[Path, int]]:
    """Per-file sizes, largest first — so an oversized deliverable names its own culprit."""
    files = _iter_files(paths)
    return sorted(((f, f.stat().st_size) for f in files), key=lambda kv: -kv[1])


def deliverable_bytes(paths: Iterable[Path] | Path) -> int:
    return sum(size for _, size in breakdown(paths))


def report(paths: Iterable[Path] | Path, limit: int = LIMIT_BYTES) -> str:
    rows = breakdown(paths)
    total = sum(size for _, size in rows)
    lines = [f"{'component':<52} {'MB':>8}"]
    for path, size in rows:
        lines.append(f"  {path.name:<50} {size / MB:8.2f}")
    verdict = "OK" if total <= limit else "OVER BUDGET"
    lines.append(f"{'TOTAL':<52} {total / MB:8.2f} / {limit / MB:.0f} MB  [{verdict}]")
    return "\n".join(lines)


def check(paths: Iterable[Path] | Path, limit: int = LIMIT_BYTES) -> int:
    """Return the total size, raising :class:`BudgetExceeded` if it is over the cap."""
    total = deliverable_bytes(paths)
    if total > limit:
        raise BudgetExceeded(
            f"deliverable is {total / MB:.2f} MB, over the {limit / MB:.0f} MB cap\n"
            + report(paths, limit)
        )
    return total


def _as_tensor_dict(name: str, component: Any) -> Mapping[str, Any]:
    """Coerce a module or state dict to a flat tensor dict, rejecting anything else.

    An optimizer's ``state_dict()`` is a dict with ``state`` and ``param_groups`` keys rather
    than a flat mapping of tensors, so this check is what stops it being saved by accident.
    """
    import torch

    state = component.state_dict() if hasattr(component, "state_dict") else component
    if not isinstance(state, Mapping):
        raise TypeError(f"component {name!r} is not a module or state dict: {type(state)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"component {name!r} key {key!r} holds {type(value).__name__}, not a Tensor "
                "— this looks like optimizer state or a pickled object, which must not ship "
                "in the deliverable"
            )
    return state


# Bit packing for codes narrower than 8 bits (quantize_state with bits < 8 and the 7-bit paths of
# the codec modules). The delivered checkpoint contains no bit-packed tensors.
def _bit_pack(values, bits: int):
    """``(n,) uint8`` in ``[0, 2**bits)`` -> a dense little-endian bitstream, as uint8."""
    import numpy as np

    # Code i fills stream bits i*bits .. (i+1)*bits - 1, lowest bit first; the unused high bits of
    # the last byte are zero.
    spread = np.unpackbits(values[:, None], axis=1, count=bits, bitorder="little")
    return np.packbits(spread.reshape(-1), bitorder="little")


def _bit_unpack(stream, n: int, bits: int):
    """Inverse of :func:`_bit_pack`."""
    import numpy as np

    spread = np.unpackbits(stream, bitorder="little")[: n * bits].reshape(n, bits)
    return np.packbits(spread, axis=1, bitorder="little")[:, 0]


def quantize_state(state: dict, bits: int = 8) -> dict:
    """Per-channel low-precision for conv/linear weights (ndim>=2), fp16 for everything else.

    The public 0.667/0.711 solutions pack a 63M R(2+1)D-34 into <100 MB exactly this way
    (welshonion's ``quantize_sd``): a weight tensor becomes an int8 array plus a per-output-
    channel fp16 scale, so 4 bytes/param drops to ~1. Rank-preserving at inference after
    ``dequantize_state``. Small 1-D tensors (norm, bias) stay fp16 -- quantising them buys
    nothing and hurts accuracy.

    ``bits`` below 8 additionally **bit-packs**, which is what the 0.711 solution's int5/int6
    does and the only way a 63.7M-parameter soup gets under 50 MB: at 8 bits torch stores one
    byte per weight however few of its levels are used, so int6 without packing saves exactly
    nothing. Packed tensors carry ``_packed`` instead of ``_int8``.

    ``bits=8`` is the default and takes the unpacked path, so every deliverable already on
    disk keeps loading and the floor's 63.84 MB does not move. That matters: the floor has a
    cold-start D=0 reproduction against its leaderboard csv, and silently changing its
    storage format would invalidate that evidence.
    """
    import torch

    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in 2..8, got {bits}")
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for k, v in state.items():
        # The test is on rank, not on layer type: every floating tensor with two or more dimensions
        # is quantised; other floating tensors become FP16 and non-floating ones are kept.
        if not (v.is_floating_point() and v.ndim >= 2):
            out[k] = v.half() if v.is_floating_point() else v
            continue
        # Symmetric per-output-channel scale: max |w| over all dimensions but the first, over qmax
        # (127 at 8 bits), floored at 1e-12 so w / scale is defined. Codes are rounded with the
        # FP32 scale; the stored scale is its FP16 copy (0 for an all-zero channel).
        w = v.float()
        scale = (w.abs().amax(dim=tuple(range(1, w.ndim)), keepdim=True) / qmax).clamp_min(1e-12)
        q = (w / scale).round().clamp(-qmax, qmax)
        if bits == 8:
            out[k] = {"_int8": q.to(torch.int8), "_scale": scale.half()}
        else:
            # Shift to unsigned so the packer sees [0, 2*qmax], then pack. Shape and count are
            # stored as tensors rather than python ints because torch.load(weights_only=True)
            # is the only loader this project uses.
            unsigned = (q + qmax).to(torch.uint8).flatten().numpy()
            out[k] = {
                "_packed": torch.from_numpy(_bit_pack(unsigned, bits)),
                "_scale": scale.half(),
                "_bits": torch.tensor(bits, dtype=torch.int16),
                "_shape": torch.tensor(list(v.shape), dtype=torch.int64),
            }
    return out


def dequantize_state(packed: dict) -> dict:
    """Inverse of :func:`quantize_state`: rebuild an fp32 state dict for model loading."""
    import numpy as np
    import torch

    out = {}
    for k, v in packed.items():
        # In the delivered run the int8 entries of B and C take the first branch and all other
        # entries (their FP16 tensors, the FP32 detector) the last; the packed branch is not used.
        if isinstance(v, dict) and "_int8" in v:
            out[k] = v["_int8"].float() * v["_scale"].float()
        elif isinstance(v, dict) and "_packed" in v:
            bits = int(v["_bits"])
            qmax = 2 ** (bits - 1) - 1
            shape = [int(x) for x in v["_shape"]]
            n = int(np.prod(shape))
            unsigned = _bit_unpack(v["_packed"].numpy(), n, bits)
            q = torch.from_numpy(unsigned.astype(np.int16)).float() - qmax
            out[k] = q.reshape(shape) * v["_scale"].float()
        else:
            out[k] = v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v
    return out


def save_deliverable(
    components: Mapping[str, Any],
    out_dir: Path,
    meta: Mapping[str, Any] | None = None,
    half: bool = False,
    int8: bool = False,
    limit: int = LIMIT_BYTES,
    bits: int = 8,
) -> dict[str, Any]:
    """Write an inference-ready deliverable and measure it.

    ``components`` maps a name to a module or a flat state dict; each is written as
    ``<name>.pt`` holding weights only. ``meta`` (config, seed, git commit, class map) is
    written as ``meta.json`` and **counts toward the budget**, because inference needs it.

    ``half`` casts floating tensors to fp16, which halves the weights — held in reserve for
    when an otherwise-good ensemble does not fit.

    Raises :class:`BudgetExceeded` after writing, so the oversized files are on disk to
    inspect rather than silently discarded.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: write each component as <name>.pt, quantised (int8=True; bit-packed if bits < 8),
    # cast to FP16 (half=True) or unchanged, then meta.json.
    for name, component in components.items():
        state = _as_tensor_dict(name, component)
        if int8:
            state = quantize_state(state, bits=bits)
        elif half:
            state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
        torch.save(state, out_dir / f"{name}.pt")

    if meta is not None:
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    # Step 2: measure the whole directory, print the report and raise if it is over the limit.
    total = deliverable_bytes(out_dir)
    manifest = {
        "dir": str(out_dir),
        "total_bytes": total,
        "total_mb": round(total / MB, 3),
        "limit_mb": limit / MB,
        "within_budget": total <= limit,
        "files": {p.name: s for p, s in breakdown(out_dir)},
    }
    print(report(out_dir, limit))
    if total > limit:
        raise BudgetExceeded(
            f"deliverable is {total / MB:.2f} MB, over the {limit / MB:.0f} MB cap; "
            f"files left in {out_dir} for inspection"
        )
    return manifest


def pack_deliverable(
    out_dir: Path,
    out_file: Path | None = None,
    limit: int = LIMIT_BYTES,
) -> dict[str, Any]:
    """Collapse a deliverable directory into ONE checkpoint file, and measure it.

    The organisers' size rule (``docs/reference/organizer-correspondence.md`` C-4) is that
    every weight inference loads is **packed into a single checkpoint file** under 100 MB.
    :func:`save_deliverable` writes a *directory* -- one ``<name>.pt`` per component plus
    ``meta.json`` -- which satisfies the byte budget but not the "one file" wording. This
    turns the directory into ``{"meta": {...}, "components": {name: state_dict}}``.

    Nothing is re-quantised: the component tensors are copied through exactly as they sit on
    disk, so a packed int8 deliverable stays int8 and a packed fp16 one stays fp16. That
    matters because re-casting here would silently change the predictions, and the S6 gate
    is that the deliverable reproduces the submitted csv with zero discordant clips.
    """
    import torch

    # Step 1: read meta.json and every <name>.pt component as stored.
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json in {out_dir}; not a deliverable directory")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    components = {}
    for path in sorted(out_dir.glob("*.pt")):
        components[path.stem] = torch.load(path, map_location="cpu", weights_only=True)
    if not components:
        raise FileNotFoundError(f"no *.pt components in {out_dir}")

    # Step 2: write a dict with "meta" and "components" as one torch file (default <out_dir>.pt)
    # and measure it.
    out_file = Path(out_file) if out_file is not None else out_dir.with_suffix(".pt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "components": components}, out_file)

    total = out_file.stat().st_size
    print(
        f"packed {len(components)} component(s) {sorted(components)} -> {out_file.name}\n"
        f"  {total / MB:.2f} / {limit / MB:.0f} MB  "
        f"[{'OK' if total <= limit else 'OVER BUDGET'}]"
    )
    if total > limit:
        raise BudgetExceeded(
            f"packed deliverable is {total / MB:.2f} MB, over the {limit / MB:.0f} MB cap; "
            f"file left at {out_file} for inspection"
        )
    return {
        "file": str(out_file),
        "total_bytes": total,
        "total_mb": round(total / MB, 3),
        "within_budget": True,
        "components": sorted(components),
        "meta": meta,
    }
