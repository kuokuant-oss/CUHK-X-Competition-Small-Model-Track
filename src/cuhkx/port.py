"""Verify that a ported checkpoint actually landed, tensor by tensor.

Cross-framework weight ports fail silently. ``load_state_dict(..., strict=False)`` returns
happily with half the network still at its random initialisation, the model trains, the loss
goes down, and the number you get is a from-scratch result wearing a pretrained label. That is
the worst failure available here because it is *measurable* -- it would be believed, recorded
in the calibration table, and shipped.

It is also a documented risk for the specific checkpoints we want: open-mmlab/mmaction2 issue
836 reports state_dict naming mismatches for exactly the ipCSN and irCSN weights.

So the port is not allowed to report success on "no exception raised". :func:`verify_load`
compares against the model's own random initialisation and requires that every tensor moved.
The check is cheap, it needs no reference implementation, and it catches the one thing that
looks like success.
"""
# Role: port of the MMAction2 ir-CSN-152 (IG-65M -> Kinetics-400) checkpoint into pytorchvideo's
# CSN: key matching by position in the network, load and forward-pass checks, and the one
# ir-CSN-152 builder (build_csn).
# Used by: cuhkx.models.PretrainedVideo3D (arch "ircsn152", via build_csn) for the ir-CSN-152 fold
# models behind the second pseudo-label teacher; the porting script itself is not included;
# training only, not used by the delivered run.

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


class PortError(RuntimeError):
    """Raised when a port cannot be trusted, with the specific tensors named."""


@dataclass
class PortReport:
    # Tensor names by outcome of verify_load: loaded (changed from its initial value, or listed in
    # allow_unmoved), unmoved (still equal to its initial value), missing from the checkpoint,
    # unexpected (in the checkpoint but not in the model), and shape mismatches.
    loaded: list[str] = field(default_factory=list)
    unmoved: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unmoved or self.missing or self.unexpected or self.shape_mismatch)

    def summary(self) -> str:
        parts = [f"{len(self.loaded)} tensors loaded"]
        for name, items in (
            ("still at random init", self.unmoved),
            ("missing from checkpoint", self.missing),
            ("unused in checkpoint", self.unexpected),
            ("shape mismatch", self.shape_mismatch),
        ):
            if items:
                shown = ", ".join(items[:6]) + (" ..." if len(items) > 6 else "")
                parts.append(f"{len(items)} {name}: {shown}")
        return "; ".join(parts)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise PortError(self.summary())


def verify_load(
    model: nn.Module,
    state: Mapping[str, Any],
    allow_unmoved: tuple[str, ...] = (),
) -> PortReport:
    """Load ``state`` into ``model`` and prove every tensor changed.

    ``allow_unmoved`` names tensors legitimately expected to match their initialisation --
    in practice zero-initialised biases and the ``num_batches_tracked`` counters, which are
    zero on both sides and cannot be distinguished from "never written". Anything not listed
    there that comes out bit-identical to its random init is treated as a failed port, not as
    a coincidence: for a float tensor of any real size the odds of an honest match are nil.
    """
    # Step 1: snapshot the model's current (randomly initialised) tensors and compare names and
    # shapes with the checkpoint; return without loading if anything differs.
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report = PortReport()

    for key, target in before.items():
        if key not in state:
            report.missing.append(key)
        elif torch.as_tensor(state[key]).shape != target.shape:
            report.shape_mismatch.append(
                f"{key} {tuple(torch.as_tensor(state[key]).shape)} != {tuple(target.shape)}"
            )
    report.unexpected = [k for k in state if k not in before]
    if report.missing or report.unexpected or report.shape_mismatch:
        return report

    # Step 2: load strictly, then list every tensor that is still bit-identical to its snapshot.
    model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})

    after = model.state_dict()
    for key, target in before.items():
        if any(key.endswith(suffix) or key == suffix for suffix in allow_unmoved):
            report.loaded.append(key)
            continue
        if target.numel() and torch.equal(target, after[key]):
            report.unmoved.append(key)
        else:
            report.loaded.append(key)
    return report


@torch.no_grad()
def sane_forward(model: nn.Module, sample: torch.Tensor, n_classes: int) -> str:
    """Run one clip through and reject the outputs that mean the port is broken.

    A ported network can pass every tensor check and still be wired wrong -- a stem with the
    channels transposed, say. NaNs and a perfectly uniform distribution are the two cheap
    tells, and both are worth catching before spending three GPU hours.
    """
    model.eval()
    logits = model(sample)
    if logits.shape[-1] != n_classes:
        raise PortError(f"head emits {logits.shape[-1]} classes, expected {n_classes}")
    if not torch.isfinite(logits).all():
        raise PortError("forward produced NaN or inf")
    probs = logits.softmax(-1)
    spread = float(probs.max() - probs.min())
    if spread < 1e-4:
        raise PortError(f"output is uniform to within {spread:.2e}; the head is not connected")
    return (
        f"forward ok: {tuple(logits.shape)}, max prob {float(probs.max()):.4f}, spread {spread:.4f}"
    )


# ---------------------------------------------------------------------------
# Key remapping: MMAction2 ResNet3dCSN -> pytorchvideo create_csn
# ---------------------------------------------------------------------------
#
# The two projects build the same network and name none of it the same way. The tempting
# shortcut is a hand-written dict of source-name -> target-name, and that is the shortcut
# mmaction2#836 is a report of: the names drift between releases, a dict written against one
# of them silently misses tensors against another, and load_state_dict(strict=False) says
# nothing.
#
# So nothing here is matched by name. Both sides are parsed into a Slot -- where a tensor sits
# in a 3D bottleneck ResNet, which is a fact about the architecture rather than about either
# library's spelling -- and matched on that. A key neither parser recognises is a hard error
# naming the key, so a naming change we did not anticipate surfaces as a refusal to port
# rather than as a partially-initialised network.


@dataclass(frozen=True)
class Slot:
    """A tensor's position in a 3D bottleneck ResNet, independent of naming convention.

    ``stage`` is 0 for the stem, 1-4 for the residual stages and -1 for the classifier.
    ``block`` indexes within a stage. ``role`` is the position inside a bottleneck: ``conv_a``
    (1x1x1 reduce), ``conv_b`` (3x3x3, depthwise in CSN), ``conv_c`` (1x1x1 expand), the
    matching ``norm_*``, and ``down_conv`` / ``down_norm`` for the shortcut projection.
    """

    stage: int
    block: int
    role: str
    part: str


# MMAction2 stem: [backbone.]conv1.conv.<tensor> and [backbone.]conv1.bn.<tensor>.
_MM_STEM = re.compile(r"^(?:backbone\.)?conv1\.(conv|bn)\.(\w+)$")
# MMAction2 wraps ir-CSN's depthwise conv2 in a Sequential, so the real keys carry an extra
# index: `layer1.0.conv2.0.conv.weight`, not `layer1.0.conv2.conv.weight`. Measured on the
# published ircsn_ig65m-pretrained-r152 checkpoint: without the optional group, exactly 300 of
# 932 tensors fail to map -- ResNet-152's 50 bottleneck blocks times conv2's 1 conv + 5 BN
# tensors. Index 0 folds into the plain role; any *other* index gets a role of its own, which
# has no target and therefore refuses the port instead of silently merging two tensors.
# _MM_BLOCK groups: stage (layerN), block, unit (conv1, conv2, conv3 or downsample), optional
# Sequential index, conv or bn, tensor name.
_MM_BLOCK = re.compile(
    r"^(?:backbone\.)?layer(\d+)\.(\d+)\.(conv1|conv2|conv3|downsample)"
    r"(?:\.(\d+))?\.(conv|bn)\.(\w+)$"
)
# MMAction2 classifier: [cls_head.]fc_cls.weight and .bias.
_MM_HEAD = re.compile(r"^(?:cls_head\.)?fc_cls\.(weight|bias)$")
# MMAction2's conv1, conv2, conv3 are the bottleneck's 1x1x1 reduce, 3x3x3 and 1x1x1 expand.
_MM_CONV_ROLE = {"conv1": "a", "conv2": "b", "conv3": "c"}

# pytorchvideo stem: blocks.0.conv.<tensor> and blocks.0.norm.<tensor>.
_PV_STEM = re.compile(r"^blocks\.0\.(conv|norm)\.(\w+)$")
# Residual block: blocks.<stage>.res_blocks.<block>.<unit>.<tensor>, where unit is branch1_conv or
# branch1_norm (shortcut projection) or branch2.conv_a ... branch2.norm_c (bottleneck, see below).
_PV_BLOCK = re.compile(r"^blocks\.(\d+)\.res_blocks\.(\d+)\.(.+?)\.(\w+)$")
# Classifier: blocks.<n>.proj (or .fc) weight and bias; blocks.5.proj in the ir-CSN-152 built here.
_PV_HEAD = re.compile(r"^blocks\.\d+\.(?:proj|fc)\.(weight|bias)$")
_PV_BRANCH2 = re.compile(r"^branch2\.(conv|norm)_([abc])$")


def parse_mmaction_key(key: str) -> Slot | None:
    """MMAction2 ``ResNet3dCSN`` key -> :class:`Slot`, or None if unrecognised."""
    if m := _MM_STEM.match(key):
        return Slot(0, 0, "stem_conv" if m[1] == "conv" else "stem_norm", m[2])
    if m := _MM_BLOCK.match(key):
        stage, block, unit, seq, kind, part = int(m[1]), int(m[2]), m[3], m[4], m[5], m[6]
        if unit == "downsample":
            role = "down_conv" if kind == "conv" else "down_norm"
        else:
            role = ("conv_" if kind == "conv" else "norm_") + _MM_CONV_ROLE[unit]
        if seq is not None and int(seq) != 0:
            role = f"{role}#{int(seq)}"
        return Slot(stage, block, role, part)
    if m := _MM_HEAD.match(key):
        return Slot(-1, 0, "head", m[1])
    return None


def parse_pytorchvideo_key(key: str) -> Slot | None:
    """pytorchvideo ``create_csn`` key -> :class:`Slot`, or None if unrecognised."""
    if m := _PV_STEM.match(key):
        return Slot(0, 0, "stem_conv" if m[1] == "conv" else "stem_norm", m[2])
    if m := _PV_HEAD.match(key):
        return Slot(-1, 0, "head", m[1])
    if m := _PV_BLOCK.match(key):
        stage, block, unit, part = int(m[1]), int(m[2]), m[3], m[4]
        if unit == "branch1_conv":
            return Slot(stage, block, "down_conv", part)
        if unit == "branch1_norm":
            return Slot(stage, block, "down_norm", part)
        if b := _PV_BRANCH2.match(unit):
            return Slot(stage, block, f"{b[1]}_{b[2]}", part)
    return None


@dataclass
class MappingReport:
    mapping: dict[str, str] = field(default_factory=dict)  # target key -> source key
    unparsed_source: list[str] = field(default_factory=list)
    unparsed_target: list[str] = field(default_factory=list)
    unmatched_target: list[str] = field(default_factory=list)
    unused_source: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.unparsed_source
            or self.unparsed_target
            or self.unmatched_target
            or self.unused_source
        )

    def summary(self) -> str:
        parts = [f"{len(self.mapping)} keys mapped"]
        for name, items in (
            ("source keys the parser does not recognise", self.unparsed_source),
            ("target keys the parser does not recognise", self.unparsed_target),
            ("target keys with no source", self.unmatched_target),
            ("source keys with no target", self.unused_source),
        ):
            if items:
                shown = ", ".join(sorted(items)[:6]) + (" ..." if len(items) > 6 else "")
                parts.append(f"{len(items)} {name}: {shown}")
        return "; ".join(parts)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise PortError(self.summary())


def build_mapping(source_keys: Iterable[str], target_keys: Iterable[str]) -> MappingReport:
    """Match an MMAction2 checkpoint's keys to a pytorchvideo model's keys by position.

    Both sides must parse completely. A key either parser does not recognise is reported
    rather than dropped, because a dropped key is precisely the tensor that would be left at
    its random initialisation.
    """
    report = MappingReport()
    # Index the recognised source keys by slot.
    src_slots: dict[Slot, str] = {}
    for key in source_keys:
        slot = parse_mmaction_key(key)
        if slot is None:
            report.unparsed_source.append(key)
        else:
            src_slots[slot] = key

    # Give each target key the source key in the same slot.
    used: set[str] = set()
    for key in target_keys:
        slot = parse_pytorchvideo_key(key)
        if slot is None:
            report.unparsed_target.append(key)
        elif slot in src_slots:
            report.mapping[key] = src_slots[slot]
            used.add(src_slots[slot])
        else:
            report.unmatched_target.append(key)

    # Recognised source keys that no target key took.
    report.unused_source = [k for k in src_slots.values() if k not in used]
    return report


def remap_state(
    source: Mapping[str, Any], target_keys: Iterable[str]
) -> tuple[dict[str, Any], MappingReport]:
    """Rename ``source`` into the target model's vocabulary. Raises if anything is left over."""
    target_keys = list(target_keys)
    report = build_mapping(source.keys(), target_keys)
    report.raise_if_bad()
    return {dst: source[src] for dst, src in report.mapping.items()}, report


def unwrap_checkpoint(obj: Any) -> dict[str, Any]:
    """Pull the tensors out of a torch.save'd file, whatever the publisher wrapped them in."""
    # Unwrap at most one level ("state_dict", "model_state" or "model"), then strip the
    # "module." prefix that (Distributed)DataParallel puts on every key.
    for key in ("state_dict", "model_state", "model"):
        if isinstance(obj, Mapping) and key in obj and isinstance(obj[key], Mapping):
            obj = obj[key]
            break
    if not isinstance(obj, Mapping):
        raise PortError(f"checkpoint is a {type(obj).__name__}, not a state dict")
    return {k.removeprefix("module."): v for k, v in obj.items()}


# ---------------------------------------------------------------------------
# ir-CSN-152 construction: ONE definition, used by both the porter and the trainer
# ---------------------------------------------------------------------------
#
# This lives here, not in either caller, because the two must not drift. The porter writes
# weights against this architecture and the trainer loads them back into it; a config that
# differs by one field between the two loads real weights into a different network, and every
# tensor check still passes because the shapes happen to match.

# ResNet3dCSN's norm_cfg default, which the mmaction2 configs never override. It matters: this
# checkpoint's running_var reaches 5.6e-45, so pytorchvideo's 1e-5 default overflows to NaN by
# stage 3 block 30 -- for every input, including zeros.
CSN_BN_EPS = 1e-3

# create_csn arguments for ir-CSN-152 as published: 3-channel stem, 400-class Kinetics head.
# For training, PretrainedVideo3D then widens the stem to 4 channels and replaces that layer.
CSN152_IG65M = {
    "model_depth": 152,
    "model_num_class": 400,
    "input_channel": 3,
    "stage_conv_a_kernel_size": (1, 1, 1),
    "stage_conv_b_kernel_size": (3, 3, 3),
    "stage_conv_b_width_per_group": 1,  # depthwise 3x3x3 -- this is what makes it "ir"
    "stem_conv_kernel_size": (3, 7, 7),
    "stem_conv_stride": (1, 2, 2),
    "dropout_rate": 0.0,
    # mmaction2's ResNet3d pools after conv1; create_csn defaults to no stem pool at all.
    # Carries no weights, so every tensor check passes while each feature map runs at twice
    # the intended spatial scale -- visible only as a disappointing fine-tune.
    "stem_pool": None,  # filled in by build_csn to avoid importing torch at module scope
    "stem_pool_kernel_size": (1, 3, 3),
    "stem_pool_stride": (1, 2, 2),
}


def build_csn(weights: Path | None = None, resolution_agnostic_head: bool = True):
    """ir-CSN-152 with mmaction2's settings, optionally loading ported weights.

    ``resolution_agnostic_head`` replaces the head's fixed ``AvgPool3d((1, 7, 7))`` -- which
    assumes the 224px training resolution and raises outright at 160px, where the final map is
    5x5 -- with ``AdaptiveAvgPool3d((None, 1, 1))``. That reproduces the original semantics
    (collapse space, keep time, let ``output_pool`` average time) at any input size.
    """
    from pytorchvideo.models.csn import create_csn

    # Step 1: build the network, refusing any setting this pytorchvideo version does not accept.
    cfg = {**CSN152_IG65M, "stem_pool": nn.MaxPool3d}
    accepted = set(inspect.signature(create_csn).parameters)
    unknown = sorted(set(cfg) - accepted)
    if unknown:
        raise PortError(
            f"this pytorchvideo's create_csn does not accept {unknown}; the architecture would "
            f"silently differ. Accepted: {sorted(accepted)}"
        )
    model = create_csn(**cfg)

    # Step 2: set eps = CSN_BN_EPS on every BatchNorm layer.
    # Set after construction, and verified. create_csn only forwards the norm *class*, and
    # pytorchvideo passes eps as an explicit call-site kwarg deeper in the factory chain, which
    # beats a functools.partial default -- doing it that way left the model at 1e-5 and the only
    # symptom was the same NaN. A silent no-op here is a hard failure instead.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._NormBase):
            module.eps = CSN_BN_EPS
    got = {m.eps for m in model.modules() if isinstance(m, nn.modules.batchnorm._NormBase)}
    if got != {CSN_BN_EPS}:
        raise PortError(f"BatchNorm eps is {sorted(got)}, expected {CSN_BN_EPS}; would NaN")

    # Step 3: optionally load the ported weights (already in pytorchvideo key names, bare or
    # under "state_dict"), strictly and as float32.
    if weights is not None:
        state = torch.load(Path(weights), map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)

    # Step 4: the input-size-independent head pool described in the docstring.
    if resolution_agnostic_head:
        model.blocks[-1].pool = nn.AdaptiveAvgPool3d((None, 1, 1))
    return model
