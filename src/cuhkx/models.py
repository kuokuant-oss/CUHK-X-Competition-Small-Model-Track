"""Per-modality encoders, plus a late-fusion head.

The small encoders below are small for **historical** reasons, not principled ones. They
were sized under two beliefs that have since been falsified, and neither should constrain
anything new:

* "pretrained backbones are banned" — wrong. The organizers confirmed on 2026-09-02 that
  the restriction is *specifically* aimed at LLMs and large vision-language foundation
  models, and that pretrained CNNs are acceptable. The one hard limit is the **delivered
  model under 100 MB** (``docs/reference/organizer-correspondence.md`` C-2).
* that 3,036 clips over 40 classes forced small models — false. The public leaderboard's
  0.716 comes from a ~63M-parameter Kinetics/IG65M-pretrained R(2+1)D-34 quantized into
  that same 100 MB budget.

**Do not read the sizes here as a target to stay near.** Capacity is not the enemy;
*unregularized* capacity is. Measured on this project at fixed capacity: the depth line
went 0.2289 -> 0.4091 (+20 pp) on regularization alone.
"""
# Role: network definitions. Members B and C are Classifier(PretrainedVideo3D(
#   "ig65m_r2plus1d34", in_channels=4)): IG-65M -> Kinetics-400 R(2+1)D-34 with a 4-channel stem,
#   Kinetics normalisation buffers and a 512 -> 40 linear head.
# Used by: fd13_inference.model_from_spec (inference) and the training entry points in training/;
#   both. The skeleton, IMU and small image encoders, FrameNet, PretrainedTSN, PretrainedR2Plus1D,
#   AttentionPool3d and the other VIDEO_BACKBONES entries are development alternatives that the
#   delivered checkpoint does not use; "ircsn152" is the second teacher, used only in training.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from cuhkx.imu import N_CHANNELS as IMU_CHANNELS
from cuhkx.imu import N_DEVICES
from cuhkx.skeleton import BONES, N_JOINTS

N_CLASSES = 40


# Skeleton and IMU encoders from development; not used by the delivered checkpoint.
def skeleton_adjacency() -> torch.Tensor:
    """Symmetrically normalised adjacency of the 17-joint graph, with self-loops."""
    adjacency = np.eye(N_JOINTS, dtype=np.float32)
    for i, j in BONES:
        adjacency[i, j] = adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    norm = np.diag(degree**-0.5)
    return torch.from_numpy(norm @ adjacency @ norm)


class GraphTemporalBlock(nn.Module):
    """One spatial graph convolution over joints, then a temporal convolution over frames."""

    def __init__(self, in_ch: int, out_ch: int, adjacency: torch.Tensor, dropout: float):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.temporal = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.norm = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T, V)
        identity = self.residual(x)
        x = self.spatial(x)
        x = torch.einsum("bctv,vw->bctw", x, self.adjacency)
        x = self.temporal(x)
        x = self.norm(x)
        return self.drop(torch.relu(x + identity))


class SkeletonEncoder(nn.Module):
    """(B, T, 17, 3) -> (B, embed).

    The input is coordinates concatenated with per-frame velocity: the pose says *what*
    posture, the velocity says *what is moving*, and several of the 40 classes differ only
    in the latter (standing up versus sitting down are the same postures in reverse).
    """

    def __init__(self, widths: tuple[int, ...] = (64, 128, 128), dropout: float = 0.3):
        super().__init__()
        adjacency = skeleton_adjacency()
        self.input_norm = nn.BatchNorm1d(6 * N_JOINTS)
        blocks, in_ch = [], 6
        for width in widths:
            blocks.append(GraphTemporalBlock(in_ch, width, adjacency, dropout))
            in_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.embed_dim = in_ch

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(poses)
        velocity[:, 1:] = poses[:, 1:] - poses[:, :-1]
        x = torch.cat([poses, velocity], dim=-1)  # (B, T, V, 6)

        batch, frames, joints, channels = x.shape
        x = self.input_norm(x.permute(0, 2, 3, 1).reshape(batch, joints * channels, frames))
        x = x.reshape(batch, joints, channels, frames).permute(0, 2, 3, 1)  # (B, C, T, V)

        x = self.blocks(x)
        return x.mean(dim=(2, 3))


class ImuEncoder(nn.Module):
    """(B, 5, T, 16) + (B, 5) mask -> (B, embed).

    Devices are encoded separately with shared weights and then averaged over the *present*
    ones, so a clip missing a sensor is not fed zeros as if they were readings — one test
    clip has no IMU at all and a few training clips are partial (data-inventory).
    """

    def __init__(self, width: int = 96, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(IMU_CHANNELS, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.device_embed = nn.Parameter(torch.zeros(N_DEVICES, width))
        self.embed_dim = width

    def forward(self, imu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, devices, frames, channels = imu.shape
        x = imu.reshape(batch * devices, frames, channels).transpose(1, 2)
        x = self.net(x).mean(dim=2).reshape(batch, devices, -1)
        x = x + self.device_embed  # which sensor a reading came from is informative

        weights = mask.float().unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


# Small image encoders trained from scratch during development, and the temporal-shift helper
# that PretrainedTSN also uses; none of them is used by the delivered checkpoint.
def temporal_shift(x: torch.Tensor, n_segment: int, fold_div: int = 8) -> torch.Tensor:
    """Temporal Shift Module (Lin et al., ICCV 2019) on ``(B*T, C, H, W)``.

    Move a slice of channels one step forward in time and another slice one step back, and
    a plain 2D convolution afterwards sees three timesteps at once. It costs **no
    parameters and no FLOPs**, which makes it the cheapest temporal modelling available --
    not a substitute for 3D convolution, just a free thing to try first. (A previous version
    of this docstring justified it as avoiding 3D convolution "on a dataset that only has
    3,036 clips". That claim was never tested and the public 0.716 contradicts it; use
    ``PretrainedR2Plus1D`` when you want real spatiotemporal modelling.)

    Only ``2/fold_div`` of the channels move; shifting more starves the spatial pathway,
    which is the failure mode the paper warns about.
    """
    frames, channels, height, width = x.shape
    batch = frames // n_segment
    x = x.view(batch, n_segment, channels, height, width)
    fold = channels // fold_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # borrow from the next frame
    out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # and from the previous one
    out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
    return out.view(frames, channels, height, width)


class ImageEncoder(nn.Module):
    """(B, T, 3, H, W) uint8 -> (B, embed).

    Per-frame 2D convolutions with the weights shared across time, then a single pooling
    step over time. This followed plan hypothesis H2 ("a 3D convolution would spend its
    parameters on a time axis that barely exists"). **H2 was never tested and the public
    0.716 -- an R(2+1)D with 3D spatiotemporal convolution -- contradicts it.** Treat this
    class as the cheap baseline it is, not as a justified design. Mean and max are
    concatenated because they
    answer different questions — mean is "what does this clip look like throughout", max is
    "did this ever happen" — and several of the 40 classes hinge on a brief moment.

    Frames arrive as uint8 and are scaled here rather than in the cache: as float32 the
    training set would be 5 GB in RAM instead of 1.2 GB, and this cast is one cheap GPU op.
    No colour augmentation is applied anywhere, by ADR 0002 — the colours *are* the depths.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 3,
        shift: bool = False,
        shift_div: int = 8,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = in_channels
        for width in widths:
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = width
        self.shift = shift
        self.shift_div = shift_div
        self.drop = nn.Dropout(dropout)
        self.embed_dim = in_ch * 2

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time = frames.shape[:2]
        x = frames.reshape(batch * time, *frames.shape[2:]).float().div_(255.0)
        x = (x - 0.45) / 0.25
        for stage in self.stages:
            if self.shift:
                x = temporal_shift(x, time, self.shift_div)
            x = stage(x)
        x = x.mean(dim=(2, 3)).reshape(batch, time, -1)
        return self.drop(torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1))


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projection shortcut when shape changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


class FrameNet(nn.Module):
    """(B, T, C, H, W) -> class logits, by classifying each frame and averaging the logits.

    Two things distinguish this from :class:`ImageEncoder` plus :class:`Classifier`, and a
    public leaderboard entry reaching 0.80+ on thermal alone uses both.

    **Residual blocks**, not a plain stack. Four plain stride-2 convolutions is a shallow
    net with no gradient shortcut; the same depth in residual form trains to a much better
    optimum on the same data.

    **Consensus over logits, not over features.** The head runs on every frame and the
    logits are averaged, so each frame is separately answerable for the class — the TSN
    formulation. Pooling features first lets the network hide an unconfident frame inside
    the average instead of being scored on it.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        width: int = 32,
        dropout: float = 0.0,
        in_channels: int = 3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, 2, 3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, 2),
            ResidualBlock(width * 2, width * 4, 2),
            ResidualBlock(width * 4, width * 8, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * 8, n_classes)
        self.embed_dim = width * 8

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time = frames.shape[:2]
        x = frames.reshape(batch * time, *frames.shape[2:]).float().div_(255.0)
        x = (x - 0.5) / 0.25
        x = self.stem(x).flatten(1)
        return self.head(self.drop(x)).reshape(batch, time, -1).mean(dim=1)


# The wrapper of the delivered members: encoder features -> dropout -> linear layer to 40 logits.
# Their state dicts hold encoder.* and head.1.* (head.0 is the dropout, inactive in eval mode).
class Classifier(nn.Module):
    """One encoder plus a linear head — the unit that single-modality baselines compare."""

    def __init__(self, encoder: nn.Module, n_classes: int = N_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.embed_dim, n_classes))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(*inputs))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Pretrained backbones tried during development. PretrainedTSN and PretrainedR2Plus1D are not
# used by the delivered checkpoint; PretrainedVideo3D further down is.
class PretrainedTSN(nn.Module):
    """(B, T, C, H, W) uint8 -> (B, 512). ImageNet ResNet18 per frame, mean over time.

    The organizers confirmed on 2026-09-02 that "small, standard pretrained CNNs such as
    ImageNet-pretrained ResNet18 (~44MB) are perfectly acceptable" in this track
    (``docs/reference/organizer-correspondence.md`` C-2). Until that answer arrived the
    project defaulted to the strictest reading of "no large pretrained backbones" and
    trained everything from scratch, which is why the image encoders above are 409K
    parameters. That default was simply wrong. (The restriction is *specifically* aimed at
    LLMs and large vision-language foundation models; the only hard limit is the delivered
    model under 100 MB.)

    Why it should matter more here than the usual "pretraining helps a bit": the 15 classes
    this project cannot do -- 25_Watch_TV at 0.000, 26_Play_games at 0.050, 18_Write at
    0.105, 27% of all clips -- are all "seated, holding a different small object", and they
    fail because 17 joint coordinates cannot encode which object. ImageNet pretraining *is*
    object recognition. The match between the deficit and the remedy is the whole argument.

    Temporal consensus rather than 3D convolution, following TSN (Wang et al., ECCV 2016):
    the frame-picking (see ``fused.tsn_picks``) is where the temporal modelling budget goes.
    ⚠️ This has a real cost that was not acknowledged for two weeks: **a mean over time is
    order-invariant**, so this class cannot tell ``32_Stand_up`` from ``34_Sit_down``. The
    old justification -- "with 3,036 clips a 3D kernel spends parameters faster than it
    earns them" -- was never tested, and the public 0.716 is a 3D R(2+1)D. Set ``tsm=True``
    for zero-parameter temporal shift, or use ``PretrainedR2Plus1D`` for real 3D.

    Normalization uses per-channel statistics of *this* dataset, not ImageNet's constants,
    per ADR 0002: Depth_Color encodes distance as colour, so borrowing natural-image
    statistics would be asserting something false about the input. The first BatchNorm
    absorbs the difference in any case.
    """

    #: Filled in from the cache by the training script; these are placeholders that keep the
    #: module usable (and testable) before any cache exists.
    DEFAULT_MEAN = (0.45, 0.45, 0.45, 0.20)
    DEFAULT_STD = (0.25, 0.25, 0.25, 0.22)

    def __init__(
        self,
        in_channels: int = 4,
        dropout: float = 0.3,
        pretrained: bool = True,
        tsm: bool = False,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        stem = backbone.conv1
        inflated = nn.Conv2d(in_channels, stem.out_channels, 7, 2, 3, bias=False)
        with torch.no_grad():
            weight = stem.weight  # (64, 3, 7, 7)
            if in_channels <= 3:
                inflated.weight.copy_(weight[:, :in_channels])
            else:
                # RGB filters copied verbatim, extra channels seeded with their mean. The
                # alternative -- rescaling everything by 3/in_channels to hold the layer's
                # response to a uniform input constant -- buys nothing here, because bn1 sits
                # directly after this convolution and absorbs any global scale. Copying
                # verbatim keeps the property worth testing: channels 0-2 are exactly the
                # pretrained filters. Seeding channel 3 with the RGB *mean* rather than one
                # colour's filter avoids asserting that IR behaves like blue.
                extra = in_channels - 3
                inflated.weight[:, :3].copy_(weight)
                seed = weight.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
                inflated.weight[:, 3:].copy_(seed)
        backbone.conv1 = inflated
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.tsm = tsm
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = 512
        n = in_channels
        self.register_buffer(
            "mean", torch.tensor(mean or self.DEFAULT_MEAN[:n]).view(1, n, 1, 1), persistent=True
        )
        self.register_buffer(
            "std", torch.tensor(std or self.DEFAULT_STD[:n]).view(1, n, 1, 1), persistent=True
        )

    def _stages(self, x: torch.Tensor, n_segment: int) -> torch.Tensor:
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        for stage in (net.layer1, net.layer2, net.layer3, net.layer4):
            if self.tsm:
                x = temporal_shift(x, n_segment)
            x = stage(x)
        return net.avgpool(x).flatten(1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, n_segment = frames.shape[0], frames.shape[1]
        x = frames.reshape(batch * n_segment, *frames.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        features = self._stages(x, n_segment)
        return self.dropout(features.view(batch, n_segment, -1).mean(dim=1))


class PretrainedR2Plus1D(nn.Module):
    """(B, T, C, H, W) uint8 -> (B, 512). Kinetics-400 R(2+1)D-18, 3D spatiotemporal.

    This exists because ``PretrainedTSN`` has **no temporal modelling at all** -- it runs a
    2D backbone per frame and takes the mean, so any ordering of the frames gives the same
    answer. Half of this dataset's 40 classes are distinguished by *how* a motion evolves
    (``32_Stand_up`` vs ``34_Sit_down`` are the same frames in the opposite order), and a
    mean over time cannot represent that.

    Why this size is allowed, since the project spent two weeks believing it was not: the
    organizers' restriction is *specifically* aimed at LLMs and large vision-language
    foundation models (``organizer-correspondence.md`` C-2). A Kinetics-pretrained video CNN
    is neither. The only hard limit is the delivered model under 100 MB, and at 31.5M
    parameters this is 63.0 MB in fp16 -- one model, inside budget, no quantization needed.

    The external reference for this recipe is the public 0.716, whose plain argmax beats
    this project's entire six-member multimodal ensemble *plus* take decoding by 14 pp. It
    uses R(2+1)D-**34** with IG65M/Kinetics pretraining, packed to int5/int6 to fit. R(2+1)D-18
    is the same family one step smaller, and it fits in fp16 without a custom packing format.

    Two deliberate departures from ``PretrainedTSN``:

    * **Kinetics normalization constants**, not this dataset's channel statistics. The
      pretrained filters were fitted to expect those, and ``conv1`` mixes channels, so a
      per-channel rescale is *not* absorbed by the BatchNorm that follows it (which is what
      the ``PretrainedTSN`` docstring assumes). The 4th channel takes the mean of the three,
      matching how its stem weights are seeded. This is not a claim that ADR 0002 is wrong:
      ADR 0002 forbids *colour augmentation*, which is untouched here.
    * **The stem is inflated 3 -> 4 channels** by copying the RGB filters verbatim and
      seeding the IR channel with their mean -- identical to ``PretrainedTSN`` and to the
      0.716 notebook's ``adapt_input_conv``. Channels 0-2 stay bit-identical to the
      pretrained filters, which is the property worth testing.
    """

    KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
    KINETICS_STD = (0.22803, 0.22145, 0.216989)

    def __init__(
        self,
        in_channels: int = 4,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        stem = backbone.stem[0]
        if in_channels != stem.in_channels:
            inflated = nn.Conv3d(
                in_channels,
                stem.out_channels,
                stem.kernel_size,
                stem.stride,
                stem.padding,
                bias=stem.bias is not None,
            )
            with torch.no_grad():
                keep = min(in_channels, stem.in_channels)
                inflated.weight[:, :keep].copy_(stem.weight[:, :keep])
                if in_channels > stem.in_channels:
                    seed = stem.weight.mean(dim=1, keepdim=True)
                    inflated.weight[:, stem.in_channels :].copy_(
                        seed.expand(-1, in_channels - stem.in_channels, -1, -1, -1)
                    )
                if stem.bias is not None:
                    inflated.bias.copy_(stem.bias)
            backbone.stem[0] = inflated
        self.embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)

        n = in_channels
        mean = (*self.KINETICS_MEAN, sum(self.KINETICS_MEAN) / 3.0)[:n]
        std = (*self.KINETICS_STD, sum(self.KINETICS_STD) / 3.0)[:n]
        self.register_buffer("mean", torch.tensor(mean).view(1, n, 1, 1), persistent=True)
        self.register_buffer("std", torch.tensor(std).view(1, n, 1, 1), persistent=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> normalize per frame -> (B, C, T, H, W) for Conv3d
        x = frames.float().div_(255.0)
        x = (x - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return self.dropout(self.backbone(x.permute(0, 2, 1, 3, 4)))


#: Kinetics-400 video backbones in torchvision that accept this project's 128px cache.
#: ``s3d`` and ``mvit_v2_s`` are deliberately absent: s3d's final pooling kernel is larger
#: than a 128px feature map, and MViT ties positional embeddings to exactly 224px. Both were
#: measured, not assumed. Sizes are parameters only; the delivered bytes are what
#: ``budget.save_deliverable`` reports.
VIDEO_BACKBONES = {
    # arch -> (builder, weights enum / hub repo / checkpoint path, parameter count).
    "r2plus1d": ("r2plus1d_18", "R2Plus1D_18_Weights", 31_505_325),
    "r3d": ("r3d_18", "R3D_18_Weights", 33_371_472),
    "swin3d_t": ("swin3d_t", "Swin3D_T_Weights", 28_158_070),
    "swin3d_s": ("swin3d_s", "Swin3D_S_Weights", 49_816_678),
    "swin3d_b": ("swin3d_b", "Swin3D_B_Weights", 88_048_984),
    # The architecture of the delivered members B and C.
    # IG-65M -> Kinetics-400 pretrained R(2+1)D-34 (63.7M), the community backbone behind the
    # public 0.667/0.711 solutions. Loaded from the moabitcoin hub repo rather than torchvision;
    # 127 MB fp16 so it ships int8 only. Builder "__hub__" routes to torch.hub below.
    "ig65m_r2plus1d34": ("__hub__", "moabitcoin/ig65m-pytorch", 63_697_175),
    # Used only in training, for the second teacher, whose predictions were averaged into member
    # B's pseudo-labels; it is not part of the delivered checkpoint.
    # ir-CSN-152, IG65M -> Kinetics-400 (82.87% top-1 against IG65M R(2+1)D-34's 79.10), and
    # only 29.7M parameters -- about 30 MB at int8 against the R(2+1)D-34 soup's 63.84 MB.
    # There is no torchvision or hub build of it: the weights come from MMAction2 and are
    # remapped into pytorchvideo's CSN by scripts/59_port_csn.py, which must be run first.
    # Builder "__csn__" routes to cuhkx.port.build_csn, the single definition of the
    # architecture that the porter also uses -- they must not drift.
    "ircsn152": ("__csn__", "ircsn152-ig65m/ircsn152_ig65m_k400.pt", 29_703_568),
}


# Optional replacement for the backbone's average pooling. The delivered members were built with
# attention_pool="none": their weights contain no scoring convolution.
class AttentionPool3d(nn.Module):
    """Learned spatial pooling, initialised to be exactly average pooling.

    Replaces the backbone's ``AdaptiveAvgPool3d`` with a weighted sum whose weights come from
    a 1x1x1 convolution over the final feature map, softmaxed across positions. The scoring
    conv is **zero-initialised**, so at step 0 every position gets weight ``1/N`` and the model
    is bit-identical to the one without it. Any deviation from uniform is something training
    chose, which is what makes the comparison against a run with ``attention_pool="none"``
    clean -- no random perturbation to disentangle from the effect.

    Why it exists here rather than a hand-cropped region. The obvious way to tell the model
    where to look is to crop to the hands, and the skeleton looked like it could supply that.
    It cannot: the joints are root-relative x/y with z as height above ground, in metres, and
    the raw JSON carries only ``keypoints`` (17,3) and ``keypoint_scores`` -- no 2D keypoints,
    no bounding box, no camera parameters. There is no joint-to-pixel mapping to be had, so
    every "crop to the wrist" design is blocked at the data, not at the effort. What survives
    is letting the classification loss find the region on its own.

    ``mode="spatial"`` normalises over (H, W) within each frame and then averages over time,
    which keeps the temporal average pooling the pretrained weights were trained under.
    ``mode="spatiotemporal"`` normalises over (T, H, W) jointly, letting the model down-weight
    whole frames -- more expressive and more able to overfit 3,036 clips.

    The attention map is worth reading, not just training: ``last_weights`` holds the most
    recent (B, T, H, W) normalised weights, which turns "which parts of the frame matter" from
    an assumption into a measurement.
    """

    def __init__(self, channels: int, mode: str = "spatial"):
        super().__init__()
        if mode not in {"spatial", "spatiotemporal"}:
            raise ValueError(f"mode must be spatial/spatiotemporal, got {mode!r}")
        self.mode = mode
        self.score = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        logits = (
            self.score(x).view(b, t, h * w)
            if self.mode == "spatial"
            else self.score(x).view(b, 1, t * h * w)
        )
        weights = logits.softmax(dim=-1)
        self.last_weights = weights.detach().view(b, t, h, w)
        if self.mode == "spatial":
            per_frame = (x.view(b, c, t, h * w) * weights.unsqueeze(1)).sum(dim=-1)
            pooled = per_frame.mean(dim=-1)
        else:
            pooled = (x.view(b, c, t * h * w) * weights).sum(dim=-1)
        return pooled.view(b, c, 1, 1, 1)


class PretrainedVideo3D(nn.Module):
    """(B, T, C, H, W) uint8 -> (B, embed). Any Kinetics-400 backbone in VIDEO_BACKBONES.

    Generalises ``PretrainedR2Plus1D`` so capacity is a flag rather than a rewrite, because
    the measured gap is capacity: our model scores 0.54228 on the leaderboard with decoding
    switched off, against 0.716 for a plain argmax from a video-pretrained 3D network -- the
    same 402 clips, neither side using structural post-processing.

    The 100 MB delivery cap is the only ceiling, and it is not reached by parameter count
    alone: at int8 storage even ``swin3d_b`` (88.0M) fits, which is how the public solution
    packs a 63M R(2+1)D-34 into 93.7 MB.

    Two families, two stem locations -- resolved by finding the first ``Conv3d`` rather than
    hard-coding a path, so adding a backbone is one line in the table above:

    * ResNet-style video nets (``r2plus1d``, ``r3d``): stem at ``stem[0]``, head at ``fc``
    * Video Swin (``swin3d_*``): stem at ``patch_embed.proj``, head at ``head``

    Channel inflation is the standard I3D trick and identical to ``PretrainedTSN``: copy the
    RGB filters verbatim into channels 0-2, seed the IR channel with their mean. Channels 0-2
    stay bit-identical to the pretrained weights, which is the property worth testing.
    """

    KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
    KINETICS_STD = (0.22803, 0.22145, 0.216989)

    def __init__(
        self,
        arch: str = "r2plus1d",
        in_channels: int = 4,
        dropout: float = 0.3,
        pretrained: bool = True,
        attention_pool: str = "none",
    ):
        super().__init__()
        if arch not in VIDEO_BACKBONES:
            raise ValueError(f"unknown arch {arch!r}; known: {sorted(VIDEO_BACKBONES)}")
        import torchvision.models.video as video

        # Step 1: build the Kinetics-400 backbone. At inference fd13_inference passes
        # pretrained=False and redirects torch.hub.load to the bundled source in
        # runtime/torch/hub, so nothing is downloaded; every weight comes from weights/model.pt.
        builder, weights_enum, _ = VIDEO_BACKBONES[arch]
        if builder == "__hub__":
            # moabitcoin's r2plus1d_34_32_kinetics: torchvision-video-resnet layout
            # (Conv3d stem + Linear fc), so the inflation and head-swap below apply unchanged.
            # num_classes must be 400 for the pretrained kinetics head; it is discarded here.
            backbone = torch.hub.load(
                weights_enum,
                "r2plus1d_34_32_kinetics",
                num_classes=400,
                pretrained=pretrained,
                trust_repo=True,
            )
        elif builder == "__csn__":
            from cuhkx.paths import p as _paths
            from cuhkx.port import build_csn

            ckpt = Path(_paths("models_root")) / weights_enum
            if pretrained and not ckpt.exists():
                raise FileNotFoundError(
                    f"{ckpt} is missing; run scripts/59_port_csn.py first (it needs the "
                    "MMAction2 checkpoint downloaded, see roadmap steps 19-20)"
                )
            # Loaded here, before the stem inflation and head swap below, exactly as the
            # __hub__ path does: the ported weights carry a 3-channel stem and a 400-class
            # head, so they have to land before those get replaced.
            backbone = build_csn(weights=ckpt if pretrained else None)
        else:
            weights = getattr(video, weights_enum).KINETICS400_V1 if pretrained else None
            backbone = getattr(video, builder)(weights=weights)

        # Step 2: widen the first Conv3d (the stem) to in_channels inputs: channels 0-2 keep the
        # RGB filters and each extra (IR) channel gets their mean. For R(2+1)D-34 this is stem.0,
        # (45, 3, 1, 7, 7) -> (45, 4, 1, 7, 7), without bias.
        stem_name = next(
            name for name, mod in backbone.named_modules() if isinstance(mod, nn.Conv3d)
        )
        stem = backbone.get_submodule(stem_name)
        if in_channels != stem.in_channels:
            inflated = nn.Conv3d(
                in_channels,
                stem.out_channels,
                stem.kernel_size,
                stem.stride,
                stem.padding,
                bias=stem.bias is not None,
            )
            with torch.no_grad():
                keep = min(in_channels, stem.in_channels)
                inflated.weight[:, :keep].copy_(stem.weight[:, :keep])
                if in_channels > stem.in_channels:
                    seed = stem.weight.mean(dim=1, keepdim=True)
                    inflated.weight[:, stem.in_channels :].copy_(
                        seed.expand(-1, in_channels - stem.in_channels, -1, -1, -1)
                    )
                if stem.bias is not None:
                    inflated.bias.copy_(stem.bias)
            # Put the widened convolution in place of the old one inside its parent module.
            parent_name, _, leaf = stem_name.rpartition(".")
            setattr(
                backbone.get_submodule(parent_name) if parent_name else backbone, leaf, inflated
            )

        # Step 3 (optional; off for the delivered members): replace the average pooling with
        # AttentionPool3d.
        if attention_pool != "none":
            pool_name = next(
                (n for n, m in backbone.named_modules() if isinstance(m, nn.AdaptiveAvgPool3d)),
                None,
            )
            if pool_name is None:
                raise ValueError(
                    f"arch {arch!r} has no AdaptiveAvgPool3d to replace; attention_pool "
                    "is only wired for the video-ResNet layouts"
                )
            channels = [m for m, x in backbone.named_modules() if isinstance(x, nn.Linear)]
            width = backbone.get_submodule(channels[-1]).in_features
            parent, _, leaf = pool_name.rpartition(".")
            setattr(
                backbone.get_submodule(parent) if parent else backbone,
                leaf,
                AttentionPool3d(width, mode=attention_pool),
            )

        # Step 4: replace the 400-class Kinetics head (the last Linear; fc for R(2+1)D-34) with
        # Identity, so the module returns pooled features of size embed_dim (512 for R(2+1)D-34).
        head_name = [n for n, m in backbone.named_modules() if isinstance(m, nn.Linear)][-1]
        head = backbone.get_submodule(head_name)
        self.embed_dim = head.in_features
        parent_name, _, leaf = head_name.rpartition(".")
        setattr(
            backbone.get_submodule(parent_name) if parent_name else backbone, leaf, nn.Identity()
        )

        self.arch = arch
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        # Step 5: normalisation buffers, Kinetics mean and std on channels 0-2 and their average
        # on the IR channel, shaped (1, C, 1, 1). They are persistent, so they are saved with the
        # weights, and at inference load_state_dict restores the checkpoint's copies.
        n = in_channels
        mean = (*self.KINETICS_MEAN, sum(self.KINETICS_MEAN) / 3.0)[:n]
        std = (*self.KINETICS_STD, sum(self.KINETICS_STD) / 3.0)[:n]
        self.register_buffer("mean", torch.tensor(mean).view(1, n, 1, 1), persistent=True)
        self.register_buffer("std", torch.tensor(std).view(1, n, 1, 1), persistent=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W) uint8, (B, 32, 4, 128, 128) for the members. Scale to [0, 1],
        # normalise each channel and permute to (B, C, T, H, W) for Conv3d; returns (B, embed_dim).
        x = frames.float().div_(255.0)
        x = (x - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return self.dropout(self.backbone(x.permute(0, 2, 1, 3, 4)))
