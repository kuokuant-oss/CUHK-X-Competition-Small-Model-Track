"""Subject-wise cross-validation harness.

Every run reports the same three numbers together — accuracy, deliverable MB, and
milliseconds per clip — because this is Small Model Track and a result without a size is
not a result (ADR 0006). Per-fold accuracies and their spread are always reported next to
the mean, since the fold-to-fold subject variation *is* the thing being fought (ADR 0004).
"""
# Role: the shared training library: TrainConfig, seeding, the training loop for one model
# (train_one_fold: AdamW, one-cycle schedule, mixup, optional AMP and EMA/SWA, resumable per-epoch
# checkpoints) and the subject-wise cross-validation harness (cross_validate).
# Used by: the training/ scripts 44, 47, 101, 103, 108 and 124 (none calls cross_validate) and,
# at inference, cuhkx.fd13_inference (seed_everything) and cuhkx.tta (_collate); both.

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

# Must be set before CUDA initialises, so it goes above the torch import rather than inside
# seed_everything: cuBLAS reads it once when it creates its handle, and a later assignment
# is silently ignored. Without it, torch.use_deterministic_algorithms raises on any matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cuhkx.budget import MB, BudgetExceeded, deliverable_bytes, save_deliverable
from cuhkx.models import count_parameters


@dataclass
class TrainConfig:
    n_frames: int = 16
    batch_size: int = 64
    epochs: int = 40
    lr: float = 3e-3  # Peak of the one-cycle schedule; the backbone's is lr * backbone_lr_mult.
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    #: Mixup strength (Beta(a,a)); 0 disables. Best-in-class cross-subject regulariser for
    #: HAR domain generalisation (measured across the DG literature). Mixes inputs and
    #: interpolates the two labels' losses.
    mixup_alpha: float = 0.0
    dropout: float = 0.3
    seed: int = 42
    ema_decay: float = 0.0  # Decay of an exponential moving average of the weights; 0 disables.
    #: Stochastic Weight Averaging (Izmailov et al. 2018): after ``swa_start`` of the epochs,
    #: the weights at the end of every epoch are averaged with equal weight and BatchNorm
    #: statistics are recomputed once at the end. 0 disables and reproduces every earlier run.
    swa_start: float = 0.0
    #: Keep BatchNorm in inference mode while training: ``"none"`` (default, reproduces every
    #: earlier run), ``"stats"`` (use the pretrained running mean/var, stop updating them, but
    #: keep learning the affine scale and shift), ``"all"`` (also freeze the affine params).
    #: Three separate observations point here and none of them has been tested: the image line
    #: trains BatchNorm3d at batch 8 while the public baseline uses 16, and small batches make
    #: BN's per-batch statistics noisy (`arXiv:2105.07576`); the one 160px run that produced
    #: the "high resolution hurts" verdict used batch **5**, so that verdict is confounded
    #: with BN and was retracted; and 12-view TTA lost 1.0 pp on the leaderboard, which is the
    #: exact signature of the train/test BN statistic mismatch that corner crops provoke
    #: (`arXiv:2604.09697`). "stats" is the variant those three share.
    freeze_bn: str = "none"
    #: Learning rate for a pretrained backbone, as a multiple of ``lr``. 1.0 keeps the
    #: single-rate behaviour every result before 2026-09-03 was produced under; 0.1 is the
    #: standard fine-tuning ratio. See :func:`_param_groups` for why it matters here.
    backbone_lr_mult: float = 1.0
    #: Exclude 1-D parameters -- biases and every normalisation affine -- from weight decay.
    #: Standard modern practice ("no bias decay"), and this project was not doing it: AdamW was
    #: applying wd uniformly to all 155 BatchNorm affines and every bias. It should matter most
    #: for ir-CSN, whose depthwise kernels are (C,1,3,3,3) -- 27 parameters per channel, so a
    #: given decay is far larger relative to the weight norm than for a full conv. False keeps
    #: the behaviour every earlier result was produced under.
    no_decay_norm_bias: bool = False
    #: Clip the global gradient norm before the optimizer step. 0 disables, which is what every
    #: earlier run did -- this trainer had no clipping at all. MMAction2 fine-tunes this exact
    #: CSN checkpoint with max_norm=40 under SGD; 1.0 is the conventional value for AdamW, whose
    #: per-parameter scaling makes 40 effectively never fire.
    clip_grad_norm: float = 0.0
    #: Accumulate this many batches per optimizer step, so the effective batch is
    #: ``batch_size * grad_accum`` at unchanged VRAM. MMAction2 fine-tunes the CSN checkpoint
    #: at an effective batch of **96** (12 per GPU x 8 GPUs) against our 8 -- a 12x gap whose
    #: BatchNorm half is already handled by ``freeze_bn`` (their config is ``bn_frozen=True``
    #: too) but whose gradient-noise half is not, and noise scales as sqrt(12) ~ 3.5x. Also
    #: the way a 224px CSN fits if batch 8 does not: batch 4 with accum 2 keeps the effective
    #: batch identical, so resolution stays a single variable. 1 is the old behaviour and is
    #: bit-for-bit unchanged -- the window closes on every batch and ``total_steps`` is the
    #: same, so nothing about the default path moves.
    grad_accum: int = 1
    #: OneCycle warmup fraction. 0.3 is torch's default and what every earlier run used.
    pct_start: float = 0.3
    log_every: int = 10  # Accuracy log line every N epochs and at the last one; 0 disables.
    log_sample: int = 512  # Clips in the un-augmented training subsample scored by that line.
    #: Where to write an every-epoch checkpoint so a killed run resumes instead of
    #: restarting. Off by default, so every result recorded before 2026-09-03 is produced by
    #: exactly the code path that produced it. `cross_validate` only writes weights when a
    #: whole fold finishes, so before this a shutdown at epoch 39 of 40 lost everything.
    checkpoint_path: Path | None = None
    #: Appended one JSON object per epoch. On a preemptible remote this is the only durable
    #: evidence the run is alive -- a session that still exists and a heartbeat that still
    #: ticks both look healthy long after the work has died, and a 378 MB checkpoint is too
    #: big to pull back every epoch. This is under 10 KB for a whole run.
    metrics_path: Path | None = None
    #: Every this many epochs, score the held-out split and keep a weights-only snapshot, so
    #: the epoch curve can be read off ONE finished run instead of re-running the whole thing
    #: per budget. This exists because val used to be computed only inside the ``log_every``
    #: print and then thrown away: nothing recorded which epoch was best, which is why
    #: ``epochs=30`` for CSN is registered as PICKED NOT KNOWN (0.79 sigma, chosen on a single
    #: fold). Snapshots go to ``<checkpoint_path>.parent/snapshots/<stem>-eNNN.pt`` -- derived
    #: from the per-fold checkpoint so four folds cannot overwrite each other's curve. Weights
    #: only, no optimizer state: this is for reading a curve and shipping the best epoch, not
    #: for resuming. 0 disables it and leaves every earlier run's behaviour unchanged.
    keep_every: int = 0
    #: Identity carried into the checkpoint and asserted on resume. ``fold`` is the one that
    #: bites: two folds train on the same clips, so every clip_id check passes while the
    #: weights come from the wrong split. All three default to None, which means "not
    #: declared" and passes, so nothing recorded before 2026-09-03 changes.
    fold: int | None = None
    recipe_hash: str | None = None
    windows_hash: str | None = None
    #: Batch size for the *evaluation* passes. 256 is fine for a per-frame 2D encoder but
    #: not for a 3D one: R(2+1)D holds spatiotemporal activations for every clip in the
    #: batch at once, and 256 clips x 16 frames OOMs an 8 GB card at the first logged
    #: epoch -- after the training loop has already run fine, which makes it look like a
    #: training problem when it is not. Default unchanged so existing runs stay identical.
    eval_batch_size: int = 256
    #: Worker processes for the training DataLoader. **0 was the default until 2026-09-03**,
    #: which means every batch was assembled on the same thread that dispatches CUDA work:
    #: a ragged memmap read, a fancy-index copy, a transpose plus ascontiguousarray, a random
    #: crop, an erase and a final from_numpy copy, all serialised in front of the GPU. The
    #: fold-0 R(2+1)D run achieved 5.1 clips/s against a model that benchmarks far above it.
    #: Determinism survives this: FusedFrameDataset derives its per-sample RNG from
    #: (seed, epoch, clip index), not from a shared stream, precisely so worker count cannot
    #: change what a sample looks like.
    num_workers: int = 0
    #: Pin cuDNN's kernel choice and forbid nondeterministic reductions. True is what every
    #: recorded result was produced under and is required for anything with a control arm --
    #: fold-0 against 0.6658, an ablation against its own baseline. It is **not** required
    #: for the full-data deliverable, which has no comparison to preserve, and B-0 §4 makes
    #: turning it off there the registered action when det_off/det_on >= 2. Reproducibility
    #: of the deliverable is protected differently: the seed, config and checkpoint are all
    #: recorded, and `scripts/79_determinism_check.py` measures what this is worth rather
    #: than assuming it.
    deterministic: bool = True
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Run-specific settings read by the calling scripts (cache name, crop, augmentation, and
    # "subjects" for the per-fold report of cross_validate).
    extra: dict = field(default_factory=dict)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every generator this project draws from, and pin the GPU kernels.

    Seeding alone does not make a CUDA run repeatable. cuDNN picks convolution algorithms by
    benchmarking them, and several backward kernels accumulate in nondeterministic order, so
    two runs of the same config diverge from the first epoch and the divergence compounds.
    Measured on this project before the pin was added: the same member at the same seed came
    back with 19-24% of its out-of-fold predictions changed and its CV score moved by up to
    0.65 pp, in both directions.

    That matters beyond tidiness. The finals award 10% for reproducibility and disqualify a
    rerun that lands more than 10% away from the submitted score, and a diagnosis run twice
    with different answers is not a diagnosis.

    ``warn_only`` is on because a handful of ops have no deterministic implementation and
    would raise rather than degrade. That means this pins most of the nondeterminism,
    not provably all of it -- which is why it is checked by measurement
    (scripts/79_determinism_check.py) rather than assumed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The escape hatch exists so the pin can be measured rather than believed: setting
    # CUHKX_DETERMINISM=0 is how scripts/79_determinism_check.py runs its control arm and
    # shows what this function is actually buying. Never set it for a real run.
    if os.environ.get("CUHKX_DETERMINISM", "1") == "0":
        deterministic = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Not merely "unpinned": cudnn.benchmark is what lets cuDNN autotune a convolution
        # algorithm per input shape, and for a 3D backbone at a fixed shape that is most of
        # what the pin costs. Leaving it False here would give up the determinism and get
        # nothing back.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    n_val: int
    val_subjects: list[str]


@dataclass
class RunResult:
    name: str
    folds: list[FoldResult]
    params: int
    megabytes: float
    ms_per_clip: float
    config: dict
    oof_probs: np.ndarray | None = None

    @property
    def accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def std(self) -> float:
        return float(np.std([f.accuracy for f in self.folds]))

    def summary(self) -> str:
        per_fold = "  ".join(f"f{f.fold}={f.accuracy:.4f}" for f in self.folds)
        return (
            f"{self.name}\n"
            f"  CV accuracy : {self.accuracy:.4f} +/- {self.std:.4f}   ({per_fold})\n"
            f"  deliverable : {self.megabytes:.2f} MB   ({self.params:,} params)\n"
            f"  inference   : {self.ms_per_clip:.2f} ms/clip"
        )

    def as_row(self) -> str:
        """A row for reports/experiments.md, whose size columns cannot be left blank."""
        per_fold = " / ".join(f"{f.accuracy:.3f}" for f in self.folds)
        return (
            f"| {self.name} | {self.accuracy:.4f} | {per_fold} | {self.std:.4f} | "
            f"{self.megabytes:.2f} | {self.ms_per_clip:.2f} |"
        )


class ArrayDataset(Dataset):
    """Holds already-cached tensors, so an epoch never touches the filesystem.

    ``augment`` maps one sample's ``{name: array}`` to an augmented one and must return the
    same keys in the same order. It is applied per access, so a sample is transformed
    differently on each epoch — and it is passed only for the training split, never for
    validation, since an augmented validation set measures a different task.
    """

    def __init__(
        self,
        tensors: dict[str, np.ndarray],
        labels: np.ndarray,
        augment: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
        indices: np.ndarray | None = None,
    ):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment
        self.indices = None if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.labels) if self.indices is None else len(self.indices)

    def __getitem__(self, i: int):
        # `indices` selects a split without copying. Slicing the arrays instead would hold
        # a train copy and a val copy of every fold alongside the originals — 591 MB per
        # split for an image modality, which ran the thermal run out of memory on fold 1.
        if self.indices is not None:
            i = int(self.indices[i])
        sample = {name: array[i] for name, array in self.tensors.items()}
        if self.augment is not None:
            sample = self.augment(sample)
        inputs = tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in sample.values())
        return inputs, int(self.labels[i])


# Collates (inputs, label) samples, where inputs is a tuple of tensors, into (tuple of stacked
# tensors, int64 label tensor). Also used at inference by cuhkx.tta.predict.
def _collate(batch):
    inputs, labels = zip(*batch, strict=True)
    stacked = tuple(torch.stack(parts) for parts in zip(*inputs, strict=True))
    return stacked, torch.tensor(labels, dtype=torch.long)


def _rng_state() -> dict:
    """Every generator a resumed run would otherwise restart from its seed."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    # torch.load(map_location="cuda") moves these byte tensors onto the GPU; set_rng_state
    # (torch 2.11) requires a CPU ByteTensor, so coerce back before restoring.
    torch.set_rng_state(state["torch"].to("cpu", torch.uint8))
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([t.to("cpu", torch.uint8) for t in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _atomic_save(payload: dict, path: Path) -> None:
    """Write, flush to the platter, then rename. A truncated checkpoint is worse than none.

    ``torch.save`` straight to the destination leaves a half-written file if the process is
    killed mid-write, which on a preemptible runtime is the expected way for it to end. The
    failure is not always loud either: a truncated file usually fails to load, but it can
    load with the tail of the payload missing. ``fsync`` before the rename is what makes the
    rename a real barrier rather than one against the page cache.
    """
    temporary = path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_metrics(path: Path, row: dict) -> None:
    """One JSON object per line, flushed and fsynced -- it is the run's proof of life."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_identity(saved: dict, key: str, expected, path: Path) -> None:
    """Refuse a resume whose checkpoint describes a different experiment.

    ``None`` on either side means the field was never declared, which is how every run
    recorded before 2026-09-03 reads, so those resume exactly as they did.
    """
    found = saved.get(key)
    if expected is None or found is None:
        return
    if found != expected:
        raise ValueError(
            f"checkpoint {path.name} has {key}={found!r} but this run declares {key}="
            f"{expected!r}. That is a different experiment, not a resume -- point --name at "
            f"its own directory, or delete the checkpoint to start over."
        )


def _loader_kwargs(config: TrainConfig) -> dict:
    """Worker settings for the training loader, valid for ``num_workers == 0`` too.

    ``persistent_workers`` and ``prefetch_factor`` are rejected outright by DataLoader when
    there are no workers, so they cannot simply be passed through with a default.
    """
    if config.num_workers <= 0:
        return {"num_workers": 0, "pin_memory": config.pin_memory}
    return {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
        "prefetch_factor": config.prefetch_factor,
    }


def _param_groups(model: nn.Module, config: TrainConfig):
    """``(groups, max_lrs)`` -- a lower learning rate for a pretrained backbone.

    Fine-tuning a pretrained network with one learning rate everywhere is the default and it
    is wrong in a specific way: the rate that a freshly initialised head needs is large
    enough to walk the pretrained filters away from what they learned. The standard remedy
    is a discriminative rate, backbone at 0.05-0.1x the head.

    That matters here more than usual. The whole reason the image line works at all is that
    Kinetics pretraining carries a *motion* prior that this dataset is too small to learn
    (measured: +22.76 pp over an ImageNet backbone, and the overfitting gap halved without a
    single regularisation knob being touched). A rate that erodes that prior is spending the
    only thing that helped.

    ``backbone_lr_mult = 1.0`` reproduces the single-rate behaviour exactly, including the
    number of parameter groups, so every result recorded before 2026-09-03 stays on the code
    path that produced it.

    The split is by module identity rather than by name matching: ``PretrainedVideo3D`` and
    ``PretrainedTSN`` both hold the pretrained weights under ``.backbone``, while the stem
    conv that was *inflated* to four channels lives inside it and is only partly pretrained.
    Treating the inflated stem as backbone is deliberate -- channels 0-2 of it are the
    pretrained filters verbatim.
    """

    def split_decay(params):
        """(decay, no_decay). 1-D catches biases and every norm affine, architecture-agnostic."""
        if not config.no_decay_norm_bias:
            return params, []
        return [q for q in params if q.ndim > 1], [q for q in params if q.ndim <= 1]

    def build(pairs):
        """pairs of (params, lr) -> optimizer groups, splitting each by weight decay."""
        groups, lrs = [], []
        for params, lr in pairs:
            decay, no_decay = split_decay(params)
            if decay:
                groups.append({"params": decay, "lr": lr})
                lrs.append(lr)
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
                lrs.append(lr)
        return groups, lrs

    # One learning rate for every parameter (split by weight decay only if requested).
    if config.backbone_lr_mult == 1.0:
        if not config.no_decay_norm_bias:
            return list(model.parameters()), config.lr
        groups, lrs = build([(list(model.parameters()), config.lr)])
        return groups, lrs

    # Two learning rates: lr * backbone_lr_mult for the pretrained backbone (model.encoder.backbone)
    # and lr for all other parameters, in practice the classifier head.
    backbone = None
    encoder = getattr(model, "encoder", model)
    backbone = getattr(encoder, "backbone", None)
    if backbone is None:
        raise ValueError(
            "backbone_lr_mult != 1.0 but the model has no `.encoder.backbone` to slow down; "
            "either the architecture changed or this is not a pretrained-backbone run"
        )
    backbone_ids = {id(p) for p in backbone.parameters()}
    slow = [p for p in model.parameters() if id(p) in backbone_ids]
    fast = [p for p in model.parameters() if id(p) not in backbone_ids]
    if not slow or not fast:
        raise ValueError(
            f"discriminative lr split degenerate: {len(slow)} backbone tensors, "
            f"{len(fast)} head tensors -- one side is empty, so this would silently be a "
            f"single-rate run"
        )
    backbone_lr = config.lr * config.backbone_lr_mult
    return build([(slow, backbone_lr), (fast, config.lr)])


def _apply_freeze_bn(net: nn.Module, mode: str) -> None:
    """Hold BatchNorm in inference mode for the rest of this epoch.

    ``model.train()`` puts every module into training mode, which for BatchNorm means
    "normalise by *this batch's* statistics and update the running estimates". At batch 8 on a
    3D net that batch is a poor estimate of anything, and the running estimates it writes are
    what test-time inference then uses — so the noise is not averaged away, it is baked into a
    buffer. ``"stats"`` keeps the pretrained running estimates and normalises by them in both
    training and inference, which removes the train/test mismatch entirely and leaves the
    affine scale and shift free to adapt. ``"all"`` additionally freezes the affine.

    Called every epoch because ``model.train()`` is called every epoch and would otherwise
    undo it. Setting ``requires_grad`` here rather than before the optimizer is built is safe:
    a parameter with no gradient is skipped by AdamW, decoupled weight decay included.
    """
    if mode == "none":
        return
    if mode not in {"stats", "all"}:
        raise ValueError(f"freeze_bn must be none/stats/all, got {mode!r}")
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            if mode == "all":
                for param in module.parameters(recurse=False):
                    param.requires_grad_(False)


def _update_bn(loader: DataLoader, net: nn.Module, device: str) -> None:
    """Recompute BatchNorm running statistics for an averaged model.

    Averaging weights does not average BatchNorm's running mean and variance — those are
    buffers, not parameters, so the averaged model inherits whichever mini-batch happened to
    be last. Skipping this is the usual reason EMA gets measured as worse than the model it
    came from and then written off.

    ``torch.optim.swa_utils.update_bn`` cannot be used here: it assumes each batch is a
    tensor (or a sequence whose first element is one), while every model in this project
    takes several tensors — IMU passes readings and a presence mask, and the collate
    function yields a tuple of them.
    """
    stats = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not stats:
        return
    momenta = {m: m.momentum for m in stats}
    for module in stats:
        module.reset_running_stats()
        module.momentum = None  # cumulative average over the pass, not exponential

    was_training = net.training
    net.train()
    with torch.no_grad():
        for inputs, _ in loader:
            net(*(t.to(device) for t in inputs))
    for module, momentum in momenta.items():
        module.momentum = momentum
    net.train(was_training)


def _clean_view(dataset: Dataset, limit: int, deterministic=None) -> Dataset | None:
    """A fixed subsample of the training split with augmentation off.

    Accuracy on this is the number that separates "cannot fit" from "fits a harder,
    augmented target", and the two demand opposite responses — capacity versus
    regularisation. It has to be logged every run, not reconstructed afterwards: this
    project spent a day reading an augmented 0.50 as underfitting, and the reading only
    came apart when someone went back and measured the clean number once, by hand.

    Fixed subsample rather than the whole split because it is evaluated repeatedly and the
    trend is what matters, not the third decimal place.
    """
    if limit <= 0:
        return None
    # A dataset that samples frames per access (fused.FusedFrameDataset) cannot be rebuilt
    # from `.tensors`, so it supplies its own deterministic view instead. Without this the
    # clean-train number silently goes missing for the image line -- and that number is the
    # only thing separating "cannot fit" from "fits an augmented target", which is the
    # confusion that already cost this project a day.
    if hasattr(dataset, "clean_view"):
        return dataset.clean_view(limit)
    if not isinstance(dataset, ArrayDataset):
        return None
    size = len(dataset)
    picked = np.arange(size) if size <= limit else np.linspace(0, size - 1, limit).astype(int)
    indices = picked if dataset.indices is None else np.asarray(dataset.indices)[picked]
    return ArrayDataset(dataset.tensors, dataset.labels, augment=deterministic, indices=indices)


def train_one_fold(
    make_model,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig,
) -> tuple[nn.Module, float, np.ndarray, float]:
    # Trains the network that make_model() builds (fresh or from given weights) on train_ds,
    # evaluates it on val_ds, and takes every setting from config.
    # Step 1: seed, build the model, and create the training, validation and clean-train loaders.
    seed_everything(config.seed, deterministic=config.deterministic)
    model = make_model().to(config.device)

    # Datasets that resample per epoch share their epoch counter with the loader's workers.
    if hasattr(train_ds, "enable_epoch_sync"):
        train_ds.enable_epoch_sync()
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
        **_loader_kwargs(config),
    )
    val_loader = DataLoader(val_ds, batch_size=config.eval_batch_size, collate_fn=_collate)
    # The deterministic view must be the *validation* one, not "no transform at all":
    # training crops 112 down to 96, so evaluating on an uncropped 112 frame feeds the
    # network a field of view it never saw and understates the fit badly enough to look
    # like underfitting. Borrowing the val split's transform keeps the two comparable.
    clean_ds = _clean_view(
        train_ds,
        config.log_sample,
        deterministic=getattr(val_ds, "augment", None),
    )
    clean_loader = (
        DataLoader(clean_ds, batch_size=config.eval_batch_size, collate_fn=_collate)
        if clean_ds
        else None
    )

    # Step 2: optimiser, one-cycle schedule, loss, mixed precision and the optional EMA/SWA
    # averages.
    groups, max_lrs = _param_groups(model, config)
    optimizer = torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
    accum = max(1, int(config.grad_accum))
    # One schedule step per OPTIMIZER step, not per batch. Leaving this as epochs*len(loader)
    # under accumulation would run the whole OneCycle in 1/accum of the run and then hold the
    # final LR, which is a different experiment than the config asks for.
    steps_per_epoch = max(1, -(-len(loader) // accum))  # That is, ceil(len(loader) / accum).
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=max(1, config.epochs * steps_per_epoch),
        pct_start=config.pct_start,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    # Mixed precision is off by default so every result recorded before 2026-09-02 stays
    # reproducible bit for bit. It is opt-in per run, and `79_determinism_check.py` must be
    # re-run with it on before any deliverable is trained under it -- autocast changes the
    # kernels selected, and the determinism pin was established without it.
    use_amp = bool(config.amp) and str(config.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    averaged = None
    if config.ema_decay > 0.0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))
    swa = None
    swa_from = int(np.ceil(config.swa_start * config.epochs)) if config.swa_start > 0 else None

    # Top-1 accuracy of `net` on a loader, in eval mode; NaN when there is no loader.
    def accuracy_on(source, net) -> float:
        if source is None:
            return float("nan")
        was_training = net.training
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in source:
                logits = net(*(t.to(config.device) for t in inputs))
                correct += int((logits.argmax(1).cpu() == labels).sum())
                total += len(labels)
        net.train(was_training)
        return correct / max(total, 1)

    # Step 3: resume from this run's per-epoch checkpoint if one exists, after checking that it
    # describes the same schedule and the same run.
    start_epoch = 1
    if config.checkpoint_path is not None and config.checkpoint_path.exists():
        saved = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
        # OneCycleLR's total_steps is epochs * len(loader), so a checkpoint taken under one
        # epoch budget describes a different schedule than another budget wants. Resuming
        # across budgets would silently train under a schedule neither config asked for --
        # the same property that made "continue for more epochs" not a continuation at all
        # (experiments #49). Refuse rather than guess.
        if int(saved.get("epochs", config.epochs)) != int(config.epochs):
            raise ValueError(
                f"checkpoint was written for epochs={saved.get('epochs')} but this run asks "
                f"for epochs={config.epochs}; the LR schedule differs, so this is not a "
                f"resume. Delete {config.checkpoint_path} to start over, or match the budget."
            )
        # Same argument for accumulation: it is the other half of total_steps. Absent in
        # checkpoints written before 2026-09-08, which all ran at accum 1.
        if int(saved.get("grad_accum", 1)) != accum:
            raise ValueError(
                f"checkpoint was written for grad_accum={saved.get('grad_accum', 1)} but this "
                f"run asks for {accum}; total_steps is epochs*ceil(len(loader)/grad_accum), "
                f"so the LR schedule differs and this is not a resume."
            )
        # The identity asserts. On a shared remote root these are the only thing standing
        # between "resume" and "silently continue a different experiment": fold 0 and fold 1
        # train on the same clips under the same filenames, so every clip_id check passes
        # while the weights come from the wrong split. Same shape for the recipe hash and
        # the crop table -- a checkpoint written before the windows were rebuilt describes a
        # model trained on different pixels.
        _assert_identity(saved, "recipe", config.recipe_hash, config.checkpoint_path)
        _assert_identity(saved, "windows", config.windows_hash, config.checkpoint_path)
        _assert_identity(saved, "fold", config.fold, config.checkpoint_path)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        scaler.load_state_dict(saved["scaler"])
        if averaged is not None and saved.get("averaged") is not None:
            averaged.load_state_dict(saved["averaged"])
        if swa_from is not None and saved.get("swa") is not None:
            from torch.optim.swa_utils import AveragedModel

            swa = AveragedModel(model)
            swa.load_state_dict(saved["swa"])
        _restore_rng(saved.get("rng"))
        start_epoch = int(saved["epoch"]) + 1
        print(
            f"      resumed from {config.checkpoint_path.name} at epoch {start_epoch}",
            flush=True,
        )

    # Step 4: the epoch loop: training batches, then (each optional) SWA update, log line,
    # snapshot, checkpoint and metrics line.
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        _apply_freeze_bn(model, config.freeze_bn)
        # Datasets that resample frames per access need to know which epoch this is, so the
        # per-sample RNG is derived from (seed, epoch, clip) rather than from a shared
        # stream that DataLoader workers would each fork identically.
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        seen = hit = 0
        # Gradient-norm telemetry. `--clip-grad-norm 1.0` has been on since 2026-09-08 and is
        # load-bearing for every run since, yet nothing recorded whether it ever fires: the
        # return value of clip_grad_norm_ was discarded. MMAction2 fine-tunes this exact CSN
        # checkpoint with max_norm=40, and the comment justifying 1.0 argued that AdamW's
        # per-parameter scaling "makes 40 effectively never fire" -- but clipping happens on
        # the RAW gradient before the optimizer, so whether a threshold fires is a property of
        # the gradients, not of the optimizer. Measure it instead of reasoning about it.
        # `gn_nonfinite` exists because the first version of this telemetry wrote
        # `if observed == observed` with the comment "skip NaN/inf". That excludes NaN and
        # NOT inf, since inf == inf is True, so a single non-finite step poisoned the whole
        # epoch's mean and max to inf -- measured on 4 epochs across two runs -- and hid how
        # many such steps there were. `scaler_skips` counts the steps GradScaler actually
        # dropped. Both are READ-ONLY: the schedule, EMA and accumulation policies are
        # deliberately untouched, because changing them here would turn the clip=1 vs clip=40
        # comparison into a two-variable experiment.
        gn_sum, gn_max, gn_clipped, gn_steps = 0.0, 0.0, 0, 0
        gn_nonfinite = scaler_skips = 0
        # Batch loop: forward (with mixup if enabled) and backward on every batch; clipping,
        # optimiser, scheduler and EMA steps when an accumulation window closes.
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs = tuple(t.to(config.device) for t in inputs)
            labels = labels.to(config.device)
            # With accumulation the gradient buffer is cleared once per window, not per batch,
            # and the window closes either every `accum` batches or at the end of the epoch --
            # a trailing partial window still has to step, or those clips contribute nothing.
            if batch_index % accum == 0:
                optimizer.zero_grad(set_to_none=True)
            closing = (batch_index + 1) % accum == 0 or (batch_index + 1) == len(loader)
            if config.mixup_alpha > 0.0:
                # Mixup: blend the batch with a shuffled copy of itself and interpolate the two
                # cross-entropies. Inputs are mixed as float; the model's internal /255 + norm
                # then applies as usual. One lambda per batch (standard mixup).
                lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                perm = torch.randperm(labels.size(0), device=config.device)
                inputs = tuple(lam * t.float() + (1.0 - lam) * t[perm].float() for t in inputs)
                labels_b = labels[perm]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(*inputs)
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(
                        logits, labels_b
                    )
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(*inputs)
                    loss = criterion(logits, labels)
            # Mean over the whole window, not over each batch, or the effective learning rate
            # would scale with `accum`.
            scaler.scale(loss / accum).backward()
            if not closing:
                hit += int((logits.argmax(1) == labels).sum())
                seen += len(labels)
                continue
            if config.clip_grad_norm > 0:
                # unscale first or the clip threshold is applied to AMP-scaled gradients,
                # which makes it fire essentially never.
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm
                )
                # clip_grad_norm_ returns the norm BEFORE clipping, so this says both how big
                # the gradients actually are and how often the threshold binds.
                observed = float(total_norm)
                if math.isfinite(observed):
                    gn_sum += observed
                    gn_max = max(gn_max, observed)
                    gn_clipped += int(observed > config.clip_grad_norm)
                    gn_steps += 1
                else:
                    gn_nonfinite += 1
            # GradScaler signals a dropped optimizer step by backing its scale off, so
            # comparing the scale across update() counts skips without changing anything.
            # Read-only on purpose: schedule.step() still runs exactly as before.
            scale_before = scaler.get_scale() if use_amp else 0.0
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                scaler_skips += 1
            schedule.step()
            if averaged is not None:
                averaged.update_parameters(model)
            hit += int((logits.argmax(1) == labels).sum())
            seen += len(labels)

        # SWA: add the weights at the end of this epoch to the equal-weight running average.
        if swa_from is not None and epoch > swa_from:
            if swa is None:
                from torch.optim.swa_utils import AveragedModel

                swa = AveragedModel(model)
            swa.update_parameters(model)

        # Log line: accuracy on the augmented batches seen this epoch, on the clean-train
        # subsample and on validation, plus gradient-norm statistics when clipping is on and
        # counts of non-finite or skipped steps when there are any.
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.epochs):
            grad = ""
            if gn_steps:
                grad = (
                    f"  grad-norm mean {gn_sum / gn_steps:.2f} max {gn_max:.2f} "
                    f"clipped {gn_clipped / gn_steps:.0%}"
                )
            # Reported unconditionally when non-zero, and separately from the finite stats:
            # a non-finite gradient and a dropped optimizer step are different events, and
            # the old telemetry could show neither.
            if gn_nonfinite or scaler_skips:
                grad += f"  nonfinite {gn_nonfinite} scaler-skips {scaler_skips}"
            print(
                f"      epoch {epoch:4d}  aug-train {hit / max(seen, 1):.4f}  "
                f"clean-train {accuracy_on(clean_loader, model):.4f}  "
                f"val {accuracy_on(val_loader, model):.4f}{grad}",
                flush=True,
            )

        # The epoch curve. Scored here rather than reconstructed later from snapshots: one
        # val pass costs seconds, where re-scoring ten snapshots per fold afterwards costs
        # tens of minutes of GPU and needs a separate script to exist.
        kept_val = None
        if config.keep_every and (epoch % config.keep_every == 0 or epoch == config.epochs):
            kept_val = round(float(accuracy_on(val_loader, model)), 5)
            if config.checkpoint_path is not None:
                snapshots = config.checkpoint_path.parent / "snapshots"
                snapshots.mkdir(parents=True, exist_ok=True)
                _atomic_save(
                    {k: v.cpu() for k, v in model.state_dict().items()},
                    snapshots / f"{config.checkpoint_path.stem}-e{epoch:03d}.pt",
                )

        if config.checkpoint_path is not None:
            # Written every epoch, and atomically: a crash during the write must not leave a
            # truncated file that then fails to load and loses the whole run anyway.
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save(
                {
                    "epoch": epoch,
                    "epochs": config.epochs,
                    # Part of the schedule's identity, like `epochs`: total_steps is
                    # epochs * ceil(len(loader)/grad_accum), so resuming across accumulation
                    # settings would continue under a schedule neither config asked for.
                    "grad_accum": accum,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "scaler": scaler.state_dict(),
                    "averaged": averaged.state_dict() if averaged is not None else None,
                    "swa": swa.state_dict() if swa is not None else None,
                    # Identity, so a resume cannot land on a different experiment (see the
                    # asserts above). None on both sides means "not declared", which passes.
                    "recipe": config.recipe_hash,
                    "windows": config.windows_hash,
                    "fold": config.fold,
                    # Without this a resume replays the same augmentation stream it already
                    # saw. The per-sample RNG is keyed on (seed, epoch, clip) so the image
                    # line survives without it, but shuffling and dropout do not.
                    "rng": _rng_state(),
                },
                config.checkpoint_path,
            )
            if config.metrics_path is not None:
                _append_metrics(
                    config.metrics_path,
                    {
                        "epoch": epoch,
                        "fold": config.fold,
                        "lr": [g["lr"] for g in optimizer.param_groups],
                        "train_acc": round(hit / max(seen, 1), 5),
                        # None except on kept epochs. This is the epoch curve: read it with
                        # `jq -c 'select(.val_acc)' metrics.jsonl` per fold, pool across
                        # folds, and the best epoch is a measurement rather than a pick.
                        "val_acc": kept_val,
                        "seconds": round(time.time() - epoch_start, 1),
                    },
                )

    # Step 5: with EMA or SWA enabled, re-estimate the average's BatchNorm statistics on the
    # training loader and return the average instead of the last-epoch weights.
    if averaged is not None:
        _update_bn(loader, averaged, config.device)
        model = averaged.module
    if swa is not None:
        # Both numbers from one run: the last-epoch weights and their average. The delta
        # between them is SWA's whole contribution, measured on the same held-out subjects.
        raw_val = accuracy_on(val_loader, model)
        _update_bn(loader, swa, config.device)
        swa_val = accuracy_on(val_loader, swa)
        print(
            f"      SWA over epochs {swa_from + 1}-{config.epochs}: "
            f"last-epoch val {raw_val:.4f} -> averaged val {swa_val:.4f} "
            f"({(swa_val - raw_val) * 100:+.2f} pp)",
            flush=True,
        )
        model = swa.module

    # Step 6: final evaluation in eval mode: softmax probabilities on the validation split and
    # accuracy on the whole training split (its un-augmented view where the dataset has one).
    model.eval()

    def evaluate(source):
        logits, targets = [], []
        with torch.no_grad():
            for inputs, labels in source:
                inputs = tuple(t.to(config.device) for t in inputs)
                logits.append(model(*inputs).cpu())
                targets.append(labels)
        return torch.cat(logits).softmax(dim=1).numpy(), torch.cat(targets).numpy()

    probs, truth = evaluate(val_loader)
    # Training accuracy, measured in eval mode on the *un-augmented* training split. It is
    # the one number that separates the two failure modes a low validation score can mean:
    # near 1.0 says the model memorised and needs regularisation, while a low value says it
    # never fitted at all and needs capacity or a better recipe. Diagnosing that by guessing
    # is how a week gets spent tuning the wrong end.
    # The same clean view the epoch log uses: the training split only (dropping `indices`
    # here silently scored train and validation together) and the validation transform.
    fit_source = _clean_view(train_ds, len(train_ds), getattr(val_ds, "augment", None))
    fit_probs, fit_truth = evaluate(
        DataLoader(fit_source or train_ds, batch_size=config.eval_batch_size, collate_fn=_collate)
    )
    train_accuracy = float((fit_probs.argmax(axis=1) == fit_truth).mean())
    # Returns the model, the validation accuracy, the validation probabilities (N_val, n_classes)
    # and the training accuracy.
    return model, float((probs.argmax(axis=1) == truth).mean()), probs, train_accuracy


def measure_ms_per_clip(model: nn.Module, sample_inputs, device: str, repeats: int = 50) -> float:
    """Latency at batch size 1 — how the model would actually be deployed on an edge device."""
    model.eval()
    single = tuple(t[:1].to(device) for t in sample_inputs)
    with torch.no_grad():
        for _ in range(10):  # warm up; the first CUDA call includes context setup
            model(*single)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(*single)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / repeats * 1000


def measure_megabytes(model: nn.Module, tmp_dir: Path) -> float:
    """Actual on-disk size of the weights, measured at the precision they ship in.

    Video 3D backbones ship in fp16 (organizer C-4 allows and encourages it, and every
    image deliverable in this repo is saved half=True), so a fp32 probe over-reports by 2x
    and, for swin3d_t (fp32 126 MB), trips the 100 MB gate on what is only a size *report*.
    A report must never abort a cross-validation run and take the OOF with it -- the same
    reasoning run_cv already applies to the folds/ directory -- so measure at fp16 and, if a
    genuinely oversized model still exceeds the cap, record the number instead of raising.
    """
    try:
        save_deliverable({"model": model}, tmp_dir, half=True)
    except BudgetExceeded:
        pass  # files are written before the gate raises; measure them below
    return deliverable_bytes(tmp_dir) / MB


def archive_existing_run(run_dir: Path) -> list[Path]:
    """Move a previous run's measurement record aside instead of overwriting it.

    Run directories are named from the configuration, so re-running the same config lands
    on the same directory and used to overwrite ``oof.npz`` in place. That is how a
    measurement stops existing: the out-of-fold probabilities of the earlier run are the
    only record of what that model thought, and once they are gone the comparison between
    the two runs can no longer be made locally at all -- it becomes a question that only an
    upload can answer, which is the most expensive way to learn anything here.

    This is not hypothetical. Re-running five members to recover their fold weights would
    have destroyed the pre-pin out-of-fold probabilities, and those turned out to carry the
    measurement that identified the training as nondeterministic in the first place. They
    survived because someone remembered to copy them by hand, which is not a mechanism.

    Only the measurement record is archived. Fold weights are left alone: training is
    deterministic now (see :func:`seed_everything`), so the same config and seed rebuilds
    them exactly, while the *older* probabilities may have come from code that no longer
    exists.

    Raises rather than continuing if the move fails -- a failed archive followed by a
    successful overwrite is the exact outcome this exists to prevent.
    """
    if not run_dir.is_dir():
        return []
    archived: list[Path] = []
    for name in ("oof.npz", "result.json"):
        source = run_dir / name
        if not source.exists():
            continue
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(source.stat().st_mtime))
        target = run_dir / f"{source.stem}-{stamp}{source.suffix}"
        attempt = 2
        while target.exists():
            target = run_dir / f"{source.stem}-{stamp}-{attempt}{source.suffix}"
            attempt += 1
        source.rename(target)
        archived.append(target)
    return archived


def restore_one_fold(make_model, val_ds: Dataset, config: TrainConfig, state_path: Path):
    """Rebuild a finished fold's validation probabilities from its saved weights.

    :func:`cross_validate` writes every fold's weights the moment that fold finishes, but
    the out-of-fold probability matrix only reaches disk after the *last* fold returns. An
    interrupted run therefore keeps all of its training and loses the one artifact that the
    weak-class gate and the fusion weight search read, so the cheapest recovery used to be
    retraining folds that had already been trained.

    Inference over one fold's validation split costs seconds against the ~24 minutes the
    fold cost to train. Accuracy is recomputed here rather than read back from ``meta.json``
    because ``save_deliverable`` rewrites that file on every call into the same directory,
    so only the last fold's metadata survives.
    """
    seed_everything(config.seed)
    model = make_model().to(config.device)
    model.load_state_dict(torch.load(state_path, map_location=config.device))
    model.eval()

    logits, targets = [], []
    with torch.no_grad():
        loader = DataLoader(val_ds, batch_size=config.eval_batch_size, collate_fn=_collate)
        for inputs, labels in loader:
            inputs = tuple(t.to(config.device) for t in inputs)
            logits.append(model(*inputs).cpu())
            targets.append(labels)
    probs = torch.cat(logits).softmax(dim=1).numpy()
    truth = torch.cat(targets).numpy()
    return model, float((probs.argmax(axis=1) == truth).mean()), probs


def cross_validate(
    name: str,
    make_model,
    build_dataset,
    labels: np.ndarray,
    folds: np.ndarray,
    config: TrainConfig,
    work_dir: Path,
    resume: bool = False,
) -> RunResult:
    """Run every fold, then report accuracy, size and latency as one result.

    Out-of-fold probabilities are kept for every clip. They are what makes take-level
    decoding measurable (ADR 0007) without retraining, and they are also the input to any
    later fusion — a modality's contribution is judged on its probabilities, not its score.
    """
    results, model, sample_inputs = [], None, None
    oof = np.zeros((len(labels), 40), dtype=np.float32)  # 40 class probabilities per clip.

    # Step 1: for each fold, validate on its clips and train on all other clips; each trained
    # model is saved as <work_dir>/folds/foldN.pt, N being the fold it held out.
    # build_dataset(mask, training) builds the dataset of the clips selected by mask.
    for fold in sorted(np.unique(folds)):
        is_val = folds == fold
        val_ds = build_dataset(is_val, False)
        checkpoint = work_dir / "folds" / f"fold{int(fold)}.pt"

        if resume and checkpoint.exists():
            # `archive_existing_run` deliberately leaves fold weights in place, so a resumed
            # run finds exactly the folds the interrupted one had finished.
            model, accuracy, probs = restore_one_fold(make_model, val_ds, config, checkpoint)
            fit = float("nan")
            print(
                f"  fold {fold}: acc {accuracy:.4f}  restored from {checkpoint.name}, "
                f"not retrained  ({int(is_val.sum())} val clips)"
            )
        else:
            train_ds = build_dataset(~is_val, True)
            # Each fold needs its own resume checkpoint and fold identity. Sharing one
            # config.checkpoint_path across folds makes fold 1 resume fold 0's epoch.pt --
            # and the fold-identity assert cannot catch it, because config.fold was never
            # advanced. Keep the per-fold checkpoint out of folds/ so it does not inflate the
            # deliverable-size measurement of the fold weights.
            fold_config = replace(
                config,
                fold=int(fold),
                checkpoint_path=(
                    work_dir / "_ckpt" / f"fold{int(fold)}.pt"
                    if config.checkpoint_path is not None
                    else None
                ),
            )
            model, accuracy, probs, fit = train_one_fold(make_model, train_ds, val_ds, fold_config)
            # Keep every fold's weights. They are trained already, each has seen a different
            # subset of subjects, and averaging them is an ensemble that costs no training at
            # all — but only if they were not discarded, which is what used to happen here.
            #
            # save_deliverable measures the whole directory, so by the third fold of a large
            # backbone the accumulated total crosses the 100 MB cap and it raises. That cap
            # is right for a deliverable and wrong here: these are N alternative models over
            # different subject subsets, and a cross-validation run's fold weights are a
            # measurement record. Letting the gate abort the run would destroy the OOF
            # probabilities -- the one artifact the whole run exists to produce -- over a
            # limit that does not apply to them. The gate still bites where it counts, on
            # the actual deliverable written by 63_predict_fused.
            #
            # The overflow is still real information and must not be swallowed silently:
            # it says this fold ensemble cannot ship as one, which was never true of the
            # skeleton line (20 fold weights totalled 20.46 MB) and is true at 44.9 MB each.
            try:
                save_deliverable(
                    {f"fold{int(fold)}": model},
                    work_dir / "folds",
                    meta={"fold": int(fold), "accuracy": accuracy, "config": asdict(config)},
                )
            except BudgetExceeded as exceeded:
                # Deliberately not re-measuring the directory here: the exception already
                # carries the size, and a second measurement inside the handler could raise
                # and replace this message with an unrelated one.
                print(
                    f"  NOTE fold {int(fold)} weights kept, but the folds/ directory is over "
                    f"the deliverable cap: {exceeded}\n"
                    f"       a cross-validation folds/ dir is a measurement record, not a "
                    f"deliverable, so the run continues -- but these {len(results) + 1} "
                    f"folds cannot ship as one ensemble, and the deliverable must come "
                    f"from the full-data models instead",
                    flush=True,
                )
            print(
                f"  fold {fold}: acc {accuracy:.4f}  train {fit:.4f}  "
                f"gap {fit - accuracy:+.4f}  ({int(is_val.sum())} val clips)"
            )

        # Record the fold's out-of-fold probabilities and held-out subjects (one subject per clip
        # in config.extra["subjects"]); the first fold also supplies one clip for timing.
        oof[is_val] = probs
        subjects = sorted(set(np.asarray(config.extra.get("subjects", []))[is_val].tolist()))
        results.append(FoldResult(int(fold), accuracy, int(is_val.sum()), subjects))
        if sample_inputs is None:
            sample_inputs, _ = _collate([val_ds[0]])

    # Step 2: parameter count, size and latency are measured on the last fold's model.
    return RunResult(
        name=name,
        folds=results,
        params=count_parameters(model),
        megabytes=measure_megabytes(model, work_dir / "size_probe"),
        ms_per_clip=measure_ms_per_clip(model, sample_inputs, config.device),
        config=asdict(config),
        oof_probs=oof,
    )
