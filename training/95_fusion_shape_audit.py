"""Audit how the *shape* of the fused probability affects decoding, and re-derive the
numbers that `docs/plans/2026-09-06-endgame-plan-to-0.85.md` §3.4 depends on.

    uv run python scripts/95_fusion_shape_audit.py                  # every section
    uv run python scripts/95_fusion_shape_audit.py --sections A C   # a subset

CPU only. It loads stored probabilities and never touches a GPU or a model, so it is safe
to run while something else has the card.

Why this exists. The fusion in `61_submit_from_probs.py` is a weighted **arithmetic** mean of
softmax probabilities, and `takes.decode_take` maximises ``sum log p + w * sum log trans``.
Dividing ``log p`` by a temperature is algebraically the same as multiplying ``w`` — proved in
`reports/experiments.md` — so anything that flattens the fused distribution raises the
*effective* transition weight. Two consequences are easy to state and were both wrong in the
first draft of that plan, which is why they are measured here rather than argued:

* the image member the OOF sees is **single fold, single pass**, while the submission uses a
  **4-fold, 4-pass** average that is both stronger and flatter. Fusion weights picked on the
  former do not transfer to the latter;
* full takes and takes thinned to the test size distribution (mean 2.41 clips) disagree about
  the sign of several decisions. The thinned condition is the one the leaderboard scores.

Section A reports both take conditions side by side for exactly that reason. Sections are
independent; each prints what it measured and the paired standard error over thinning repeats,
because a difference of 0.5 pp between two decode settings is inside the repeat-to-repeat
spread unless the repeats are paired on the same subsample.

⚠️ 2026-09-06: sections A-E quote the wrong replication unit, and it cost five submissions.
``paired()`` takes its SE over thinning repeats with the model and the *subjects* held fixed,
so "+2.11 pp, SE 0.22, t=9.6" in section C said only that the thinning is reproducible -- it
never spoke to whether another draw of people would flip the sign, which is the only thing
every decision made with it depended on. The leaderboard then returned -1.99 pp on that exact
comparison. **Section F re-prices both instruments by resampling subjects** (cluster bootstrap
plus jackknife). A-E keep the old statistic so their published anchors still reproduce; read F
for any decision. When a gate compares two separately trained runs, even F is a floor, because
it holds the training run fixed and so excludes seed and optimisation noise.
"""
# Role: CPU audit, in sections A-F, of how the shape of fused probabilities (single- or four-pass
# image member, arithmetic or geometric fusion, transition weight) changes take-decoded accuracy
# on complete and on test-like thinned training takes. Also provides subsample_contiguous.
# Used by: training/44_self_training.py and 48_evaluate_matched.py import subsample_contiguous,
# and training/next_common.py compiles that function from this file's source with ast; training
# only, not used by the delivered run.
# The sections analyse a blend of one image model and three skeleton models, from stored
# probabilities (not distributed); the delivered system uses neither this blend nor any skeleton
# model. The usage lines above name scripts/; the file is in training/.

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# Make the cuhkx package (src/) importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuhkx.paths import p  # noqa: E402
from cuhkx.takes import decode_all, transition_matrix  # noqa: E402

# training/ itself, for the digit-named sibling module. Its `subsample` is the random thinner: it
# keeps randomly chosen positions of each take, in recording order.
sys.path.insert(0, str(Path(__file__).resolve().parent))
subsample = import_module("52_simulate_incomplete_takes").subsample

# Stored inputs under models_root: out-of-fold probabilities of the image model (the R(2+1)D-34
# fold models on the det view) and of three skeleton models, fold-0 image probabilities under
# several test-time views, and the blend's test-set probabilities (section E).
IMAGE_RUN = "ig65m-t32-det-60ep"
VIEWS_NPZ = "ig65m-t32-mv/fold0_views.npz"
# the 4-pass TTA actually used at inference (90_infer_tta.py): plain, hflip, roll+1, roll-1
BASELINE4 = ("view:c:0.5", "view:c:0.5:flip", "view:c:0.5:roll+1", "view:c:0.5:roll-1")
SKELETON = (
    "skeleton-T16-e120-s42",
    "skeleton-T16-e120-yaw30-bone-s42",
    "skeleton-T32-e120-yaw30-bone-s42",
)
TEST_RUN = "ig65mt32v1"
TEST_MEMBERS = (
    "skeleton-T16-e120",
    "skeleton-T16-e120-yaw30-bone",
    "skeleton-T32-e120-yaw30-bone",
    "image_tta",
)
# Transition weights swept by the sections; 0.75 is the weight the take decoder uses.
TWS = (0.0, 0.25, 0.5, 0.75, 1.0)
SHIPPED_TW = 0.75
# Image member weight in the blend, against 1 for each skeleton member (an image share of 5/8).
SHIPPED_W_IMAGE = 5.0


# (clip id -> (40,) out-of-fold probabilities, clip id -> fold) from <models_root>/<name>/oof.npz.
def load_oof(name: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    blob = np.load(p("models_root") / name / "oof.npz", allow_pickle=False)
    clip_ids = [str(c) for c in blob["clip_ids"]]
    return (
        dict(zip(clip_ids, blob["probs"], strict=True)),
        dict(zip(clip_ids, (int(f) for f in blob["folds"]), strict=True)),
    )


def fuse(
    image: dict[str, np.ndarray],
    skeleton: list[dict[str, np.ndarray]],
    clips: list[str],
    w_image: float | None,
    geometric: bool = False,
) -> dict[str, np.ndarray]:
    """``w_image=None`` means image only. Weights are normalised exactly as 61_submit does."""
    out: dict[str, np.ndarray] = {}
    for clip in clips:
        if w_image is None:
            out[clip] = np.asarray(image[clip], dtype=np.float64)
            continue
        # Member weights [w_image, 1, 1, ...] normalised to sum 1. Arithmetic fusion is the
        # weighted mean of the probabilities; geometric fusion the weighted mean of the
        # log-probabilities (clipped at 1e-12), exponentiated and renormalised.
        members = [np.asarray(image[clip], dtype=np.float64)]
        members += [np.asarray(s[clip], dtype=np.float64) for s in skeleton]
        weights = np.array([w_image, *([1.0] * len(skeleton))])
        weights = weights / weights.sum()
        if geometric:
            logp = sum(
                w * np.log(np.clip(m, 1e-12, None)) for w, m in zip(weights, members, strict=True)
            )
            # Subtracting the maximum avoids underflow in exp; renormalising cancels the shift.
            shifted = np.exp(logp - logp.max())
            out[clip] = shifted / shifted.sum()
        else:
            out[clip] = np.tensordot(weights, np.stack(members), axes=(0, 0))
    return out


# Accuracy over every clip of `takes` after decoding them with transition weight `tw`.
def decoded_accuracy(takes, fused, transitions, truth, tw: float) -> float:
    decoded = decode_all(takes, fused, transitions, tw)
    covered = [c for members in takes.values() for c in members]
    return float(np.mean([decoded[c] == truth[c] for c in covered]))


def thinned_curve(takes, fused, transitions, truth, target, repeats: int) -> np.ndarray:
    """(len(TWS), repeats) accuracies, paired on the same subsample across transition weights."""
    per = np.zeros((len(TWS), repeats))
    # Thinning j uses a generator seeded j: calls on the same takes get the same subsamples, and
    # all tw values of one call are scored on them.
    for j in range(repeats):
        thin = THIN(takes, target, np.random.default_rng(j))
        for i, tw in enumerate(TWS):
            per[i, j] = decoded_accuracy(thin, fused, transitions, truth, tw)
    return per


def paired(a: np.ndarray, b: np.ndarray) -> str:
    """SE over thinning repeats -- the WRONG replication unit; see section F.

    The model and the subjects are held fixed across the repeats, so this quantifies how
    reproducible the thinning is, not whether a different draw of people would flip the
    sign. It is a lower bound on the uncertainty that matters and can understate it by an
    order of magnitude. Kept as-is because sections A-E and their published anchors were
    computed with it; section F reports the subject-clustered version alongside.
    """
    d = a - b
    return f"{d.mean() * 100:+.2f} pp  SE {d.std(ddof=1) / np.sqrt(len(d)) * 100:.2f}"


# training/next_common.py extracts this function by name from the source with ast and runs it
# with only `np` in scope, so it has to stay self-contained.
def subsample_contiguous(takes, target, rng):
    """Thin each take to a *contiguous* run of the drawn size, keeping the order.

    The default thinner (``52_simulate_incomplete_takes.subsample``) draws random positions,
    so a thinned take can pair position 0 with position 4 and the bigram prior is then asked
    for a transition it was never estimated on. Measured against the real test takes, that is
    unfaithful: consecutive clips inside a real test take sit a median 3.4 s apart -- the same
    spacing as *adjacent* training clips (3.7 s) -- while random thinning stretches the median
    to 4.5 s and the p90 from 7.5 s to 12.1 s. Contiguous thinning reproduces the real spacing
    (median 3.8 s, p90 8.4 s), so it is the faithful simulation of how the test set is shaped.
    Both are kept because every published number so far used the random one.
    """
    out = {}
    for take_id, members in takes.items():
        # Two draws per take: the run length (a random entry of `target`, at most the take's
        # length), then the start, uniform over the len(members) - keep + 1 possible positions.
        keep = min(len(members), int(rng.choice(target)))
        if keep <= 0:
            continue
        start = int(rng.integers(0, len(members) - keep + 1))
        out[take_id] = members[start : start + keep]
    return out


# --thinning picks one of these; the sections call whatever the module-level THIN holds.
THINNERS = {"random": subsample, "contiguous": subsample_contiguous}
THIN = subsample  # rebound by main() from --thinning


def subject_of(clip_id: str) -> str:
    """``0_Wash_face/user16/1-1-1`` -> ``user16``."""
    return clip_id.split("/")[1]


def subject_counts(val, fused, transitions, truth, target, repeats: int):
    """Per-subject hit counts at every tw, with the thinning noise averaged *inside*.

    Returns ``(subjects, correct[n_subj, n_tw], total[n_subj])`` where the counts are
    summed over the thinning repeats, so the only variation left between rows is the one
    the leaderboard actually resamples: which people you happened to get.
    """
    correct: dict[str, np.ndarray] = {}
    total: dict[str, float] = {}
    for j in range(repeats):
        thin = THIN(val, target, np.random.default_rng(j))
        covered = [c for members in thin.values() for c in members]
        subs = np.array([subject_of(c) for c in covered])
        # hits[i, k] is 1 when covered clip i is decoded correctly at transition weight TWS[k].
        hits = np.zeros((len(covered), len(TWS)))
        for i, tw in enumerate(TWS):
            decoded = decode_all(thin, fused, transitions, tw)
            hits[:, i] = [decoded[c] == truth[c] for c in covered]
        for s in np.unique(subs):
            mask = subs == s
            correct[s] = correct.get(s, np.zeros(len(TWS))) + hits[mask].sum(axis=0)
            total[s] = total.get(s, 0.0) + float(mask.sum())
    subjects = sorted(correct)
    return (
        subjects,
        np.stack([correct[s] for s in subjects]),
        np.array([total[s] for s in subjects]),
    )


def cluster_se(num: np.ndarray, total: np.ndarray, reps: int = 4000, seed: int = 0):
    """Point estimate and subject-clustered SE of the clip-weighted ratio ``num/total``.

    ``num`` is a per-subject numerator (hits, or a paired hit difference). Subjects are
    resampled with replacement, which is the cluster bootstrap; the jackknife SE is
    returned too because with five subjects the bootstrap grid is coarse.
    """
    n = len(total)
    point = num.sum() / total.sum()
    # Bootstrap: `reps` resamples of the n subjects with replacement; the 90% interval is their
    # 5th to 95th percentile. Jackknife: leave out one subject at a time.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot = num[idx].sum(axis=1) / total[idx].sum(axis=1)
    keep = ~np.isnan(boot)
    lo, hi = np.percentile(boot[keep], [5, 95])
    jack = np.array(
        [(num.sum() - num[s]) / (total.sum() - total[s]) for s in range(n)],
    )
    se_jack = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return point, float(boot[keep].std(ddof=1)), float(se_jack), float(lo), float(hi)


# The shared inputs, as a tuple in this order: clip -> true label, single-pass image probabilities,
# four-pass image probabilities (fold 0 only), the skeleton models' probabilities, clip -> fold,
# training takes, take -> fold (None if its first clip is not in the image model's out-of-fold
# file), and the clip counts of the test takes.
def build_context():
    folds_table = pd.read_parquet(p("processed_root") / "folds.parquet")
    truth = dict(zip(folds_table["clip_id"], folds_table["action_id"], strict=True))
    image_sp, fold_of = load_oof(IMAGE_RUN)
    skeleton = [load_oof(name)[0] for name in SKELETON]

    views = np.load(p("models_root") / VIEWS_NPZ, allow_pickle=True)
    view_ids = [str(c) for c in views["clip_ids"]]
    # Four-pass image member: the mean of the plain, flipped and time-rolled (+1, -1) views.
    image_4p = dict(zip(view_ids, np.mean([views[v] for v in BASELINE4], axis=0), strict=True))

    table = pd.read_parquet(p("processed_root") / "takes_train.parquet")
    takes = {
        take_id: group.sort_values("position")["clip_id"].tolist()
        for take_id, group in table.groupby("take_id")
    }
    take_fold = {t: fold_of.get(m[0]) for t, m in takes.items()}
    target = pd.read_parquet(p("processed_root") / "takes_test.parquet")
    target = target.groupby("take_id").size().to_numpy()
    return truth, image_sp, image_4p, skeleton, fold_of, takes, take_fold, target


# val: the takes of `fold`; others: the takes of the other folds (for the transition prior),
# leaving out takes whose fold is unknown.
def fold_slice(takes, take_fold, fold: int):
    val = {t: m for t, m in takes.items() if take_fold.get(t) == fold}
    others = {t: m for t, m in takes.items() if take_fold.get(t) not in (fold, None)}
    return val, others


# Section A (fold 0): single- or four-pass image member, alone or blended with the skeleton
# members; decoded accuracy on complete and on thinned takes, at tw 0.75 and at the best tw.
def section_a(ctx, repeats: int) -> None:
    truth, image_sp, image_4p, skeleton, fold_of, takes, take_fold, target = ctx
    fold0 = sorted(c for c, f in fold_of.items() if f == 0)
    val, others = fold_slice(takes, take_fold, 0)
    transitions = transition_matrix(others, truth)

    print("\n=== A. image-only vs blend, single-pass vs 4-pass image (fold 0) ===")
    print("anchors that must match the roadmap (0.7289 / 0.7434 / 0.7987):")
    for tag, probs in (("single-pass", image_sp), ("4-pass", image_4p)):
        acc = np.mean([np.asarray(probs[c]).argmax() == truth[c] for c in fold0])
        print(f"  image {tag:12s} clip-level argmax {acc:.4f}")

    for tag, image in (("single-pass", image_sp), ("4-pass", image_4p)):
        for label, w in (("image-only", None), (f"blend w={SHIPPED_W_IMAGE:g}", SHIPPED_W_IMAGE)):
            fused = fuse(image, skeleton, fold0, w)
            full = {tw: decoded_accuracy(val, fused, transitions, truth, tw) for tw in TWS}
            thin = thinned_curve(val, fused, transitions, truth, target, repeats).mean(axis=1)
            best_full = max(full, key=full.get)
            best_thin = TWS[int(thin.argmax())]
            print(
                f"  {tag:12s} {label:14s} full take {full[SHIPPED_TW]:.4f} "
                f"(best tw {best_full} -> {full[best_full]:.4f})   "
                f"thinned {thin[TWS.index(SHIPPED_TW)]:.4f} "
                f"(best tw {best_thin} -> {thin.max():.4f})"
            )
    print("  full takes and thinned takes disagree on how much the skeleton members add;")
    print("  the real test takes average 2.41 clips, so the thinned column is the comparable one.")


# Section B (fold 0, four-pass image, thinned takes): image shares 0.625, 0.75, 0.85 and 1.0 of
# the blend (w_image 5, 9, 17 against three skeleton members at 1, or image only) at every tw.
def section_b(ctx, repeats: int) -> None:
    truth, _, image_4p, skeleton, fold_of, takes, take_fold, target = ctx
    fold0 = sorted(c for c, f in fold_of.items() if f == 0)
    val, others = fold_slice(takes, take_fold, 0)
    transitions = transition_matrix(others, truth)

    print("\n=== B. image-share ladder, 4-pass image, thinned takes (fold 0) ===")
    print(f"{'share':>7}  {'w_image':>8}" + "".join(f"   tw={t:<4}" for t in TWS))
    curves = {}
    for share, w in ((0.625, 5.0), (0.75, 9.0), (0.85, 17.0), (1.0, None)):
        fused = fuse(image_4p, skeleton, fold0, w)
        per = thinned_curve(val, fused, transitions, truth, target, repeats)
        curves[share] = per
        label = "-" if w is None else f"{w:g}"
        print(f"{share:>7}  {label:>8}" + "".join(f"  {m:.4f}" for m in per.mean(axis=1)))
    for share, per in curves.items():
        best = TWS[int(per.mean(axis=1).argmax())]
        print(f"  share {share}: best tw {best} -> {per.mean(axis=1).max():.4f}")
    base = curves[0.625]
    b_i, o_i = int(base.mean(axis=1).argmax()), int(curves[1.0].mean(axis=1).argmax())
    print(
        f"  blend at its best tw minus image-only at its best tw: "
        f"{paired(base[b_i], curves[1.0][o_i])}"
    )


# Section C (thinned takes): accuracy at every tw for the single-pass blend on each fold and
# pooled over the four folds, then for the four-pass blend on fold 0; tw 0.25 against 0.75.
def section_c(ctx, repeats: int) -> None:
    truth, image_sp, image_4p, skeleton, fold_of, takes, take_fold, target = ctx

    print("\n=== C. transition weight: single-pass across all four folds vs 4-pass on fold 0 ===")
    print(f"{'fold':>6}" + "".join(f"   tw={t:<4}" for t in TWS))
    per_fold = {}
    clips_by_fold = {}
    for fold in range(4):
        clips_by_fold[fold] = sorted(c for c, f in fold_of.items() if f == fold)
        val, others = fold_slice(takes, take_fold, fold)
        transitions = transition_matrix(others, truth)
        fused = fuse(image_sp, skeleton, clips_by_fold[fold], SHIPPED_W_IMAGE)
        per_fold[fold] = thinned_curve(val, fused, transitions, truth, target, repeats)
        print(f"{fold:>6}" + "".join(f"  {m:.4f}" for m in per_fold[fold].mean(axis=1)))
    pooled = np.concatenate(list(per_fold.values()), axis=1)
    print(f"{'pooled':>6}" + "".join(f"  {m:.4f}" for m in pooled.mean(axis=1)))
    print(
        f"  single-pass, pooled, tw=0.25 minus tw={SHIPPED_TW}: "
        f"{paired(pooled[TWS.index(0.25)], pooled[TWS.index(SHIPPED_TW)])}"
    )

    val, others = fold_slice(takes, take_fold, 0)
    transitions = transition_matrix(others, truth)
    fused = fuse(image_4p, skeleton, clips_by_fold[0], SHIPPED_W_IMAGE)
    four = thinned_curve(val, fused, transitions, truth, target, repeats)
    print(f"{'0 (4p)':>6}" + "".join(f"  {m:.4f}" for m in four.mean(axis=1)))
    print(
        f"  4-pass, fold 0, tw=0.25 minus tw={SHIPPED_TW}: "
        f"{paired(four[TWS.index(0.25)], four[TWS.index(SHIPPED_TW)])}"
    )
    print("  the shipped tw was chosen on single-pass probabilities; the submission uses a")
    print("  4-fold 4-pass average, which is flatter, and a flatter p wants a lower nominal w.")
    print("  4-pass exists for fold 0 only, so this is five subjects -- not a decision on its own.")


# Section D (fold 0, four-pass image): arithmetic against geometric fusion, each at its best tw.
def section_d(ctx, repeats: int) -> None:
    truth, _, image_4p, skeleton, fold_of, takes, take_fold, target = ctx
    fold0 = sorted(c for c, f in fold_of.items() if f == 0)
    val, others = fold_slice(takes, take_fold, 0)
    transitions = transition_matrix(others, truth)

    print("\n=== D. arithmetic vs geometric (product-of-experts) fusion, 4-pass, fold 0 ===")
    best = {}
    for geometric in (False, True):
        fused = fuse(image_4p, skeleton, fold0, SHIPPED_W_IMAGE, geometric=geometric)
        full = {tw: decoded_accuracy(val, fused, transitions, truth, tw) for tw in TWS}
        per = thinned_curve(val, fused, transitions, truth, target, repeats)
        best[geometric] = per[int(per.mean(axis=1).argmax())]
        name = "geometric" if geometric else "arithmetic"
        print(
            f"  {name:11s} full take best {max(full.values()):.4f}   "
            f"thinned best {per.mean(axis=1).max():.4f}"
        )
    print(
        f"  geometric minus arithmetic, thinned, each at its own best tw: "
        f"{paired(best[True], best[False])}"
    )


# Section E (test set, no labels): how many decoded test labels change when tw or the member set
# changes, against the blend decoded at tw 0.75 with a prior counted on all training takes.
def section_e(ctx) -> None:
    truth, _, _, _, _, takes, _, _ = ctx
    print("\n=== E. how many test predictions actually differ (is a submission resolvable?) ===")
    blob = np.load(p("models_root") / TEST_RUN / "test_probs.npz", allow_pickle=False)
    clip_ids = [str(c) for c in blob["clip_ids"]]
    members = {k[len("member:") :]: blob[k] for k in blob.files if k.startswith("member:")}
    # Weights in TEST_MEMBERS order: the three skeleton members at 1 and the image member at 5,
    # normalised; the blend is the weighted mean of the members' probabilities.
    weights = np.array([1.0, 1.0, 1.0, SHIPPED_W_IMAGE])
    weights = weights / weights.sum()
    stacked = np.stack([members[m] for m in TEST_MEMBERS])
    blend = dict(zip(clip_ids, np.tensordot(weights, stacked, axes=(0, 0)), strict=True))
    image_only = dict(zip(clip_ids, members["image_tta"], strict=True))

    table = pd.read_parquet(p("processed_root") / "takes_test.parquet")
    test_takes = {
        take_id: group.sort_values("position")["clip_id"].tolist()
        for take_id, group in table.groupby("take_id")
    }
    transitions = transition_matrix(takes, truth)
    reference = decode_all(test_takes, blend, transitions, SHIPPED_TW)

    # D counts the test clips whose decoded label differs from the reference; `scored` rescales D
    # from the 405 clips to the 201 that this script takes to be scored.
    def report(label: str, decoded: dict[str, int]) -> None:
        d = sum(1 for c in clip_ids if reference[c] != decoded[c])
        scored = d * 201 / len(clip_ids)
        print(f"  {label:34s} D={d:3d}/405  ~{scored:4.0f} scored clips")

    for tw in TWS:
        if tw != SHIPPED_TW:
            report(f"blend tw={tw} vs shipped", decode_all(test_takes, blend, transitions, tw))
    for tw in (0.5, SHIPPED_TW):
        report(
            f"image-only tw={tw} vs shipped",
            decode_all(test_takes, image_only, transitions, tw),
        )
    print("  resolvable needs D >= (1.96/(2p-1))^2 on the *scored* clips, not D >= 21:")
    print("  a +2 pp effect (~4 scored clips) inside D=19 needs D >= 87. None of these qualify,")
    print("  so decide these zero-cost candidates on the thinned estimator and ship them.")


def section_f(ctx, repeats: int) -> None:
    """The same two instruments, re-priced against the unit that actually varies.

    Sections A-E report an SE over thinning repeats with the subjects held fixed. That
    number answers "would I get this again with another subsample", but every decision it
    is used for asks "would I get this again with other people" -- and the leaderboard
    draws four people we have never seen. So resample subjects, not takes.
    """
    truth, image_sp, image_4p, skeleton, fold_of, takes, take_fold, target = ctx
    i75 = TWS.index(SHIPPED_TW)

    print("\n=== F. subject-clustered uncertainty (bootstrap + jackknife over people) ===")

    # Per-subject hit counts on fold 0 with the four-pass image member (see subject_counts).
    def counts_fold0(w, geometric=False):
        fold0 = sorted(c for c, f in fold_of.items() if f == 0)
        val, others = fold_slice(takes, take_fold, 0)
        fused = fuse(image_4p, skeleton, fold0, w, geometric=geometric)
        return subject_counts(val, fused, transition_matrix(others, truth), truth, target, repeats)

    # The same, pooled over the four folds, with the single-pass image member.
    def counts_pooled(w):
        subjects, correct, total = [], [], []
        for fold in range(4):
            clips = sorted(c for c, f in fold_of.items() if f == fold)
            val, others = fold_slice(takes, take_fold, fold)
            fused = fuse(image_sp, skeleton, clips, w)
            subs, corr, tot = subject_counts(
                val, fused, transition_matrix(others, truth), truth, target, repeats
            )
            subjects += subs
            correct.append(corr)
            total.append(tot)
        return subjects, np.concatenate(correct), np.concatenate(total)

    # instrument A: fold-0 only, 4-pass image -- the shape that ships, five subjects
    # instrument B: all four folds, single pass -- eighteen subjects, flatter-than-shipped shape
    rows = {
        "A fold-0 4-pass": counts_fold0(SHIPPED_W_IMAGE),
        "B 4-fold single-pass": counts_pooled(SHIPPED_W_IMAGE),
    }
    extra = {
        "A image-only": counts_fold0(None),
        "B image-only": counts_pooled(None),
        "A geometric": counts_fold0(SHIPPED_W_IMAGE, geometric=True),
    }

    print("\n(i) decoded accuracy at the shipped tw -- the scale of subject idiosyncrasy")
    for label, (subjects, correct, total) in rows.items():
        acc = correct[:, i75] / total
        point, se_boot, se_jack, lo, hi = cluster_se(correct[:, i75], total)
        print(
            f"  {label:22s} n={len(subjects):2d}  {point:.4f}  "
            f"SE_boot {se_boot * 100:.2f} pp  SE_jack {se_jack * 100:.2f} pp  "
            f"90% [{lo:.4f}, {hi:.4f}]"
        )
        print(
            f"  {'':22s} per-subject spread {acc.min():.3f}..{acc.max():.3f} "
            f"(SD {acc.std(ddof=1) * 100:.1f} pp across people)"
        )

    print(f"\n(ii) every tw minus the shipped tw={SHIPPED_TW}, subject-clustered")
    for label, (subjects, correct, total) in rows.items():
        print(f"  {label} (n={len(subjects)})")
        for i, tw in enumerate(TWS):
            if i == i75:
                continue
            d = correct[:, i] - correct[:, i75]
            point, se_boot, se_jack, lo, hi = cluster_se(d, total)
            pos = int((d > 0).sum())
            print(
                f"    tw={tw:<5} {point * 100:+.2f} pp  SE_boot {se_boot * 100:.2f}  "
                f"SE_jack {se_jack * 100:.2f}  90% [{lo * 100:+.2f}, {hi * 100:+.2f}] pp  "
                f"{pos}/{len(subjects)} subjects positive"
            )

    print(f"\n(iii) member composition and fusion form, at the shipped tw={SHIPPED_TW}")
    print("  the two decisions still resting on numbers measured under the random thinner")
    for label, (a_key, b_key) in {
        "skeleton members worth keeping? (blend - image-only)": (
            "A fold-0 4-pass",
            "A image-only",
        ),
        "  same question, 18 subjects": ("B 4-fold single-pass", "B image-only"),
        "geometric - arithmetic fusion": ("A geometric", "A fold-0 4-pass"),
    }.items():
        # Both configurations were scored on the same thinnings, so their per-subject clip totals
        # must agree, which the assert checks.
        src_a = rows.get(a_key) or extra[a_key]
        src_b = rows.get(b_key) or extra[b_key]
        subjects, corr_a, total = src_a
        _, corr_b, total_b = src_b
        assert np.allclose(total, total_b), "pairing broken"
        d = corr_a[:, i75] - corr_b[:, i75]
        point, se_boot, se_jack, lo, hi = cluster_se(d, total)
        print(
            f"  {label:52s} n={len(subjects):2d}  {point * 100:+.2f} pp  "
            f"SE_boot {se_boot * 100:.2f}  90% [{lo * 100:+.2f}, {hi * 100:+.2f}] pp  "
            f"{int((d > 0).sum())}/{len(subjects)} subjects positive"
        )

    print("\n  What this replaces. Instrument A shipped as '+2.11 pp, SE 0.22, t=9.6' and the")
    print("  leaderboard came back -1.99 pp. Read the interval above: the 0.22 was an SE over")
    print("  thinning repeats with the same five people, so it never had anything to say about")
    print("  a different draw of people. The subject-clustered interval is the honest one, and")
    print("  whether it covers -1.99 pp is the whole question.")
    print("  For the N1/N3 gate this is a FLOOR, not the answer: comparing two separately")
    print("  trained runs adds seed and optimisation noise on top of what is measured here.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sections",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        choices=["A", "B", "C", "D", "E", "F"],
    )
    ap.add_argument("--repeats", type=int, default=20, help="thinning repeats, paired by seed")
    ap.add_argument(
        "--thinning",
        default="random",
        choices=sorted(THINNERS),
        help="how takes are thinned to the test size distribution; 'random' is what every "
        "published number used, 'contiguous' is the one that matches the real test spacing",
    )
    args = ap.parse_args()

    # Rebind the module-level thinner once, before any section runs.
    global THIN
    THIN = THINNERS[args.thinning]

    # ctx[4], ctx[5] and ctx[7] below are clip -> fold, the training takes and the test take sizes.
    ctx = build_context()
    print(f"thinning={args.thinning}")
    print(
        f"fold-0 clips {sum(1 for f in ctx[4].values() if f == 0)}, "
        f"train takes {len(ctx[5])}, test take sizes mean {ctx[7].mean():.2f}"
    )
    runners = {
        "A": section_a,
        "B": section_b,
        "C": section_c,
        "D": section_d,
        "F": section_f,
    }
    # Section E has no thinning, so it takes no repeat count.
    for name in args.sections:
        if name == "E":
            section_e(ctx)
        else:
            runners[name](ctx, args.repeats)


if __name__ == "__main__":
    main()
