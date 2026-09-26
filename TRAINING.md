# Training

Every weight used at inference is in `weights/model.pt`, so reproducing the submission needs no
training. This file documents how those weights were produced. The training entry points are
included as the record of the exact procedure; they are the last stages of a longer pipeline and read
intermediate artifacts that are not distributed (listed at the end), so they do not retrain the model
from the raw data on their own.

## Lineage

```
IG-65M→K400 R(2+1)D-34 ──► 4 fold models (60 epochs, each on 3 of 4 folds) ──► uniform weight average ──┬─► member C
                                                                                                        └─► member B ◄── soft pseudo-labels
IG-65M→K400 ir-CSN-152 ──► 4 fold models (60 epochs, 224² crops of det248) ──► uniform weight average ──► teacher 2 ─┐   for the 405 test clips
R(2+1)D-34 weight average ─────────────────────────────────────────────────────────────────────────► teacher 1 ─┴─► (mean of both)

VideoMAEv2 ViT-S (public, frozen) ──► region features of the 2,931 training clips ──► logistic-regression readout
training takes ──► 40×40 transition prior
```

## Recipes

**Fold models behind the initialisation** (`ig65m-t32-det-60ep`). R(2+1)D-34 from IG-65M→Kinetics-400,
4-channel stem, `det` view, 32 frames, 60 epochs, batch 8, AdamW, peak learning rate 3e-4 (backbone
×0.1), one-cycle schedule with 15% warm-up, weight decay 0.01, label smoothing 0.1, mixup 0.2,
dropout 0.3, seed 42. Fold k trains on the other three folds. The four are averaged uniformly (float64
mean of every floating-point tensor, `cuhkx/soup.py`), and the average's BatchNorm statistics are
re-estimated on 1,600 training clips.

**Second teacher** (`csn152-4f224-clip40`). ir-CSN-152 from IG-65M→Kinetics-400 (MMAction2 checkpoint
`ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb`, ported with `cuhkx/port.py`), 4-channel
stem, 224² crops of `det248`, 32 frames, 60 epochs, the same schedule with weight decay 1e-4; four fold
models averaged uniformly.

**Member C** (`training/124_c_full_student.py`). Starts from the R(2+1)D-34 average and trains on all
2,931 labelled clips for 10 epochs: batch 8, AdamW, peak learning rate 3e-5 for the classifier and
3e-6 for the backbone, one-cycle with 15% warm-up, weight decay 0.01, dropout 0.3 on the backbone
features and 0.5 before the classifier, label smoothing 0.1, mixup 0.2, random erasing 0.25,
horizontal flip 0.5, random 128² crop of the 144² view, one random frame per temporal segment, mixed
precision, seed 42. Each clip is drawn from the `det` or `miw` view per epoch with equal probability
(`cuhkx/shared_views.py`).

**Member B** (`training/103_b_final_student.py`). Same start and recipe on the `det` view, with the
405 test clips added. A test clip's target is the mean four-pass softmax (temperature 1) of the two
teachers, with no confidence threshold; its loss weight is 0.5 against 1 for a labelled clip
(label-smoothed one-hot), so pseudo-labels carry 0.5 × 405 / 2,931 = 6.9% of the labelled loss weight
(`cuhkx/replay.py`). Both teachers saw labelled training clips only.

**BatchNorm re-estimation** (`training/108_finalize_calibrated.py` for B; inside
`124_c_full_student.py` for C). Running statistics are reset and re-estimated as a cumulative mean over
200 batches of 8 training clips with inference-time frames and crops, dropout off, no gradients. The
learned weights do not change.

**Readout head** (`training/el22_p1_heads.py --full`). Support-weighted region features of the frozen
encoder for the 2,931 training clips (both flips, each at sample weight 0.5), per-modality
standardisation, design [I, D, I⊙D, |I−D|] (7,680 dimensions), second standardisation, scikit-learn
multinomial logistic regression with C = 0.01 (L-BFGS).

**Transition prior.** Counts of adjacent activity pairs in the 791 training takes, add-one smoothing,
rows normalised (`cuhkx/takes.transition_matrix`).

**Packing.** Post-training int8 quantisation with per-output-channel scales, member C as a lossless
delta against B, and the zlib envelope (`cuhkx/fd_lzma_codec.py`, `fd13_codec.py`, `fd18_codec.py`).

Training is seeded but not bit-reproducible, because `adaptive_avg_pool3d_backward_cuda` has no
deterministic CUDA kernel. Inference is deterministic.

## Cost

One NVIDIA GeForce RTX 5070 Ti.

| Stage | Time |
|---|---|
| R(2+1)D-34 fold models (4 × 60 epochs) | about 8 h |
| ir-CSN-152 fold models (4 × 60 epochs) | 7.8 h |
| Pseudo-labels for the 405 test clips | 75 s |
| Member B (10 epochs) | 1,728 s |
| Member B BatchNorm re-estimation | 50 s |
| Member C (10 epochs) | 1,613 s |
| Transformer features | about 5 min |
| Readout head | 11 s |

About 17 GPU-hours in total, not counting development runs.

## Intermediate artifacts

The entry points expect these under the paths set in `configs/paths.yaml` (`models_root`,
`processed_root`, `cache_root`, `train_root`). They are not distributed; their SHA256 values identify
the exact files used. The shipped `configs/paths.yaml` holds only the keys that inference needs, so
`models_root` and `test_csv` (a CSV whose `path` column lists the 405 test clips, such as the sample
submission; read by `103_b_final_student.py`) have to be added before training.

| Artifact | SHA256 |
|---|---|
| R(2+1)D-34 weight average (`t32-soup/soup.pt`) | `9f9f82eb1790cdd08cb8b65827cd0f7e9f5ffb495cccb478c47acd14d237a414` |
| ir-CSN-152 weight average (`csn152-4f224-clip40-soup/soup.pt`) | `f09b357da5ceaddb89cd082682a2fbcca5c7c10c7788911cab679f48b73ca1fb` |
| VideoMAEv2 checkpoint `vit_s_k710_dl_from_giant.pth` (repo commit `29eab1e`) | `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172` |
| Member B, packed int8 before merging | `d6439464cfec28ca07ed765c77575fc49ce2a0a3b3d03afcd3fada8bdda208b6` |
| Member C, packed int8 before merging | `bed1c09019f1090179e185ff2ee72afee5380c86003f0e8d9980e4e9a09631c5` |
| Readout head (`head.pt`) | `4346fce6c3b1d684cb41a2ac6915a5c9e3f40fc8a87fe4446459e797b43bdaa0` |

Besides these, a full retrain needs: the extracted training data; the clip index, fold and take tables
(`clip_index_listing.parquet`, `folds.parquet`, `takes_train.parquet`; `evidence/folds.csv` gives the
fold of every clip); the training caches built with `scripts/22_person_windows.py` and
`scripts/21_build_fused_cache.py` for the `det`, `miw` and `det248` views; the fold models and the two
weight averages; the transformer features; and the JSON records that pin each run's configuration.
The entry points also check the SHA256 of their own source files against those records, so they need
the original, uncommented files in their original layout. `scikit-learn` (1.9.0) and `psutil` are needed for training but not for
inference.
