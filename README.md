# CUHK-X Competition - Small Model Track

**Take-structured decoding of a Depth+IR video ensemble.** Inference code, checkpoint and training
code for our entry in the
[CUHK-X Multimodal Human Activity Challenge](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track),
Small Model Track: 40-class activity recognition on 405 test clips of unseen subjects, with every
weight used at inference in one file under 100 MB.

| | |
|---|---|
| Public leaderboard | 0.84079 |
| Private leaderboard | 0.83823 (22nd of 326 teams) |
| Checkpoint | `weights/model.pt`, 97,478,875 bytes |
| Technical report | `CUHKX_TechnicalReport_SmallModelTrack.pdf` |

The system has three parts:

1. **Recognition.** Two R(2+1)D-34 networks read person-cropped Depth_Color and infrared (IR) video
   as one pixel-aligned four-channel input; a linear readout of a frozen VideoMAEv2 ViT-S is blended
   into their average in log space.
2. **Take-structured decoding.** Frame filenames keep the recorder's clock, which regroups the test
   clips into continuous recordings ("takes"). Each take is decoded as a set of distinct labels under
   a transition prior, and takes that repeat the same routine share evidence first.
3. **Packing.** The second network is stored as a lossless int8 delta against the first, which fits
   both networks, the transformer and the person detector into one 97.5 MB file.

`METHOD.md` describes each step with pointers into the code; the technical report gives the
evaluation.

## Running inference

Requirements: a CUDA GPU (there is no CPU path), Python 3.12, and the competition test data.

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
pip install -r requirements.txt
```

The checkpoint is `weights/model.pt`. Check it before running:

```bash
shasum -a 256 -c checksums.txt
```

Then run, with `--data_dir` pointing at the folder that directly contains `SM_test_0001` …
`SM_test_0405`:

```bash
./run_inference.sh \
  --data_dir /path/to/small_model_track_test \
  --sample_submission /path/to/sample_submission.csv \
  --output submission.csv
```

The run takes about 6 minutes and rebuilds every intermediate file from the raw folders. It writes
405 rows (`path,prediction`, with `prediction` in 0–39) and prints two hashes next to their expected
values:

| | SHA256 |
|---|---|
| `submission.csv` | `aef1ed637146b2c8ef6235b092c88b5869dcbd5e09b88f909e6198bd3cff6191` |
| prediction vector (int64) | `982db37510d0dd2d86a29358dfe3116f65623a5ece537bb91f2d2b8de87f1860` |

These are the hashes of the submitted CSV. The run also writes `outputs/run/prediction-original.csv`,
the same probabilities decoded without repeat-chain borrowing, which differs from the submission in 10
rows.

### Notes

- `run_inference.sh` writes the two paths into `configs/paths.yaml`, empties `outputs/cache` and
  `outputs/run`, and calls `inference.sh`, which passes `--active-readout S1-G`. That flag selects the delivered readout and is
  not the argparse default, so call `inference.sh` rather than the Python entry point directly.
- The runtime loads no file other than `weights/model.pt`, rejects any other `torch.load`, and
  disables network connections before it starts.
- Seeds, single-threaded BLAS, disabled TF32 and disabled cuDNN benchmarking make repeated runs on
  one GPU identical (checked twice). Bit-identical output across GPU models is not guaranteed.
- `scipy` is pinned as a range; it is used only on a decoder branch that the delivered checkpoint
  never takes.

## Repository layout

```
weights/model.pt      the only weight file loaded at inference
run_inference.sh      wrapper: sets paths, clears the cache and run folders, runs inference.sh
inference.sh          the command that produced the submission
configs/paths.yaml    test folder and sample_submission.csv (written by run_inference.sh)
scripts/              inference entry point and preprocessing steps
src/cuhkx/            library used by inference and training
training/             training entry points and the modules they import
runtime/              R(2+1)D architecture code from ig65m-pytorch (no weights)
tools/                path setup and the preprocessing figure of the report
evidence/             records of the submitted runs and the fold assignment
third_party/          licences of bundled third-party code
```

File names keep the prefixes they had during development (`fd18_`, `el22_`, `el25_`, numbered
scripts) so that they match the recorded runs in `evidence/`. `METHOD.md` §8 explains the names that
appear in the code, and every source file starts with a comment block saying what it does and who
uses it. These comments were added for this release. Apart from them, the code differs from the
submitted version only in the wording of two docstrings and in the name of an environment variable
that the training helpers read but never use.

### `scripts/`

| File | Role |
|---|---|
| `el25r_p1_repeat_runtime.py` | Entry point: checks the checkpoint, builds the inputs, runs the networks, fuses, decodes and writes the CSV |
| `22_person_windows.py` | Person window per clip from IR (called by the entry point) |
| `26_build_test_caches.py` | Builds the cache commands from the checkpoint's metadata and checks the result |
| `fd18_raw_builder.py`, `21_build_fused_cache.py` | Crop, resize and store the four-channel frames of one view |
| `12_build_take_index.py` | Take tables for training data; at inference, takes come from `src/cuhkx/el22_clock_adapter.py` |

### `src/cuhkx/`

| Role | Modules |
|---|---|
| Preprocessing | `person` (detector cascade), `depth` (palette lookup table, motion box), `fused` (shared crop, cache, frame sampling), `data`, `images`, `paths` |
| Recognition | `fd13_inference` (members B and C), `models`, `tta`, `thermal_pair` (joins C's two views; the name is historical), `el22_support_inference`, `el22_support_pool`, `el22_runtime_support`, `fd15_vit`, `fd15_official_vit` (transformer) |
| Fusion and readout | `fd18_readout`, `fd16_readout`, `fd15_codec` |
| Decoding | `el22_clock_adapter`, `el21_take_runtime`, `el21_take_geometry` (take grouping), `takes` (beam decoder), `el25_repeat_decode` (repeat chains), `h_mpm_runtime` (decoder driver and fallbacks) |
| Checkpoint format | `fd18_codec`, `fd16_codec`, `fd13_codec`, `fd_lzma_codec` (int8 states, member-C delta), `fd_codec`, `fd_entropy_codec`, `fd_mixed_codec`, `budget` |
| Training | `train`, `replay`, `shared_views`, `soup`, `interpolation`, `port` |

`imu`, `skeleton`, `take_runtime_v2` and `h_mpm_prototype` are imported but none of their functions
runs in the delivered configuration.

## Training

Retraining is not needed to reproduce the submission. `TRAINING.md` documents how each weight was
produced, the recipes and their costs, and which intermediate artifacts a full retrain needs beyond
this repository.

## Data and licence

No competition data and no file derived from the test set are included. The code is released under the Apache License 2.0 (`LICENSE`).
The checkpoint was trained on CUHK-X data, so its use is also subject to the CUHK-X dataset licence
(research use). Bundled and pretrained third-party components keep their own licences; see
`THIRD_PARTY_NOTICES.md`.

## Acknowledgements

The backbone choice, four-channel Depth_Color+IR input, stem widening, Kinetics normalisation,
four-pass test-time averaging and person-window geometry follow the public notebooks
[LB 0.711 | YOLO Person Crop + R2Plus1D <100MB](https://www.kaggle.com/code/phuongncn/lb-0-711-yolo-person-crop-r2plus1d-100mb)
and [YOLO for CUHK-X](https://www.kaggle.com/code/kunaldesale2408/yolo-for-cuhk-x); the per-channel
int8 scheme follows
[[LB0.667] baseline with YOLO person crop](https://www.kaggle.com/code/welshonionman/lb0-667-baseline-with-yolo-person-crop).

If you use this work, please cite the CUHK-X dataset paper:

> S. Jiang, M. Yuan, X. Ji, et al. A Large-Scale Multimodal Dataset and Benchmarks for Human
> Activity Scene Understanding and Reasoning. MobiSys '26, 2026.
> [arXiv:2512.07136](https://arxiv.org/abs/2512.07136)
