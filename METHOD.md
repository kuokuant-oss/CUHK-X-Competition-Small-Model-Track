# Method

This file describes what the delivered system computes, step by step, with the module that
implements each step. The technical report covers the same system with its evaluation.

## 1. Input

One folder per clip, `SM_test_XXXX/`, containing:

- `Depth_Color/*.png`: depth rendered through a jet-like colour map, 640×480 RGB, 10 fps;
- `IR/*.png`: 8-bit infrared from the same sensor, on the same pixel grid, with matching file names;
- `Skeleton/`: only the file names are read, as a fallback source of timestamps.

Test clips have 2–101 frames (median 20). IMU, radar and thermal data are not used.

## 2. Preprocessing

All steps are deterministic and run inside `inference.sh`; nothing is precomputed.

**Frames.** Clips longer than 32 frames keep 32 evenly spaced frames (`fused.load_clip_fused`).

**Person window** (`scripts/22_person_windows.py`, `cuhkx/person.py`). A COCO-pretrained
`ssdlite320_mobilenet_v3_large` (torchvision; weights inside `model.pt`) runs on 8 evenly spaced IR
frames and keeps the best person box per frame. The clip's boxes are accepted if one scores at least
0.30, or if at least 4 score at least 0.10 and their centres scatter by less than 10% of the frame
width. Otherwise the detector reruns on four overlapping corner tiles (60% of the frame's width and
height) after a 1st–99.5th percentile contrast stretch; then a motion box on decoded depth is used,
and finally the whole frame. The window is a square on the median box centre with side
max(1.4 × largest box side, 0.35 × frame width), not clipped to the frame. On the test set: 385
confident, 11 agreement, 5 tiles, 4 motion, 0 whole frame.

**Depth decoding** (`cuhkx/depth.py`). A lookup table orders the 254 palette colours along the jet
colour map and maps each to a rank 0–254; black and unknown colours mean "no return". The table is
stored in the checkpoint and checked by SHA256 on load. Decoded depth is used only to find motion
boxes; the networks receive the Depth_Color RGB values.

**Shared crop and views** (`scripts/21_build_fused_cache.py`, `cuhkx/fused.py`). The same window and
frame indices cut Depth_Color and IR; window parts outside the frame repeat the edge pixels. Frames
are stored as uint8 with channels Depth_Color R, G, B and IR, in three views:

| View | Window | Size | Used by |
|---|---|---|---|
| `det` | person window | 144² | members B and C |
| `miw` | tighter box inside the person window around the pixels in the top 2% of depth motion (mean absolute deviation from the per-pixel temporal median), 10% margin, squared; falls back to the person window | 144² | member C |
| `det248` | person window | 248² | transformer |

**Frame sampling** (`fused.tsn_picks`). The stored frames are split into K equal segments and the
centre frame of each is taken: K = 32 for the members, K = 16 for the transformer. Short clips repeat
frames.

**Normalisation.** Members: pixels / 255, Kinetics mean and standard deviation on channels 0–2 and
their average on the IR channel. These constants are buffers of the member (`models.PretrainedVideo3D`)
and are stored int8-quantised like its weights.
Transformer: ImageNet mean and standard deviation (`fd15_vit.preprocess`).

## 3. Recognition

**Members B and C** (`cuhkx/fd13_inference.py`, `cuhkx/models.py`). R(2+1)D-34, IG-65M → Kinetics-400
pretrained (architecture code in `runtime/`, from ig65m-pytorch), with the first convolution widened
to 4 channels. Input: centre 128² crop of the 144² view, 32 frames. Each member averages its softmax
over four passes (`cuhkx/tta.py`): unchanged, horizontal flip, and the frame sequence rolled by ±1.
B reads `det`; C reads `det` and `miw` and averages the two (`thermal_pair.fixed_pair`; if `miw` is
missing, C falls back to `det`). F0 = ½ B + ½ C.

**Transformer readout** (`cuhkx/el22_support_inference.py`, `el22_support_pool.py`,
`fd18_readout.py`). The public VideoMAEv2 ViT-S/16 checkpoint `vit_s_k710_dl_from_giant` (int8 in
`model.pt`) is frozen. Depth_Color and IR (copied to three channels) are encoded separately from the
centre 224² crop of `det248`, 16 frames, plain and horizontally flipped, giving 8×14×14 tokens of
dimension 384. After the last block the tokens are averaged over five regions — the whole 14×14 grid
and its four 7×7 quadrants — and layer-normalised. Each patch is weighted by the fraction of its
pixels that lie inside the camera frame (computed from the raw frame size and the window with the
same bilinear geometry as the crop, `el22_runtime_support.support_for_ids`); regions whose patches
are all inside the frame use the plain mean.

This gives I, D ∈ R^(5×384) per flip. Each is standardised per dimension with training statistics;
the head input is x = [I, D, I⊙D, |I−D|] ∈ R^7680 (elementwise product and absolute difference),
standardised again, followed by a 7680 → 40 linear layer and softmax. The head is a multinomial
logistic regression (L2, C = 0.01, L-BFGS) fitted on all 2,931 training clips and stored as FP32
tensors. The two flips are averaged into q.

**Fusion** (`fd18_readout.compose`). On clips where both IR and depth decoded (401 of 405):

    f = normalize(exp(0.8 · log F0 + 0.2 · log q)),   probabilities clipped at 1e-12

The other clips keep f = F0.

## 4. Decoding

**Takes** (`cuhkx/el22_clock_adapter.py`, `el21_take_runtime.py`, `el21_take_geometry.py`). A frame
file name carries a date, a 100 ms timestamp and the recorder's frame counter, e.g.
`Depth_2025-06-11_13-21-46.616_00000067_Color.png`. For each clip, `first timestamp − 100 ms ×
first counter` is the time the recorder started; clips with the same date and start time form a take
and are ordered by (start, end). A clip whose span does not follow an exact 100 ms clock is left
out of all takes. On the training set this reproduces the 791 known recordings (`user/trial`)
exactly; on the test set it gives 144 takes of 1–8 clips (median 2) covering all 405 clips.

**Repeat chains** (`cuhkx/el25_repeat_decode.py`). Consecutive takes of the same date are chained when
the later one starts less than 120 s after the start of the earlier one's last clip and both have the
same number of clips. Within a chain, each take's per-position log-probabilities get λ = 0.25 times
the sum of the other takes' log-probabilities at the same position and are renormalised per clip. On
the test set: 39 chains (35 pairs, 4 triples), 235 clips.

**Joint decoding** (`cuhkx/takes.py: decode_take`, driven by `h_mpm_runtime.submission_frame`). For
each take, labels are chosen to be pairwise distinct and to maximise
Σ log f_i(y_i) + 0.75 Σ log T(y_{i−1}, y_i), where T is the 40×40 transition matrix counted in the
training takes with add-one smoothing (stored in the checkpoint). Beam search: 64 beams, the 16 most
probable classes per clip. A single-clip take keeps its argmax.

**Fallbacks.** If fewer than 50% of the clips fall into takes, decoding and chaining are skipped and
every clip takes its argmax; clips outside any take also take their argmax
(`evidence/runs/fallbacks/receipt.json` exercises these cases). The run writes both the final CSV
and the decode without chaining (`prediction-original.csv`); they differ in 10 of 405 rows.

## 5. Checkpoint

`weights/model.pt` (97,478,875 bytes) holds every weight loaded at inference:

| Component | Stored values | Storage | Stored bytes |
|---|---:|---|---:|
| Member B | 63,590,251 | int8, per-output-channel scales, LZMA/zlib per tensor | 51.3 MB |
| Member C | 63,590,251 | mod-256 difference of its int8 codes from B's (3 tensors predicted from B's rescaled codes), LZMA/zlib | 12.8 MB |
| ViT-S encoder | 21,921,792 | int8, per-output-channel scales, zlib | 19.9 MB |
| SSDLite detector | 3,473,074 | FP32, compressed | 11.6 MB |
| Readout head | 330,320 | 307,240 linear parameters, 23,040 standardisation statistics, 40-entry class map; FP32/FP64 | 1.4 MB |

The five components hold 97.0 MB of tensor data. Per-tensor SHA256 manifests, metadata (palette
table, transition prior, clip lists) and serialisation add 1.9 MB, giving a 98,899,037-byte payload that
an outer zlib layer compresses to the file size. Every tensor has a SHA256 manifest that is checked on load, and
member C is rebuilt from B and the difference and checked against its own SHA256
(`fd_lzma_codec.restore_delta`). The runtime asserts the file size is below 100,000,000 bytes.

## 6. Training (summary)

B and C are fine-tuned from the uniform weight average of four R(2+1)D-34 models, each trained for 60
epochs on three of the four folds. B additionally sees the 405 test clips with soft pseudo-labels,
the mean four-pass softmax of two weight-averaged teachers (R(2+1)D-34 and ir-CSN-152) trained on
labelled clips only, at loss weight 0.5 per test clip (0.5 × 405 / 2,931 = 6.9% of the labelled loss
weight). C uses labelled clips only and draws the `det` or `miw` view per clip and epoch. Both then
have their BatchNorm statistics re-estimated on 1,600 training clips without gradient steps.
`TRAINING.md` has the full recipe.

## 7. Validation protocol

Four subject-disjoint folds of the 2,931 training clips (`evidence/folds.csv`: 760/843/629/699 clips,
5/5/4/4 subjects). For the local estimates every component is retrained per fold with that fold's
subjects held out: fold-specific starting weights, teachers, members, readout head and transition
prior. Member B of fold k is self-trained on a test-like subset of fold k's clips (one contiguous run
per take, run lengths drawn from the test take sizes), mirroring what the delivered B does on the test
set. Four-fold means after take decoding: 0.7999 on the test-like subset, 0.8092 on complete takes.

## 8. Names used in the code

| Name | Meaning |
|---|---|
| F0 | ½ B + ½ C, the member average |
| `det`, `miw`, `det248` | the three cache views (section 2) |
| S1 | readout head on five pooled regions with features [I, D, I⊙D, \|I−D\|] |
| S1-G | S1 blended geometrically with F0 at weight 0.2 (`--active-readout S1-G`); S1-A would be the arithmetic blend |
| P1 / P0 | region pooling with / without the in-frame support weights |
| repeat-v1 | repeat-chain borrowing (λ = 0.25, 120 s) |
| `clock100ms-start-end-v1` | take grouping by recorder start time |
| EL25R-P1-repeat-v1 | name of the submitted configuration (Kaggle submission 56254406) |
| `fd13`–`fd18`, `el21`–`el25` prefixes | development iterations in which a module was written |

Code comments also refer to internal design notes as "ADR 0002"–"ADR 0007" and to the organisers'
answers as "C-1"–"C-11". The design notes record, in order: keeping the Depth_Color colour semantics
(no colour augmentation), subject-wise validation, the four-fold subject split, the skeleton format,
measuring the 100 MB limit on disk, and decoding per take. The organisers' answers allowed
deterministic test-time preprocessing inside `inference.sh`, public pretrained models, a small
person detector counted inside the 100 MB file, and self-training on unlabelled test clips.
