# Third-party notices

The code in this repository is released under the Apache License 2.0 (`LICENSE`), except for the
third-party material listed below, which keeps its own licence.

## Code

| Path | Origin | Licence |
|---|---|---|
| `runtime/torch/hub/moabitcoin_ig65m-pytorch_master/` | [ig65m-pytorch](https://github.com/moabitcoin/ig65m-pytorch), R(2+1)D-34 architecture code | MIT, see `LICENSE.md` in that folder |
| `src/cuhkx/fd15_official_vit.py` | [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) (commit `29eab1e`), itself based on BEiT, timm, DINO and DeiT | MIT, see `third_party/VideoMAEv2/LICENSE` |

## Weights inside `weights/model.pt`

| Component | Derived from | Terms |
|---|---|---|
| Members B and C | R(2+1)D-34 pretrained on IG-65M and Kinetics-400, as released by [ig65m-pytorch](https://github.com/moabitcoin/ig65m-pytorch); fine-tuned on CUHK-X | Upstream terms of the pretrained weights and the CUHK-X dataset licence |
| ViT-S encoder | VideoMAEv2 checkpoint `vit_s_k710_dl_from_giant` from [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), quantised to int8, not fine-tuned | VideoMAEv2 terms (MIT) |
| Person detector | torchvision `ssdlite320_mobilenet_v3_large`, COCO weights ([torchvision](https://github.com/pytorch/vision)) | BSD-3-Clause |
| Readout head, transition prior, palette table | Fitted on CUHK-X training data | CUHK-X dataset licence |

## Used only during training

The second pseudo-label teacher was initialised from MMAction2's ir-CSN-152 IG-65M→Kinetics-400
checkpoint ([MMAction2](https://github.com/open-mmlab/mmaction2), Apache-2.0). None of its weights are
in `weights/model.pt`.

## Data

No CUHK-X data is included. The dataset is available to challenge participants under the CUHK-X
dataset licence (v2.0), which also applies to models trained on it.
