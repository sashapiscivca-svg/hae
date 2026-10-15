# NOTICE — third-party material and attribution

[Українська](NOTICE.uk.md)

The hae source code and trained weights are released under the [MIT License](LICENSE), Copyright (c) 2026 The hae authors. This file lists the third-party material that hae relies on, its licence, and the attribution it requires. Licence statements were read on each dataset's primary page (Zenodo records via the Zenodo API, the authors' sites or repositories). Check the current terms on those pages before you redistribute anything.

## 1. Datasets

**No images from these datasets are redistributed in this repository.** `data/annotations/` contains our *converted annotations* (COCO JSON: bounding boxes, file names, split membership) for the exact splits used to train and evaluate the released model. These are adaptations of the original annotations, distributed under each dataset's licence as listed below. Changes made: conversion to COCO, mapping the person class to id 0 and "don't care"/occluded regions to an ignore class (id 3), dropping all other classes (they become background), resizing frames to 640×512, and splitting. See [DATA_CARD.md](DATA_CARD.md) for details.

| Dataset | Licence | Use in hae | Attribution |
|---|---|---|---|
| **HIT-UAV** | CC0 1.0 | thermal training (trainval) and test | Suo, J. et al. *HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection.* Scientific Data 10, 227 (2023), doi:[10.1038/s41597-023-02066-6](https://doi.org/10.1038/s41597-023-02066-6). Data: Zenodo, doi:[10.5281/zenodo.7633134](https://doi.org/10.5281/zenodo.7633134). CC0 does not require attribution; we cite it as good practice. |
| **AIResQ Benchmark** | CC BY 4.0 | thermal training and validation | Büttner, J. *AIResQ Benchmark.* Zenodo (2025), doi:[10.5281/zenodo.17405074](https://doi.org/10.5281/zenodo.17405074). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: see above. |
| **WiSARD** (thermal part) | MIT | evaluation only (transfer test, `wisard_ir_val`) | Broyles, D., Hayner, C. R., Leung, K. *WiSARD: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue.* IEEE/RSJ IROS 2022, pp. 9467–9474, doi:[10.1109/IROS47612.2022.9981298](https://doi.org/10.1109/IROS47612.2022.9981298). https://sites.google.com/uw.edu/wisard. The dataset is distributed under the MIT License, copyright (c) 2022 Daniel Broyles, Christopher Hayner, Karen Leung. Its permission notice: *"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files … THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND …"* (full text as in [LICENSE](LICENSE), with the copyright line above). |

The README figures (`docs/assets/hae-*.jpg`) show HIT-UAV test frames (CC0 1.0), shown with a colour palette and overlaid with our detections.

Whether attribution duties extend to model weights trained on these datasets is not settled in law. We give the attribution anyway, and the ONNX metadata of the model names its training datasets (`training_data` key).

### Datasets considered and not used

We rejected datasets with non-commercial (NC) or no-derivatives (ND) terms, with no stated licence, available only on request or by subscription, or of military origin. Examples: LLVIP, VisDrone, ForestPersons, POP, NII-CU (non-commercial); SARD, VTUAV-det, TinyPerson, NOMAD (licence unclear). BIRDSAI was excluded by project decision even though its licence would allow use. [DATA_CARD.md](DATA_CARD.md#7-rejected-datasets) has the full list.

## 2. Pretrained weights

None. The released model was trained from random initialisation. No ImageNet or other third-party checkpoints, teachers or backbones were used.

## 3. Published ideas the design draws on

All code in this repository was written by the hae authors. No third-party source code was copied. The design draws on these published ideas:

- Objects as Points / CenterNet — Zhou, Wang, Krähenbühl (2019)
- FCOS: Fully Convolutional One-Stage Object Detection — Tian, Shen, Chen, He (2019)
- Generalized Focal Loss (Quality Focal Loss) — Li et al. (2020)
- Generalized IoU — Rezatofighi et al. (2019)
- Normalized Wasserstein Distance for tiny objects — Wang, Xu, Yang, Yu (2021)
- GhostNet — Han et al. (2020); RepVGG — Ding et al. (2021)
- Sub-pixel convolution / space-to-depth — Shi et al. (2016)
- Slicing Aided Hyper Inference (SAHI) — Akyon, Altinuc, Temizel (2022)
- AdamW and SGDR — Loshchilov and Hutter (2017, 2019); weight EMA

## 4. Runtime dependencies (installed by the user, not bundled)

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause (and bundled components) |
| onnxruntime | MIT (see its ThirdPartyNotices) |
| onnx | Apache-2.0 |
| opencv-python-headless | Apache-2.0. The wheel bundles FFmpeg under LGPL-2.1; this matters if you redistribute the wheel itself. |
| pillow | MIT-CMU (HPND) |
| huggingface_hub (optional) | Apache-2.0 |
| torch, tensorboard, pyyaml, pycocotools (training only) | BSD-style / Apache-2.0 / MIT / BSD |

hae does not ship these packages. `pip` installs them from their publishers under their own licences.
