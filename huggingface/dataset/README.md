---
pretty_name: hae thermal SAR annotations
license: other
license_name: per-file (CC0-1.0, CC-BY-4.0, MIT)
license_link: https://github.com/sashapiscivca-svg/hae/blob/main/NOTICE.md
task_categories:
  - object-detection
tags:
  - search-and-rescue
  - thermal
  - infrared
  - drone
  - uav
  - tiny-object-detection
  - coco
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: hituav_trainval
        path: annotations/hituav_trainval.json
      - split: hituav_test
        path: annotations/hituav_test.json
      - split: airesq_train
        path: annotations/airesq_train.json
      - split: airesq_val
        path: annotations/airesq_val.json
      - split: wisard_ir_val
        path: annotations/wisard_ir_val.json
---

# hae thermal SAR annotations (v1.0.0)

These are the exact COCO annotations of every split used to train and evaluate the open thermal person detector [hae-s-ir](https://huggingface.co/sashapiscivca-svg/hae-s-ir), converted from three openly licensed civilian datasets. **The images are not included.** Download them from the original sources, then run `scripts/prepare_data.py` from https://github.com/sashapiscivca-svg/hae to convert and align them.

| File | Source | Licence | Frames | People | Role |
|---|---|---|---:|---:|---|
| `annotations/hituav_trainval.json` | [HIT-UAV](https://zenodo.org/records/7633134) | CC0 1.0 | 2,319 | 9,701 | training |
| `annotations/hituav_test.json` | HIT-UAV | CC0 1.0 | 579 | 2,611 | test |
| `annotations/airesq_train.json` | [AIResQ Benchmark](https://zenodo.org/records/17405074) | CC BY 4.0 | 1,486 | 1,710 | training |
| `annotations/airesq_val.json` | AIResQ Benchmark | CC BY 4.0 | 502 | 933 | validation |
| `annotations/wisard_ir_val.json` | [WiSARD](https://sites.google.com/uw.edu/wisard) (thermal) | MIT | 1,633 | 2,439 | transfer test |

Categories: `0 = person`, `3 = ignore` (regions not scored either way). All frames are 640×512 thermal frames, and boxes are `[x, y, w, h]` in pixels. Each file's `info` block records its source, licence and the changes we made.

**Licences.** Each file is an adaptation of its source dataset's annotations and is distributed under that dataset's licence. Changes: conversion to COCO, class mapping (person → 0; HIT-UAV DontCare and bicycle → ignore; vehicles dropped), resizing to 640×512 where needed, and splitting by scene. Attribution and full references: [NOTICE.md](https://github.com/sashapiscivca-svg/hae/blob/main/NOTICE.md). Documentation: [DATA_CARD.md](https://github.com/sashapiscivca-svg/hae/blob/main/DATA_CARD.md).

**Intended use.** Training and benchmarking person detection for civilian search and rescue. Not for identifying or tracking specific individuals. See [RESPONSIBLE_USE.md](https://github.com/sashapiscivca-svg/hae/blob/main/RESPONSIBLE_USE.md).

**Citation.** Please cite the source datasets (HIT-UAV: Suo et al., *Scientific Data* 2023; AIResQ: Büttner, Zenodo 2025; WiSARD: Broyles, Hayner, Leung, *IROS* 2022) and hae.
