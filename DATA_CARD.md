# Data card: the hae data pool v1.0.0

[Українська](DATA_CARD.uk.md)

This card follows the spirit of *Datasheets for Datasets* (Gebru et al., 2021). The **hae data pool** is the set of openly licensed, civilian thermal aerial datasets that the hae model is trained and evaluated on. It is distributed as:

1. **Converted annotations** for every split, in `data/annotations/*.json` (COCO format, included in this repository).
2. **Deterministic conversion code**, `scripts/prepare_data.py`, which rebuilds the images from the official sources and aligns them to those annotations.

The images are **not** re-hosted. Download them from the original publishers, so that their licence terms and any updates reach you directly.

## 1. Motivation

People in thermal search-and-rescue imagery are tiny, partly hidden and rare. Public detectors are mostly trained on street scenes or on datasets whose licences forbid commercial use, which shuts out the companies and NGOs that build rescue equipment. We assembled a pool that is (a) civilian, (b) licensed for commercial use and derivatives, and (c) split by scene, so that the scores mean something for unseen terrain.

## 2. Composition

| Split file | Source | Licence | Frame size | Frames | Persons | Ignore boxes | Median person height (px) | Role |
|---|---|---|---|---:|---:|---:|---:|---|
| `hituav_trainval.json` | HIT-UAV train + val | CC0 1.0 | 640×512 | 2,319 | 9,701 | 4,301 | 19.0 | training |
| `hituav_test.json` | HIT-UAV test | CC0 1.0 | 640×512 | 579 | 2,611 | 827 | 19.0 | test |
| `airesq_train.json` | AIResQ Benchmark | CC BY 4.0 | 640×512 | 1,486 | 1,710 | 0 | 9.3 | training |
| `airesq_val.json` | AIResQ Benchmark | CC BY 4.0 | 640×512 | 502 | 933 | 0 | 8.1 | validation |
| `wisard_ir_val.json` | WiSARD thermal | MIT | 640×512 | 1,633 | 2,439 | 0 | 32.0 | transfer test |
| **Total** | | | | **6,519** | **17,394** | | | |

All frames are thermal (LWIR), 8-bit grayscale.

**Classes.** `0 = person` (including partly hidden people). `3 = ignore`: regions that should not count as an error either way (HIT-UAV "DontCare" regions and bicycles). Vehicles and every other class are **dropped on purpose**. They become background, so the model learns not to fire on warm cars of a similar size.

**Instances.** Each frame is a single still image with zero or more axis-aligned boxes `[x, y, w, h]` in pixels of the converted frame. The pool also contains frames without people, which act as negatives.

**Personal data.** The images show people from 30–120 m above ground, a few to a few dozen pixels tall. Nobody is identifiable at that scale. No names, faces or other identifiers are present in the annotations.

## 3. Collection and conversion

The original authors collected the data (see [NOTICE.md](NOTICE.md) for references). The hae conversion, implemented in `src/hae/data/` and driven by `scripts/prepare_data.py`, works as follows:

| Source | Conversion | Split rule (no scene appears in both train and evaluation) |
|---|---|---|
| HIT-UAV | `Person` → person; `Bicycle`, `DontCare` → ignore; `Car`, `OtherVehicle` → background. Frames kept at 640×512. File names `hituav_{split}_{original}.jpg`. | official train + val → `trainval`; official test → `test` |
| AIResQ | YOLO labels, class 0 → person. 1280×1024 frames resized to 640×512 (Lanczos); boxes scaled. File names `airesq_{original stem}.jpg`. | grouped by recording hour or clip id; every 7th group starting at the 4th held out whole (~14 %) |
| WiSARD thermal | 640×512, YOLO class 0 → person; only labelled IR sequences; original file names. | by **site**: Karen, MtErie, Airfield → val. The other sites were not used in v1.0.0. |

The published annotation files are the ground truth of what the model saw. `prepare_data.py` aligns the rebuilt images to those files by file name and reports any mismatch.

## 4. Uses

- **Used for:** training and evaluating hae-s-ir v1.0.0 (see the [model card](MODEL_CARD.md#5-evaluation-data)).
- **Good for:** small-object detection research, thermal aerial SAR benchmarks, studies of thermal domain shift (train on HIT-UAV/AIResQ, test on WiSARD-IR), and int8/edge-deployment studies.
- **Not suitable for:** identifying or tracking specific individuals, crowd counting in dense crowds, or ground-level pedestrian detection.

## 5. Known issues and biases

- **Box conventions differ between sources.** Some sources draw tight boxes around the visible person; others include a margin. When we fitted a single box-size correction to our detections, HIT-UAV labels wanted boxes about 13 % larger, while another thermal dataset (RGBTDronePerson) wanted them about 6 % smaller. Box centres are also noisy for people under 10 px. Expect AP at IoU 0.5 to understate how well tiny people are actually found; the centre-based metric in `hae.recall` is less sensitive to this.
- **Geography and seasons.** Each source covers a handful of regions (for example China for HIT-UAV and the US Pacific Northwest for WiSARD). Tropical, desert and snow scenes are scarce or absent.
- **Staged scenes.** In several sources many people are volunteers standing or walking in open terrain. Real victims are more often lying down, partly covered, or under trees.
- **Small validation set.** AIResQ val has 933 people in 502 frames from a few recording groups, so differences under ~0.04 AP50 there are not significant.
- **AIResQ** is the open benchmark subset; its authors provide the full dataset on request, and we did not use it.

## 6. Distribution and maintenance

- Annotations: `data/annotations/` in this repository (git) and on the Hugging Face Hub (see `huggingface/dataset/README.md`). Each file's `info` block names its source dataset, URL, licence and the changes made. Integrity: `data/annotations/SHA256SUMS`.
- Licence of each annotation file: **the licence of its source dataset** (CC0 1.0, CC BY 4.0 or MIT), not the hae MIT licence.
- Errata and additions: open a GitHub issue. Corrections will be released as new versioned files, never as silent edits.

## 7. Rejected datasets

We checked the licence of the data itself on its primary source (not the licence of the paper or the code) and accepted only CC0, CC BY, MIT, Apache-2.0, BSD or CDLA-Permissive. "Non-commercial", "no derivatives", "academic only", "on request", "subscription" and "unspecified" were all treated as rejections, as was any military origin or funding.

| Dataset | Reason |
|---|---|
| LLVIP | non-commercial terms |
| VisDrone, ForestPersons/ForestPersonsIR, POP, NII-CU, AFO | CC BY-NC(-SA/-ND) or commercial use only by agreement |
| SARD | paid subscription (IEEE DataPort), no licence stated |
| VTUAV-det, VTSaR, TinyPerson, NOMAD, Caltech Aerial RGB-T | data licence not stated or not confirmed |
| Manipal-UAV | academic use only |
| BIRDSAI | licence allows use; excluded by project decision (anti-poaching wildlife footage, off-target) |
| DOTA, DroneVehicle and other military-themed sets | out of scope by principle |
