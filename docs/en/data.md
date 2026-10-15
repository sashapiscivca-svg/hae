# Data: rebuilding the exact training and evaluation sets

[Українська](../uk/data.md) · [Docs index](README.md) · [Data card](../../DATA_CARD.md)

The repository ships the **annotations** of every split in `data/annotations/` (COCO JSON with SHA-256 sums). The images stay with their publishers: download them yourself, then let `scripts/prepare_data.py` convert them and align them to the published annotations. Nothing is fetched from the network automatically.

## 1. Download the sources

```bash
python scripts/prepare_data.py sources      # prints URL, licence and the expected folder layout for each dataset
```

| Dataset | Where | Licence | What `--src` must contain |
|---|---|---|---|
| HIT-UAV | https://zenodo.org/records/7633134 | CC0 1.0 | the unpacked archive with `normal_json/{annotations,train,val,test}` |
| AIResQ Benchmark | https://zenodo.org/records/17405074 | CC BY 4.0 | a folder with `images/*.jpg` and YOLO `labels/*.txt` |
| WiSARD (thermal) | https://sites.google.com/uw.edu/wisard | MIT | the `<date>_<Site>_<platform>_IR_<n>/` sequence folders (frames and YOLO `.txt`) |

Only WiSARD is needed for the transfer test. HIT-UAV and AIResQ are enough to train.

## 2. Convert and align

```bash
python scripts/prepare_data.py hituav    --src ~/dl/HIT-UAV --out data   # → hituav_trainval, hituav_test
python scripts/prepare_data.py airesq    --src ~/dl/AIResQ  --out data   # → airesq_train, airesq_val
python scripts/prepare_data.py wisard-ir --src ~/dl/WiSARD  --out data   # → wisard_ir_val
```

Each command:

1. converts the original files into the unified format (640×512 grayscale frames; class mapping as in the [data card](../../DATA_CARD.md#3-collection-and-conversion)) under `data/work/<dataset>/`;
2. **aligns** to `data/annotations/<set>.json` (the default): it keeps exactly the frames listed there, uses the published boxes, and reports missing frames, size mismatches and label differences. With `--no-align`, it keeps the converter's own split instead;
3. writes `data/sets/<set>/{ann.json, thermal/}` (hard links, no copies) and the training caches `data/cache/<set>_t.u8`, `_meta.json` and `_ann.json`.

The uint8 caches of the five sets take about 2.1 GB, on top of the downloads.

## 3. Check

The frame and person counts reported at the end must match the data card: `hituav_trainval` 2,319 / 9,701; `hituav_test` 579 / 2,611; `airesq_train` 1,486 / 1,710; `airesq_val` 502 / 933; `wisard_ir_val` 1,633 / 2,439. If frames are reported missing, your download probably differs from the version we used (Zenodo records are versioned). Open an issue with the report.

## Using your own data

Any COCO file with `person` boxes (category 0) and optional ignore regions (category 3) on 8-bit thermal frames can be added: put the frames under `data/sets/<name>/thermal/` and the annotations in `data/sets/<name>/ann.json`, then build the cache with `hae.data.cache.build_cache_thermal(<data>/sets/<name>, "<name>", <data>/cache)` and copy `ann.json` to `<data>/cache/<name>_ann.json` (this is what `write_set` in `scripts/prepare_data.py` does). After that, `<name>` can be used in a recipe's `train_sets` or `val_sets` and in `scripts/evaluate.py --sets`.
