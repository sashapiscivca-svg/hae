# data/

`annotations/` holds the exact COCO annotations of every split used to train and evaluate hae-s-ir (see [DATA_CARD.md](../DATA_CARD.md)). Each file keeps the licence of its source dataset (CC0 1.0, CC BY 4.0 or MIT); see [NOTICE.md](../NOTICE.md). Integrity: `annotations/SHA256SUMS`.

Images are not stored here. `scripts/prepare_data.py` writes converted images and training caches into this directory (ignored by git); see [docs/en/data.md](../docs/en/data.md).
