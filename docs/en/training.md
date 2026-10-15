# Training and fine-tuning

[Українська](../uk/training.md) · [Docs index](README.md)

## Setup

```bash
pip install -e ".[train]"      # PyTorch, pycocotools, tensorboard, pyyaml, safetensors, rich, matplotlib
```

A CUDA GPU is recommended. The released model trained in 42 minutes on one NVIDIA T4 (90 epochs × 4,900 random 256×320 crops). The same recipe runs on a CPU, but it is many times slower. Prepare the data first ([data.md](data.md)).

## Reproduce hae-s-ir

```bash
python scripts/train.py --config configs/hae-s-ir.yaml --data data --out runs/hae-s-ir --device cuda
python scripts/export.py --checkpoint runs/hae-s-ir/last.pt --config configs/hae-s-ir.yaml --data data --out models-local/hae-s-ir --conf 0.5 --sensitive-conf 0.45
python scripts/evaluate.py --model models-local/hae-s-ir/hae-s-ir_lwir_int8.onnx --data data --sets hituav_test,airesq_val,wisard_ir_val
```

`train.py` writes to `--out`: `last.pt`, `best.pt` and `resume.pt` (resume with `--resume` after an interruption); `log.jsonl` and `results.csv` per epoch; ready-made figures (`dashboard.png`, `results.png`, `curves.png`, `val_pred.png`, `errors.png`, `labels.png`, `train_batch*.png`); and `final.json`, with the metrics of the final EMA weights on every `val_sets` entry of the recipe.

**Expected variation.** The recipe is the one stored in `models/hae-s-ir/config.json`. The order in which training crops are sampled differs slightly from the original run, and GPU kernels are not deterministic, so expect AP50 within about ±0.02 of the released numbers on HIT-UAV test and ±0.04–0.08 on AIResQ val, not identical weights.

## The recipe file

```yaml
name: hae-s-ir
train_sets:                      # first = main set; weight = share of training crops (sum 1)
  - {name: hituav_trainval, weight: 0.6}
  - {name: airesq_train, weight: 0.4}
val_sets: [hituav_test, airesq_val, wisard_ir_val]   # first = in-training validation; all = final evaluation
eval_stride: 4                   # in-training validation on every 4th frame
train:                           # any hae.train.TrainConfig field
  epochs: 90
  size: s                        # n | t | s | m (architecture preset)
  scale_range: [0.9, 2.4]        # crop scale jitter: teaches the ×1.5 inference view
  eval_scale: 1.5
  copy_paste: 0.3
  tnorm: true                    # per-frame z-score input
  ctx: true                      # stride-32 context level
  neg_decay: 1.0                 # soft negatives
  ...
```

`hae.train.TrainConfig` in `src/hae/train.py` documents every field. Command-line flags (`--epochs`, `--crops-per-epoch`, `--seed`, `--device`, `--threads`, `--init`) override the recipe.

## Fine-tuning on your own camera

Fine-tuning helps most when your camera, altitude or terrain differ from HIT-UAV and AIResQ (see the transfer results in the [model card](../../MODEL_CARD.md#7-quantitative-analysis)).

1. Label a few hundred frames from your sensor in COCO format: `person` as category 0, and optionally regions to ignore as category 3. CVAT and Label Studio both export COCO. Hold out whole flights for validation.
2. Add them as sets `my_train` and `my_val` ([data.md](data.md#using-your-own-data)).
3. Copy `configs/hae-s-ir.yaml` to `configs/my.yaml`. Put `my_train` first in `train_sets` and keep some HIT-UAV/AIResQ (for example 0.5 / 0.3 / 0.2) so the model does not forget. Set `val_sets: [my_val, hituav_test]`, fewer epochs (20–30) and a lower `lr` (for example 0.001).
4. Start from the released weights: `python scripts/train.py --config configs/my.yaml --data data --out runs/my --init models/hae-s-ir/hae-s-ir.safetensors` (or `init_weights:` in the recipe's `train` block). The recipe must keep the architecture of hae-s-ir (`size: s`, `ctx: true`, `tnorm: true`); a mismatch stops with an error. The released model can also serve as a teacher for distillation (`distill` in `TrainConfig`, see `src/hae/distill.py`).
5. Export with `scripts/export.py` (calibrate int8 on `my_train`), and choose your thresholds from the `scripts/evaluate.py --conf 0.3,0.35,0.4,0.45,0.5` sweep on `my_val`.

Do not judge a fine-tuned model on the frames you trained on, and keep `wisard_ir_val` or another untouched set to watch for regressions.

## Rules we followed (and recommend)

- **No third-party pretrained weights.** `build_model` refuses `backbone`/`pretrained`. This keeps the MIT licence of the weights clean.
- **Split by scene** (flight, site or recording group), never by frame. Neighbouring frames are near-duplicates.
- **Two seeds before claiming an improvement.** Single runs vary by about 0.02 AP50.
