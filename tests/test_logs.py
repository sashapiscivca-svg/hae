import io
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

from hae import ui
from hae.data.annotations import FrameAnn
from hae.dataset_stats import label_stats, save_batch_preview

sys.path.insert(0, str(Path(__file__).parent))


def _ann(i, boxes, labels=None):
    b = np.array(boxes, np.float32).reshape(-1, 4)
    return FrameAnn(i, f"{i}.jpg", 64, 96, b, np.zeros((0, 4), np.float32), np.array(labels if labels is not None else [0] * len(b), np.int64))


def test_label_stats_counts_classes_sizes_and_empty_frames():
    anns = [_ann(0, [[0, 0, 8, 8], [20, 20, 40, 44]], [0, 1]), _ann(1, []), _ann(2, [[5, 5, 9, 9]], [0])]
    s = label_stats(anns, ["person", "rider"])
    assert s["frames"] == 3 and s["empty_frames"] == 1 and s["instances"] == 3
    assert s["per_class"] == {"person": 2, "rider": 1} and s["small_lt10px"] == 2 / 3
    assert 4 <= s["size_p50"] <= 8.5 and s["per_frame_mean"] == 1.0


def test_epoch_line_shows_components_speed_and_quick_validation():
    rec = {"epoch": 3, "loss": 1.5, "epoch_s": 9.0, "lr": 1e-3, "loss_cls": 0.9, "loss_box": 0.6, "inst_per_crop": 3.2, "crops_per_s": 540.0,
           "data_wait_pct": 12.0, "gpu_gb": 1.4, "grad_norm": 2.3, "qval_lwir_AP50": 0.21, "qval_lwir_AP": 0.05, "qval_lwir_AR": 0.2, "qval_recall": 0.61, "qval_fp": 3.4}
    line = ui.epoch_line(rec, epochs=10, best=True)
    assert line.startswith("epoch 3/10 loss=1.500 9.0 s/epoch")
    for frag in ("cls 0.90", "box 0.60", "540 crops/s", "data wait 12%", "GPU 1.4G", "val AP50 0.210", "recall 0.61", "FP 3.4", "★"):
        assert frag in line, frag


def test_batch_preview_writes_png(tmp_path):
    from hae.data.sampler import Batch
    import torch
    b = Batch(torch.rand(4, 3, 32, 48), torch.rand(4, 1, 32, 48), [np.array([[5, 5, 15, 15]], np.float32)] * 4, [np.zeros((0, 4), np.float32)] * 4,
              [np.zeros(1, np.int64)] * 4, ("lwir",), None)
    p = save_batch_preview(b, tmp_path / "b.png", mode="lwir", n=4)
    assert p.exists() and p.stat().st_size > 500


def test_train_logs_components_quick_val_and_files(tmp_path):
    import test_ckpt as T
    from hae.train import train
    ui.set_console(Console(file=io.StringIO(), force_terminal=False, width=140, highlight=False))
    root = T._data(tmp_path)
    train(T._cfg(tmp_path, root, epochs=2, eval_every=2, eval_split="val", eval_frames=[0, 1, 2, 3], quick_val=3, verbose=True, ckpt_every=1))
    h = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    for k in ("loss_cls", "loss_box", "inst_per_crop", "crops_per_s", "data_wait_pct", "grad_norm", "qval_lwir_AP50", "qval_recall", "qval_fp"):
        assert k in h[0], k
    out = tmp_path / "run"
    assert (out / "train.log").exists() and "epoch 1/2" in (out / "train.log").read_text()
    for f in ("labels.png", "train_batch0.png", "train_batch1.png", "train_batch2.png", "results.png", "curves.png", "val_pred.png", "errors.png", "dashboard.png"):
        assert (out / f).exists() and (out / f).stat().st_size > 300, f   # ready-made images with default settings, without TensorBoard
    ui.set_console(None)
