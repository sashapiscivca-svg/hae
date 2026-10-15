import json

import numpy as np
import pytest
import torch
from PIL import Image

from hae.data.cache import build_cache
from hae.train import TrainConfig, train


def _data(tmp_path, n=6):
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
        (root / "val" / m).mkdir(parents=True)
    for sp in ("train", "val"):
        imgs, anns = [], []
        for i in range(n):
            im = np.zeros((64, 96, 3), np.uint8)
            im[20:30, 6 + 12 * i:16 + 12 * i] = 255
            for m in ("visible", "thermal"):
                Image.fromarray(im).save(root / sp / m / f"{i}.jpg", quality=100)
            imgs.append({"id": i, "file_name": f"{i}.jpg", "height": 64, "width": 96})
            anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [6 + 12 * i, 20, 10, 10], "ignore": 0, "iscrowd": 0})
        (root / f"{sp}_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
        build_cache(root, sp, tmp_path / "c")
    return root


def _cfg(tmp_path, root, **kw):
    base = dict(cache_dir=str(tmp_path / "c"), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2, crops_per_epoch=16, batch=8, threads=2,
                crop_hw=(32, 64), size="n", classes=1, prefetch=0, mode_probs=(0, 0, 1), eval_modes=("lwir",), tb=False)
    base.update(kw)
    return TrainConfig(**base)


def test_periodic_checkpoint_is_atomic_and_resume_continues_without_repeating_epochs(tmp_path):
    root = _data(tmp_path)
    train(_cfg(tmp_path, root, epochs=2, ckpt_every=1))
    out = tmp_path / "run"
    assert (out / "resume.pt").exists() and (out / "last.pt").exists() and not list(out.glob("*.tmp"))
    w_before = torch.load(out / "last.pt", weights_only=False)["model"]
    res = train(_cfg(tmp_path, root, epochs=4, ckpt_every=1, resume=True))
    epochs = [json.loads(l)["epoch"] for l in (out / "log.jsonl").read_text().splitlines()]
    assert epochs == [1, 2, 3, 4]                                      # continued from the 3rd, no repeats
    assert [h["epoch"] for h in res["history"]] == [1, 2, 3, 4]
    w_after = torch.load(out / "last.pt", weights_only=False)["model"]
    assert any(not torch.equal(w_before[k], w_after[k]) for k in w_before if w_before[k].dtype.is_floating_point)   # training really continued


def test_resume_of_finished_run_is_noop(tmp_path):
    root = _data(tmp_path)
    train(_cfg(tmp_path, root, epochs=2, ckpt_every=1))
    w1 = torch.load(tmp_path / "run" / "last.pt", weights_only=False)["model"]
    train(_cfg(tmp_path, root, epochs=2, ckpt_every=1, resume=True))
    w2 = torch.load(tmp_path / "run" / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


def test_best_checkpoint_follows_validation_metric(tmp_path):
    root = _data(tmp_path)
    train(_cfg(tmp_path, root, epochs=3, eval_every=1, eval_split="val", eval_frames=[0, 1, 2], ckpt_every=1))
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    hist = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    metric = [h["eval_lwir_AP50"] for h in hist]
    assert best["metric"] == max(metric) and best["epoch"] == 1 + int(np.argmax(metric)) and "cfg" in best and "model" in best


def test_tensorboard_events_are_written(tmp_path):
    pytest.importorskip("tensorboard")
    root = _data(tmp_path)
    train(_cfg(tmp_path, root, epochs=2, eval_every=1, eval_split="val", eval_frames=[0, 1], tb=True))
    assert list((tmp_path / "run" / "tb").glob("events.out.tfevents.*"))
