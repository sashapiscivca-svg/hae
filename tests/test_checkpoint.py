"""Released weights (.safetensors + config.json) reproduce the released fp32 ONNX graphs."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hae.checkpoint import load_checkpoint
from hae.train import TrainConfig

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("safetensors")
MODELS = sorted(p for p in (Path(__file__).resolve().parents[1] / "models").glob("hae-*") if (p / "config.json").exists())


@pytest.mark.skipif(not MODELS, reason="no released models in models/")
@pytest.mark.parametrize("d", MODELS, ids=lambda p: p.name)
def test_released_safetensors_match_fp32_onnx(d):
    conf = json.loads((d / "config.json").read_text())
    TrainConfig(**conf["training"])   # the sanitized training config loads as is
    model, cfg = load_checkpoint(d / f"{d.name}.safetensors")
    assert cfg == conf["training"]
    mode = conf.get("mode") or ("lwir" if d.name.endswith("-ir") else "rgb")
    m = model.for_mode(mode).eval()
    m.fuse()   # as hae.export does before tracing
    x = np.random.default_rng(0).random((1, 1 if mode == "lwir" else 3, 256, 320), dtype=np.float32)
    key = "t" if mode == "lwir" else "rgb"
    with torch.no_grad():
        o = m(**{key: torch.from_numpy(x)})
    s = ort.InferenceSession(str(d / f"{d.name}_{mode}.onnx"), providers=["CPUExecutionProvider"])
    cls, box = s.run(None, {key: x})
    assert np.abs(cls - o["cls"].numpy()).max() < 1e-3 and np.abs(box - o["box"].numpy()).max() < 1e-3
