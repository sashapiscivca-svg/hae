"""Every released int8 model in models/ loads through Detector and runs on a synthetic 512x640 frame."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from hae.detector import Detector

RELEASED = sorted((Path(__file__).resolve().parents[1] / "models").glob("*/*_int8.onnx"))


@pytest.mark.skipif(not RELEASED, reason="released models directory is missing")
@pytest.mark.parametrize("path", RELEASED, ids=lambda p: p.parent.name)
def test_released_model_runs_on_synthetic_frame(path):
    det = Detector(path)
    for k in ("mode", "recommended_scale", "recommended_conf", "license"):
        assert k in det.meta
    r = np.random.default_rng(0)
    frame = r.integers(40, 90, (512, 640), np.uint8)
    frame[200:216, 300:308] = 230                       # warm spot
    out = det.detect(np.stack([frame] * 3, -1) if det.mode == "rgb" else frame)
    assert out.ndim == 2 and out.shape[1] == 6 and det.meta["license"] == "MIT"
