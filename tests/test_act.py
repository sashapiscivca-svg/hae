import torch.nn as nn

from hae.export import export_onnx
from hae.model.blocks import make_act
from hae.model.presets import build


def test_default_is_relu_and_hardswish_option_changes_every_activation():
    assert not any(isinstance(m, nn.Hardswish) for m in build("n").modules())
    hs = build("n", overrides={"act": "hardswish"})
    assert any(isinstance(m, nn.Hardswish) for m in hs.modules())
    assert not any(isinstance(m, nn.ReLU) for m in hs.modules())


def test_global_activation_is_restored_after_build():
    build("n", overrides={"act": "hardswish"})
    assert isinstance(make_act(), nn.ReLU)


def test_hardswish_model_exports(tmp_path):
    m = build("n", block="repghost", overrides={"act": "hardswish"}).eval()
    assert export_onnx(m, tmp_path / "hs.onnx", mode="lwir", hw=(64, 96))["bytes"] > 0
