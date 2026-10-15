import pytest

from hae.detector import resolve_model


@pytest.mark.parametrize("name,mode", [("hae-n-ir", "lwir"), ("hae-n-ir-ext", "lwir"), ("hae-n-ir-ext-kdf", "lwir"), ("hae-n-rgb", "rgb"), ("hae-s-rgb-ext", "rgb")])
def test_resolve_model_infers_mode_from_name_with_suffix(tmp_path, name, mode):
    d = tmp_path / name
    d.mkdir()
    (d / f"{name}_{mode}_int8.onnx").write_bytes(b"x")
    (d / f"{name}_{mode}.onnx").write_bytes(b"x")
    assert resolve_model(name, tmp_path).name == f"{name}_{mode}_int8.onnx"
    assert resolve_model(name, tmp_path, int8=False).name == f"{name}_{mode}.onnx"
