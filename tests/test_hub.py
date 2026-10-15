"""hae.hub and the Hugging Face fallback of model resolution; no network: huggingface_hub is replaced by a fake module."""
import sys
import types

import pytest

from hae import hub


def _fake_hub(monkeypatch, root=None):
    calls = []

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, "cache_dir": cache_dir})
        if root is None:
            return f"/cache/{filename}"
        p = root / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"onnx")
        return str(p)

    m = types.ModuleType("huggingface_hub")
    m.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", m)
    return calls


def test_download_builds_repo_paths(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch)
    monkeypatch.setattr(hub, "HF_ORG", "someone")
    assert hub.download("hae-n-ir").as_posix() == "/cache/hae-n-ir_lwir_int8.onnx"
    hub.download("hae-m-rgb", int8=False, revision="v1.0.0", cache_dir=tmp_path)
    assert calls[1] == {"repo_id": "someone/hae-m-rgb", "filename": "hae-m-rgb_rgb.onnx", "revision": "v1.0.0", "cache_dir": str(tmp_path)}
    assert hub.model_mode("hae-n-ir-ext") == "lwir" and hub.model_mode("hae-n-rgb") == "rgb" and hub.model_mode("hae-irx") == "rgb"


def test_download_without_huggingface_hub_explains_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert not hub.available()
    with pytest.raises(ImportError, match=r'pip install "hae\[hub\]"'):
        hub.download("hae-n-ir")


@pytest.fixture()
def no_local_models(monkeypatch, tmp_path):
    from hae import detector
    monkeypatch.delenv("HAE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(detector, "REPO_MODELS", tmp_path / "no-repo-models")
    return detector


def test_resolve_falls_back_to_hub_for_released_names(no_local_models, monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path / "hf")
    p = no_local_models.resolve_model("hae-s-ir")
    assert p.exists() and p.name == "hae-s-ir_lwir_int8.onnx" and calls[0]["repo_id"].endswith("/hae-s-ir") and calls[0]["filename"] == "hae-s-ir_lwir_int8.onnx"
    with pytest.raises(FileNotFoundError, match="hae-zzz"):
        no_local_models.resolve_model("hae-zzz")                                   # unknown names never go to the network
    assert len(calls) == 1


def test_resolve_prefers_local_models_and_explains_options(no_local_models, monkeypatch, tmp_path):
    local = tmp_path / "models" / "hae-n-rgb"
    local.mkdir(parents=True)
    (local / "hae-n-rgb_rgb_int8.onnx").write_bytes(b"x")
    calls = _fake_hub(monkeypatch)
    assert no_local_models.resolve_model("hae-n-rgb") == tmp_path / "models" / "hae-n-rgb" / "hae-n-rgb_rgb_int8.onnx"   # ./models
    assert no_local_models.list_models() == ["hae-n-rgb"] and not calls
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match=r"hae\[hub\]"):
        no_local_models.resolve_model("hae-s-ir")
