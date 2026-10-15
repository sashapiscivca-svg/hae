import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
from hae.export import export_onnx
from hae.model.presets import build

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(*args, cwd=None, env=None):
    return subprocess.run([sys.executable, "-m", "hae.cli", *args], capture_output=True, text=True, cwd=cwd, env={"PYTHONPATH": SRC, "PATH": "", **(env or {})})


@pytest.fixture()
def models(tmp_path):
    root = tmp_path / "models" / "hae-n-ir"
    root.mkdir(parents=True)
    export_onnx(build("n", ncls=3, tnorm=True).eval(), root / "hae-n-ir_lwir_int8.onnx", "lwir", (64, 96))
    return tmp_path / "models"


def _imgs(d, n=2):
    d.mkdir()
    for k in range(n):
        Image.fromarray(np.random.default_rng(k).integers(0, 255, (64, 96), np.uint8)).save(d / f"{k}.png")
    return d


def test_models_info_and_predict_by_name_on_folder_saves_results(tmp_path, models):
    env = {"HAE_MODELS": str(models)}
    assert "hae-n-ir" in _run("models", env=env).stdout
    info = _run("info", "hae-n-ir", env=env)
    assert info.returncode == 0 and "lwir" in info.stdout and "person" in info.stdout
    src = _imgs(tmp_path / "imgs")
    r = _run("predict", "hae-n-ir", str(src), "--conf", "0.0", "--json", env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    exp = tmp_path / "runs" / "predict" / "exp"
    assert sorted(p.name for p in exp.glob("*.png")) == ["0.png", "1.png"] and (exp / "results.json").exists()
    assert set(json.loads((exp / "results.json").read_text())) == {"0.png", "1.png"}
    r2 = _run("predict", "hae-n-ir", str(src), "--conf", "0.0", env=env, cwd=tmp_path)     # a second run -> exp2, not an overwrite
    assert (tmp_path / "runs" / "predict" / "exp2").exists() and r2.returncode == 0


def test_clear_errors(tmp_path, models):
    env = {"HAE_MODELS": str(models)}
    r = _run("predict", "hae-zzz", str(tmp_path), env=env)
    assert r.returncode != 0 and "hae-n-ir" in (r.stderr + r.stdout)                         # suggests the available models
    r = _run("predict", "hae-n-ir", str(tmp_path / "missing.png"), env=env)
    assert r.returncode != 0 and "not found" in (r.stderr + r.stdout)


def test_download_command_uses_hub(monkeypatch, capsys):
    import types
    from hae import cli
    from hae import hub
    monkeypatch.setattr(hub, "HF_ORG", "sashapiscivca-svg")
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = lambda repo_id, filename, revision=None, cache_dir=None: calls.append((repo_id, filename, revision)) or f"/cache/{filename}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert cli.main(["download", "hae-n-ir", "hae-s-rgb", "--fp32"]) == 0
    assert calls == [("sashapiscivca-svg/hae-n-ir", "hae-n-ir_lwir.onnx", None), ("sashapiscivca-svg/hae-s-rgb", "hae-s-rgb_rgb.onnx", None)]
    calls.clear()
    assert cli.main(["download", "--all"]) == 0 and len(calls) == 1 and all(c[1].endswith("_int8.onnx") for c in calls)
    assert "hae-n-ir" in capsys.readouterr().out


def test_download_without_huggingface_hub_is_a_clear_error(monkeypatch):
    from hae import cli
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit, match="hae\\[hub\\]"):
        cli.main(["download", "hae-n-ir"])


def test_export_help_lists_checkpoint_formats():
    r = _run("export", "--help")
    assert r.returncode == 0 and "safetensors" in r.stdout


def test_export_released_safetensors_to_onnx(tmp_path):
    pytest.importorskip("hae.checkpoint")
    st = Path(__file__).resolve().parents[1] / "models" / "hae-s-ir" / "hae-s-ir.safetensors"
    if not st.exists():
        pytest.skip("released models directory is missing")
    from hae import cli
    from hae.infer import read_metadata
    assert cli.main(["export", str(st), str(tmp_path), "--name", "hae-s-ir", "--mode", "lwir", "--hw", "64,96"]) == 0
    meta = read_metadata(tmp_path / "hae-s-ir_lwir.onnx")
    assert meta["mode"] == "lwir" and meta["license"] == "MIT"


def test_runtime_commands_work_without_rich(tmp_path, models):
    code = ("import sys; sys.modules['rich'] = None\n"            # makes `import rich` fail as if it were not installed
            "from hae.cli import main\n"
            "assert main(['models']) == 0 and main(['info', 'hae-n-ir']) == 0\n"
            f"assert main(['predict', 'hae-n-ir', {str(_imgs(tmp_path / 'imgs'))!r}, '--conf', '0.0']) == 0\n"
            "from hae import ui; assert not ui.HAVE_RICH")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env={"PYTHONPATH": SRC, "PATH": "", "HAE_MODELS": str(models)})
    assert r.returncode == 0, r.stderr
    assert "hae-n-ir" in r.stdout and "lwir" in r.stdout and "total detections" in r.stdout and "conf (balanced)" in r.stdout
