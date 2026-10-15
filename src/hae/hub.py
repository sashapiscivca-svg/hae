"""Download released hae models from the Hugging Face Hub (optional dependency: `pip install "hae[hub]"`).

One Hub repository per model (`<org>/<name>`, e.g. alexshynkarenk0/hae-s-ir) with files at the repository root: `<name>_<mode>.onnx`,
`<name>_<mode>_int8.onnx`, `<name>.safetensors`, `config.json`, `metrics.json`. Mode is `lwir` for thermal models (`hae-*-ir`)
and `rgb` otherwise. Files are cached by `huggingface_hub` (see `cache_dir`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Default Hugging Face namespace of the released models; override with $HAE_HF_ORG.
HF_ORG = os.environ.get("HAE_HF_ORG", "alexshynkarenk0")
RELEASED_MODELS = ("hae-s-ir",)   # models published on the Hub; `hae download --all` fetches these


def model_mode(name: str) -> str:
    """Input mode from the model name: `-ir` (also with suffixes such as `hae-n-ir-ext`) -> lwir, anything else -> rgb."""
    return "lwir" if re.search(r"-ir(-|$)", str(name)) else "rgb"


def onnx_name(name: str, int8: bool = True) -> str:
    return f"{name}_{model_mode(name)}{'_int8' if int8 else ''}.onnx"


def model_filename(name: str, int8: bool = True) -> str:
    """Path of the ONNX file inside a local model directory: `<name>/<name>_<mode>[_int8].onnx`."""
    return f"{name}/{onnx_name(name, int8)}"


def repo_id(name: str) -> str:
    return f"{HF_ORG}/{name}"


def available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def download(name: str, int8: bool = True, revision: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """Download the ONNX file of one released model (int8 by default, fp32 with int8=False) and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError('downloading models needs huggingface_hub: pip install "hae[hub]"') from e
    p = hf_hub_download(repo_id=repo_id(name), filename=onnx_name(name, int8), revision=revision,
                        cache_dir=str(cache_dir) if cache_dir is not None else None)
    return Path(p)
