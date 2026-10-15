"""Loading hae models for PyTorch: training checkpoints (`.pt`) and released weights (`.safetensors` + `config.json`).

The model is always built from our own config and our own weights (no third-party weights).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from hae.train import TrainConfig, build_model


def _check_architecture(model, arch: dict | None, path: Path) -> None:
    """The released config.json also records the resolved HaeConfig; a mismatch means the config and weights disagree."""
    if not arch:
        return
    got = asdict(model.cfg)
    diff = {k: (v, got.get(k)) for k, v in arch.items() if k in got and _norm(got[k]) != _norm(v)}
    if diff:
        raise ValueError(f"{path}: architecture in config.json does not match the model built from its training config: {diff}")


def _norm(v):
    return list(v) if isinstance(v, tuple) else v


def load_checkpoint(path: str | Path):
    """(model.eval(), training config dict) from a training checkpoint `.pt` (keys `model`, `cfg`) or from a released
    `<name>.safetensors` with a sibling `config.json` (`training` = TrainConfig fields, `architecture` = HaeConfig fields)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        conf_path = path.with_name("config.json")
        if not conf_path.exists():
            raise FileNotFoundError(f"{conf_path} not found (needed next to {path.name})")
        conf = json.loads(conf_path.read_text())
        cfg = conf["training"]
        model = build_model(TrainConfig.from_dict(cfg))
        _check_architecture(model, conf.get("architecture"), path)
        model.load_state_dict(load_file(str(path)))
        return model.eval(), cfg
    ck = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = build_model(TrainConfig.from_dict(cfg))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg


def save_safetensors(model, cfg: dict, out_dir: str | Path, name: str, extra: dict | None = None) -> Path:
    """Write `<name>.safetensors` (state_dict of the training model, unfused) + `config.json` in the released layout."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.safetensors"
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(p))
    drop = ("cache_dir", "ann_root", "out_dir", "eval_frames", "threads", "device", "verbose", "prefetch", "resume", "tb", "eval_split", "train_set", "init_weights")
    training = {k: v for k, v in cfg.items() if k not in drop}
    if training.get("extra"):   # local annotation paths are not part of the recipe
        training["extra"] = [{"name": e["name"], "weight": e["weight"]} for e in training["extra"]]
    conf = {"name": name, "architecture": asdict(model.cfg), "training": training, **(extra or {})}
    (out / "config.json").write_text(json.dumps(conf, indent=1, default=list))
    return p
