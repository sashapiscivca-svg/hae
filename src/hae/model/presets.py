"""The hae-n/t/s/m/l/x lineup: one architecture design, six scales.

n, t, s fit the Pi Zero 2 W budget (<= 0.3 GMAC per 320x256 tile); m, l, x are for stronger hardware
and for chasing accuracy. `t` is the proven v0 baseline.
"""
from __future__ import annotations

from dataclasses import replace

from hae.model.hae import HaeConfig, HaeNet

SIZES = ("n", "t", "s", "m", "l", "x")

# size -> (width, depth, block); params/GMAC were measured per size
_PRESETS = {
    "n": (0.67, 1.0, "ghost"),
    "t": (1.0, 1.0, "ghost"),
    "s": (1.67, 1.0, "ghost"),
    "m": (2.67, 2.0, "ghost"),
    "l": (5.0, 3.0, "ghost"),
    "x": (8.0, 4.0, "ghost"),
}


def preset_config(size: str) -> HaeConfig:
    if size not in _PRESETS:
        raise ValueError(f"unknown size {size!r}; available: {', '.join(SIZES)}")
    w, d, blk = _PRESETS[size]
    return HaeConfig(block=blk).scaled(w, d)


def build(size: str | None = None, block: str = "ghost", width: float = 1.0, depth: float = 1.0,
          align: bool = False, ncls: int = 1, blob: bool = False, overrides: dict | None = None, ctx: bool = False, tnorm: bool = False, aux: bool = False) -> HaeNet:
    """size given -> preset; otherwise an arbitrary scale (width/depth/block) for ablations."""
    cfg = preset_config(size) if size else HaeConfig(block=block).scaled(width, depth)
    if align or blob or ctx or tnorm or aux or ncls != 1:
        cfg = replace(cfg, align=align, ncls=ncls, blob=blob, ctx=ctx, tnorm=tnorm, aux=aux)
    if overrides:   # arbitrary HaeConfig fields for ablations (lists from JSON -> tuples)
        bad = set(overrides) - set(cfg.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown HaeConfig fields: {sorted(bad)}")
        cfg = replace(cfg, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})
    return HaeNet(cfg)
