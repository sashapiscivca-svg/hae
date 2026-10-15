"""HaeNet v0: stem (space-to-depth) -> RGB/LWIR gate -> backbone s4/s8/s16 -> top-down neck -> s4 head.

The mode (rgb / lwir / both) is defined by which tensors are passed; an unused stem
is not computed (saves inference on a cheap drone without a thermal camera).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from hae.model.align import Align
from hae.model.blob import BlobBank
from hae.model.blocks import Down, cbr, fuse_model, make_act, make_block, set_act


@dataclass
class HaeConfig:
    block: str = "ghost"
    stem_c: int = 16
    widths: tuple = (24, 32, 48)   # channels at stride 4 / 8 / 16
    depths: tuple = (1, 2, 2)
    neck_c: int = 24
    head_c: int = 24
    stride: int = 4
    ncls: int = 1         # 1 = person+rider together; 3 = person/rider/crowd following the dataset protocol
    aux: bool = False     # auxiliary HALO branch (mask), training only
    tnorm: bool = False   # LWIR input is fed as a z-score using per-frame statistics
    ctx: bool = False     # stride-32 context level: cheaply widens the receptive field for rider/crowd
    blob: bool = False    # fixed DoG bank at the input
    act: str = "relu"    # relu | hardswish
    decoupled: bool = False  # decoupled head: separate 3x3 trunk for box; cls stays on the shared trunk
    fine_stem: bool = False  # stem with stride 2 -> 4: less loss on tiny people at the cost of MACs in the first layer
    head_s2: bool = False    # head at stride 2 (requires fine_stem; single-modality modes only)
    lnorm: bool = False   # extra LWIR channel: local contrast normalization, for tiny blobs
    lnorm_k: int = 31     # window of the lnorm channel; 31 as in all earlier models
    pre_lnorm: int = 0    # LWIR input is the frame after local_contrast_u8(k), 0 = raw frame; changes preprocessing only, not weights
    modebn: bool = False  # separate BN statistics for both/rgb/lwir modes; free at deployment: one file per mode
    align: bool = False   # learned RGB->LWIR alignment; mode both only

    def __post_init__(self):
        if self.head_s2:
            self.fine_stem, self.stride = True, 2

    def scaled(self, width: float = 1.0, depth: float = 1.0) -> "HaeConfig":
        """Width/depth scaling; channels are multiples of 8 so ghost/SIMD work without waste."""
        def r8(c):
            return max(8, int(round(c * width / 8)) * 8)
        return HaeConfig(block=self.block, stem_c=r8(self.stem_c), widths=tuple(r8(c) for c in self.widths),
                         depths=tuple(max(1, int(round(d * depth))) for d in self.depths),
                         neck_c=r8(self.neck_c), head_c=r8(self.head_c), stride=self.stride, ncls=self.ncls, aux=self.aux, tnorm=self.tnorm, ctx=self.ctx, blob=self.blob, align=self.align, modebn=self.modebn, lnorm=self.lnorm, lnorm_k=self.lnorm_k, pre_lnorm=self.pre_lnorm, fine_stem=self.fine_stem, head_s2=self.head_s2, decoupled=self.decoupled, act=self.act)


class Stem(nn.Module):
    """space-to-depth(2) with no MACs, then a single 3x3 stride 2 -> total stride 4."""
    def __init__(self, cin: int, c: int, fine: bool = False):
        super().__init__()
        self.fine = fine
        if fine:   # 3x3 stride 1 at stride 2, then 3x3 stride 2 -> stride 4; the stride-2 map is passed to the head
            self.net1, self.net2 = cbr(cin * 4, c, 3, 1), cbr(c, c, 3, 2)
        else:
            self.net = cbr(cin * 4, c, 3, 2)

    def forward(self, x, with_s2: bool = False):
        z = F.pixel_unshuffle(x, 2)
        if not self.fine:
            return self.net(z)
        f2 = self.net1(z)
        f4 = self.net2(f2)
        return (f4, f2) if with_s2 else f4


def local_norm(x: torch.Tensor, k: int = 31, floor: float = 0.25) -> torch.Tensor:
    """(x - local mean) / max(local sigma, floor): removes slow background, keeps small warm blobs.
    floor avoids amplifying noise in flat areas (x is already in per-frame z units). Only avg_pool/arithmetic -> clean ONNX."""
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    m = F.avg_pool2d(xp, k, 1)
    v = (F.avg_pool2d(xp * xp, k, 1) - m * m).clamp_min(0)
    return (x - m) / v.sqrt().clamp_min(floor)


MODE_INDEX = {"both": 0, "rgb": 1, "lwir": 2}


class ModeBN(nn.Module):
    """Three BatchNorms (both/rgb/lwir) with shared convolutions: features of different modes have different distributions,
    and shared BN statistics destroyed the rgb mode (AP50~0). The active index is the model's shared `state[0]`."""
    def __init__(self, bn: nn.BatchNorm2d, state: list):
        super().__init__()
        self.bns = nn.ModuleList([copy.deepcopy(bn) for _ in MODE_INDEX])
        self.state = state

    def forward(self, x):
        return self.bns[self.state[0]](x)


def _swap_bn(mod: nn.Module, state: list) -> None:
    for name, ch in mod.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            setattr(mod, name, ModeBN(ch, state))
        else:
            _swap_bn(ch, state)


def _collapse_bn(mod: nn.Module, idx: int) -> None:
    for name, ch in mod.named_children():
        if isinstance(ch, ModeBN):
            setattr(mod, name, ch.bns[idx])
        else:
            _collapse_bn(ch, idx)


class Gate(nn.Module):
    """Per-pixel modality trust gate."""
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(2 * c, c, 1)

    def forward(self, fr, ft):
        g = torch.sigmoid(self.conv(torch.cat([fr, ft], 1)))
        return g * fr + (1 - g) * ft


class Context(nn.Module):
    """Dilated 3x3 (dilation 2) for context at stride 16."""
    def __init__(self, c: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(c), make_act(inplace=True))

    def forward(self, x):
        return x + self.f(x)


def _stage(kind, c, n):
    return nn.Sequential(*[make_block(kind, c) for _ in range(n)])


def attach_neck_head(m: nn.Module, chans, neck_c: int, head_c: int, ncls: int, decoupled: bool = False) -> None:
    """Top-down neck (s16->s8->s4) and a single-level stride-4 head."""
    c4, c8, c16 = chans
    m.lat16, m.lat8, m.lat4 = nn.Conv2d(c16, neck_c, 1), nn.Conv2d(c8, neck_c, 1), nn.Conv2d(c4, neck_c, 1)
    m.fuse8, m.fuse4 = cbr(neck_c, neck_c, 3), cbr(neck_c, neck_c, 3)
    m.trunk = cbr(neck_c, head_c, 3)
    if decoupled:
        m.trunk_box = cbr(neck_c, head_c, 3)
    m.cls = nn.Conv2d(head_c, ncls, 1)
    m.box = nn.Conv2d(head_c, 4, 1)
    nn.init.constant_(m.cls.bias, -math.log(99.0))   # prior probability ~0.01
    nn.init.constant_(m.box.bias, 0.9)               # softplus(0.9)~1.25 -> box ~5 px per side at stride 4


def neck_head_forward(m: nn.Module, c4, c8, c16, c2=None) -> dict:
    p8 = m.fuse8(m.lat8(c8) + F.interpolate(m.lat16(c16), scale_factor=2, mode="nearest"))
    p4 = m.fuse4(m.lat4(c4) + F.interpolate(p8, scale_factor=2, mode="nearest"))
    feat = p4
    if c2 is not None:   # head at stride 2: thin stem branch + upsampled p4
        feat = m.fuse2(m.lat2(c2) + F.interpolate(p4, scale_factor=2, mode="nearest"))
    x = m.trunk(feat)
    xb = m.trunk_box(feat) if hasattr(m, "trunk_box") else x   # decoupled head
    out = {"cls": m.cls(x), "box": m.box(xb)}
    if getattr(m, "aux", None) is not None and m.training:
        out["aux"] = m.aux(p4)          # auxiliary mask, training only; absent at runtime and in ONNX
    return out


class HaeNet(nn.Module):
    def __init__(self, cfg: HaeConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or HaeConfig()
        prev = set_act(cfg.act)
        try:
            self._build(cfg)
        finally:
            set_act(prev)

    def _build(self, cfg: HaeConfig) -> None:
        w0, w1, w2 = cfg.widths
        self.blob = BlobBank() if cfg.blob else None
        nb = len(self.blob.scales) if cfg.blob else 0
        self.stem_rgb, self.stem_t = Stem(3 + nb, cfg.stem_c, cfg.fine_stem), Stem(1 + nb + int(cfg.lnorm), cfg.stem_c, cfg.fine_stem)
        self.gate = Gate(cfg.stem_c)
        self.align = Align(cfg.stem_c, cfg.stride) if cfg.align else None
        self.s4 = nn.Sequential(cbr(cfg.stem_c, w0, 1), _stage(cfg.block, w0, cfg.depths[0]))
        self.s8 = nn.Sequential(Down(w0, w1), _stage(cfg.block, w1, cfg.depths[1]))
        self.s16 = nn.Sequential(Down(w1, w2), _stage(cfg.block, w2, cfg.depths[2]), Context(w2))
        self.s32 = nn.Sequential(Down(w2, w2), _stage(cfg.block, w2, 1), Context(w2)) if cfg.ctx else None
        self.up32 = nn.Conv2d(w2, w2, 1) if cfg.ctx else None
        self.aux = nn.Conv2d(cfg.neck_c, 1, 1) if cfg.aux else None
        attach_neck_head(self, (w0, w1, w2), cfg.neck_c, cfg.head_c, cfg.ncls, cfg.decoupled)
        if cfg.head_s2:
            self.lat2, self.fuse2 = nn.Conv2d(cfg.stem_c, cfg.neck_c, 1), cbr(cfg.neck_c, cfg.neck_c, 3)
        self._state = [0]
        if cfg.modebn:
            _swap_bn(self, self._state)

    def _stem_out(self, stem, x):
        if self.cfg.head_s2:
            return stem(x, with_s2=True)
        return stem(x), None

    def for_mode(self, mode: str) -> "HaeNet":
        """Copy with ONE plain BN in place of every ModeBN - for exporting/fusing a specific mode."""
        m = copy.deepcopy(self)
        _collapse_bn(m, MODE_INDEX[mode])
        m.cfg = replace(m.cfg, modebn=False)
        return m

    def forward(self, rgb: torch.Tensor | None = None, t: torch.Tensor | None = None):
        if rgb is None and t is None:
            raise ValueError("At least one input is required: rgb or t")
        self._state[0] = 0 if (rgb is not None and t is not None) else 1 if rgb is not None else 2
        ref = rgb if rgb is not None else t
        h, w = ref.shape[-2:]
        if h % 16 or w % 16:
            raise ValueError(f"Input size must be a multiple of 16, got {h}x{w}")
        if self.s32 is not None and (h % 32 or w % 32):
            raise ValueError(f"With the context level (ctx) the input size must be a multiple of 32, got {h}x{w}")
        ln = local_norm(t, self.cfg.lnorm_k) if (self.cfg.lnorm and t is not None) else None
        if self.blob is not None:   # extra channels: blob responses at 3 scales (from luminance)
            rgb = None if rgb is None else torch.cat([rgb, self.blob(rgb)], 1)
            t = None if t is None else torch.cat([t, self.blob(t)], 1)
        if ln is not None:
            t = torch.cat([t, ln], 1)
        flow, c2 = None, None
        if self.cfg.head_s2 and rgb is not None and t is not None:
            raise ValueError("The stride-2 head supports single-modality modes only (rgb or lwir)")
        if rgb is not None and t is not None:
            fr, ft = self.stem_rgb(rgb), self.stem_t(t)
            if self.align is not None:
                fr, flow = self.align(fr, ft)
            f = self.gate(fr, ft)
        elif rgb is not None:
            f, c2 = self._stem_out(self.stem_rgb, rgb)
        else:
            f, c2 = self._stem_out(self.stem_t, t)
        c4 = self.s4(f)
        c8 = self.s8(c4)
        c16 = self.s16(c8)
        if self.s32 is not None:
            c16 = c16 + self.up32(F.interpolate(self.s32(c16), scale_factor=2, mode="nearest"))
        out = neck_head_forward(self, c4, c8, c16, c2)
        if flow is not None:
            out["flow"] = flow
        return out

    def fuse(self) -> None:
        fuse_model(self)
