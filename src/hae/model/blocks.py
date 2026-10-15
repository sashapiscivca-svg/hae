"""Backbone block candidates. The final choice is made by measured speed."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = ["relu"]   # ReLU: fastest on CPU/ARM and simplest for int8; hardswish is an option, set while building the model


def make_act(inplace: bool = False) -> nn.Module:
    return nn.Hardswish(inplace=inplace) if _ACT[0] == "hardswish" else nn.ReLU(inplace=inplace)


def set_act(name: str) -> str:
    """Returns the previous name: call as a pair set_act(new) ... set_act(previous) in try/finally."""
    if name not in ("relu", "hardswish"):
        raise ValueError(f"act must be relu or hardswish, got {name!r}")
    prev, _ACT[0] = _ACT[0], name
    return prev


def cbr(cin: int, cout: int, k: int = 3, s: int = 1, g: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(cout), make_act(inplace=True))


class DenseBlock(nn.Module):
    """x + ReLU(BN(conv3x3(x))): the simplest one, maps well onto oneDNN."""
    def __init__(self, c: int):
        super().__init__()
        self.f = cbr(c, c, 3)

    def forward(self, x):
        return x + self.f(x)


class DWBlock(nn.Module):
    """Inverted bottleneck: pw(×2) → dw3x3 → pw."""
    def __init__(self, c: int):
        super().__init__()
        self.f = nn.Sequential(cbr(c, 2 * c, 1), cbr(2 * c, 2 * c, 3, g=2 * c),
                               nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.f(x)


class GhostBlock(nn.Module):
    """Half of the channels via 1x1, the other half via a cheap dw3x3 of the first half."""
    def __init__(self, c: int):
        super().__init__()
        assert c % 2 == 0
        self.p = cbr(c, c // 2, 1)
        self.q = cbr(c // 2, c // 2, 3, g=c // 2)

    def forward(self, x):
        a = self.p(x)
        return x + torch.cat([a, self.q(a)], 1)


def _fuse_conv_bn(w: torch.Tensor, bn: nn.BatchNorm2d):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    return w * t.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * t


class RepBlock(nn.Module):
    """Training: 3x3+BN, 1x1+BN, identity+BN. Inference: a single 3x3 (after fuse)."""
    def __init__(self, c: int):
        super().__init__()
        self.c3, self.b3 = nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c)
        self.c1, self.b1 = nn.Conv2d(c, c, 1, 1, 0, bias=False), nn.BatchNorm2d(c)
        self.bi = nn.BatchNorm2d(c)
        self.act = make_act(inplace=False)
        self.fused: nn.Conv2d | None = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        return self.act(self.b3(self.c3(x)) + self.b1(self.c1(x)) + self.bi(x))

    @torch.no_grad()
    def fuse(self):
        c = self.c3.in_channels
        w3, b3 = _fuse_conv_bn(self.c3.weight, self.b3)
        w1, b1 = _fuse_conv_bn(self.c1.weight, self.b1)
        wid = torch.zeros_like(self.c3.weight)
        wid[torch.arange(c), torch.arange(c), 1, 1] = 1.0
        wi, bi = _fuse_conv_bn(wid, self.bi)
        conv = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        conv.weight.copy_(w3 + F.pad(w1, [1, 1, 1, 1]) + wi)
        conv.bias.copy_(b3 + b1 + bi)
        self.fused = conv


class RepDW(nn.Module):
    """Depthwise 3x3 + 1x1 + identity (each with BN) in training; a single depthwise 3x3 after fuse - zero extra MACs (RepGhost approach)."""
    def __init__(self, c: int):
        super().__init__()
        self.c3, self.b3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False), nn.BatchNorm2d(c)
        self.c1, self.b1 = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False), nn.BatchNorm2d(c)
        self.bi = nn.BatchNorm2d(c)
        self.act = make_act(inplace=False)
        self.fused: nn.Conv2d | None = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        return self.act(self.b3(self.c3(x)) + self.b1(self.c1(x)) + self.bi(x))

    @torch.no_grad()
    def fuse(self):
        c = self.c3.in_channels
        w3, b3 = _fuse_conv_bn(self.c3.weight, self.b3)
        w1, b1 = _fuse_conv_bn(self.c1.weight, self.b1)
        wid = torch.zeros_like(self.c3.weight)
        wid[:, 0, 1, 1] = 1.0
        wi, bi = _fuse_conv_bn(wid, self.bi)
        conv = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True)
        conv.weight.copy_(w3 + F.pad(w1, [1, 1, 1, 1]) + wi)
        conv.bias.copy_(b3 + b1 + bi)
        self.fused = conv


class RepGhostBlock(nn.Module):
    """Ghost block whose "cheap" operation is RepDW: more capacity in training without extra MACs at runtime."""
    def __init__(self, c: int):
        super().__init__()
        assert c % 2 == 0
        self.p = cbr(c, c // 2, 1)
        self.q = RepDW(c // 2)

    def forward(self, x):
        a = self.p(x)
        return x + torch.cat([a, self.q(a)], 1)


BLOCKS = {"dense": DenseBlock, "dw": DWBlock, "ghost": GhostBlock, "rep": RepBlock, "repghost": RepGhostBlock}


def make_block(kind: str, c: int) -> nn.Module:
    return BLOCKS[kind](c)


class Down(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(cbr(cin, cout, 3, 2))


def fuse_model(m: nn.Module) -> None:
    """Call after m.eval(): merges every RepBlock into a single 3x3."""
    for mod in m.modules():
        if isinstance(mod, (RepBlock, RepDW)):
            mod.fuse()
