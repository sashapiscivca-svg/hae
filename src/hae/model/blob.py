"""Fixed DoG (difference of Gaussians) bank that amplifies "blobs".

Tiny people (sqrt(area) 7-18 px, median 11) are blobs of radius ~2.5-6 px. An early Laplacian/DoG response at the
right scales is the LF-DETR trick (SOTA 50.9 on RGBTDronePerson). Here it is a cheap, fixed (parameter-free)
input that a small, capacity-limited network gets for free. For RGB the luminance is used.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RATIO = 1.6   # sigma_large / sigma_small for DoG ~ LoG


def _gauss1d(sigma: float) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


class BlobBank(nn.Module):
    def __init__(self, scales=(1.5, 2.5), gain: float = 8.0):
        super().__init__()
        self.scales, self.gain = tuple(scales), gain
        sigmas = sorted({s for sc in self.scales for s in (sc, sc * RATIO)})
        self.sigmas = sigmas
        for i, s in enumerate(sigmas):
            self.register_buffer(f"k{i}", _gauss1d(s), persistent=False)

    def _blur(self, g: torch.Tensor, i: int) -> torch.Tensor:
        k = getattr(self, f"k{i}")
        r = (k.numel() - 1) // 2
        g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(g, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x if x.shape[1] == 1 else (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        blurred = {s: self._blur(g, i) for i, s in enumerate(self.sigmas)}
        return torch.cat([self.gain * (blurred[sc] - blurred[sc * RATIO]) for sc in self.scales], 1)
