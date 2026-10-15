"""Learned cross-modal RGB->LWIR alignment.

In RGBTDronePerson the RGB image is shifted relative to the thermal one by ~13 px (median; p90 ~35 px) for targets of ~11 px.
The flow regressor works on a stride-16 grid with a large receptive field (dilation); the flow is upsampled to stride 4
and shifts the RGB features via grid_sample (exportable to ONNX). Training: detection loss + a self-supervised
NCC loss between edge maps (their structure is shared by RGB and thermal) + flow smoothness.
Zero initialization: at the start the module is the identity, so baseline behaviour is not harmed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(feat: torch.Tensor, flow_cells: torch.Tensor) -> torch.Tensor:
    """out(x) = feat(x + flow); flow in feature-map cells, shape (B,2,H,W): channel 0 - dx, 1 - dy."""
    b, _, h, w = feat.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=feat.device, dtype=feat.dtype),
                            torch.arange(w, device=feat.device, dtype=feat.dtype), indexing="ij")
    gx, gy = xs[None] + flow_cells[:, 0], ys[None] + flow_cells[:, 1]
    grid = torch.stack([2 * gx / max(w - 1, 1) - 1, 2 * gy / max(h - 1, 1) - 1], -1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)


class Align(nn.Module):
    def __init__(self, c: int, stride: int = 4, max_shift_px: float = 48.0):
        super().__init__()
        self.stride, self.max_shift = stride, max_shift_px
        self.net = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(c, 2, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f_rgb: torch.Tensor, f_t: torch.Tensor):
        x = F.avg_pool2d(torch.cat([f_rgb, f_t], 1), 4)                      # stride-16 grid
        flow_px = torch.tanh(self.net(x)) * self.max_shift                    # input pixel shifts
        flow_px = F.interpolate(flow_px, size=f_rgb.shape[-2:], mode="bilinear", align_corners=False)
        return warp(f_rgb, flow_px / self.stride), flow_px


def edge_map(x: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """Edge map on the stride grid; normalized per sample (zero mean, unit sigma)."""
    g = x if x.shape[1] == 1 else (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    e = F.avg_pool2d(torch.sqrt(gx * gx + gy * gy + 1e-8), stride)
    e = e - e.mean(dim=(1, 2, 3), keepdim=True)
    return e / (e.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


def edge_alignment_loss(rgb: torch.Tensor, t: torch.Tensor, flow_px: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """1 - NCC(shifted RGB edge map, thermal edge map), averaged over the batch."""
    w = warp(edge_map(rgb, stride), flow_px / stride)
    e = edge_map(t, stride)
    w = w - w.mean(dim=(1, 2, 3), keepdim=True)
    ncc = (w * e).sum(dim=(1, 2, 3)) / (w.flatten(1).norm(dim=1) * e.flatten(1).norm(dim=1) + 1e-6)
    return (1 - ncc).mean()


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    return (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean() + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
