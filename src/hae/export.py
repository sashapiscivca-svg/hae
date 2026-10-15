"""Export hae to ONNX for low-end hardware (e.g. Raspberry Pi Zero 2 W with onnxruntime).

One file per input mode: both (rgb + t), rgb, lwir; height/width are dynamic (multiples of 16/32 depending on the model config),
so one file works both on a 320x256 SAHI tile and on a full 640x512 frame. Outputs are raw: cls (logits) and box
(raw l,t,r,b); decoding (3x3 peaks + softplus * stride) is done by the application (`hae.decode_np` or `hae.model.decode`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn


class _Wrap(nn.Module):
    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        self.m, self.mode = model, mode

    def forward(self, *x):
        out = self.m(rgb=x[0], t=x[1]) if self.mode == "both" else self.m(rgb=x[0]) if self.mode == "rgb" else self.m(t=x[0])
        return out["cls"], out["box"]


def export_onnx(model: nn.Module, path: str | Path, mode: str = "both", hw=(256, 320), opset: int = 17) -> dict:
    if mode not in ("both", "rgb", "lwir"):
        raise ValueError(f"mode must be both/rgb/lwir, got {mode!r}")
    m = (model.for_mode(mode) if hasattr(model, "for_mode") else copy.deepcopy(model)).eval()
    if hasattr(m, "fuse"):
        m.fuse()                                    # fuses RepBlocks (if any) into a single 3x3 conv
    h, w = hw
    specs = {"both": [("rgb", 3), ("t", 1)], "rgb": [("rgb", 3)], "lwir": [("t", 1)]}[mode]
    dummy = tuple(torch.rand(1, c, h, w) for _, c in specs)
    names = [n for n, _ in specs]
    path = Path(path)
    torch.onnx.export(_Wrap(m, mode), dummy, str(path), input_names=names, output_names=["cls", "box"], opset_version=opset,
                      dynamic_axes={**{n: {0: "b", 2: "h", 3: "w"} for n in names}, "cls": {0: "b", 2: "h4", 3: "w4"},
                                    "box": {0: "b", 2: "h4", 3: "w4"}}, dynamo=False)
    from hae.infer import default_metadata, embed_metadata
    cfg = getattr(m, "cfg", None)
    embed_metadata(path, default_metadata("rgb" if mode == "rgb" else "lwir" if mode == "lwir" else "both", getattr(cfg, "ncls", 1), bool(getattr(cfg, "tnorm", False)), getattr(cfg, "stride", 4),
                                          int(getattr(cfg, "pre_lnorm", 0) or 0)))   # pre_lnorm key only for such models
    return {"path": str(path), "mode": mode, "bytes": path.stat().st_size, "params": sum(p.numel() for p in m.parameters())}
