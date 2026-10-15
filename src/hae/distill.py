"""Teacher -> student distillation for the anchor-free stride-4 detector.

Training only: student inference is unchanged. The teacher (a larger hae: s/m, a .pt or .onnx checkpoint) is frozen, in eval mode,
without gradients; it sees the same input upscaled `teacher_scale_ratio` times (cross-scale: x2 for the teacher with x1.5 for the student => 1.333).

Components (each disabled at neutral values; `alpha=0` and `feat=False` => train() is identical to plain training):
  1. cls distillation: QFL against the teacher's soft target s = sigmoid(z_t / tau). Student target = comb(hard GT, s):
     comb="max": T = max(qt_GT, s), loss_cls = (1-alpha)*hard + alpha*QFL(T);  comb="mix": T = (1-alpha)*qt_GT + alpha*s (replaces hard).
     small_boost > 0: cells near (3x3) teacher peaks with a small box (< small_px in student coordinates) get weight 1 + small_boost.
  2. box distillation (box_alpha, default = alpha): GIoU+NWD to the teacher box only where teacher confidence > tau_box (weight = confidence).
  3. feat=True: FGD-like imitation of neck features (levels fuse4 [+ fuse8]) with a foreground mask from GT boxes,
     teacher spatial attention and relative norm; a 1x1 adapter from student to teacher channels (training only).

Teacher and student grids differ when ratio != 1: confidence is reduced to the student grid by max-pooling (peaks are kept), the box is taken from the max cell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hae.boxes import giou_pairs, nwd_pairs
from hae.model.decode import raw_to_boxes

TEACHER_BASES = ("teachers", "runs")


@dataclass
class DistillConfig:
    teacher: str = ""                 # path to a .pt/.onnx or a relative "<dir>/<file>" (searched in TEACHER_BASES)
    alpha: float = 0.0                # cls distillation weight, 0 = off
    tau: float = 1.0                  # teacher target temperature: sigmoid(z_t / tau); 1 = unchanged
    comb: str = "max"                 # max | mix (see module docstring)
    box_alpha: float | None = None    # box distillation weight; None = alpha
    tau_box: float = 0.3              # teacher positives: confidence > tau_box
    small_px: float = 16.0            # "small" target: sqrt(w*h) of the teacher box in student coordinates
    small_boost: float = 0.0          # extra cls-loss weight near teacher peaks for small targets, 0 = off
    peak_thr: float = 0.1             # peak confidence threshold for small_boost
    feat: bool = False                # FGD-like feature distillation
    feat_w: float = 1.0
    feat_levels: int = 1              # 1 = fuse4 (stride 4), 2 = + fuse8 (stride 8)
    feat_fg: float = 1.0              # foreground / background weight (both normalized by mask area)
    feat_bg: float = 0.25
    feat_T: float = 0.5               # temperature of teacher spatial attention
    teacher_scale_ratio: float = 1.0  # teacher input = student input x ratio (e.g. 2/1.5)

    @property
    def box_a(self) -> float:
        return self.alpha if self.box_alpha is None else self.box_alpha

    @property
    def enabled(self) -> bool:
        return self.alpha > 0 or self.box_a > 0 or self.feat

    @classmethod
    def from_dict(cls, d: dict | None) -> "DistillConfig":
        d = dict(d or {})
        bad = set(d) - {f.name for f in fields(cls)}
        if bad:
            raise ValueError(f"unknown distill fields: {sorted(bad)}")
        c = cls(**d)
        if c.comb not in ("max", "mix"):
            raise ValueError(f"distill.comb must be max/mix, got {c.comb!r}")
        if c.tau <= 0 or c.teacher_scale_ratio <= 0 or not 0 <= c.alpha <= 1:
            raise ValueError("distill: tau>0, teacher_scale_ratio>0, 0<=alpha<=1")
        if c.feat_levels not in (1, 2):
            raise ValueError("distill.feat_levels must be 1 or 2")
        return c


def find_teacher(spec: str, bases=TEACHER_BASES) -> Path:
    """An existing path as is; otherwise `spec` relative to each base; otherwise rglob by the last two path components."""
    p = Path(spec)
    if p.is_file():
        return p
    for b in bases:
        if (Path(b) / spec).is_file():
            return Path(b) / spec
    tail = Path(*p.parts[-2:]) if len(p.parts) > 1 else p
    for b in bases:
        if Path(b).is_dir():
            hits = sorted(h for h in Path(b).rglob(p.name) if h.is_file() and str(h).endswith(str(tail)))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"teacher checkpoint {spec!r} not found (bases: {list(bases)})")


class OnnxTeacher:
    """Teacher from .onnx (onnxruntime, CPU): logits/boxes only, no features. The input mode is fixed by the export."""
    ncls = None

    def __init__(self, path, stride: int = 4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.s.get_inputs()]
        self.cfg = type("C", (), {"stride": stride, "ctx": False, "tnorm": None, "ncls": None})()

    def __call__(self, rgb=None, t=None):
        have = {"rgb": rgb, "t": t}
        if set(self.names) != {k for k, v in have.items() if v is not None}:
            raise ValueError(f"ONNX teacher takes inputs {self.names}, but got {[k for k, v in have.items() if v is not None]}")
        cls, box = self.s.run(None, {k: have[k].detach().float().cpu().numpy() for k in self.names})
        dev = (rgb if rgb is not None else t).device
        return {"cls": torch.from_numpy(cls).to(dev), "box": torch.from_numpy(box).to(dev)}


def load_teacher(spec: str, device, channels_last: bool = True, bases=TEACHER_BASES):
    """Frozen teacher in eval mode. .pt: training checkpoint with "model" (EMA) and "cfg"; .safetensors: released weights with
    config.json next to them; .onnx: OnnxTeacher."""
    path = find_teacher(spec, bases)
    if path.suffix == ".onnx":
        return OnnxTeacher(path)
    from hae.checkpoint import load_checkpoint
    m, _ = load_checkpoint(path)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(memory_format=torch.channels_last) if channels_last and device.type != "cpu" else m


def _round_to(x: float, k: int) -> int:
    return max(k, int(round(x / k)) * k)


def _fg_mask(boxes, hg: int, wg: int, stride: float, dev) -> torch.Tensor:
    """(B,1,hg,wg): 1 in cells intersecting GT boxes (at least the centre cell); stride - cell step in student pixels."""
    m = torch.zeros(len(boxes), 1, hg, wg, device=dev)
    for b, bx in enumerate(boxes):
        for x1, y1, x2, y2 in np.asarray(bx, np.float32).reshape(-1, 4):
            cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
            xa, xb = int(math.floor(min(x1 / stride, cx - 0.5))), int(math.ceil(max(x2 / stride, cx + 0.5)))
            ya, yb = int(math.floor(min(y1 / stride, cy - 0.5))), int(math.ceil(max(y2 / stride, cy + 0.5)))
            m[b, 0, max(ya, 0):min(yb, hg), max(xa, 0):min(xb, wg)] = 1.0
    return m


class Distiller:
    """Holds the teacher and feature adapters and computes the extra loss terms. Created BEFORE the optimizer (adapters join its groups)."""

    LEVELS = ("fuse4", "fuse8")

    def __init__(self, cfg: DistillConfig, student, device, channels_last: bool = True, teacher=None):
        self.cfg, self.dev, self.cl = cfg, device, channels_last
        self.teacher = teacher if teacher is not None else load_teacher(cfg.teacher, device, channels_last)
        sc, tc = student.cfg, self.teacher.cfg
        if getattr(tc, "tnorm", None) is not None and bool(tc.tnorm) != bool(sc.tnorm):
            raise ValueError(f"student tnorm ({sc.tnorm}) and teacher tnorm ({tc.tnorm}) differ: the batch is normalized one way")
        if getattr(tc, "ncls", None) is not None and tc.ncls != sc.ncls:
            raise ValueError(f"student ({sc.ncls}) and teacher ({tc.ncls}) class counts differ")
        if cfg.feat and isinstance(self.teacher, OnnxTeacher):
            raise ValueError("feature distillation needs a .pt teacher (ONNX has no intermediate features)")
        self.mult = 32 if getattr(tc, "ctx", False) else 16
        self.adapters = nn.ModuleList()
        self._feats = {"s": {}, "t": {}}
        self._hooks = []
        if cfg.feat:
            for lv in self.LEVELS[:cfg.feat_levels]:
                self._hook(student, lv, "s")
                self._hook(self.teacher, lv, "t")
            self.adapters = nn.ModuleList(nn.Conv2d(sc.neck_c, tc.neck_c, 1) for _ in self.LEVELS[:cfg.feat_levels]).to(device)
        self.sstride = float(sc.stride)
        self.last_logs: dict = {}

    def _hook(self, model, lv, who):
        self._hooks.append(getattr(model, lv).register_forward_hook(lambda _m, _i, o, lv=lv, who=who: self._feats[who].__setitem__(lv, o)))

    def parameters(self):
        return self.adapters.parameters()

    def state_dict(self):
        return self.adapters.state_dict()

    def load_state_dict(self, sd):
        self.adapters.load_state_dict(sd)

    # ---- teacher ----
    @torch.no_grad()
    def teach(self, kw: dict, autocast: bool = False) -> dict:
        """Teacher forward pass on the upscaled input. Returns {cls, box (raw), hw: (rh, rw)} - scales relative to the student input."""
        ref = kw.get("rgb") if kw.get("rgb") is not None else kw["t"]
        h, w = ref.shape[-2:]
        r = self.cfg.teacher_scale_ratio
        ht, wt = (h, w) if r == 1.0 else (_round_to(h * r, self.mult), _round_to(w * r, self.mult))
        x = {}
        for k, v in kw.items():
            if (ht, wt) != (h, w):
                v = F.interpolate(v, size=(ht, wt), mode="bilinear", align_corners=False)
                if self.cl:
                    v = v.contiguous(memory_format=torch.channels_last)
            x[k] = v
        self._feats["t"].clear()
        with torch.autocast("cuda", torch.float16, enabled=autocast):
            out = self.teacher(**x)
        return {"cls": out["cls"].float(), "box": out["box"].float(), "rh": ht / h, "rw": wt / w, "stride": float(self.teacher.cfg.stride)}

    def _to_student_grid(self, tout: dict, hs: int, ws: int):
        """Teacher confidence/logits and boxes on the student grid: (z (B,K,hs,ws), boxes (B,hs,ws,4) in student coordinates)."""
        z = tout["cls"]
        bsz, k, ht, wt = z.shape
        boxes = raw_to_boxes(tout["box"], tout["stride"])                                    # (B,ht,wt,4), teacher pixels
        boxes = boxes / boxes.new_tensor([tout["rw"], tout["rh"], tout["rw"], tout["rh"]])    # -> student pixels
        if (ht, wt) == (hs, ws):
            return z, boxes
        zmax = z.amax(1, keepdim=True)
        _, idx = F.adaptive_max_pool2d(zmax, (hs, ws), return_indices=True)                   # index of the max cell in the window
        zp = F.adaptive_max_pool2d(z, (hs, ws))
        bg = boxes.reshape(bsz, ht * wt, 4).gather(1, idx.flatten(1)[..., None].expand(-1, -1, 4)).view(bsz, hs, ws, 4)
        return zp, bg

    # ---- loss ----
    def loss(self, out_s: dict, tout: dict, loss_fn, boxes, enable_feat_masks=True):
        """Addition to the total loss (tensor) and logs. out_s - student outputs (float); loss_fn.last - hard-loss state of this step."""
        c, last = self.cfg, loss_fn.last
        cls_s, hs, ws = out_s["cls"], out_s["cls"].shape[2], out_s["cls"].shape[3]
        z_t, b_t = self._to_student_grid(tout, hs, ws)
        p_t = z_t.sigmoid()
        pmax = p_t.amax(1)                                                                    # (B,hs,ws)
        total = cls_s.sum() * 0.0
        logs = {}
        npos = max(last["npos"], 1)
        weight = last["weight"]
        if c.alpha > 0:
            soft = (z_t / c.tau).sigmoid()
            qt = last["qt"]
            tgt = torch.maximum(qt, soft) if c.comb == "max" else (1 - c.alpha) * qt + c.alpha * soft
            wmap = weight
            if c.small_boost > 0:
                size = ((b_t[..., 2] - b_t[..., 0]).clamp(min=0) * (b_t[..., 3] - b_t[..., 1]).clamp(min=0)).sqrt()
                peak = ((pmax == F.max_pool2d(pmax[:, None], 3, 1, 1)[:, 0]) & (pmax > c.peak_thr) & (size < c.small_px)).float()
                near = F.max_pool2d(peak[:, None], 3, 1, 1)[:, 0]
                wmap = weight * (1 + c.small_boost * near)
                logs["kd_small_cells"] = float(near.sum())
            prob = cls_s.sigmoid()
            qfl = F.binary_cross_entropy_with_logits(cls_s, tgt, reduction="none") * (prob - tgt).abs().pow(2)
            if loss_fn.class_weights is not None:
                qfl = qfl * torch.tensor(loss_fn.class_weights, dtype=qfl.dtype, device=qfl.device).view(1, -1, 1, 1)
            kd_cls = (qfl * wmap[:, None]).sum() / npos
            a = 1.0 if c.comb == "mix" else c.alpha            # mix: kd_cls REPLACES hard; max: blend (1-a)*hard + a*kd
            total = total + loss_fn.cls_w * (a * kd_cls - a * last["cls_loss"])
            logs["kd_cls"] = float(kd_cls.detach())
        if c.box_a > 0:
            pos = (pmax > c.tau_box) & (weight > 0)
            if bool(pos.any()):
                pb = raw_to_boxes(out_s["box"], loss_fn.stride)[pos]
                tb, w = b_t[pos].detach(), pmax[pos].detach()
                ok = ((tb[:, 2] - tb[:, 0]) > 0) & ((tb[:, 3] - tb[:, 1]) > 0)
                if bool(ok.any()):
                    pb, tb, w = pb[ok], tb[ok], w[ok]
                    l = ((1 - giou_pairs(pb, tb)) + (1 - nwd_pairs(pb, tb, loss_fn.nwd_c)))
                    kd_box = (l * w).sum() / w.sum().clamp(min=1e-6)
                    total = total + loss_fn.box_w * c.box_a * kd_box
                    logs["kd_box"] = float(kd_box.detach())
                    logs["kd_box_n"] = float(ok.sum())
        if c.feat:
            lf = self._feat_loss(boxes, hs, ws)
            total = total + c.feat_w * lf
            logs["kd_feat"] = float(lf.detach())
        self.last_logs = logs
        return total, logs

    def _feat_loss(self, boxes, hs: int, ws: int) -> torch.Tensor:
        c, tot = self.cfg, 0.0
        for k, lv in enumerate(self.LEVELS[:c.feat_levels]):
            fs, ft = self._feats["s"][lv].float(), self._feats["t"][lv].float().detach()
            fs = self.adapters[k](fs)
            if ft.shape[-2:] != fs.shape[-2:]:
                ft = F.adaptive_avg_pool2d(ft, fs.shape[-2:])
            d = (fs - ft).pow(2).mean(1, keepdim=True) / (ft.pow(2).mean().detach() + 1e-6)   # relative error, (B,1,h,w)
            h, w = fs.shape[-2:]
            att = F.softmax(ft.abs().mean(1).flatten(1) / c.feat_T, 1).view(-1, 1, h, w) * (h * w)   # mean 1
            m = _fg_mask(boxes, h, w, hs * self.sstride / h, fs.device)
            fg = (m * att * d).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)
            bg = ((1 - m) * att * d).flatten(1).sum(1) / (1 - m).flatten(1).sum(1).clamp(min=1)
            tot = tot + (c.feat_fg * fg + c.feat_bg * bg).mean()
        return tot / c.feat_levels

    def close(self):
        for h in self._hooks:
            h.remove()
