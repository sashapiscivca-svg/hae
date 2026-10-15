"""Crop sampler over the cache: 70 % of crops contain a target, the rest are random/background."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch

from hae.data.annotations import FrameAnn
from hae.data.cache import Cache

MIN_VISIBLE = 0.5  # a box visible on less than 50 % of its area becomes ignore


@dataclass
class Batch:
    rgb: torch.Tensor   # (B,3,h,w) float32 0..1
    t: torch.Tensor     # (B,1,h,w) float32 0..1
    boxes: list         # list[np.ndarray (n,4)] xyxy in crop coordinates
    ignore: list        # list[np.ndarray (m,4)]
    labels: list = None # list[np.ndarray (n,) int64] - class of each positive
    modes: tuple = ("both", "rgb", "lwir")   # modes this batch can be trained in (LWIR-only sets -> ("lwir",))
    halo: object = None  # (B,1,h,w) HALO mask 0..1 (auxiliary target) or None


def _clip(b: np.ndarray, cw: int, ch: int):
    """Returns clipped boxes and the visible area fraction."""
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, cw)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, ch)
    carea = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    return c, carea / np.maximum(area, 1e-6)


def photometric(x: np.ndarray, rng: np.random.Generator, strength: float, color: float = 1.0) -> np.ndarray:
    """Photometric augmentation of an RGB crop: brightness, contrast, gamma, white balance, saturation, grayscale, blur, noise.
    strength in [0,1] scales the amplitudes; 0 = identity. color in [0,1] separately scales the COLOUR transforms (white balance,
    saturation, grayscale): on HERIDAL a person is visible by clothing colour against vegetation, and strong colour shifts hurt. Fast implementation: tone - one cv2.LUT table per channel,
    colour - one 3x3 matrix (cv2.transform), no full-size float arrays (was +275 ms per batch, now a fraction of that)."""
    if strength <= 0:
        return x
    st = float(strength)
    m = float(x.mean())
    v = np.arange(256, dtype=np.float32)
    b, ct, g = rng.uniform(1 - 0.4 * st, 1 + 0.4 * st), rng.uniform(1 - 0.4 * st, 1 + 0.4 * st), float(np.exp(rng.uniform(-0.35 * st, 0.35 * st)))
    cs = st * float(color)
    wb = rng.uniform(1 - 0.12 * cs, 1 + 0.12 * cs, size=3).astype(np.float32)
    f = (v * b - m * b) * ct + m * b                                                  # brightness + contrast around the mean
    lut = np.stack([255.0 * np.clip(f * wb[c] / 255.0, 0, 1) ** g for c in range(3)], -1)   # white balance + gamma
    out = cv2.LUT(x, np.clip(lut, 0, 255).round().astype(np.uint8).reshape(256, 1, 3))
    sat = float(rng.uniform(1 - 0.5 * cs, 1 + 0.5 * cs))
    if rng.random() < 0.15 * cs:
        sat = 0.0                                                                      # grayscale
    luma = np.array([0.114, 0.587, 0.299], np.float32)                                 # BGR/RGB symmetric in the sum; enough for equal channels
    mat = sat * np.eye(3, dtype=np.float32) + (1 - sat) * np.tile(luma[None, :], (3, 1))
    out = cv2.transform(out, mat)
    if rng.random() < 0.2 * st:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3 * st:
        amp = int(max(1, round(rng.uniform(1.0, 6.0) * st)))
        noise = rng.integers(-amp, amp + 1, size=out.shape, dtype=np.int8)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def thermal_aug(t: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Thermal crop augmentation: sensor and optics non-linearities that per-frame normalization (tnorm) does NOT remove:
    gamma, range compression (AGC), slow gradient (vignetting/drift), blur, noise, reduced bit depth.
    At the end the crop mean and sigma are matched to the original - normalization removes a global brightness shift/scale anyway."""
    if strength <= 0:
        return t
    st = float(strength)
    m0, s0 = float(t.mean()), float(t.std()) + 1e-6
    f = t.astype(np.float32)
    f = 255.0 * (f / 255.0) ** float(np.exp(rng.uniform(-0.4 * st, 0.4 * st)))               # gamma
    if rng.random() < 0.5 * st:                                                               # AGC: clip tails and stretch
        lo, hi = np.percentile(f, [rng.uniform(0, 2) * st, 100 - rng.uniform(0, 2) * st])
        f = np.clip((f - lo) / max(hi - lo, 1.0), 0, 1) * 255.0
    if rng.random() < 0.5 * st:                                                               # slow gradient across the crop
        h, w = f.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = f + rng.uniform(-10, 10) * st * (xx / max(w - 1, 1) - 0.5) + rng.uniform(-10, 10) * st * (yy / max(h - 1, 1) - 0.5)
    if rng.random() < 0.3 * st:
        f = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5 * st:
        f = f + rng.normal(0, rng.uniform(1.0, 5.0) * st, f.shape).astype(np.float32)
    if rng.random() < 0.2 * st:                                                               # lower bit depth (coarser quantization)
        q = float(rng.choice([4, 8, 16]))
        f = np.round(f / q) * q
    f = (f - f.mean()) / (f.std() + 1e-6) * s0 + m0                                            # global mean and sigma - same as the original crop
    return np.clip(f, 0, 255).round().astype(np.uint8)


def fourier_low_amp(x: np.ndarray, band: float = 0.1) -> np.ndarray:
    """Low-frequency block of the amplitude spectrum of an RGB crop (3, 2bh+1, 2bw+1): the domain "style" (lighting, tone, large textures)."""
    h, w = x.shape[:2]
    amp = np.abs(np.fft.fftshift(np.fft.fft2(x.astype(np.float32), axes=(0, 1)), axes=(0, 1)))
    bh, bw = max(1, int(round(band * h / 2))), max(1, int(round(band * w / 2)))
    return amp[h // 2 - bh:h // 2 + bh + 1, w // 2 - bw:w // 2 + bw + 1].transpose(2, 0, 1).astype(np.float32)


def fourier_mix(x: np.ndarray, bank: np.ndarray, lam: float, band: float = 0.1) -> np.ndarray:
    """Replaces the crop's low-frequency amplitude with a mix with another crop's "style" (the phase, i.e. shapes and fine details, is kept);
    idea from FACT/FOUND (arXiv 2105.11120, 2511.10352), own implementation. lam=0 -> identity."""
    if lam <= 0:
        return x
    h, w = x.shape[:2]
    f = np.fft.fftshift(np.fft.fft2(x.astype(np.float32), axes=(0, 1)), axes=(0, 1))
    amp, pha = np.abs(f), np.angle(f)
    bh, bw = max(1, int(round(band * h / 2))), max(1, int(round(band * w / 2)))
    sl = (slice(h // 2 - bh, h // 2 + bh + 1), slice(w // 2 - bw, w // 2 + bw + 1))
    amp[sl] = (1 - lam) * amp[sl] + lam * bank.transpose(1, 2, 0)
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(0, 1)), axes=(0, 1)).real
    return np.clip(out, 0, 255).round().astype(np.uint8)


def fourier_mix_fast(x: np.ndarray, bank: np.ndarray, lam: float, band: float = 0.1, down: int = 4) -> np.ndarray:
    """Same low-frequency amplitude mixing, but on a copy downscaled `down` times: the difference (low frequencies only)
    is added to the full-size crop. ~10-20x faster than a full FFT; on 256x320x3 a full FFT in CPU workers made an epoch 6x longer."""
    if lam <= 0:
        return x
    h, w = x.shape[:2]
    hl, wl = h // down, w // down
    small = cv2.resize(x, (wl, hl), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(small, axes=(0, 1)), axes=(0, 1))
    bh, bw = (bank.shape[1] - 1) // 2, (bank.shape[2] - 1) // 2
    bhl, bwl = max(1, int(round(bh / down))), max(1, int(round(bw / down)))
    ri = np.clip(bh + down * np.arange(-bhl, bhl + 1), 0, bank.shape[1] - 1)       # same physical frequencies in the full-resolution bank
    ci = np.clip(bw + down * np.arange(-bwl, bwl + 1), 0, bank.shape[2] - 1)
    tgt = bank[:, ri][:, :, ci].transpose(1, 2, 0) / float(down * down)               # amplitude of the downscaled image is down^2 smaller
    sl = (slice(hl // 2 - bhl, hl // 2 + bhl + 1), slice(wl // 2 - bwl, wl // 2 + bwl + 1))
    amp, pha = np.abs(f), np.angle(f)
    new = f.copy()
    amp_new = amp.copy()
    amp_new[sl] = (1 - lam) * amp[sl] + lam * tgt
    new = amp_new * np.exp(1j * pha)
    delta = np.fft.ifft2(np.fft.ifftshift(new - f, axes=(0, 1)), axes=(0, 1)).real.astype(np.float32)
    delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(x.astype(np.float32) + delta, 0, 255).round().astype(np.uint8)


def build_style_bank(caches, crop_hw, band: float = 0.1, n_per: int = 120, seed: int = 0) -> np.ndarray:
    """Bank of "styles" from all RGB domains (n_per random crops per domain) - shared by all samplers."""
    rng = np.random.default_rng(seed)
    ch, cw = crop_hw
    bank = []
    for c in caches:
        if c.rgb is None:
            continue
        for _ in range(n_per):
            i = int(rng.integers(c.n))
            y, x = int(rng.integers(0, max(c.h - ch, 0) + 1)), int(rng.integers(0, max(c.w - cw, 0) + 1))
            bank.append(fourier_low_amp(np.asarray(c.rgb[i, y:y + ch, x:x + cw]), band))
    return np.stack(bank) if bank else np.zeros((0, 3, 1, 1), np.float32)


class CropSampler:
    def __init__(self, cache: Cache, anns: Sequence[FrameAnn], crop_hw=(256, 320),
                 pos_frac: float = 0.7, seed: int = 0, flip: bool = True, frames=None,
                 scale_range=(1.0, 1.0), copy_paste: float = 0.0, cp_max: int = 3, cp_pool: int = 1500, t_norm: bool = False, rare_crop: float = 0.0, use_halo: bool = False, photo: float = 0.0, photo_color: float = 1.0, taug: float = 0.0, t_invert: float = 0.0):
        self.cache, self.anns = cache, anns
        self.taug = float(taug)   # thermal augmentation strength, 0 = off
        self.t_invert = float(t_invert)   # probability of inverting thermal crop polarity 255-t, 0 = off
        self.photo, self.photo_color = float(photo), float(photo_color)   # strength of RGB photometric augmentation and of its colour part
        self.style_bank, self.fourier_p, self.fourier_band = None, 0.0, 0.1   # cross-domain spectrum mixing
        self.ch, self.cw = crop_hw
        self.pos_frac, self.flip = pos_frac, flip
        if not 0.0 <= copy_paste <= 1.0:
            raise ValueError(f"copy_paste is a probability in [0,1], got {copy_paste}")
        self.t_norm = t_norm
        self.use_halo = use_halo and cache.halo is not None
        self.rare_crop = rare_crop    # probability of centring a crop on a rare class (rider/crowd), if present
        self.copy_paste, self.cp_max, self.cp_pool_size, self._pool = copy_paste, cp_max, cp_pool, None
        lo, hi = scale_range
        if not 0 < lo <= hi:
            raise ValueError(f"scale_range must be 0 < lo <= hi, got {scale_range}")
        self.scale_range = (float(lo), float(hi))   # >1 - upscaling (larger targets), <1 - downscaling
        self.rng = np.random.default_rng(seed)
        self.frames = list(range(len(anns))) if frames is None else list(frames)
        self.pos_frames = [i for i in self.frames if len(anns[i].boxes) > 0]
        self.hn, self.hn_frac = None, 0.0   # hard negatives: (N,4) [frame, cx, cy, score]
        self.rare = [(i, k) for i in self.pos_frames if anns[i].labels is not None for k in range(len(anns[i].boxes)) if anns[i].labels[k] > 0]

    def _build_pool(self, m: int = 3):
        """Pool of people with local background (margin m px) and a blurred alpha mask of the edges (copy-paste)."""
        cands = [(i, k) for i in self.pos_frames for k in range(len(self.anns[i].boxes))]
        self.rng.shuffle(cands)
        pool = []
        for i, k in cands:
            if len(pool) >= self.cp_pool_size:
                break
            x1, y1, x2, y2 = (int(round(v)) for v in self.anns[i].boxes[k])
            bw, bh = x2 - x1, y2 - y1
            if bw < 3 or bh < 3 or x1 - m < 0 or y1 - m < 0 or x2 + m > self.cache.w or y2 + m > self.cache.h:
                continue
            ph, pw = bh + 2 * m, bw + 2 * m
            core = np.zeros((ph, pw), np.float32)
            core[m:m + bh, m:m + bw] = 1.0
            alpha = np.clip(cv2.GaussianBlur(core, (0, 0), 1.5) * 1.6, 0.0, 1.0)
            pool.append({"t": np.array(self.cache.t[i, y1 - m:y2 + m, x1 - m:x2 + m]),
                         "rgb": np.array(self.cache.rgb[i, y1 - m:y2 + m, x1 - m:x2 + m]) if self.cache.rgb is not None else None,
                         "label": 0 if self.anns[i].labels is None else int(self.anns[i].labels[k]),
                         "halo": np.array(self.cache.halo[i, y1 - m:y2 + m, x1 - m:x2 + m]) if self.use_halo else None,
                         "alpha": alpha, "m": m, "bw": bw, "bh": bh})
        self._pool = pool

    def _paste(self, r, t, hl, boxes, ign, labels):
        if self._pool is None:
            self._build_pool()
        if not self._pool:
            return r, t, hl, boxes, labels
        r, t = r.copy(), t.copy()
        hl = None if hl is None else hl.copy()
        new_b, new_l = [], []
        for _ in range(int(self.rng.integers(1, self.cp_max + 1))):
            it = self._pool[int(self.rng.integers(len(self._pool)))]
            ph, pw = it["alpha"].shape
            if pw >= self.cw or ph >= self.ch:
                continue
            for _try in range(8):
                x, y = int(self.rng.integers(0, self.cw - pw + 1)), int(self.rng.integers(0, self.ch - ph + 1))
                core = np.array([x + it["m"], y + it["m"], x + it["m"] + it["bw"], y + it["m"] + it["bh"]], np.float32)
                occupied = np.concatenate([boxes.reshape(-1, 4), ign.reshape(-1, 4), np.array(new_b, np.float32).reshape(-1, 4)])
                inter = (np.minimum(occupied[:, 2], core[2]) - np.maximum(occupied[:, 0], core[0])).clip(0) * \
                        (np.minimum(occupied[:, 3], core[3]) - np.maximum(occupied[:, 1], core[1])).clip(0)
                if not (inter > 0).any():
                    a = it["alpha"]
                    t[y:y + ph, x:x + pw] = (a * it["t"] + (1 - a) * t[y:y + ph, x:x + pw]).round().astype(np.uint8)
                    if it["rgb"] is not None and r.any():
                        r[y:y + ph, x:x + pw] = (a[..., None] * it["rgb"] + (1 - a[..., None]) * r[y:y + ph, x:x + pw]).round().astype(np.uint8)
                    if hl is not None and it["halo"] is not None:
                        hl[y:y + ph, x:x + pw] = np.maximum(hl[y:y + ph, x:x + pw], (a * it["halo"]).round().astype(np.uint8))
                    new_b.append(core)
                    new_l.append(it["label"])
                    break
        if new_b:
            boxes = np.concatenate([boxes.reshape(-1, 4), np.array(new_b, np.float32)])
            labels = np.concatenate([labels, np.array(new_l, np.int64)])
        return r, t, hl, boxes, labels

    def _fit(self, b: np.ndarray) -> np.ndarray:
        """After scaling, float32 may give a coordinate 1e-6 beyond the crop border."""
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.cw)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.ch)
        return b

    def _draw_scale(self) -> float:
        lo, hi = self.scale_range
        lo = max(lo, self.ch / self.cache.h, self.cw / self.cache.w)   # the crop source must fit in the frame
        hi = max(hi, lo)
        return lo if hi <= lo else float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _boxes_for_crop(self, ann: FrameAnn, x0: int, y0: int, flipx: bool, flipy: bool, sw=None, sh=None):
        """sw, sh - size of the crop SOURCE (before scaling); boxes are converted to output pixel coordinates."""
        sw, sh = sw or self.cw, sh or self.ch
        sx, sy = self.cw / sw, self.ch / sh
        scl = np.array([sx, sy, sx, sy], np.float32)
        shift = np.array([x0, y0, x0, y0], np.float32)
        out_boxes, out_ign = [], []
        labels = np.zeros(0, np.int64)
        if len(ann.boxes):
            c, vis = _clip(ann.boxes - shift, sw, sh)
            c = self._fit(c * scl)
            keep = vis >= MIN_VISIBLE
            labels = (np.zeros(len(ann.boxes), np.int64) if ann.labels is None else ann.labels)[keep]
            out_boxes.append(c[keep])
            part = (~keep) & (vis > 0)
            out_ign.append(c[part])
        if len(ann.ignore):
            c, vis = _clip(ann.ignore - shift, sw, sh)
            out_ign.append(self._fit(c * scl)[vis > 0])
        boxes = np.concatenate(out_boxes) if out_boxes else np.zeros((0, 4), np.float32)
        ign = np.concatenate(out_ign) if out_ign else np.zeros((0, 4), np.float32)
        for arr in (boxes, ign):
            if flipx and len(arr):
                arr[:, [0, 2]] = self.cw - arr[:, [2, 0]]
            if flipy and len(arr):
                arr[:, [1, 3]] = self.ch - arr[:, [3, 1]]
        return boxes.astype(np.float32), ign.astype(np.float32), labels

    def set_style_bank(self, bank, p: float, band: float = 0.1) -> None:
        self.style_bank = None if bank is None or len(bank) == 0 else bank
        self.fourier_p, self.fourier_band = float(p), float(band)

    def set_hard_negatives(self, points, frac: float) -> None:
        self.hn = None if points is None or len(points) == 0 else np.asarray(points, np.float32)
        self.hn_frac = float(frac)

    def _origin(self, sw: int, sh: int):
        H, W = self.cache.h, self.cache.w
        if self.hn is not None and self.rng.random() < self.hn_frac:
            p = self.hn[int(self.rng.integers(len(self.hn)))]
            x0 = int(np.clip(p[1] - self.rng.uniform(0.1, 0.9) * sw, 0, W - sw))
            y0 = int(np.clip(p[2] - self.rng.uniform(0.1, 0.9) * sh, 0, H - sh))
            return int(p[0]), x0, y0
        if self.pos_frames and self.rng.random() < self.pos_frac:
            if self.rare and self.rng.random() < self.rare_crop:
                i, k = self.rare[int(self.rng.integers(len(self.rare)))]
            else:
                i = int(self.rng.choice(self.pos_frames))
                k = int(self.rng.integers(len(self.anns[i].boxes)))
            bx = self.anns[i].boxes[k]
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            x0 = int(np.clip(cx - self.rng.uniform(0.1, 0.9) * sw, 0, W - sw))
            y0 = int(np.clip(cy - self.rng.uniform(0.1, 0.9) * sh, 0, H - sh))
        else:
            i = int(self.rng.choice(self.frames))
            x0 = int(self.rng.integers(0, W - sw + 1))
            y0 = int(self.rng.integers(0, H - sh + 1))
        return i, x0, y0

    def sample_batch(self, bs: int) -> Batch:
        rgbs, ts, boxes, ign, labs, hls = [], [], [], [], [], []
        tm, tsd = self.cache.tstats if self.t_norm else (None, None)
        for _ in range(bs):
            sc = self._draw_scale()
            sw, sh = min(int(round(self.cw / sc)), self.cache.w), min(int(round(self.ch / sc)), self.cache.h)
            i, x0, y0 = self._origin(sw, sh)
            fx = self.flip and self.rng.random() < 0.5
            fy = self.flip and self.rng.random() < 0.5
            has_rgb = self.cache.rgb is not None
            r = np.asarray(self.cache.rgb[i, y0:y0 + sh, x0:x0 + sw]) if has_rgb else np.zeros((self.ch, self.cw, 3), np.uint8)
            t = np.asarray(self.cache.t[i, y0:y0 + sh, x0:x0 + sw])
            hl = np.asarray(self.cache.halo[i, y0:y0 + sh, x0:x0 + sw]) if self.use_halo else None
            if (sw, sh) != (self.cw, self.ch):
                interp = cv2.INTER_AREA if sw > self.cw else cv2.INTER_LINEAR
                if has_rgb:
                    r = cv2.resize(np.ascontiguousarray(r), (self.cw, self.ch), interpolation=interp)
                t = cv2.resize(np.ascontiguousarray(t), (self.cw, self.ch), interpolation=interp)
                if hl is not None:
                    hl = cv2.resize(np.ascontiguousarray(hl), (self.cw, self.ch), interpolation=interp)
            if self.taug > 0:
                t = thermal_aug(np.ascontiguousarray(t), self.rng, self.taug)
            if self.photo > 0 and has_rgb:
                r = photometric(np.ascontiguousarray(r), self.rng, self.photo, self.photo_color)
            if self.style_bank is not None and has_rgb and self.rng.random() < self.fourier_p:
                r = fourier_mix_fast(np.ascontiguousarray(r), self.style_bank[int(self.rng.integers(len(self.style_bank)))], float(self.rng.uniform(0.3, 1.0)), self.fourier_band)
            if fx:
                r, t = r[:, ::-1], t[:, ::-1]
                hl = None if hl is None else hl[:, ::-1]
            if fy:
                r, t = r[::-1], t[::-1]
                hl = None if hl is None else hl[::-1]
            b, g, lab = self._boxes_for_crop(self.anns[i], x0, y0, fx, fy, sw, sh)
            if self.copy_paste > 0 and self.rng.random() < self.copy_paste:
                r, t, hl, b, lab = self._paste(r, t, hl, b, g, lab)
            inv = self.t_invert > 0 and self.rng.random() < self.t_invert   # after all thermal augmentations and pastes; rng untouched when off
            if inv:
                t = 255 - np.asarray(t)   # frame statistics for tnorm: mean -> 255 - mu, same sigma
            rgbs.append(np.ascontiguousarray(r))
            ts.append(np.ascontiguousarray(t) if tm is None else ((t.astype(np.float32) - (255.0 - tm[i] if inv else tm[i])) / max(float(tsd[i]), 1.0)))
            boxes.append(b)
            ign.append(g)
            labs.append(lab)
            if hl is not None:
                hls.append(np.ascontiguousarray(hl))
        rgb = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().div_(255)
        t = torch.from_numpy(np.stack(ts))[:, None].float()
        t = t if self.t_norm else t.div_(255)
        return Batch(rgb, t, boxes, ign, labs, self.cache.modes if self.cache.rgb is not None else ('lwir',),
                     torch.from_numpy(np.stack(hls))[:, None].float().div_(255) if hls else None)


class MixedSampler:
    """Picks the batch source by weight: the main set and the extra open sets."""
    def __init__(self, samplers, weights, seed: int = 0):
        if len(samplers) != len(weights):
            raise ValueError("samplers and weights must have the same length")
        self.samplers, self.p = list(samplers), np.asarray(weights, float) / float(sum(weights))
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, bs: int):
        return self.samplers[int(self.rng.choice(len(self.samplers), p=self.p))].sample_batch(bs)


def load_frames(cache: Cache, idxs: Sequence[int], t_norm: bool = False, pre_lnorm: int = 0):
    """Full frames (for evaluation/SAHI): RGB 0..1; thermal 0..1 or (t_norm) z-scores using per-frame statistics.
    pre_lnorm=k: thermal after local_contrast_u8(k) on the full frame. If the cache was already converted with the same k
    (hae.data.cache.ensure_lnorm_cache), frames are taken as is; a raw cache is converted on the fly (tnorm statistics come from converted frames)."""
    pre = int(pre_lnorm or 0)
    have = int(getattr(cache, "pre_lnorm", 0) or 0)
    if have and have != pre:
        raise ValueError(f"cache was converted with local_contrast_u8(k={have}), but the model expects pre_lnorm={pre}")
    on_fly = pre > 0 and have == 0
    r = (np.stack([np.asarray(cache.rgb[i]) for i in idxs]) if cache.rgb is not None
         else np.zeros((len(idxs), cache.h, cache.w, 3), np.uint8))   # LWIR-only set: rgb placeholder (lwir mode does not read it)
    if on_fly:
        from hae.infer import local_contrast_u8
        t = np.stack([local_contrast_u8(np.asarray(cache.t[i]), pre) for i in idxs])
    else:
        t = np.stack([np.asarray(cache.t[i]) for i in idxs])
    tt = torch.from_numpy(t)[:, None].float()
    if t_norm and on_fly:
        f = t.astype(np.float32).reshape(len(idxs), -1)
        mu = torch.from_numpy(f.mean(1))[:, None, None, None]
        sd = torch.from_numpy(f.std(1)).clamp(min=1.0)[:, None, None, None]
        tt = (tt - mu) / sd
    elif t_norm:
        mean, std = cache.tstats
        mu, sd = torch.from_numpy(mean[list(idxs)])[:, None, None, None], torch.from_numpy(std[list(idxs)]).clamp(min=1.0)[:, None, None, None]
        tt = (tt - mu) / sd
    else:
        tt = tt.div_(255)
    return torch.from_numpy(r).permute(0, 3, 1, 2).float().div_(255), tt


class Prefetcher:
    """Background threads prepare batches (each with its own CropSampler), the main thread only collects them."""
    def __init__(self, make_sampler, batch: int, n_batches: int, workers: int = 2, depth: int = 4):
        self.n, self.q = n_batches, queue.Queue(maxsize=max(depth, 1))
        per = [n_batches // workers + (1 if i < n_batches % workers else 0) for i in range(workers)]
        self.threads = [threading.Thread(target=self._work, args=(make_sampler, i, batch, k), daemon=True)
                        for i, k in enumerate(per) if k > 0]
        for t in self.threads:
            t.start()

    def _work(self, make_sampler, i, batch, k):
        try:
            s = make_sampler(i)
            for _ in range(k):
                self.q.put(s.sample_batch(batch))
        except BaseException as e:  # pass the exception to the main thread so it does not hang
            self.q.put(e)

    def __iter__(self):
        for _ in range(self.n):
            item = self.q.get()
            if isinstance(item, BaseException):
                raise item
            yield item
