"""Detection result for one image with convenience methods (an Ultralytics-like interface, independent implementation)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hae.viz import draw_detections


@dataclass
class Results:
    orig_img: np.ndarray
    boxes: np.ndarray                      # (n,4) xyxy in original image coordinates
    conf: np.ndarray                       # (n,)
    cls: np.ndarray                        # (n,) class index
    names: dict = field(default_factory=lambda: {0: "person"})
    path: str | None = None
    mode: str = "lwir"
    speed: dict = field(default_factory=dict)   # ms: preprocess / inference / postprocess

    @classmethod
    def from_array(cls, img: np.ndarray, dets: np.ndarray, names, path: str | None = None, mode: str = "lwir") -> "Results":
        d = np.asarray(dets, np.float32).reshape(len(dets), -1) if len(dets) else np.zeros((0, 6), np.float32)
        return cls(img, d[:, :4], d[:, 4], d[:, 5].astype(int) if d.shape[1] > 5 else np.zeros(len(d), int),
                   {i: n for i, n in enumerate(names)}, path, mode)

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def orig_shape(self) -> tuple:
        return tuple(self.orig_img.shape[:2])

    def plot(self, thr: float = 0.0, show_score: bool = True) -> np.ndarray:
        d = np.concatenate([self.boxes, self.conf[:, None], self.cls[:, None].astype(np.float32)], 1) if len(self) else np.zeros((0, 6), np.float32)
        return draw_detections(self.orig_img, d, thr=thr, show_score=show_score)

    def save(self, path: str | Path, thr: float = 0.0) -> Path:
        from PIL import Image
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.plot(thr)).save(p)
        return p

    def to_json(self) -> str:
        return json.dumps([{"box": [round(float(v), 1) for v in b], "score": round(float(c), 4), "class": self.names.get(int(k), int(k))}
                           for b, c, k in zip(self.boxes, self.conf, self.cls)], ensure_ascii=False, indent=1)

    def summary(self) -> str:
        per = {}
        for k in self.cls:
            per[self.names.get(int(k), int(k))] = per.get(self.names.get(int(k), int(k)), 0) + 1
        what = ", ".join(f"{n}: {c}" for n, c in sorted(per.items())) or "nothing"
        ms = f", {sum(self.speed.values()):.1f} ms" if self.speed else ""
        return f"{Path(self.path).name if self.path else 'image'} {self.orig_shape[1]}x{self.orig_shape[0]}: {len(self)} detections ({what}){ms}"
