"""Static int8 quantisation (QDQ) of hae ONNX models for low-end CPUs (onnxruntime).

Calibration uses real tiles passed in by the caller; no external models are needed.
"""
from __future__ import annotations

from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class _Reader(CalibrationDataReader):
    def __init__(self, feeds: dict):
        n = len(next(iter(feeds.values())))
        self._it = iter([{k: v[i] for k, v in feeds.items()} for i in range(n)])

    def get_next(self):
        return next(self._it, None)


def head_nodes(model_path: str | Path) -> list:
    """Output 1x1 convs of cls/box: quantising them costs up to 7 AP50 points in rgb mode, while their weights are tiny."""
    g = onnx.load(str(model_path)).graph
    return [n.name for n in g.node if n.op_type == "Conv" and any(o in ("cls", "box") for o in n.output)]


def quantize_int8(fp32_path: str | Path, out_path: str | Path, calib: dict) -> dict:
    """calib: {input name: list of float32 arrays (1,C,H,W)}. Names must match the model inputs."""
    names = [i.name for i in ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).get_inputs()]
    missing = [n for n in names if n not in calib]
    if missing:
        raise ValueError(f"no calibration data for inputs: {missing}")
    quantize_static(str(fp32_path), str(out_path), _Reader({n: calib[n] for n in names}), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    nodes_to_exclude=head_nodes(fp32_path))
    from hae.infer import embed_metadata, read_metadata
    embed_metadata(out_path, {k: v for k, v in read_metadata(fp32_path).items() if not k.startswith("onnx.")})   # quantisation drops metadata
    return {"bytes_fp32": Path(fp32_path).stat().st_size, "bytes_int8": Path(out_path).stat().st_size}
