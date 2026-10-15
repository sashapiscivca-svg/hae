"""hae (해, "sun"): tiny person detectors for search-and-rescue drones (thermal LWIR + RGB).

`import hae` stays light (no PyTorch, no onnxruntime): `hae.Detector` is imported lazily on first access.
"""
__version__ = "1.0.0"
__all__ = ["Detector", "__version__"]


def __getattr__(name: str):
    if name == "Detector":
        from hae.detector import Detector
        return Detector
    raise AttributeError(f"module 'hae' has no attribute {name!r}")
