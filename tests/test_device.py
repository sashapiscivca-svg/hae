import numpy as np
import pytest

from hae.data.annotations import load_annotations
from hae.data.cache import Cache
from hae.data.sampler import CropSampler, Prefetcher
from hae.train import TrainConfig, resolve_device, train
from test_train_smoke import _tiny_dataset


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    with pytest.raises(ValueError):
        resolve_device("tpu")


def _sampler_factory(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    cache, anns = Cache(cache_dir, "train"), load_annotations(root / "train_thermal.json")
    return (lambda i: CropSampler(cache, anns, (32, 64), seed=100 + i)), root, cache_dir


def test_prefetcher_yields_exact_count_and_stops(tmp_path):
    make, _, _ = _sampler_factory(tmp_path)
    got = list(Prefetcher(make, batch=4, n_batches=7, workers=3, depth=2))
    assert len(got) == 7 and all(b.rgb.shape == (4, 3, 32, 64) for b in got)


def test_prefetcher_propagates_worker_error():
    def bad(i):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        list(Prefetcher(bad, batch=2, n_batches=3, workers=1))


def test_train_with_prefetch_explicit_cpu_and_amp_flag_ignored_on_cpu(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2,
                      crops_per_epoch=32, batch=8, threads=2, crop_hw=(32, 64), device="cpu", amp=True, prefetch=2)
    hist = train(cfg)["history"]
    assert len(hist) == 2 and all(np.isfinite(h["loss"]) for h in hist)
