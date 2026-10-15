import numpy as np
import torch

from hae.decode_np import decode_np
from hae.model.decode import decode


def _rand_out(k, seed):
    g = torch.Generator().manual_seed(seed)
    return {"cls": torch.randn(2, k, 16, 20, generator=g) * 2 - 3, "box": torch.randn(2, 4, 16, 20, generator=g) * 0.5 + 0.9}


def test_numpy_decode_matches_torch_for_one_and_three_classes():
    for k in (1, 3):
        out = _rand_out(k, k)
        ref = decode(out, stride=4, topk=60, score_thr=0.02)
        got = decode_np(out["cls"].numpy(), out["box"].numpy(), stride=4, topk=60, score_thr=0.02)
        assert len(ref) == len(got) == 2
        for a, b in zip(ref, got):
            assert a.shape == b.shape
            np.testing.assert_allclose(a[np.argsort(-a[:, 4], kind="stable")], b[np.argsort(-b[:, 4], kind="stable")], atol=1e-4)


def test_numpy_decode_empty_and_columns():
    cls, box = np.full((1, 3, 8, 8), -10.0, np.float32), np.zeros((1, 4, 8, 8), np.float32)
    assert decode_np(cls, box)[0].shape == (0, 6)
    assert decode_np(cls[:, :1], box)[0].shape == (0, 5)


def _decode_np_reference(cls, box, stride=4, topk=100, score_thr=0.05):
    """Previous implementation (softplus over the whole map): reference for the faster version (hot path on a Pi)."""
    from hae.decode_np import _peak_mask, _sigmoid, _softplus
    out = []
    b_, k, hg, wg = cls.shape
    ys = (np.arange(hg, dtype=np.float32) + 0.5) * stride
    xs = (np.arange(wg, dtype=np.float32) + 0.5) * stride
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    for b in range(b_):
        score = _sigmoid(cls[b].astype(np.float32))
        peaks = np.where(_peak_mask(score), score, 0.0).reshape(-1)
        idx = np.argsort(-peaks, kind="stable")[:min(topk, peaks.size)]
        idx = idx[peaks[idx] >= score_thr]
        cell = idx % (hg * wg)
        d = _softplus(box[b].astype(np.float32)).reshape(4, -1)[:, cell] * stride
        cxs, cys = cx.reshape(-1)[cell], cy.reshape(-1)[cell]
        cols = [cxs - d[0], cys - d[1], cxs + d[2], cys + d[3], peaks[idx]] + ([(idx // (hg * wg)).astype(np.float32)] if k > 1 else [])
        out.append(np.stack(cols, 1).astype(np.float32))
    return out


def test_numpy_decode_is_bit_identical_to_previous_implementation():
    for k in (1, 3):
        rng = np.random.default_rng(k)
        cls = (rng.normal(size=(2, k, 24, 31)) * 2 - 2).astype(np.float32)
        box = (rng.normal(size=(2, 4, 24, 31)) * 0.7 + 0.9).astype(np.float32)
        for thr in (0.0, 0.05, 0.4):
            for a, b in zip(_decode_np_reference(cls, box, 4, 80, thr), decode_np(cls, box, 4, 80, thr)):
                np.testing.assert_array_equal(a, b)
