import numpy as np
import pytest

pytest.importorskip("matplotlib")
from hae.gallery import error_gallery, pred_grid, split_errors


def _frames():
    img = np.full((64, 96), 80, np.uint8)
    gts = [np.array([[10, 10, 18, 18], [50, 40, 58, 48]], np.float32)]
    dets = [np.array([[10, 10, 18, 18, 0.9, 0],       # hit
                      [70, 10, 78, 18, 0.8, 0]], np.float32)]   # false alarm; the person at (50,40) is missed
    return [img], gts, dets


def test_split_errors_finds_missed_people_and_false_alarms():
    imgs, gts, dets = _frames()
    fn, fp = split_errors(gts, dets, conf=0.3)
    assert len(fn) == 1 and fn[0][0] == 0 and np.allclose(fn[0][1], [50, 40, 58, 48])
    assert len(fp) == 1 and fp[0][0] == 0 and abs(fp[0][2] - 0.8) < 1e-6


def test_pred_grid_and_error_gallery_write_nonempty_pngs(tmp_path):
    imgs, gts, dets = _frames()
    g = pred_grid(imgs * 4, gts * 4, dets * 4, conf=0.3, cols=2, width=96)
    assert g.dtype == np.uint8 and g.shape[2] == 3 and g.shape[1] == 2 * 96
    p = error_gallery(imgs, gts, dets, 0.3, tmp_path / "errors.png")
    assert p.exists() and p.stat().st_size > 3000


def test_error_gallery_handles_perfect_and_empty_cases(tmp_path):
    imgs = [np.zeros((32, 32), np.uint8)]
    p = error_gallery(imgs, [np.zeros((0, 4), np.float32)], [np.zeros((0, 6), np.float32)], 0.3, tmp_path / "e.png")
    assert p.exists()
