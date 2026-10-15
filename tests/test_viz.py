import numpy as np

from hae.viz import draw_detections


def test_draws_boxes_and_scores_without_mutating_input_and_handles_gray():
    gray = np.full((64, 96), 100, np.uint8)
    dets = np.array([[10, 10, 30, 30, 0.9, 0], [50, 20, 70, 40, 0.1, 0]], np.float32)
    gts = np.array([[12, 12, 28, 28]], np.float32)
    out = draw_detections(gray, dets, gts=gts, thr=0.3)
    assert out.shape == (64, 96, 3) and out.dtype == np.uint8
    assert (gray == 100).all()                                   # input is not modified
    assert not np.array_equal(out[10:12, 10:30], np.full((2, 20, 3), 100, np.uint8))   # box is drawn
    below = draw_detections(gray, dets[1:], thr=0.3)
    assert (below == 100).all()                                   # detection below threshold is not drawn


def test_rgb_float_input_and_empty_detections():
    img = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    out = draw_detections(img, np.zeros((0, 5), np.float32))
    assert out.shape == (32, 48, 3) and out.dtype == np.uint8
