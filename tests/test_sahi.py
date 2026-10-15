import numpy as np
import pytest

from hae.sahi import merge_detections, sliced_predict, tile_origins


def test_tile_origins_cover_frame_with_overlap_and_aligned_last_tile():
    org = tile_origins(512, 640, 256, 320, overlap=0.2)
    ys = sorted({y for y, _ in org}); xs = sorted({x for _, x in org})
    assert ys[0] == 0 and ys[-1] == 512 - 256 and xs[0] == 0 and xs[-1] == 640 - 320
    cover = np.zeros((512, 640), bool)
    for y, x in org:
        cover[y:y + 256, x:x + 320] = True
    assert cover.all()
    assert max(b - a for a, b in zip(xs, xs[1:])) <= 320 * 0.8 + 1               # overlap >= 20 %
    assert tile_origins(100, 100, 256, 320, 0.2) == [(0, 0)]                      # frame smaller than the tile


def test_merge_removes_duplicates_keeps_best_and_respects_classes():
    d = np.array([[10, 10, 20, 20, 0.9, 0], [11, 10, 21, 20, 0.6, 0], [10, 10, 20, 20, 0.8, 1], [50, 50, 60, 60, 0.7, 0]], np.float32)
    out = merge_detections(d, iou_thr=0.5)
    assert len(out) == 3 and out[0, 4] == pytest.approx(0.9)
    assert sorted(out[:, 5].tolist()) == [0.0, 0.0, 1.0]
    assert merge_detections(np.zeros((0, 6), np.float32)).shape == (0, 6)


def test_sliced_predict_offsets_boxes_and_merges_overlap_duplicates():
    H, W = 128, 192
    rgb, t = np.zeros((3, H, W), np.float32), np.zeros((1, H, W), np.float32)

    target = np.array([100, 60, 110, 70])       # x1,y1,x2,y2 in frame coordinates

    def infer(rgb_tile, t_tile, origin):
        y0, x0 = origin
        gx1, gy1, gx2, gy2 = target
        if gx1 >= x0 and gx2 <= x0 + t_tile.shape[-1] and gy1 >= y0 and gy2 <= y0 + t_tile.shape[-2]:
            return np.array([[gx1 - x0, gy1 - y0, gx2 - x0, gy2 - y0, 0.8, 0]], np.float32)
        return np.zeros((0, 6), np.float32)

    out = sliced_predict(infer, rgb, t, tile_hw=(64, 96), overlap=0.5)
    assert len(out) == 1
    np.testing.assert_allclose(out[0, :4], target)


def test_sliced_predict_pads_small_frames_to_multiple_of_32_and_clips_boxes():
    H, W = 50, 70
    rgb, t = np.zeros((3, H, W), np.float32), np.zeros((1, H, W), np.float32)
    shapes = []

    def infer(rgb_tile, t_tile, origin):
        shapes.append(t_tile.shape[-2:])
        return np.array([[60, 40, 90, 70, 0.9, 0]], np.float32)       # extends beyond the 70x50 frame (from padding)

    out = sliced_predict(infer, rgb, t, tile_hw=(64, 96), overlap=0.2)
    assert shapes == [(64, 96)]
    assert (out[:, [0, 2]] <= W).all() and (out[:, [1, 3]] <= H).all()
