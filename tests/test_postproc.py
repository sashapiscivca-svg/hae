import numpy as np

from hae.fusion import fuse_wbf
from hae.temporal import temporal_confirm


def _d(*rows):
    return np.array(rows, np.float32).reshape(-1, 6)


def test_wbf_merges_agreeing_boxes_and_halves_confidence_of_singletons():
    a = _d([10, 10, 20, 20, 0.8, 0], [50, 50, 60, 60, 0.6, 0])
    b = _d([11, 10, 21, 20, 0.6, 0])                                  # agrees only with the first box of a
    out = fuse_wbf([a, b], iou_thr=0.4)
    assert len(out) == 2
    top = out[np.argmax(out[:, 4])]
    assert abs(top[4] - 0.7) < 1e-6 and 10 < top[0] < 11 and 20 < top[2] < 21   # mean confidence, coordinates between the two
    single = out[np.argmin(out[:, 4])]
    assert abs(single[4] - 0.3) < 1e-6 and single[0] == 50                      # lone detection: 0.6 * 1/2


def test_wbf_handles_empty_inputs_and_keeps_classes_separate():
    assert fuse_wbf([_d(), _d()]).shape == (0, 6)
    a, b = _d([10, 10, 20, 20, 0.9, 0]), _d([10, 10, 20, 20, 0.9, 1])
    assert len(fuse_wbf([a, b])) == 2                                           # different classes are not fused


def test_temporal_confirmation_removes_one_frame_blips_but_keeps_persistent_and_moving_objects():
    persistent = [10 + 3 * t for t in range(6)]                                # a slowly moving person
    seq = []
    for t in range(6):
        rows = [[persistent[t], 20, persistent[t] + 8, 28, 0.7, 0]]
        if t == 3:
            rows.append([100, 100, 108, 108, 0.9, 0])                          # one-off flash
        seq.append(_d(*rows))
    out = temporal_confirm(seq, k=2, n=3, radius=10)
    assert [len(o) for o in out] == [0, 1, 1, 1, 1, 1]                         # first frame unconfirmed; the flash is removed
    assert all(o[0, 4] == 0.7 for o in out[1:])
    assert temporal_confirm([_d()] * 3, k=2, n=3, radius=10)[0].shape == (0, 6)
