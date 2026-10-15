import math

import numpy as np
import torch

from hae.model.decode import cell_centers, decode, raw_to_boxes

RAW5PX = math.log(math.expm1(5 / 4))  # softplus(raw)*4 = 5 px


def _out(hg=16, wg=20):
    return {"cls": torch.full((1, 1, hg, wg), -10.0), "box": torch.full((1, 4, hg, wg), RAW5PX)}


def test_cell_centers():
    cy, cx = cell_centers(2, 3, 4)
    assert cx[0].tolist() == [2.0, 6.0, 10.0] and cy[:, 0].tolist() == [2.0, 6.0]


def test_single_peak_decodes_to_expected_box():
    out = _out()
    out["cls"][0, 0, 5, 7] = 4.0
    d = decode(out)[0]
    assert d.shape == (1, 5)
    np.testing.assert_allclose(d[0, :4], [25, 17, 35, 27], atol=1e-3)  # centre (30,22) +- 5
    assert abs(d[0, 4] - 1 / (1 + math.exp(-4))) < 1e-5


def test_empty_output_is_empty_array():
    assert decode(_out())[0].shape == (0, 5)


def test_non_peak_neighbours_suppressed_and_topk():
    out = _out()
    out["cls"][0, 0, 5, 7] = 4.0
    out["cls"][0, 0, 5, 8] = 3.0   # a lower neighbour - suppressed by 3x3 max-pool
    out["cls"][0, 0, 12, 15] = 2.0
    d = decode(out, topk=10)[0]
    assert len(d) == 2 and d[0, 4] > d[1, 4]
    assert len(decode(out, topk=1)[0]) == 1


def test_raw_to_boxes_shape():
    b = raw_to_boxes(torch.zeros(2, 4, 8, 10), 4)
    assert b.shape == (2, 8, 10, 4)


def test_multiclass_decode_returns_class_column_and_keeps_both_classes_in_one_cell():
    out = {"cls": torch.full((1, 3, 16, 20), -10.0), "box": torch.full((1, 4, 16, 20), RAW5PX)}
    out["cls"][0, 1, 5, 7] = 4.0
    out["cls"][0, 2, 5, 7] = 3.0      # another class in the same cell - a separate detection
    out["cls"][0, 0, 12, 15] = 2.0
    d = decode(out)[0]
    assert d.shape == (3, 6)
    assert d[:, 5].tolist() == [1.0, 2.0, 0.0] and d[0, 4] > d[1, 4] > d[2, 4]
    np.testing.assert_allclose(d[0, :4], [25, 17, 35, 27], atol=1e-3)
