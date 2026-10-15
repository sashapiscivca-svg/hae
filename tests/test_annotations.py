import json
from pathlib import Path

import numpy as np
import pytest

from hae.data.annotations import load_annotations


def _write(tmp_path, anns, n_images=3):
    d = {
        "images": [{"id": i, "file_name": f"{i:05d}.jpg", "height": 512, "width": 640} for i in range(n_images)],
        "categories": [{"id": 0, "name": "person"}],
        "annotations": [
            {"id": k, "image_id": a[0], "category_id": a[1], "bbox": a[2], "ignore": a[3] if len(a) > 3 else 0, "iscrowd": 0}
            for k, a in enumerate(anns)
        ],
    }
    p = tmp_path / "x.json"
    p.write_text(json.dumps(d))
    return p


def test_class_mapping_and_xyxy(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 20, 8, 12]), (0, 1, [50, 50, 6, 6]), (0, 2, [100, 100, 30, 30]), (0, 3, [5, 5, 4, 4])])
    fr = load_annotations(p)[0]
    np.testing.assert_allclose(fr.boxes, [[10, 20, 18, 32], [50, 50, 56, 56]])
    np.testing.assert_allclose(fr.ignore, [[100, 100, 130, 130], [5, 5, 9, 9]])


def test_ignore_flag_moves_positive_to_ignore(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 10, 8, 8], 1)])
    fr = load_annotations(p)[0]
    assert fr.boxes.shape == (0, 4) and fr.ignore.shape == (1, 4)


def test_frame_without_annotations_has_empty_arrays(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 10, 8, 8])])
    frames = load_annotations(p)
    assert len(frames) == 3
    assert frames[2].boxes.shape == (0, 4) and frames[2].ignore.shape == (0, 4)
    assert frames[2].boxes.dtype == np.float32


def test_degenerate_boxes_are_dropped(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 10, 0, 8]), (0, 0, [10, 10, 8, 0]), (0, 0, [1, 1, 5, 5])])
    fr = load_annotations(p)[0]
    np.testing.assert_allclose(fr.boxes, [[1, 1, 6, 6]])


@pytest.mark.data
def test_real_train_counts():
    root = Path(__file__).resolve().parents[1] / "data" / "RGBTDronePerson"
    if not root.exists():
        pytest.skip("dataset not available")
    frames = load_annotations(root / "train_thermal.json")
    assert len(frames) == 4900
    total = sum(len(f.boxes) + len(f.ignore) for f in frames)
    assert 0.99 * 56314 <= total <= 56314  # only degenerate boxes are dropped (w or h <= 0)
    # in the JSON 3 frames (00065, 00067, 00076) have stale 630x500 metadata; the real JPEGs are 640x512
    odd = [f.file_name for f in frames if (f.height, f.width) != (512, 640)]
    assert odd == ["00065.jpg", "00067.jpg", "00076.jpg"]


def test_coco3_mode_keeps_three_classes_and_ignores_only_uncertain(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 20, 8, 12]), (0, 1, [50, 50, 6, 6]), (0, 2, [100, 100, 30, 30]), (0, 3, [5, 5, 4, 4])])
    fr = load_annotations(p, mode="coco3")[0]
    np.testing.assert_allclose(fr.boxes, [[10, 20, 18, 32], [50, 50, 56, 56], [100, 100, 130, 130]])
    assert fr.labels.tolist() == [0, 1, 2]
    np.testing.assert_allclose(fr.ignore, [[5, 5, 9, 9]])


def test_person_mode_labels_are_all_zero_and_aligned_with_boxes(tmp_path):
    p = _write(tmp_path, [(0, 0, [10, 20, 8, 12]), (0, 1, [50, 50, 6, 6]), (0, 2, [100, 100, 30, 30])])
    fr = load_annotations(p)[0]
    assert fr.labels.tolist() == [0, 0] and len(fr.labels) == len(fr.boxes)
    empty = load_annotations(p)[2]
    assert empty.labels.shape == (0,) and empty.labels.dtype == np.int64


def test_unknown_mode_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="mode"):
        load_annotations(_write(tmp_path, []), mode="x")
