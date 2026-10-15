import numpy as np

from hae.recall import recall_at


def test_recall_precision_and_fp_per_frame_at_threshold():
    gts = [np.array([[10, 10, 20, 20], [40, 40, 50, 50]], np.float32), np.zeros((0, 4), np.float32)]
    ign = [np.zeros((0, 4), np.float32)] * 2
    dets = [np.array([[10, 10, 20, 20, 0.9, 0],       # TP
                      [30, 30, 38, 38, 0.8, 0],       # FP
                      [40, 40, 50, 50, 0.2, 0]], np.float32),   # TP, but below threshold 0.5
            np.array([[5, 5, 12, 12, 0.7, 0]], np.float32)]     # FP on an empty frame
    r = recall_at(gts, ign, dets, thr=0.5)
    assert r["gt"] == 2 and r["tp"] == 1 and r["fp"] == 2
    assert abs(r["recall"] - 0.5) < 1e-9 and abs(r["fp_per_frame"] - 1.0) < 1e-9 and abs(r["precision"] - 1 / 3) < 1e-9
    r2 = recall_at(gts, ign, dets, thr=0.1)
    assert r2["tp"] == 2 and r2["recall"] == 1.0


def test_detections_inside_ignore_regions_are_not_false_positives():
    gts = [np.zeros((0, 4), np.float32)]
    ign = [np.array([[0, 0, 30, 30]], np.float32)]
    dets = [np.array([[10, 10, 18, 18, 0.9, 0]], np.float32)]
    assert recall_at(gts, ign, dets, thr=0.5)["fp"] == 0


def test_empty_detections_and_empty_gt_do_not_crash():
    gts = [np.zeros((0, 4), np.float32), np.array([[1, 1, 5, 5]], np.float32)]
    ign = [np.zeros((0, 4), np.float32)] * 2
    r = recall_at(gts, ign, [np.zeros((0,), np.float32), np.zeros((0, 6), np.float32)], thr=0.3)
    assert r["tp"] == 0 and r["fp"] == 0 and r["recall"] == 0.0 and r["gt"] == 1


def test_center_matching_counts_clustered_duplicates_once_and_tolerates_loose_boxes():
    from hae.recall import recall_center
    gts = [np.array([[40, 40, 48, 48]], np.float32)]
    ign = [np.zeros((0, 4), np.float32)]
    dets = [np.array([[44, 40, 54, 50, 0.9, 0],       # IoU with the person < 0.5, but the centre is inside the box -> hit
                      [43, 41, 53, 51, 0.8, 0],       # duplicate of the same detection (centre within 8 px) -> not a separate alarm
                      [10, 10, 18, 18, 0.7, 0]], np.float32)]   # a real false alarm
    r = recall_center(gts, ign, dets, thr=0.5)
    assert r["tp"] == 1 and r["fp"] == 1 and r["recall"] == 1.0 and r["fp_per_frame"] == 1.0
    strict = recall_at(gts, ign, dets, thr=0.5)
    assert strict["tp"] == 0 and strict["fp"] == 3          # with IoU 0.5 the same gives 0 hits and 3 false alarms


def test_frame_level_alarms():
    from hae.recall import frame_level
    gts = [np.array([[1, 1, 9, 9]], np.float32), np.zeros((0, 4), np.float32), np.array([[1, 1, 9, 9]], np.float32), np.zeros((0, 4), np.float32)]
    ign = [np.zeros((0, 4), np.float32)] * 4
    dets = [np.array([[2, 2, 8, 8, 0.9, 0]], np.float32), np.array([[20, 20, 30, 30, 0.9, 0]], np.float32), np.zeros((0, 5), np.float32), np.zeros((0, 5), np.float32)]
    r = frame_level(gts, ign, dets, thr=0.5)
    assert r["frame_recall"] == 0.5 and r["false_alarm_frames"] == 0.5     # 1 of 2 frames with people found; a false alarm on 1 of 2 empty frames


def test_presets_balanced_by_f1_and_sensitive_by_f2():
    from hae.recall import choose_presets
    sets = {"a": {"0.1": {"recall": 0.98, "precision": 0.25}, "0.2": {"recall": 0.6, "precision": 0.6}, "0.4": {"recall": 0.2, "precision": 0.8}}}
    p = choose_presets(sets)
    assert p["balanced"] == 0.2 and p["sensitive"] == 0.1                 # best F1 at 0.2; F2 (recall-weighted) at 0.1
