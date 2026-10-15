import numpy as np

from hae.metrics import evaluate_detections

HW = (512, 640)


def _scene(rng, n_frames=20, per=8, size=10):
    gts = []
    for _ in range(n_frames):
        xy = rng.uniform(20, 500, size=(per, 2))
        gts.append(np.concatenate([xy, xy + size], 1).astype(np.float32))
    return gts


def _dets_from(gts, shift=0.0, score=0.9):
    out = []
    for g in gts:
        d = np.concatenate([g + np.array([shift, 0, shift, 0], np.float32), np.full((len(g), 1), score, np.float32)], 1)
        out.append(d.astype(np.float32))
    return out


def test_perfect_detections_give_ap_one():
    gts = _scene(np.random.default_rng(0))
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    r = evaluate_detections(gts, empty, _dets_from(gts), HW)
    assert r["AP50"] > 0.999 and r["AP"] > 0.999


def test_ap_decreases_monotonically_with_localisation_error():
    gts = _scene(np.random.default_rng(1))
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    aps = [evaluate_detections(gts, empty, _dets_from(gts, s), HW)["AP"] for s in (0, 1, 2, 3)]  # IoU 1 / 0.82 / 0.67 / 0.54 for 10x10 boxes
    assert all(a > b for a, b in zip(aps, aps[1:])), aps


def test_no_detections_zero_and_empty_frames_ok():
    gts = _scene(np.random.default_rng(2), n_frames=4)
    gts[1] = np.zeros((0, 4), np.float32)
    empty = [np.zeros((0, 4), np.float32)] * 4
    r = evaluate_detections(gts, empty, [np.zeros((0, 5), np.float32)] * 4, HW)
    assert r["AP50"] == 0.0
    # detections in a frame without targets are FP, not a crash
    dets = _dets_from(gts)
    dets[1] = np.array([[10, 10, 20, 20, 0.9]], np.float32)
    assert 0.0 < evaluate_detections(gts, empty, dets, HW)["AP50"] <= 1.0


def test_detections_inside_ignore_regions_are_not_false_positives():
    gts = [np.array([[100, 100, 110, 110]], np.float32)]
    ign = [np.array([[300, 300, 400, 400]], np.float32)]
    clean = [np.array([[100, 100, 110, 110, 0.9]], np.float32)]
    noisy = [np.array([[100, 100, 110, 110, 0.9], [320, 320, 330, 330, 0.95]], np.float32)]
    a = evaluate_detections(gts, ign, clean, HW)["AP50"]
    b = evaluate_detections(gts, ign, noisy, HW)["AP50"]
    assert abs(a - b) < 1e-6 and a > 0.999


def _mc_scene(rng):
    gts, labels = [], []
    for _ in range(12):
        n = 9
        xy = rng.uniform(20, 480, size=(n, 2))
        wh = np.where(np.arange(n)[:, None] < 6, 10.0, 34.0)       # 6 tiny (10 px) + 3 larger (34 px)
        gts.append(np.concatenate([xy, xy + wh], 1).astype(np.float32))
        labels.append((np.arange(n) % 3).astype(np.int64))
    return gts, labels


def _mc_dets(gts, labels, keep=None, swap=False):
    out = []
    for g, l in zip(gts, labels):
        m = np.ones(len(g), bool) if keep is None else keep(g)
        cls = ((l + 1) % 3 if swap else l)[m]
        out.append(np.concatenate([g[m], np.full((m.sum(), 1), 0.9), cls[:, None]], 1).astype(np.float32))
    return out


def test_protocol_perfect_detections_all_metrics_one():
    from hae.metrics import evaluate_protocol
    gts, labels = _mc_scene(np.random.default_rng(0))
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    r = evaluate_protocol(gts, labels, empty, _mc_dets(gts, labels), HW)
    assert r["AP50"] > 0.999 and r["AP25"] > 0.999 and r["AP50_tiny"] > 0.999
    assert set(r["per_class"]) == {"person", "rider", "crowd"} and all(v > 0.999 for v in r["per_class"].values())


def test_protocol_wrong_class_gets_zero():
    from hae.metrics import evaluate_protocol
    gts, labels = _mc_scene(np.random.default_rng(1))
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    assert evaluate_protocol(gts, labels, empty, _mc_dets(gts, labels, swap=True), HW)["AP50"] < 0.01


def test_protocol_tiny_bucket_and_ignore_regions():
    from hae.metrics import evaluate_protocol
    gts, labels = _mc_scene(np.random.default_rng(2))
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    tiny_only = _mc_dets(gts, labels, keep=lambda g: (g[:, 2] - g[:, 0]) < 20)
    r = evaluate_protocol(gts, labels, empty, tiny_only, HW)
    assert r["AP50_tiny"] > 0.999 and r["AP50"] < 0.9          # large targets not found -> overall AP is lower
    ign = [np.array([[0, 0, 640, 512]], np.float32)] * len(gts)  # "uncertain" over the whole frame: FPs are absorbed
    extra = [np.concatenate([d, np.array([[300, 300, 310, 310, 0.99, 1]], np.float32)]) for d in _mc_dets(gts, labels)]
    r2 = evaluate_protocol(gts, labels, ign, extra, HW)
    assert r2["AP50"] > 0.999


def test_protocol_class_without_gt_is_excluded_from_mean():
    from hae.metrics import evaluate_protocol
    gts, labels = _mc_scene(np.random.default_rng(3))
    labels = [np.zeros_like(l) for l in labels]
    empty = [np.zeros((0, 4), np.float32)] * len(gts)
    r = evaluate_protocol(gts, labels, empty, _mc_dets(gts, labels), HW)
    assert r["AP50"] > 0.999 and "rider" not in r["per_class"]
