import json

import numpy as np
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import build_cache_thermal
from hae.data.wisard_ir import convert_wisard_ir, is_ir_sequence, site_of

H, W = 64, 96


def _seq(root, name, n, people=(), unlabelled=False, ext="jpg"):
    """n frames; people = indices of frames with one person; the rest have an empty .txt (true negatives)."""
    d = root / name
    d.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((H, W, 3), 100 + i, np.uint8)).save(d / f"{name}_{i:08d}.{ext}")
        if not unlabelled:
            (d / f"{name}_{i:08d}.txt").write_text("0 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.1 0.1\n" if i in people else "")
    if unlabelled:
        (d / "count.txt").write_text("0")


def _src(tmp_path):
    root = tmp_path / "src"
    _seq(root, "200704_Baker_FLIR_IR_1", 40, people=range(0, 40, 2))          # 20 frames with people, 20 empty
    _seq(root, "200929_Karen_FLIR_IR_1", 30, people=range(0, 30, 2), ext="jpeg")
    _seq(root, "200910_Carnation_VIS_1", 10, people=range(10))                 # VIS is ignored
    _seq(root, "210924_FHL_Enterprise_IR_0127", 20, unlabelled=True)           # unlabelled: not negatives
    _seq(root, "210327_Airfield_FLIR_IR_1", 12, people=[0, 1, 2])
    _seq(root, "200626_Everson_FLIR_IR_1", 12, people=range(12))               # site in neither train nor val
    return root


def test_site_and_ir_filter():
    assert site_of("200704_Baker_FLIR_IR_1") == "Baker"
    assert is_ir_sequence("210417_MtErie_Enterprise_IR_0004") and not is_ir_sequence("210327_Airfield_FLIR_VIS_1")


def test_split_by_site_subsample_and_negatives(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    r = convert_wisard_ir(src, out, train_every=4, val_every=3, neg_frac=0.2)
    # train = Baker: 20 frames with people, every 4th -> 5; negatives ~20 % of 5 = 1
    assert r["wisard_ir_train"]["frames"] == 5 + 1 and r["wisard_ir_train"]["negatives"] == 1
    # val = Karen (15 with people -> 5) + Airfield (3 -> 1); 20 % negatives: Karen 1, Airfield round(0.2)=0
    v = r["wisard_ir_val"]
    assert v["boxes"] == v["frames"] - v["negatives"]          # one person per frame (class 1 dropped)
    fr = load_annotations(out / "wisard_ir_train" / "ann.json")
    names = [f.file_name for f in fr]
    assert all("Baker" in n for n in names) and not any("Karen" in n or "VIS" in n or "Everson" in n or "FHL" in n for n in names)
    assert not any(" " in n for n in names)


def test_no_leakage_between_train_and_val(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    convert_wisard_ir(src, out)
    tr = {f.file_name.split("_")[1] for f in load_annotations(out / "wisard_ir_train" / "ann.json")}
    va = {f.file_name.split("_")[1] for f in load_annotations(out / "wisard_ir_val" / "ann.json")}
    assert tr == {"Baker"} and va == {"Karen", "Airfield"}


def test_ann_format_and_boxes_in_pixels(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    convert_wisard_ir(src, out)
    d = json.loads((out / "wisard_ir_train" / "ann.json").read_text())
    assert {c["id"] for c in d["categories"]} == {0, 3}
    im = d["images"][0]
    assert (im["width"], im["height"]) == (W, H) and (out / "wisard_ir_train" / "thermal" / im["file_name"]).exists()
    pos = [a for a in d["annotations"] if a["image_id"] == im["id"]]
    assert pos and pos[0]["category_id"] == 0
    np.testing.assert_allclose(pos[0]["bbox"], [(0.5 - 0.125) * W, (0.5 - 0.25) * H, 0.25 * W, 0.5 * H])
    assert Image.open(out / "wisard_ir_train" / "thermal" / im["file_name"]).mode == "L"


def test_missing_label_file_gives_empty_frame_in_labelled_sequence(tmp_path):
    src = tmp_path / "src"
    _seq(src, "200704_Baker_FLIR_IR_1", 8, people=[0, 4])
    (src / "200704_Baker_FLIR_IR_1" / "200704_Baker_FLIR_IR_1_00000002.txt").unlink()   # no label file
    out = tmp_path / "out"
    r = convert_wisard_ir(src, out, train_every=1, neg_frac=3.0)
    names = [f.file_name for f in load_annotations(out / "wisard_ir_train" / "ann.json")]
    assert r["wisard_ir_train"]["frames"] == 2 + 6            # 2 with people + 6 empty (5 with an empty .txt and 1 without .txt)
    assert any("00000002" in n for n in names)


def test_cache_builds_from_output(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    r = convert_wisard_ir(src, out)
    build_cache_thermal(out / "wisard_ir_train", "wisard_ir_train", tmp_path / "cache")
    meta = json.loads((tmp_path / "cache" / "wisard_ir_train_meta.json").read_text())
    assert meta["n"] == r["wisard_ir_train"]["frames"] and (meta["h"], meta["w"]) == (H, W)
