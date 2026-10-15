"""scripts/prepare_data.py: conversion -> sets -> caches, alignment to the published annotation files. No network access."""
import json
import sys
from pathlib import Path

import pytest

from hae.data.annotations import load_annotations
from hae.data.cache import Cache
from test_extra import _fake_hituav

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data as P  # noqa: E402

COUNTS = {"hituav_test": (579, 2611), "airesq_val": (502, 933), "wisard_ir_val": (1633, 2439)}


def test_sources_list_urls_and_licences(capsys):
    assert P.main(["sources"]) == 0
    out = capsys.readouterr().out
    for frag in ("zenodo.org/records/7633134", "CC0-1.0", "zenodo.org/records/17405074", "CC-BY-4.0", "sites.google.com/uw.edu/wisard", "MIT"):
        assert frag in out, frag
    with pytest.raises(SystemExit):
        P.main(["--help"])
    assert "7633134" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(COUNTS))
def test_published_annotation_files_parse_and_counts_match(name):
    f = ROOT / "data" / "annotations" / f"{name}.json"
    if not f.exists():
        pytest.skip(f"{f} not present")
    d = json.loads(f.read_text())
    assert (len(d["images"]), sum(a["category_id"] == 0 for a in d["annotations"])) == COUNTS[name]
    assert {c["id"] for c in d["categories"]} == {0, 3}
    assert {(i["height"], i["width"]) for i in d["images"]} == {(512, 640)}


def test_published_trainval_sets_exist_for_the_recipe():
    for n in ("hituav_trainval", "airesq_train"):
        if (ROOT / "data" / "annotations" / f"{n}.json").exists():
            assert json.loads((ROOT / "data" / "annotations" / f"{n}.json").read_text())["images"]


def test_hituav_no_align_builds_converter_split_and_caches(tmp_path):
    _fake_hituav(tmp_path / "src")
    rep = P.prepare("hituav", tmp_path / "src", tmp_path / "data", do_align=False)
    assert rep["hituav_trainval"]["kept"] == 2 and rep["hituav_test"]["kept"] == 1 and rep["hituav_test"]["boxes"] == 1
    for name, n in (("hituav_trainval", 2), ("hituav_test", 1)):
        c = Cache(tmp_path / "data" / "cache", name)
        assert c.n == n and c.rgb is None
        anns = load_annotations(tmp_path / "data" / "cache" / f"{name}_ann.json")
        assert len(anns) == n and all(len(a.boxes) == 1 for a in anns)


def test_hituav_aligned_to_published_keeps_listed_frames_and_reports_missing(tmp_path):
    _fake_hituav(tmp_path / "src")
    pub = tmp_path / "pub"
    pub.mkdir()
    cats = [{"id": 0, "name": "person"}, {"id": 3, "name": "ignore"}]
    (pub / "hituav_test.json").write_text(json.dumps({"categories": cats,
        "images": [{"id": 0, "file_name": "hituav_test_test_a.jpg", "height": 64, "width": 96}, {"id": 1, "file_name": "hituav_test_nope.jpg", "height": 64, "width": 96}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [11, 11, 8, 12], "ignore": 0, "iscrowd": 0},
                        {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5], "ignore": 0, "iscrowd": 0}]}))
    rep = P.prepare("hituav", tmp_path / "src", tmp_path / "data", do_align=True, annotations=pub)
    t = rep["hituav_test"]
    assert t["aligned"] and t["kept"] == 1 and t["missing"] == ["hituav_test_nope.jpg"] and t["label_mismatch"] == 1
    a = load_annotations(tmp_path / "data" / "cache" / "hituav_test_ann.json")
    assert len(a) == 1 and a[0].boxes.tolist() == [[11, 11, 19, 23]]                   # published labels are used
    assert rep["hituav_trainval"]["aligned"] is False and rep["hituav_trainval"]["kept"] == 2   # no published file -> converter split
    assert (tmp_path / "data" / "sets" / "hituav_test" / "thermal" / "hituav_test_test_a.jpg").exists()


def test_airesq_split_holds_out_whole_recordings():
    assert P._airesq_group("airesq_DJI_20240101120000_0001_T.jpg") == P._airesq_group("airesq_DJI_20240101125959_0002_T.jpg")
    assert P._airesq_group("airesq_DJI_0783_T.jpg") == "DJI_0783"


def test_missing_src_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="normal_json"):
        P.prepare("hituav", tmp_path / "nope", tmp_path / "data")
