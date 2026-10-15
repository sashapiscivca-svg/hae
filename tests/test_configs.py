"""configs/<name>.yaml reproduce the training recipe recorded in models/<name>/config.json exactly."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train as T  # noqa: E402  (scripts/train.py)
from hae.train import TrainConfig  # noqa: E402

CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
LEGACY_NAMES = {"sds_train": "seadronessee_train", "hrd_train": "heridal_train"}


def _norm(v):
    if isinstance(v, (tuple, list)):
        return [_norm(x) for x in v]
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.stem)
def test_recipe_matches_released_training_config(cfg, tmp_path):
    rec = T.load_recipe(cfg)
    assert rec["name"] == cfg.stem
    released = ROOT / "models" / cfg.stem / "config.json"
    if not released.exists():
        pytest.skip(f"{released} not present")
    tr = json.loads(released.read_text())["training"]
    mine = asdict(TrainConfig(**rec["train"]))
    for k, v in tr.items():
        if k in ("extra",):
            continue
        assert _norm(mine[k]) == _norm(v), k
    # the original run mounted an unused main set with weight 0; here the first listed set is the main one with the same weights
    want = {LEGACY_NAMES.get(e["name"], e["name"]): e["weight"] for e in tr["extra"]}
    got = {s["name"]: s["weight"] for s in rec["train_sets"]}
    assert {k: v for k, v in got.items() if k in want} == want and abs(sum(got.values()) - 1) < 1e-9


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.stem)
def test_build_config_maps_first_set_to_main_and_rest_to_extras(cfg, tmp_path):
    rec = T.load_recipe(cfg)
    cache = tmp_path / "cache"
    cache.mkdir()
    for s in [x["name"] for x in rec["train_sets"]] + rec["val_sets"]:
        (cache / f"{s}_meta.json").write_text(json.dumps({"n": 10, "h": 32, "w": 32}))
    c = T.build_config(rec, tmp_path, tmp_path / "run", epochs=2)
    assert c.train_set == rec["train_sets"][0]["name"] and c.epochs == 2
    assert [e["name"] for e in c.extra or []] == [s["name"] for s in rec["train_sets"][1:]]
    assert abs(1.0 - sum(e["weight"] for e in c.extra or []) - rec["train_sets"][0]["weight"]) < 1e-9
    assert c.eval_split == rec["val_sets"][0] and c.eval_frames == list(range(0, 10, rec.get("eval_stride", 1)))


def test_build_config_reports_unprepared_sets(tmp_path):
    rec = T.load_recipe(CONFIGS[0])
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        T.build_config(rec, tmp_path, tmp_path / "run")
