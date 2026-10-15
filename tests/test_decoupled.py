import torch

from hae.export import export_onnx
from hae.model.presets import build


def test_decoupled_head_branches_are_independent():
    m = build("n", overrides={"decoupled": True}, ncls=3).eval()
    x = torch.rand(1, 1, 64, 96)
    with torch.no_grad():
        a = m(t=x)
        for p in m.trunk.parameters():          # touch only the cls branch
            p.add_(0.5)
        b = m(t=x)
    assert torch.equal(a["box"], b["box"]) and not torch.equal(a["cls"], b["cls"])


def test_decoupled_adds_one_trunk_and_keeps_shapes():
    base, dec = build("n"), build("n", overrides={"decoupled": True})
    assert sum(p.numel() for p in dec.parameters()) > sum(p.numel() for p in base.parameters())
    x = torch.rand(1, 1, 64, 96)
    assert dec(t=x)["box"].shape == base(t=x)["box"].shape == (1, 4, 16, 24)


def test_decoupled_works_with_head_s2_and_exports(tmp_path):
    m = build("n", block="repghost", overrides={"decoupled": True, "head_s2": True}).eval()
    assert m(t=torch.rand(1, 1, 64, 96))["cls"].shape[-2:] == (32, 48)
    assert export_onnx(m, tmp_path / "d.onnx", mode="lwir", hw=(64, 96))["bytes"] > 0
