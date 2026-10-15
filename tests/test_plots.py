import pytest

pytest.importorskip("matplotlib")
from hae.plots import plot_curves, plot_results, write_results_csv


def _hist(n=6):
    return [{"epoch": i + 1, "loss": 2.0 - 0.2 * i, "lr": 1e-3 * (1 - i / n), "epoch_s": 9.0, "eval_lwir_AP50": 0.05 * i, "eval_lwir_AP": 0.01 * i,
             "eval_lwir_AR": 0.1 + 0.02 * i, "eval_lwir_per_class": {"person": 0.06 * i, "rider": 0.03 * i}} for i in range(n)]


def test_results_csv_and_png_are_written(tmp_path):
    h = _hist()
    csv = write_results_csv(h, tmp_path / "results.csv")
    lines = csv.read_text().splitlines()
    assert lines[0].split(",")[:3] == ["epoch", "loss", "lr"] and len(lines) == 1 + len(h) and "eval_lwir_per_class_person" in lines[0]
    png = plot_results(h, tmp_path / "results.png")
    assert png.exists() and png.stat().st_size > 5000


def test_curves_png_and_best_threshold(tmp_path):
    thr = [0.05, 0.1, 0.2, 0.3, 0.5]
    curves = {t: {"recall": r, "precision": p, "fp_per_frame": fp} for t, r, p, fp in zip(thr, [0.9, 0.8, 0.6, 0.4, 0.05], [0.1, 0.3, 0.6, 0.8, 0.9], [9, 2, 0.4, 0.1, 0.0])}
    png, best = plot_curves(curves, tmp_path / "curves.png")
    assert png.exists() and png.stat().st_size > 5000 and best == 0.2          # best F1 at threshold 0.2
