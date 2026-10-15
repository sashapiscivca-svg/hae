import io

from rich.console import Console

from hae import ui


def _capture(width=120):
    buf = io.StringIO()
    ui.set_console(Console(file=buf, force_terminal=False, width=width, highlight=False))
    return buf


REC = {"epoch": 7, "loss": 1.2347, "lr": 0.0021, "epoch_s": 9.8, "eval_lwir_AP50": 0.1503, "eval_lwir_AP": 0.0381, "eval_lwir_AR": 0.1912,
       "eval_lwir_per_class": {"person": 0.22, "rider": 0.11}}


def test_epoch_line_is_compact_and_keeps_machine_readable_prefix():
    line = ui.epoch_line(REC, epochs=30, best=True)
    assert line.startswith("epoch 7/30 loss=1.235 9.8 s/epoch")         # format parsed by log-monitoring scripts
    assert "lwir" in line and "AP50 0.150" in line and "★" in line
    assert "★" not in ui.epoch_line(REC, epochs=30, best=False)
    assert "AP50" not in ui.epoch_line({k: v for k, v in REC.items() if not k.startswith("eval_")}, epochs=30, best=False)


def test_banner_and_tables_render_without_terminal():
    buf = _capture()
    ui.banner("hae-n-ir", {"Model": "53 647 parameters, 0.052 GMAC", "Device": "cpu"})
    ui.table("Data", ["dataset", "frames", "weight"], [["RGBTDronePerson", 4900, "0,65"], ["HIT-UAV", 2319, "0,15"]])
    out = buf.getvalue()
    assert "hae-n-ir" in out and "53 647" in out and "RGBTDronePerson" in out and "4900" in out


def test_final_summary_lists_per_class_and_artifacts(tmp_path):
    buf = _capture()
    ui.final_summary(best={"epoch": 5, "metric": 0.21, "metric_key": "eval_lwir_AP50"}, rec=REC, total_s=125.0, out_dir=tmp_path, recommended_conf=0.15)
    out = buf.getvalue()
    assert "person" in out and "rider" in out and "0.220" in out and "best" in out.lower() and "0.15" in out and "2 min" in out


def test_epoch_bar_is_noop_without_terminal_and_does_not_fail():
    _capture()
    with ui.EpochBar(total=10, epoch=1, epochs=3) as bar:
        for k in range(10):
            bar.update(1.0, 1e-3)
    ui.set_console(None)
