"""Plots and results.csv for training runs: training curves and P/R/F1 vs. confidence threshold. matplotlib (Agg backend)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def _flat(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update({f"{k}_{ck}": cv for ck, cv in v.items() if isinstance(cv, (int, float))})
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def write_results_csv(history: list, path: str | Path) -> Path:
    rows = [_flat(r) for r in history]
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = Path(path)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def plot_results(history: list, path: str | Path) -> Path:
    rows = [_flat(r) for r in history]
    ep = [r["epoch"] for r in rows]
    modes = sorted({k.split("_")[1] for r in rows for k in r if k.startswith("eval_") and k.endswith("_AP50")})
    panels = [("training loss", [("loss", "loss")]), ("learning rate", [("lr", "lr")]), ("epoch time, s", [("epoch_s", "s")])]
    for m in modes:
        panels.append((f"AP50 ({m})", [(f"eval_{m}_AP50", "AP50")]))
        panels.append((f"AP and AR ({m})", [(f"eval_{m}_AP", "AP"), (f"eval_{m}_AR", "AR")]))
        pcs = sorted({k for r in rows for k in r if k.startswith(f"eval_{m}_per_class_")})
        if pcs:
            panels.append((f"AP50 per class ({m})", [(k, k.rsplit("_", 1)[1]) for k in pcs]))
    n = len(panels)
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 3.0 * ((n + cols - 1) // cols)), squeeze=False)
    for ax, (title, series) in zip(axes.flat, panels):
        for key, label in series:
            xs = [e for e, r in zip(ep, rows) if key in r]
            ys = [r[key] for r in rows if key in r]
            ax.plot(xs, ys, marker="o" if len(xs) < 25 else None, ms=3, lw=1.6, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _f1_best(curves: dict):
    thr = sorted(curves)
    p = [curves[t]["precision"] for t in thr]
    r = [curves[t]["recall"] for t in thr]
    fp = [curves[t]["fp_per_frame"] for t in thr]
    f1 = [2 * a * b / max(a + b, 1e-9) for a, b in zip(p, r)]
    return thr, p, r, f1, fp, thr[max(range(len(thr)), key=lambda i: f1[i])]


def draw_curves(a1, a2, curves: dict) -> float:
    """Draws P/R/F1 vs. confidence threshold and recall vs. false alarms on two axes; returns the threshold with the best F1."""
    thr, p, r, f1, fp, best = _f1_best(curves)
    a1.plot(thr, p, label="precision", lw=1.8)
    a1.plot(thr, r, label="recall", lw=1.8)
    a1.plot(thr, f1, label="F1", lw=2.2, color="k")
    a1.axvline(best, color="r", ls="--", lw=1)
    a1.set_title(f"P / R / F1 vs. confidence threshold (best F1 at {best:.2f})", fontsize=10)
    a1.set_xlabel("confidence threshold")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.plot(fp, r, marker="o", ms=3, lw=1.8)
    a2.set_xscale("symlog", linthresh=0.1)
    a2.set_title("recall vs. false alarms per frame", fontsize=10)
    a2.set_xlabel("false alarms per frame")
    a2.set_ylabel("recall")
    a2.grid(alpha=0.3)
    return best


def plot_curves(curves: dict, path: str | Path) -> tuple:
    """curves: {threshold: {recall, precision, fp_per_frame}} -> PNG and the threshold with the best F1."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    best = draw_curves(a1, a2, curves)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, best
