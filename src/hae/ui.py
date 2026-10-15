"""hae console UI: banner, tables, epoch progress, metrics line, summary. Uses rich only.

In a terminal: coloured panels/tables and a progress bar with ETA; in file logs (e.g. cloud notebooks): compact lines without animation.
The epoch line starts with `epoch N/M loss=... s/epoch` (a format parsed by monitoring scripts), followed by metrics.
"""
from __future__ import annotations

from pathlib import Path

try:                       # rich is optional (training extra); without it everything is printed as plain text
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


class PlainConsole:
    """Minimal stand-in for rich.Console when rich is not installed: plain print, records output for export_text()."""
    is_terminal = False

    def __init__(self):
        self._buf: list = []

    def print(self, *objs, **kw) -> None:
        text = " ".join(str(o) for o in objs)
        self._buf.append(text)
        print(text, flush=True)

    def export_text(self) -> str:
        return "\n".join(self._buf) + "\n"


_console = None
_lines: list = []   # copy of epoch lines for train.log if the console does not record


def console():
    global _console
    if _console is None:
        _console = Console(highlight=False, record=True) if HAVE_RICH else PlainConsole()
    return _console


def set_console(c) -> None:
    global _console
    _console = c


def banner(title: str, info: dict) -> None:
    if not HAVE_RICH:
        w = max((len(str(k)) for k in info), default=0)
        console().print("\n".join([f"== {title} =="] + [f"  {str(k).ljust(w)}  {v}" for k, v in info.items()]))
        return
    t = Table.grid(padding=(0, 2))
    for k, v in info.items():
        t.add_row(f"[bold cyan]{k}[/]", str(v))
    console().print(Panel(t, title=f"[bold]{title}[/]", border_style="cyan", box=box.ROUNDED, expand=False))


def table(title: str, columns: list, rows: list, **kw) -> None:
    if not HAVE_RICH:
        cells = [[str(c) for c in columns]] + [[str(x) for x in r] for r in rows]
        w = [max(len(r[i]) for r in cells if i < len(r)) for i in range(len(columns))]
        fmt = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
        console().print("\n".join([title, fmt(cells[0]), "  ".join("-" * x for x in w)] + [fmt(r) for r in cells[1:]]))
        return
    t = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold", title_style="bold cyan", **kw)
    for c in columns:
        t.add_column(str(c), justify="left" if c == columns[0] else "right")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console().print(t)


def _eval_modes(rec: dict) -> list:
    return sorted({k.split("_")[1] for k in rec if k.startswith("eval_") and k.endswith("_AP50")})


def epoch_line(rec: dict, epochs: int, best: bool = False) -> str:
    """One informative epoch line. The prefix is machine-readable: `epoch N/M loss=... s/epoch`; then loss components, instances, speed, memory,
    quick validation (after every epoch) and the full evaluation when present."""
    s = f"epoch {rec['epoch']}/{epochs} loss={rec['loss']:.3f} {rec['epoch_s']:.1f} s/epoch"
    if "loss_cls" in rec:
        s += f" | cls {rec['loss_cls']:.2f} box {rec['loss_box']:.2f}"
    if "inst_per_crop" in rec:
        s += f" · {rec['inst_per_crop']:.1f} inst/crop"
    s += f" | lr {rec['lr']:.1e}"
    if "grad_norm" in rec:
        s += f" gn {rec['grad_norm']:.1f}"
    if "crops_per_s" in rec:
        s += f" | {rec['crops_per_s']:.0f} crops/s, data wait {rec.get('data_wait_pct', 0):.0f}%"
    if rec.get("gpu_gb"):
        s += f" GPU {rec['gpu_gb']:.1f}G"
    for m in sorted({k.split("_")[1] for k in rec if k.startswith("qval_") and k.endswith("_AP50")}):
        s += f" | val AP50 {rec[f'qval_{m}_AP50']:.3f} AP {rec.get(f'qval_{m}_AP', 0):.3f} AR {rec.get(f'qval_{m}_AR', 0):.3f}"
        if "qval_recall" in rec:
            s += f" recall {rec['qval_recall']:.2f} FP {rec['qval_fp']:.1f}"
    for m in _eval_modes(rec):
        s += f" | FULL {m} AP50 {rec[f'eval_{m}_AP50']:.3f} AP {rec.get(f'eval_{m}_AP', 0):.3f} AR {rec.get(f'eval_{m}_AR', 0):.3f}"
    return s + ("  ★ best" if best else "")


def save_log(path) -> None:
    """Saves the whole training console output to a file (train.log) without colours."""
    try:
        text = console().export_text()
    except Exception:
        text = "\n".join(_lines) + "\n"
    Path(path).write_text(text)


def print_epoch(rec: dict, epochs: int, best: bool = False) -> None:
    line = epoch_line(rec, epochs, best)
    _lines.append(line)
    if console().is_terminal:
        console().print(f"[green]{line}[/]" if best else line, soft_wrap=True)
    else:
        console().print(line, soft_wrap=True)


class EpochBar:
    """Progress bar of epoch steps (terminal only): epoch, bar, steps, loss, lr, elapsed and remaining time. Prints nothing outside a terminal."""

    def __init__(self, total: int, epoch: int, epochs: int):
        self.total, self.epoch, self.epochs = total, epoch, epochs
        self.progress = None

    def __enter__(self):
        if console().is_terminal:
            self.progress = Progress(SpinnerColumn(), TextColumn("[bold]epoch {task.fields[ep]}/{task.fields[eps]}"), BarColumn(bar_width=28),
                                     TextColumn("{task.completed}/{task.total}"), TextColumn("loss {task.fields[loss]:.3f}"), TextColumn("lr {task.fields[lr]:.1e}"),
                                     TimeElapsedColumn(), TimeRemainingColumn(), console=console(), transient=True)
            self.progress.start()
            self.task = self.progress.add_task("", total=self.total, ep=self.epoch, eps=self.epochs, loss=0.0, lr=0.0)
        return self

    def update(self, loss: float, lr: float, n: int = 1) -> None:
        if self.progress is not None:
            self.progress.update(self.task, advance=n, loss=loss, lr=lr)

    def __exit__(self, *exc):
        if self.progress is not None:
            self.progress.stop()


def final_summary(best: dict, rec: dict, total_s: float, out_dir, recommended_conf: float | None = None) -> None:
    """Summary: best epoch, per-class AP50 of the last evaluation, time, recommended threshold, list of artefacts."""
    modes = _eval_modes(rec)
    for m in modes:
        pc = rec.get(f"eval_{m}_per_class") or {}
        rows = [[k, f"{v:.3f}"] for k, v in pc.items()] + [["all (mAP50)", f"{rec[f'eval_{m}_AP50']:.3f}"]]
        table(f"AP50 per class ({m})", ["class", "AP50"], rows)
    info = {"Best epoch": f"{best.get('epoch', '-')} ({best.get('metric_key', '')} = {best.get('metric', float('nan')):.3f})" if best.get("epoch") else "-",
            "Training time": f"{total_s / 60:.0f} min" if total_s >= 60 else f"{total_s:.0f} s"}
    if recommended_conf is not None:
        info["Recommended confidence threshold"] = f"{recommended_conf:.2f} (max F1 on validation)"
    out = Path(out_dir)
    arts = [n for n in ("best.pt", "last.pt", "dashboard.png", "results.png", "curves.png", "val_pred.png", "errors.png", "train_batch0.png", "labels.png", "val_samples", "tb") if (out / n).exists()]
    info["Results"] = f"{out}  ({', '.join(arts)})"
    banner("training summary", info)
