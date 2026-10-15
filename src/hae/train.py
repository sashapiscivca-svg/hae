"""hae training loop: modality dropout, EMA, cosine lr, CPU frequency log.

Every model is trained from scratch: no third-party pretrained weights are used anywhere."""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_halo_cache
from hae.data.sampler import CropSampler, MixedSampler, Prefetcher, load_frames
from hae.distill import DistillConfig, Distiller
from hae.loss import HaeLoss, soft_iou_loss
from hae import ui
from hae.dataset_stats import label_stats, plot_labels, save_batch_preview
from hae.mining import mine_hard_negatives
from hae.recall import recall_at, recall_center
from hae.data.sampler import build_style_bank
from hae.metrics import evaluate_detections, evaluate_protocol
from hae.model.decode import decode
from hae.model.align import edge_alignment_loss, flow_smoothness
from hae.model.presets import build

MODES = ("both", "rgb", "lwir")


@dataclass
class TrainConfig:
    cache_dir: str = "cache"
    ann_root: str = "data"
    out_dir: str = "runs/dev"
    block: str = "ghost"
    epochs: int = 1
    crops_per_epoch: int = 4900
    batch: int = 32
    lr: float = 3e-3
    wd: float = 1e-2
    warmup_steps: int = 50
    seed: int = 0
    mode_probs: tuple = (0.6, 0.2, 0.2)
    threads: int = 4
    crop_hw: tuple = (256, 320)
    frames: list | None = None          # subset of training frames (overfitting checks)
    eval_every: int = 1
    eval_frames: list | None = None     # frame indices in the eval_split cache
    eval_split: str = "train"           # cache set used for evaluation; a missing cache raises FileNotFoundError
    eval_modes: tuple = ("both", "rgb", "lwir")
    channels_last: bool = True          # noticeably faster training steps on CPU
    ema_decay: float = 0.998
    ckpt_every: int = 5                 # how often to write resume.pt/last.pt (epochs); always after the last epoch too
    resume: bool = False                # continue from out_dir/resume.pt if it exists
    init_weights: str | None = None     # start from these weights (fine-tuning): released .safetensors + config.json, or your own .pt
    best_metric: str = ""               # metric key for best.pt; empty = (q)eval_<first mode>_AP50
    tb: bool = True                     # TensorBoard (out_dir/tb); silently disabled without the tensorboard package
    tb_images: int = 3                  # validation frames with detections written to TensorBoard after each evaluation
    tb_every_steps: int = 25            # step interval for loss/grad-norm TensorBoard scalars
    quick_val: int = 64                 # frames of quick validation after EVERY epoch (YOLO-style), 0 = off; full eval every eval_every
    device: str = "auto"                # auto = cuda if available, else cpu
    amp: bool = False                   # fp16 autocast, cuda only; ignored on cpu
    size: str | None = None             # model size preset n/t/s/m/l/x; overrides block/width/depth when set
    width: float = 1.0                  # model width multiplier
    depth: float = 1.0
    scale_range: tuple = (1.0, 1.0)     # crop scale augmentation: larger/smaller targets (flight altitude)
    eval_scale: float = 1.0             # input scale at evaluation (for models trained on upscaled crops)
    halo: bool = False                  # auxiliary HALO mask (training only)
    aux_w: float = 1.0
    copy_paste: float = 0.0             # probability of pasting people with their local background into a crop
    tnorm: bool = False                 # per-frame thermal normalisation (frame statistics)
    class_weights: tuple | None = None  # per-class QFL weights, e.g. (1, 2.5, 1.5) for person/rider/crowd
    rare_crop: float = 0.0              # fraction of crops centred on rider/crowd
    ctx: bool = False                   # stride-32 context level
    blob: bool = False                  # DoG blob bank at the input
    align: bool = False                 # RGB->LWIR alignment + self-supervised NCC loss for the "both" mode
    align_w: float = 1.0
    smooth_w: float = 0.1
    box_w: float = 2.0                  # box regression loss weight
    cls_w: float = 1.0
    cfg_overrides: dict | None = None   # arbitrary HaeConfig fields for ablations, e.g. {"depths": [3,2,2]}
    backbone: str | None = None         # legacy field kept so that released config.json files load; must be None
    pretrained: bool = False            # legacy field; must be False (all models are trained from scratch)
    bb_lr_mult: float = 0.3             # legacy field; unused
    hn_frac: float = 0.0                # fraction of crops centred on mined hard negatives, 0 = off
    hn_start_epoch: int = 10            # epoch at which mining happens
    hn_frames: int = 1500               # training frames scanned by the model during mining
    hn_score: float = 0.2               # confidence threshold of a false positive
    fourier_p: float = 0.0              # probability of mixing the low-frequency spectrum with another domain, 0 = off
    fourier_band: float = 0.1           # fraction of the spectrum that is mixed
    taug: float = 0.0                   # thermal augmentation strength, 0 = off
    t_invert: float = 0.0               # probability of inverting thermal polarity of a crop, 0 = off
    pre_lnorm: int = 0                  # LWIR input = local_contrast_u8(k) of the full frame: 0 = raw frame; k = window (e.g. 61)
    photo_color: float = 1.0            # share of colour transforms in photometric augmentation (0 = tone, noise, blur only)
    photo: float = 0.0                  # RGB photometric augmentation strength, 0 = off
    neg_decay: float = 0.0              # soft negatives near targets, 0 = off
    main_modes: tuple | None = None     # modes allowed for the main set when it has all modalities, e.g. ("both","lwir")
    extra: list | None = None           # additional training sets: [{"name", "weight"[, "ann" | "unified"]}]
    classes: int = 1                    # 1 = person (+rider); 3 = person/rider/crowd
    verbose: bool = False               # print a line per epoch
    distill: dict | None = None         # teacher -> student distillation: {teacher, alpha, tau, feat, teacher_scale_ratio, ...}; None = off
    prefetch: int = 0                   # background batch-preparation threads (0 = main thread)
    train_set: str = "train"            # name of the main training set in cache_dir

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Build from a dict (checkpoint / released config.json), ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def ann_path(cfg: TrainConfig, name: str) -> Path:
    """Annotations of cache set `name`: <cache_dir>/<name>_ann.json (layout of scripts/prepare_data.py),
    otherwise <ann_root>/<name>_thermal.json (RGBTDronePerson-style dataset root)."""
    p = Path(cfg.cache_dir) / f"{name}_ann.json"
    return p if p.exists() else Path(cfg.ann_root) / f"{name}_thermal.json"


class ModelEma:
    def __init__(self, model, decay=0.998):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, s in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(s.detach(), alpha=1 - d)
            else:
                e.copy_(s)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto/cpu/cuda, got {name!r}")
    return torch.device(name)


def allowed_modes(batch_modes, main_modes=None) -> list:
    """Modes allowed for a batch. main_modes restricts only batches with all modalities (an RGBT main set whose RGB labels
    are in thermal coordinates, so the rgb mode must not learn from them); RGB-only sets stay rgb."""
    ok = [m for m in MODES if m in batch_modes]
    if main_modes and tuple(batch_modes) == tuple(MODES):
        ok = [m for m in ok if m in main_modes]
    return ok


def _atomic_save(obj, path: Path) -> None:
    """Write via a temporary file + os.replace: an interruption mid-write does not corrupt the previous checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _make_writer(cfg, out: Path):
    if not cfg.tb:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(out / "tb"))
    except Exception:   # no tensorboard package: training does not depend on it
        return None


@torch.no_grad()
def _tb_images(writer, model, ecache, eanns, frames, mode: str, scale: float, step: int, n: int) -> None:
    """A few validation frames with detections (red) and ground truth (green): the quickest view of what the model sees."""
    from hae.viz import draw_detections
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    for k, i in enumerate(list(frames)[:n]):
        rgb, t = (x.to(dev) for x in load_frames(ecache, [i], t_norm=getattr(model.cfg, "tnorm", False), pre_lnorm=getattr(model.cfg, "pre_lnorm", 0)))
        if scale != 1.0:
            rgb, t = (F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False) for x in (rgb, t))
        kw = {"both": dict(rgb=rgb, t=t), "rgb": dict(rgb=rgb), "lwir": dict(t=t)}[mode]
        d = decode(model(**kw), model.cfg.stride, 100, 0.05)[0].copy()
        d[:, :4] /= scale
        img = ecache.rgb[i] if (mode != "lwir" and ecache.rgb is not None) else np.repeat(np.asarray(ecache.t[i])[..., None], 3, -1)
        writer.add_image(f"val/{mode}/{k}", draw_detections(np.asarray(img), d, eanns[i].boxes, thr=0.3).transpose(2, 0, 1), step)
    model.train(was)


def _model_gmac(model, cfg, hw=(256, 320)) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    dev = next(model.parameters()).device
    mode = max(("both", "rgb", "lwir"), key=lambda m: cfg.mode_probs[("both", "rgb", "lwir").index(m)])
    kw = {"both": dict(rgb=torch.rand(1, 3, *hw, device=dev), t=torch.rand(1, 1, *hw, device=dev)), "rgb": dict(rgb=torch.rand(1, 3, *hw, device=dev)),
          "lwir": dict(t=torch.rand(1, 1, *hw, device=dev))}[mode]
    was = model.training
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(**kw)
    model.train(was)
    return fc.get_total_flops() / 2e9


def _print_header(cfg, model, cache, extras, dev, steps_per_epoch, start_ep, anns_main):
    import platform
    name = Path(cfg.out_dir).name
    params = sum(p.numel() for p in model.parameters())
    dname = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"CPU ({torch.get_num_threads()} threads)"
    ui.banner(f"hae · training · {name}", {
        "Model": f"{cfg.size or 'custom'} ({cfg.block}) · {params:,} parameters · {_model_gmac(model, cfg):.3f} GMAC per 320×256 tile".replace(",", " "),
        "Recipe": f"{cfg.epochs} epochs × {steps_per_epoch} steps, batch {cfg.batch}, lr {cfg.lr:g}, classes {cfg.classes}, modes {tuple(cfg.mode_probs)}",
        "Augmentation": f"scale {tuple(cfg.scale_range)}, copy-paste {cfg.copy_paste}, photometric {cfg.photo}, hard-neg {cfg.hn_frac}, neg_decay {cfg.neg_decay}",
        "Device": f"{dname} · seed {cfg.seed} · fp32" if not cfg.amp else f"{dname} · seed {cfg.seed} · AMP",
        "Versions": f"hae {__import__('hae').__version__} · torch {torch.__version__} · python {platform.python_version()}",
        **({"Resumed": f"from epoch {start_ep}"} if start_ep else {})})
    rows = [[cfg.train_set, cache.n, f"{1.0 - sum(w for _, _, w in extras):.2f}"]] + [[e["name"], c.n, f"{w:.2f}"] for e, (c, _, w) in zip(cfg.extra or [], extras)]
    ui.table("data (frames, share of batches)", ["set", "frames", "weight"], rows)
    st = label_stats(anns_main, ["person", "rider", "crowd"][:max(cfg.classes, 1)])
    ui.table("labels of the main set", ["statistic", "value"], [["instances", st["instances"]], ["frames without people", f"{st['empty_frames']} of {st['frames']}"], ["people per frame (mean)", f"{st['per_frame_mean']:.1f}"],
             ["size √area p10/p50/p90, px", f"{st['size_p10']:.0f} / {st['size_p50']:.0f} / {st['size_p90']:.0f}"], ["smaller than 10 px", f"{100 * st['small_lt10px']:.0f}%"], ["per class", ", ".join(f"{k} {v}" for k, v in st["per_class"].items())]])


def _finish_artifacts(cfg, out, ema, ecache, eanns, history, best, total_s, writer) -> None:
    """Final artifacts: results.csv/png, P/R/F1 vs threshold curves, recommended threshold in the checkpoints, validation samples, console summary."""
    rec_conf = None
    try:
        from hae.plots import plot_curves, plot_results, write_results_csv
        write_results_csv(history, out / "results.csv")
        plot_results(history, out / "results.png")
        if ecache is not None and cfg.eval_frames is not None and cfg.eval_modes:
            mode = cfg.eval_modes[0]
            frames = list(cfg.eval_frames)
            dets = detect_frames(ema.m, ecache, frames, mode, scale=cfg.eval_scale)
            gts, igs = [eanns[i].boxes for i in frames], [eanns[i].ignore for i in frames]
            curves = {t: recall_at(gts, igs, dets, t) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)}
            _, rec_conf = plot_curves(curves, out / "curves.png")
            from hae.gallery import dashboard, pred_grid
            from hae.viz import draw_detections
            from PIL import Image
            _refresh_images(out, ecache, eanns, frames[:200], dets[:200], mode, rec_conf)
            pick = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
            grid = pred_grid([_frame_img(ecache, frames[k], mode) for k in pick], [eanns[frames[k]].boxes for k in pick], [dets[k] for k in pick], rec_conf, cols=3, width=420)
            dashboard(history, curves, grid, out / "dashboard.png", f"hae · {out.name}")
            (out / "val_samples").mkdir(exist_ok=True)
            for k, (i, d) in enumerate(list(zip(frames, dets))[:6]):
                img = ecache.rgb[i] if (mode != "lwir" and ecache.rgb is not None) else np.repeat(np.asarray(ecache.t[i])[..., None], 3, -1)
                Image.fromarray(draw_detections(np.asarray(img), d, eanns[i].boxes, thr=rec_conf)).save(out / "val_samples" / f"{k}.png")
            for name in ("last.pt", "best.pt"):   # the threshold from the curve of the final EMA model goes into both checkpoints
                f = out / name
                if f.exists():
                    ck = torch.load(f, weights_only=False, map_location="cpu")
                    ck["recommended_conf"] = rec_conf
                    _atomic_save(ck, f)
    except ImportError:
        pass
    if cfg.verbose and history:
        ui.final_summary(best, history[-1], total_s, out, rec_conf)
    if cfg.verbose:
        ui.save_log(out / "train.log")


def _frame_img(ecache, i, mode):
    return np.asarray(ecache.rgb[i]) if (mode != "lwir" and ecache.rgb is not None) else np.asarray(ecache.t[i])


def _refresh_images(out, ecache, eanns, frames, dets, mode, conf) -> None:
    """Ready-made images (no TensorBoard needed): ground truth/prediction grid and error gallery; refreshed at checkpoint epochs and at the end."""
    from hae.gallery import error_gallery, pred_grid
    from PIL import Image
    imgs = [_frame_img(ecache, i, mode) for i in frames]
    gts = [eanns[i].boxes for i in frames]
    pick = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    Image.fromarray(pred_grid([imgs[k] for k in pick], [gts[k] for k in pick], [dets[k] for k in pick], conf, cols=3)).save(out / "val_pred.png")
    error_gallery(imgs, gts, dets, conf, out / "errors.png")


def build_model(cfg: "TrainConfig"):
    """Model from a training config (shared by train(), checkpoint loading and export)."""
    if cfg.backbone or cfg.pretrained:
        raise ValueError("third-party backbones / pretrained weights are not supported: hae models are trained from scratch")
    ov = cfg.cfg_overrides
    if cfg.pre_lnorm:   # input preprocessing is stored in HaeConfig so evaluation, mining and ONNX export all see it
        ov = {**(ov or {}), "pre_lnorm": int(cfg.pre_lnorm)}
    return build(cfg.size, cfg.block, cfg.width, cfg.depth, align=cfg.align, ncls=cfg.classes, blob=cfg.blob, ctx=cfg.ctx,
                 tnorm=cfg.tnorm, aux=cfg.halo, overrides=ov)


def cpu_freq_mhz():
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) / 1000
    except OSError:
        return None


def _lr(step, total, base, warm):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


@torch.no_grad()
def detect_frames(model, cache, frames, mode="both", batch=8, score_thr=0.01, topk=300, scale=1.0) -> list:
    """Per-frame detections in original frame coordinates (shared by metrics, threshold curves and sample images)."""
    model.eval()
    dev = next(model.parameters()).device
    frames = list(frames)
    dets = []
    for s in range(0, len(frames), batch):
        idx = frames[s:s + batch]
        rgb, t = (x.to(dev) for x in load_frames(cache, idx, t_norm=getattr(model.cfg, "tnorm", False), pre_lnorm=getattr(model.cfg, "pre_lnorm", 0)))
        h0, w0 = t.shape[-2:]
        # input size is a multiple of 32 (required by the ctx level); for 512×640 frames and ×1.25/×1.5 this is exactly `scale`,
        # for other sizes (e.g. 600×800 ×1.5 = 900×1200) the nearest multiple of 32; boxes are mapped back separately in x and y
        H, W = (max(32, int(round(v * scale / 32)) * 32) for v in (h0, w0))
        if (H, W) != (h0, w0):   # resized input: boxes are mapped back to original coordinates below
            rgb, t = (F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) for x in (rgb, t))
        kw = {"both": dict(rgb=rgb, t=t), "rgb": dict(rgb=rgb), "lwir": dict(t=t)}[mode]
        for d in decode(model(**kw), model.cfg.stride, topk, score_thr):
            d = d.copy()
            d[:, [0, 2]] /= W / w0
            d[:, [1, 3]] /= H / h0
            dets.append(d)
    return dets


def _metrics_from_dets(model, cache, anns, frames, dets):
    gts, igs, labs = [anns[i].boxes for i in frames], [anns[i].ignore for i in frames], [anns[i].labels for i in frames]
    if model.cfg.ncls > 1:
        return evaluate_protocol(gts, labs, igs, dets, (cache.h, cache.w))
    return evaluate_detections(gts, igs, dets, (cache.h, cache.w))


def evaluate(model, cache, anns, frames=None, mode="both", batch=8, score_thr=0.01, topk=300, scale=1.0):
    frames = list(range(cache.n)) if frames is None else list(frames)
    return _metrics_from_dets(model, cache, anns, frames, detect_frames(model, cache, frames, mode, batch, score_thr, topk, scale))


def train(cfg: TrainConfig) -> dict:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if cfg.pre_lnorm:   # the thermal channel of every cached set is transformed ONCE on full frames (sibling dir <cache>_lnorm<k>)
        from hae.data.cache import ensure_lnorm_cache
        cfg = replace(cfg, cache_dir=str(ensure_lnorm_cache(cfg.cache_dir, cfg.pre_lnorm, cfg.verbose)))
    cache = Cache(cfg.cache_dir, cfg.train_set)
    mode = "coco3" if cfg.classes == 3 else "person"
    if cfg.classes not in (1, 3):
        raise ValueError(f"classes must be 1 or 3, got {cfg.classes}")
    anns = load_annotations(ann_path(cfg, cfg.train_set), mode)
    extras = []
    for e in cfg.extra or []:
        ap = Path(e["ann"]) if e.get("ann") else Path(e["unified"]) / "ann.json" if e.get("unified") else ann_path(cfg, e["name"])
        extras.append((Cache(cfg.cache_dir, e["name"]), load_annotations(ap, mode), float(e["weight"])))
    w_main = 1.0 - sum(w for _, _, w in extras)
    if cfg.halo:   # HALO masks are built once and stored next to the cache
        if cache.halo is None:
            build_halo_cache(cache, anns, cfg.train_set, cfg.cache_dir)
            cache = Cache(cfg.cache_dir, cfg.train_set)
        for k, (c, a, w_) in enumerate(extras):
            if c.halo is None:
                nm = cfg.extra[k]["name"]
                build_halo_cache(c, a, nm, cfg.cache_dir)
                extras[k] = (Cache(cfg.cache_dir, nm), a, w_)

    hn_state = {"pts": None}   # hard negatives of the main set; filled at epoch hn_start_epoch

    style_bank = None
    if cfg.fourier_p > 0:   # "styles" of all domains for cross-domain spectrum mixing
        style_bank = build_style_bank([cache] + [c for c, _, _ in extras], cfg.crop_hw, cfg.fourier_band, seed=cfg.seed)

    def make_sampler(seed):
        main = CropSampler(cache, anns, cfg.crop_hw, seed=seed, frames=cfg.frames, scale_range=tuple(cfg.scale_range), copy_paste=cfg.copy_paste, t_norm=cfg.tnorm, rare_crop=cfg.rare_crop, use_halo=cfg.halo, photo=cfg.photo, photo_color=cfg.photo_color, taug=cfg.taug, t_invert=cfg.t_invert)
        main.set_hard_negatives(hn_state["pts"], cfg.hn_frac)
        main.set_style_bank(style_bank, cfg.fourier_p, cfg.fourier_band)
        if not extras:
            return main
        subs = [CropSampler(c, a, cfg.crop_hw, seed=seed + 7 * (k + 1), scale_range=tuple(cfg.scale_range),
                            copy_paste=cfg.copy_paste, t_norm=cfg.tnorm, use_halo=cfg.halo, photo=cfg.photo, photo_color=cfg.photo_color, taug=cfg.taug, t_invert=cfg.t_invert)
                for k, (c, a, _) in enumerate(extras)]
        for sub in subs:
            sub.set_style_bank(style_bank, cfg.fourier_p, cfg.fourier_band)
        return MixedSampler([main] + subs, [w_main] + [w for _, _, w in extras], seed)
    sampler = make_sampler(cfg.seed)
    if cfg.eval_frames is None:
        ecache = eanns = None
    elif cfg.eval_split == cfg.train_set:
        ecache, eanns = cache, anns
    else:
        ecache = Cache(cfg.cache_dir, cfg.eval_split)
        eanns = load_annotations(ann_path(cfg, cfg.eval_split), mode)
    dev = resolve_device(cfg.device)
    model = build_model(cfg).to(dev).train()
    if cfg.init_weights and not (cfg.resume and (Path(cfg.out_dir) / "resume.pt").exists()):
        from hae.checkpoint import load_checkpoint
        init, _ = load_checkpoint(cfg.init_weights)
        model.load_state_dict(init.state_dict())   # strict: the recipe's architecture must match the initial weights
    cl = torch.channels_last
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")
    if cfg.channels_last:
        model = model.to(memory_format=cl)
    ema, loss_fn = ModelEma(model, cfg.ema_decay), HaeLoss(stride=model.cfg.stride, box_w=cfg.box_w, cls_w=cfg.cls_w, class_weights=cfg.class_weights, neg_decay=cfg.neg_decay)
    dcfg = DistillConfig.from_dict(cfg.distill)
    distiller = None
    if dcfg.enabled:   # after ModelEma: the EMA copy must not inherit the feature hooks
        if not dcfg.teacher:
            raise ValueError("distill is enabled but no teacher is set (path to the teacher .pt/.onnx)")
        distiller = Distiller(dcfg, model, dev, cfg.channels_last)
    groups = {}
    for p in model.parameters():   # weight decay only on weights (ndim > 1), not on biases / norm parameters
        groups.setdefault(p.ndim > 1, []).append(p)
    opt = torch.optim.AdamW([{"params": ps, "weight_decay": cfg.wd if dec else 0.0, "lr_scale": 1.0}
                             for dec, ps in groups.items()] +
                            ([{"params": list(distiller.parameters()), "weight_decay": 0.0, "lr_scale": 1.0}] if distiller is not None and dcfg.feat else []), lr=cfg.lr)
    steps_per_epoch = max(cfg.crops_per_epoch // cfg.batch, 1)
    total = steps_per_epoch * cfg.epochs
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, step, start_ep = [], 0, 0
    best = {"metric": -1.0, "epoch": 0}
    best_key = cfg.best_metric or ((f"qval_{cfg.eval_modes[0]}_AP50" if cfg.quick_val else f"eval_{cfg.eval_modes[0]}_AP50") if cfg.eval_modes else "")
    if cfg.resume and (out / "resume.pt").exists():   # continue after an interrupted session
        ck = torch.load(out / "resume.pt", weights_only=False, map_location=dev)
        model.load_state_dict(ck["model"])
        ema.m.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if distiller is not None and dcfg.feat and "kd" in ck:
            distiller.load_state_dict(ck["kd"])
        history, step, start_ep, best = ck["history"], ck["step"], ck["epoch"], ck["best"]
        if cfg.verbose:
            print(f"resumed from epoch {start_ep}/{cfg.epochs} (step {step})", flush=True)
    writer = _make_writer(cfg, out)
    t_train0 = time.perf_counter()
    if cfg.verbose:
        _print_header(cfg, model, cache, extras, dev, steps_per_epoch, start_ep, anns)
    try:   # like YOLO labels/train_batch: see the data and augmentations before training
        plot_labels(anns, out / "labels.png", ["person", "rider", "crowd"][:max(cfg.classes, 1)])
        pmode = max(("both", "rgb", "lwir"), key=lambda m: cfg.mode_probs[("both", "rgb", "lwir").index(m)])
        for k in range(3):
            save_batch_preview(make_sampler(cfg.seed + 777 + k).sample_batch(8), out / f"train_batch{k}.png", "rgb" if pmode == "rgb" else "lwir")
    except Exception as e:   # previews must never break training
        print("data previews not created:", e)
    if writer is not None:
        writer.add_text("cfg", "<pre>" + json.dumps(asdict(cfg), ensure_ascii=False, default=str, indent=1) + "</pre>", 0)
    last_q = None
    batches = None
    if cfg.prefetch > 0 and start_ep < cfg.epochs:
        batches = iter(Prefetcher(lambda i: make_sampler(cfg.seed + 1000 * i + 17 * start_ep),
                                  cfg.batch, total - step, workers=cfg.prefetch))
    for ep in range(start_ep, cfg.epochs):
        if cfg.hn_frac > 0 and (ep == cfg.hn_start_epoch or (ep == start_ep and ep > cfg.hn_start_epoch)):   # mine false positives with the current EMA model
            pool = list(cfg.frames) if cfg.frames is not None else list(range(len(anns)))
            sub = sorted(rng.choice(pool, size=min(cfg.hn_frames, len(pool)), replace=False).tolist())
            mmode = max((m for m in MODES if (m != "rgb" or cache.rgb is not None) and m in cache.modes), key=lambda m: cfg.mode_probs[MODES.index(m)])
            hn_state["pts"] = mine_hard_negatives(ema.m, cache, anns, sub, mmode, cfg.eval_scale, dev, score_thr=cfg.hn_score)
            if batches is not None:
                batches = iter(Prefetcher(lambda i: make_sampler(cfg.seed + 5000 + 1000 * i), cfg.batch, total - step, workers=cfg.prefetch))
            else:
                (sampler.samplers[0] if isinstance(sampler, MixedSampler) else sampler).set_hard_negatives(hn_state["pts"], cfg.hn_frac)
            if cfg.verbose:
                print(f"hard negatives: {len(hn_state['pts'])} points in {len(sub)} frames (mode {mmode})", flush=True)
        t0, acc, accn, acc_al, n_al, n_extra, acc_ax, n_ax = time.perf_counter(), 0.0, 0, 0.0, 0, 0, 0.0, 0
        acc_cls = acc_box = acc_inst = acc_gn = data_s = 0.0
        acc_kd: dict = {}
        crops = 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        bar = ui.EpochBar(steps_per_epoch, ep + 1, cfg.epochs) if cfg.verbose else None
        if bar is not None:
            bar.__enter__()
        improved = False
        for _ in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = _lr(step, total, cfg.lr, cfg.warmup_steps) * g["lr_scale"]
            t_data = time.perf_counter()
            b = next(batches) if batches is not None else sampler.sample_batch(cfg.batch)
            data_s += time.perf_counter() - t_data
            rgb, t = b.rgb.to(dev, non_blocking=True), b.t.to(dev, non_blocking=True)
            if cfg.channels_last:
                rgb, t = rgb.contiguous(memory_format=cl), t.contiguous(memory_format=cl)
            allowed = allowed_modes(b.modes, cfg.main_modes)      # an LWIR-only set allows only lwir; an RGB-only set only rgb
            pm = np.array([cfg.mode_probs[MODES.index(m)] for m in allowed], float)
            pm = pm / pm.sum() if pm.sum() > 0 else np.full(len(allowed), 1.0 / len(allowed))
            mode = allowed[int(rng.choice(len(allowed), p=pm))]
            n_extra += int(b.modes == ("lwir",))
            kw = {"both": dict(rgb=rgb, t=t), "rgb": dict(rgb=rgb), "lwir": dict(t=t)}[mode]
            with torch.autocast("cuda", torch.float16, enabled=scaler.is_enabled()):
                out_ = model(**kw)
            loss, lg = loss_fn({k: v.float() for k, v in out_.items()}, b.boxes, b.ignore, b.labels)
            acc_cls, acc_box = acc_cls + lg["cls"], acc_box + lg["box"]
            if distiller is not None:   # teacher: frozen, eval, no gradients; extra loss term on the same student outputs
                tout = distiller.teach(kw, autocast=scaler.is_enabled())
                kd_loss, kd_logs = distiller.loss({k: v.float() for k, v in out_.items()}, tout, loss_fn, b.boxes)
                loss = loss + kd_loss
                for k_, v_ in kd_logs.items():
                    acc_kd[k_] = acc_kd.get(k_, 0.0) + v_
            acc_inst, crops = acc_inst + sum(len(x) for x in b.boxes), crops + len(b.boxes)
            if "aux" in out_ and b.halo is not None:   # auxiliary mask: SoftIoU against HALO pooled to the head grid
                tgt = F.avg_pool2d(b.halo.to(dev), model.cfg.stride)
                axl = soft_iou_loss(out_["aux"].float(), tgt)
                loss = loss + cfg.aux_w * axl
                acc_ax, n_ax = acc_ax + float(axl.detach()), n_ax + 1
            if "flow" in out_:   # self-supervised alignment: only when both modalities are fed
                al = edge_alignment_loss(rgb.float(), t.float(), out_["flow"].float(), model.cfg.stride)
                loss = loss + cfg.align_w * al + cfg.smooth_w * flow_smoothness(out_["flow"].float())
                acc_al, n_al = acc_al + float(al.detach()), n_al + 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))   # norm BEFORE clipping
            acc_gn += gn
            if writer is not None and step % cfg.tb_every_steps == 0:
                writer.add_scalar("train/grad_norm", gn, step)
                writer.add_scalar("train/loss_step", float(loss.detach()), step)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            acc, accn, step = acc + float(loss.detach()), accn + 1, step + 1
            if bar is not None:
                bar.update(acc / accn, opt.param_groups[0]["lr"])
        if bar is not None:
            bar.__exit__()
        rec = {"epoch": ep + 1, "loss": acc / accn, "epoch_s": time.perf_counter() - t0, "cpu_mhz": cpu_freq_mhz(), "lr": opt.param_groups[0]["lr"]}
        rec.update({"loss_cls": acc_cls / accn, "loss_box": acc_box / accn, "inst_per_crop": acc_inst / max(crops, 1), "grad_norm": acc_gn / accn,
                    "crops_per_s": crops / max(rec["epoch_s"], 1e-9), "data_wait_pct": 100 * data_s / max(rec["epoch_s"], 1e-9)})
        if dev.type == "cuda":
            rec["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if acc_kd:
            rec.update({k_: v_ / accn for k_, v_ in acc_kd.items()})
        if n_al:
            rec["align_loss"] = acc_al / n_al
        if n_ax:
            rec["aux_loss"] = acc_ax / n_ax
        if extras:
            rec["extra_batches"] = n_extra
        if hn_state["pts"] is not None:
            rec["hn_points"] = int(len(hn_state["pts"]))
        if cfg.quick_val and cfg.eval_frames is not None and cfg.eval_modes:   # quick validation after every epoch (frame subset): AP and operating recall/FP
            qm = cfg.eval_modes[0]
            fr = list(cfg.eval_frames)
            qf = [fr[i] for i in np.linspace(0, len(fr) - 1, min(cfg.quick_val, len(fr))).astype(int)]
            qd = detect_frames(ema.m, ecache, qf, qm, scale=cfg.eval_scale)
            last_q = (qf, qd, qm)
            qr = _metrics_from_dets(ema.m, ecache, eanns, qf, qd)
            rec.update({f"qval_{qm}_{k}": v for k, v in qr.items() if isinstance(v, (int, float))})
            cr = recall_center([eanns[i].boxes for i in qf], [eanns[i].ignore for i in qf], qd, 0.25)
            rec["qval_recall"], rec["qval_fp"] = cr["recall"], cr["fp_per_frame"]
        if (ep + 1) % cfg.eval_every == 0 and cfg.eval_frames is not None:
            for mode in cfg.eval_modes:
                r = evaluate(ema.m, ecache, eanns, cfg.eval_frames, mode=mode, scale=cfg.eval_scale)
                rec.update({f"eval_{mode}_{k}": v for k, v in r.items()})
                if writer is not None and cfg.tb_images:
                    _tb_images(writer, ema.m, ecache, eanns, cfg.eval_frames, mode, cfg.eval_scale, ep + 1, cfg.tb_images)
        if best_key in rec and rec[best_key] > best["metric"]:   # best EMA model by the validation metric (selection biases the estimate on that split)
            best = {"metric": float(rec[best_key]), "epoch": ep + 1, "metric_key": best_key}
            improved = True
            _atomic_save({"model": ema.m.state_dict(), "cfg": asdict(cfg), "epoch": ep + 1, "metric": best["metric"], "metric_key": best_key}, out / "best.pt")
        history.append(rec)
        if writer is not None:
            writer.add_scalar("train/loss", rec["loss"], ep + 1)
            writer.add_scalar("train/lr", rec["lr"], ep + 1)
            writer.add_scalar("train/epoch_seconds", rec["epoch_s"], ep + 1)
            for k, v in rec.items():
                if k.startswith("eval_") and isinstance(v, (int, float)):
                    writer.add_scalar("val/" + k[len("eval_"):].replace("_", "/", 1), v, ep + 1)
                elif k.startswith("eval_") and isinstance(v, dict):   # per-class AP50 (person/rider/crowd)
                    for ck, cv in v.items():
                        if isinstance(cv, (int, float)):
                            writer.add_scalar(f"val/{k[len('eval_'):].replace('_per_class', '')}/AP50_{ck}", cv, ep + 1)
            writer.flush()
        if cfg.verbose:
            ui.print_epoch(rec, cfg.epochs, improved)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        try:   # ready-made images refresh during training: results.png every epoch, samples and errors at checkpoint epochs
            from hae.plots import plot_results, write_results_csv
            write_results_csv(history, out / "results.csv")
            plot_results(history, out / "results.png")
            if last_q is not None and ((ep + 1) % max(cfg.ckpt_every, 1) == 0 or ep + 1 == cfg.epochs):
                _refresh_images(out, ecache, eanns, last_q[0], last_q[1], last_q[2], 0.25)
        except ImportError:
            pass
        if (ep + 1) % max(cfg.ckpt_every, 1) == 0 or ep + 1 == cfg.epochs:
            _atomic_save({"model": ema.m.state_dict(), "cfg": asdict(cfg), "epoch": ep + 1}, out / "last.pt")
            _atomic_save({"model": model.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                          "history": history, "step": step, "epoch": ep + 1, "best": best, "cfg": asdict(cfg),
                          **({"kd": distiller.state_dict()} if distiller is not None and dcfg.feat else {})}, out / "resume.pt")
    _finish_artifacts(cfg, out, ema, ecache, eanns, history, best, time.perf_counter() - t_train0, writer)
    if start_ep >= cfg.epochs and not (out / "last.pt").exists():
        _atomic_save({"model": ema.m.state_dict(), "cfg": asdict(cfg), "epoch": start_ep}, out / "last.pt")
    if writer is not None:
        writer.close()
    if distiller is not None:
        distiller.close()
    return {"model": ema.m, "raw": model.eval(), "history": history}
