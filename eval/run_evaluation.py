#!/usr/bin/env python3
"""Evaluation harness -- pixel-level and point-level mIoU with per-distance-band breakdown.

All paths and model parameters come from config YAML + env overrides.
Outputs: results/eval_{timestamp}.json, confusion_matrices.png,
         per_distance_band.png, per_class_iou.png, eval_summary.md
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

try:
    import psutil
except ImportError:
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train import miou_from_confusion, update_confusion, CLASS_NAMES_5, CLASS_NAMES_4
from src.common.config import (
    load_config,
    resolve_device,
    resolve_precision,
    resolve_path,
    repo_root,
)
from src.data.webdataset_loader import create_val_loader
from src.data.label_mapping import bin_5_to_4, remap_labels, load_class_mapping
from src.models.unet import RangeImageUNet
from src.models.predict import Segmenter

load_dotenv(repo_root() / ".env", override=False)


DISTANCE_BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]
BAND_NAMES = [f"{lo}-{hi}m" for lo, hi in DISTANCE_BANDS]


def _safe_floats(iou_list):
    return [float(x) if not np.isnan(x) else None for x in iou_list]


def _latency_stats(latencies):
    if not latencies:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p90": 0.0}
    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


class _MemTracker:
    """Tracks peak process RSS (MiB) using psutil when available, with a
    tracemalloc fallback. Sampled explicitly (not on a timer) so the caller
    controls overhead, typically once per batch/scan."""

    def __init__(self):
        self.peak_rss_mib = 0.0
        if psutil is not None:
            self._proc = psutil.Process()
        else:
            self._proc = None
            import tracemalloc as _tm
            _tm.start()
            self._tm = _tm

    def sample(self):
        if self._proc is not None:
            rss = self._proc.memory_info().rss / (1024 * 1024)
            if rss > self.peak_rss_mib:
                self.peak_rss_mib = rss
        else:
            _, peak = self._tm.get_traced_memory()
            peak_mib = peak / (1024 * 1024)
            if peak_mib > self.peak_rss_mib:
                self.peak_rss_mib = peak_mib

    def finish(self) -> dict:
        self.sample()
        if self._proc is None:
            self._tm.stop()
        return {"peak_rss_mb": round(self.peak_rss_mib, 1)}


class _BandStats:
    """Shared confusion + per-distance-band accumulator for the pixel- and
    point-level eval paths so both produce byte-identical result shapes.
    `preds/tgts` may be image-shaped (H,W) or flat point vectors;
    `update_confusion` flattens and drops ignore_index (255) rows on its own."""

    def __init__(self, num_classes: int):
        self.cm_5 = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.cm_4 = np.zeros((4, 4), dtype=np.int64)
        self.band_cms_5 = {b: np.zeros((num_classes, num_classes), dtype=np.int64) for b in DISTANCE_BANDS}
        self.band_cms_4 = {b: np.zeros((4, 4), dtype=np.int64) for b in DISTANCE_BANDS}
        self.latencies = []

    def observe(self, preds_5, tgts_5, ranges=None, latency_ms=None):
        update_confusion(self.cm_5, preds_5, tgts_5)
        preds_4 = bin_5_to_4(preds_5)
        tgts_4 = bin_5_to_4(tgts_5)
        update_confusion(self.cm_4, preds_4, tgts_4)
        if latency_ms is not None:
            self.latencies.append(latency_ms)
        if ranges is None:
            return
        for lo, hi in DISTANCE_BANDS:
            mask = (ranges >= lo) & (ranges < hi)
            if mask.any():
                update_confusion(self.band_cms_5[(lo, hi)], preds_5[mask], tgts_5[mask])
                update_confusion(self.band_cms_4[(lo, hi)], preds_4[mask], tgts_4[mask])

    def result(self, count_key: str, count_value: int) -> dict:
        miou_5, ci_5 = miou_from_confusion(self.cm_5)
        miou_4, ci_4 = miou_from_confusion(self.cm_4)
        band_5, band_4 = {}, {}
        for (lo, hi), name in zip(DISTANCE_BANDS, BAND_NAMES):
            m5, c5 = miou_from_confusion(self.band_cms_5[(lo, hi)])
            m4, c4 = miou_from_confusion(self.band_cms_4[(lo, hi)])
            band_5[name] = {"miou": m5, "class_ious": dict(zip(CLASS_NAMES_5, _safe_floats(c5)))}
            band_4[name] = {"miou": m4, "class_ious": dict(zip(CLASS_NAMES_4, _safe_floats(c4)))}
        return {
            count_key: count_value,
            "miou_5class": miou_5,
            "miou_4class": miou_4,
            "class_ious_5": dict(zip(CLASS_NAMES_5, _safe_floats(ci_5))),
            "class_ious_4": dict(zip(CLASS_NAMES_4, _safe_floats(ci_4))),
            "per_distance_band_5class": band_5,
            "per_distance_band_4class": band_4,
            "confusion_5class": self.cm_5.tolist(),
            "confusion_4class": self.cm_4.tolist(),
            "latency_ms": _latency_stats(self.latencies),
        }


def _resolve_checkpoint(cfg):
    ckpt = os.getenv("PC2D_CHECKPOINT", "")
    if ckpt:
        return resolve_path(ckpt)
    ckpt = cfg.get("model", {}).get("checkpoint")
    if ckpt:
        return resolve_path(ckpt)
    ckpt_dir = resolve_path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
    best_name = cfg.get("checkpoint", {}).get("best_name", "best_miou.pt")
    return ckpt_dir / best_name


def _resolve_seq_dir(cfg, cli_override=None):
    if cli_override:
        return resolve_path(cli_override)
    env = os.getenv("PC2D_SEQ_DIR")
    if env:
        return Path(env)
    src = cfg.get("source", {}).get("seq_dir")
    if src:
        return resolve_path(src)
    raw_root = cfg.get("data", {}).get("raw_root")
    if raw_root:
        val_seqs = cfg.get("data", {}).get("val_seqs", [8])
        return resolve_path(raw_root) / f"{val_seqs[0]:02d}"
    return None


def evaluate_pixel_level(model, val_loader, device, num_classes, use_amp, max_range, limit=None, mem=None):
    model.eval()
    acc = _BandStats(num_classes)
    n_samples = 0

    with torch.inference_mode():
        # Warmup: run 2 forward passes on the first batch so CUDA init,
        # cuDNN autotuner, and torch.compile (if active) finish before
        # any latency is recorded. These iterations are not timed.
        warmup_batch = None
        for batch_idx, batch in enumerate(val_loader):
            ri_batch, li_batch = batch
            ri_gpu = ri_batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(ri_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(ri_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmup_batch = batch
            break

        # Re-iterate from the start (including the warmup batch for mIoU).
        for batch_idx, batch in enumerate(val_loader):
            if limit is not None and batch_idx >= limit:
                break

            ri_batch, li_batch = batch
            ri_gpu = ri_batch.to(device, non_blocking=True)

            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ri_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            preds_5 = logits.argmax(dim=1).cpu().numpy()
            tgts_5 = li_batch.numpy()
            ri_np = ri_batch.numpy()
            for b in range(preds_5.shape[0]):
                ranges = ri_np[b, 0] * max_range
                acc.observe(preds_5[b], tgts_5[b], ranges=ranges, latency_ms=latency_ms)

            n_samples += preds_5.shape[0]
            if mem is not None:
                mem.sample()
            if (batch_idx + 1) % 20 == 0:
                print(f"  pixel: {n_samples} samples ({batch_idx + 1} batches)")

    return acc.result("n_samples", n_samples)


def evaluate_point_level(seq_dir, checkpoint, model_cfg, train_cfg,
                         device_str, precision, num_classes, limit=None, mem=None):
    seq_path = Path(resolve_path(seq_dir)) if not isinstance(seq_dir, Path) else seq_dir
    velodyne_dir = seq_path / "velodyne"
    labels_dir = seq_path / "labels"

    if not velodyne_dir.exists():
        print(f"  Skipping point-level: {velodyne_dir} not found")
        return None

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if not bin_files:
        print(f"  Skipping point-level: no .bin in {velodyne_dir}")
        return None

    mapping = load_class_mapping()
    max_range = train_cfg.get("max_range", 80.0)

    segmenter = Segmenter(
        checkpoint=checkpoint,
        in_channels=model_cfg.get("in_channels", 5),
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 32),
        use_groupnorm=model_cfg.get("use_groupnorm", True),
        groups=model_cfg.get("groups", 8),
        h=model_cfg.get("h", 64),
        w=model_cfg.get("w", 2048),
        fov_top_deg=model_cfg.get("fov_top_deg", 2.0),
        fov_bottom_deg=model_cfg.get("fov_bottom_deg", -24.8),
        max_range=max_range,
        device=device_str,
        precision=precision,
    )

    # Warmup with a realistic-sized scan (~120k points) so the CUDA kernels
    # for the actual input shape are compiled before timing starts. The
    # segmenter's built-in warmup only exercises the UNet forward (fixed
    # 1x5x64x2048 shape); this exercises the full projection+segment
    # pipeline at live scan size.
    warmup_pts = np.random.randn(120_000, 4).astype(np.float32)
    warmup_pts[:, 3] = 0.5
    segmenter.segment(warmup_pts)
    segmenter.segment(warmup_pts)

    eval_files = bin_files[:limit] if limit else bin_files

    acc = _BandStats(num_classes)
    n_scans = 0

    for bf in eval_files:
        points = np.fromfile(str(bf), dtype=np.float32).reshape(-1, 4)
        label_path = labels_dir / f"{bf.stem}.label"
        if not label_path.exists():
            continue

        raw_labels = np.fromfile(str(label_path), dtype=np.uint32)
        semantic = (raw_labels & 0xFFFF).astype(np.uint8)
        gt_5 = remap_labels(semantic, mapping)
        gt_4 = bin_5_to_4(gt_5)

        pred_5, timings = segmenter.segment(points)

        valid = gt_5 != 255
        if valid.any():
            ranges = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
            acc.observe(pred_5[valid], gt_5[valid], ranges=ranges[valid], latency_ms=timings["total"])
        else:
            acc.latencies.append(timings["total"])

        n_scans += 1
        if mem is not None:
            mem.sample()
        if n_scans % 50 == 0:
            print(f"  point: {n_scans}/{len(eval_files)} scans")

    return acc.result("n_scans", n_scans)


def plot_confusion_matrices(cm_5, cm_4, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _norm(cm):
        cm_f = cm.astype(float)
        rs = cm_f.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return cm_f / rs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(_norm(cm_5), cmap="Blues", vmin=0, vmax=1)
    ax1.set_title("5-Class (row-normalized)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Ground Truth")
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(CLASS_NAMES_5, rotation=45, ha="right", fontsize=7)
    ax1.set_yticklabels(CLASS_NAMES_5, fontsize=7)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(_norm(cm_4), cmap="Blues", vmin=0, vmax=1)
    ax2.set_title("4-Class (row-normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Ground Truth")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(CLASS_NAMES_4, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(CLASS_NAMES_4, fontsize=8)
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_distance_bands(pixel_bands, point_bands, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BAND_NAMES))
    width = 0.35

    pixel_mious = [pixel_bands[b]["miou"] for b in BAND_NAMES]
    bars1 = ax.bar(x - width / 2, pixel_mious, width, label="Pixel-level", color="steelblue")

    point_mious = None
    if point_bands is not None:
        point_mious = [point_bands[b]["miou"] for b in BAND_NAMES]
        ax.bar(x + width / 2, point_mious, width, label="Point-level", color="coral")

    ax.set_xlabel("Distance Band")
    ax.set_ylabel("mIoU (4-class)")
    ax.set_title("mIoU by Distance Band (4-class)")
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    if point_mious is not None:
        for bar in ax.patches[len(bars1):]:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "per_distance_band.png", dpi=150)
    plt.close(fig)


def plot_class_ious(pixel_ci, point_ci, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#22c55e", "#a16207", "#ef4444", "#3b82f6"]

    n_plots = 1 + (1 if point_ci else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, ci_dict, title in [(axes[0], pixel_ci, "Pixel-level Per-Class IoU (4-class)")]:
        names = list(ci_dict.keys())
        values = [v if v is not None else 0.0 for v in ci_dict.values()]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("IoU")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    if point_ci and len(axes) > 1:
        names = list(point_ci.keys())
        values = [v if v is not None else 0.0 for v in point_ci.values()]
        bars = axes[1].bar(names, values, color=colors)
        axes[1].set_title("Point-level Per-Class IoU (4-class)")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("IoU")
        axes[1].tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_iou.png", dpi=150)
    plt.close(fig)


def generate_summary(results, output_dir):
    lines = ["# Evaluation Summary", ""]
    lines.append(f"- **Timestamp**: {results['timestamp']}")
    lines.append(f"- **Device**: {results['device']} | **Precision**: {results['precision']}")
    lines.append("")

    pixel = results.get("pixel")
    if pixel:
        lines.append("## Pixel-Level")
        lines.append("")
        lines.append(f"- **Samples**: {pixel['n_samples']}")
        lines.append(f"- **mIoU (5-class)**: {pixel['miou_5class']:.4f}")
        lines.append(f"- **mIoU (4-class)**: {pixel['miou_4class']:.4f}")
        lines.append("")
        lines.append("### Per-Class IoU (4-class)")
        lines.append("")
        lines.append("| Class | IoU |")
        lines.append("|-------|-----|")
        for name, iou in pixel["class_ious_4"].items():
            v = f"{iou:.4f}" if iou is not None else "N/A"
            lines.append(f"| {name} | {v} |")
        lines.append("")
        lines.append("### Per-Distance-Band mIoU (4-class)")
        lines.append("")
        lines.append("| Band | mIoU |")
        lines.append("|------|------|")
        for bn, data in pixel["per_distance_band_4class"].items():
            lines.append(f"| {bn} | {data['miou']:.4f} |")
        lines.append("")
        lines.append(f"### Latency: {pixel['latency_ms']['median']:.1f} ms/batch (mean {pixel['latency_ms']['mean']:.1f} +/- {pixel['latency_ms']['std']:.1f}, p90 {pixel['latency_ms']['p90']:.1f})")
        lines.append("")

    point = results.get("point")
    if point:
        lines.append("## Point-Level")
        lines.append("")
        lines.append(f"- **Scans**: {point['n_scans']}")
        lines.append(f"- **mIoU (5-class)**: {point['miou_5class']:.4f}")
        lines.append(f"- **mIoU (4-class)**: {point['miou_4class']:.4f}")
        lines.append("")
        lines.append("### Per-Class IoU (4-class)")
        lines.append("")
        lines.append("| Class | IoU |")
        lines.append("|-------|-----|")
        for name, iou in point["class_ious_4"].items():
            v = f"{iou:.4f}" if iou is not None else "N/A"
            lines.append(f"| {name} | {v} |")
        lines.append("")
        lines.append("### Per-Distance-Band mIoU (4-class)")
        lines.append("")
        lines.append("| Band | mIoU |")
        lines.append("|------|------|")
        for bn, data in point["per_distance_band_4class"].items():
            lines.append(f"| {bn} | {data['miou']:.4f} |")
        lines.append("")
        lines.append(f"### Latency: {point['latency_ms']['median']:.1f} ms/scan (mean {point['latency_ms']['mean']:.1f} +/- {point['latency_ms']['std']:.1f}, p90 {point['latency_ms']['p90']:.1f})")
        lines.append("")
    else:
        lines.append("## Point-Level: Skipped (no raw data)")
        lines.append("")

    mem = results.get("memory", {})
    lines.append("## Memory")
    lines.append("")
    lines.append(f"- **Peak RSS**: {mem.get('peak_rss_mb', 0):.1f} MB")
    if mem.get("gpu_peak_mb") is not None:
        lines.append(f"- **GPU Allocated**: {mem['gpu_peak_mb']:.1f} MB")
    if mem.get("gpu_reserved_mb") is not None:
        lines.append(f"- **GPU Reserved**: {mem['gpu_reserved_mb']:.1f} MB")
    lines.append("")

    (output_dir / "eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="pc2d evaluation harness")
    parser.add_argument("--config", type=str, default="train_range_image",
                        help="Config name (without .yaml)")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit batches (pixel) or scans (point)")
    parser.add_argument("--seq-dir", type=str, default=None,
                        help="Sequence dir for point-level eval (overrides config/env)")
    parser.add_argument("--pixel-only", action="store_true")
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    device_str = resolve_device(args.device or model_cfg.get("device", "auto"))
    precision = resolve_precision(args.precision or model_cfg.get("precision", "fp16"))
    device = torch.device(device_str)
    use_amp = device.type == "cuda" and precision == "fp16"
    num_classes = model_cfg.get("num_classes", 5)
    max_range = train_cfg.get("max_range", 80.0)

    output_dir = Path(resolve_path(args.output_dir)) if args.output_dir else repo_root() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== pc2d Evaluation ===")
    print(f"  Device: {device_str}  Precision: {precision}  AMP: {use_amp}")
    print(f"  Num classes: {num_classes}  Max range: {max_range}")
    print(f"  Output: {output_dir}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device_str,
        "precision": precision,
        "config": args.config,
        "pixel": None,
        "point": None,
        "memory": {},
    }

    mem = _MemTracker()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if not args.point_only:
        print("\n--- Pixel-Level Evaluation ---")
        processed_root = resolve_path(cfg.get("data", {}).get("processed_root", "data/processed"))
        checkpoint = _resolve_checkpoint(cfg)

        if not checkpoint.exists():
            print(f"  ERROR: checkpoint not found: {checkpoint}")
            print(f"  Set PC2D_CHECKPOINT or PC2D_CKPT_DIR env var, or download with scripts/download_checkpoint.py")
        elif not (Path(processed_root) / "val").exists():
            print(f"  ERROR: val shards not found: {processed_root}/val/")
            print(f"  Set PC2D_PROCESSED_ROOT env var, or run scripts/preprocess_to_shards.py")
        else:
            print(f"  Checkpoint: {checkpoint}")
            print(f"  Val shards: {processed_root}")

            val_loader = create_val_loader(str(processed_root), batch_size=1, num_workers=0)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            model = RangeImageUNet(
                in_channels=model_cfg.get("in_channels", 5),
                num_classes=num_classes,
                base_channels=model_cfg.get("base_channels", 32),
                use_groupnorm=model_cfg.get("use_groupnorm", True),
                groups=model_cfg.get("groups", 8),
            )
            ckpt_data = torch.load(str(checkpoint), map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.to(device).eval()
            if device.type == "cuda" and precision == "fp16":
                model.half()

            pixel_results = evaluate_pixel_level(
                model, val_loader, device, num_classes, use_amp, max_range, args.limit, mem=mem)
            results["pixel"] = pixel_results

            print(f"  mIoU 5-class: {pixel_results['miou_5class']:.4f}")
            print(f"  mIoU 4-class: {pixel_results['miou_4class']:.4f}")
            print(f"  Latency: {pixel_results['latency_ms']['median']:.1f} ms/batch (mean {pixel_results['latency_ms']['mean']:.1f} +/- {pixel_results['latency_ms']['std']:.1f}, p90 {pixel_results['latency_ms']['p90']:.1f})")

            del model, ckpt_data
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not args.pixel_only:
        print("\n--- Point-Level Evaluation ---")
        seq_dir = _resolve_seq_dir(cfg, args.seq_dir)
        if seq_dir is None:
            print("  Skipping: no seq_dir available (set --seq-dir or PC2D_SEQ_DIR)")
        else:
            checkpoint = _resolve_checkpoint(cfg)
            if not checkpoint.exists():
                print(f"  ERROR: checkpoint not found: {checkpoint}")
            else:
                print(f"  Seq dir: {seq_dir}")
                point_results = evaluate_point_level(
                    seq_dir=seq_dir,
                    checkpoint=checkpoint,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    device_str=device_str,
                    precision=precision,
                    num_classes=num_classes,
                    limit=args.limit,
                    mem=mem,
                )
                if point_results is not None:
                    results["point"] = point_results
                    print(f"  mIoU 5-class: {point_results['miou_5class']:.4f}")
                    print(f"  mIoU 4-class: {point_results['miou_4class']:.4f}")
                    print(f"  Latency: {point_results['latency_ms']['median']:.1f} ms/scan (mean {point_results['latency_ms']['mean']:.1f} +/- {point_results['latency_ms']['std']:.1f}, p90 {point_results['latency_ms']['p90']:.1f})")

    mem_results = mem.finish()
    results["memory"]["peak_rss_mb"] = mem_results["peak_rss_mb"]
    if device.type == "cuda":
        alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        results["memory"]["gpu_peak_mb"] = round(alloc_mib, 1)
        results["memory"]["gpu_reserved_mb"] = round(reserved_mib, 1)
    else:
        results["memory"]["gpu_peak_mb"] = None
        results["memory"]["gpu_reserved_mb"] = None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results JSON: {json_path}")

    if results["pixel"] is not None:
        cm_5 = np.array(results["pixel"]["confusion_5class"], dtype=np.int64)
        cm_4 = np.array(results["pixel"]["confusion_4class"], dtype=np.int64)
        plot_confusion_matrices(cm_5, cm_4, output_dir)
        print(f"  Confusion matrices: {output_dir / 'confusion_matrices.png'}")

        pixel_bands = results["pixel"]["per_distance_band_4class"]
        point_bands = (results["point"]["per_distance_band_4class"]
                       if results.get("point") else None)
        plot_distance_bands(pixel_bands, point_bands, output_dir)
        print(f"  Distance bands: {output_dir / 'per_distance_band.png'}")

        pixel_ci = results["pixel"]["class_ious_4"]
        point_ci = results["point"]["class_ious_4"] if results.get("point") else None
        plot_class_ious(pixel_ci, point_ci, output_dir)
        print(f"  Per-class IoU: {output_dir / 'per_class_iou.png'}")

    generate_summary(results, output_dir)
    print(f"  Summary: {output_dir / 'eval_summary.md'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
