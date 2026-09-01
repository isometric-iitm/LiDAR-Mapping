#!/usr/bin/env python3
"""Inject latest evaluation results into README.md between markers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARKER_START = "<!-- RESULTS_START -->"
MARKER_END = "<!-- RESULTS_END -->"


def find_latest_json(results_dir):
    results_dir = Path(results_dir)
    jsons = sorted(results_dir.glob("eval_*.json"))
    return jsons[-1] if jsons else None


def format_results(data):
    lines = [""]
    pixel = data.get("pixel")
    point = data.get("point")

    if pixel:
        lines.append("## Evaluation Results")
        lines.append("")
        lines.append(f"**Device**: {data['device']} | **Precision**: {data['precision']} | **Date**: {data['timestamp'][:10]}")
        lines.append("")
        lines.append("### Pixel-Level mIoU")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| mIoU (5-class) | {pixel['miou_5class']:.4f} |")
        lines.append(f"| mIoU (4-class) | {pixel['miou_4class']:.4f} |")
        lines.append(f"| Samples | {pixel['n_samples']} |")
        lines.append(f"| Avg latency | {pixel['latency_ms']['mean']:.1f} ms/batch |")
        lines.append("")
        lines.append("#### Per-Class IoU (4-class)")
        lines.append("")
        lines.append("| Class | IoU |")
        lines.append("|-------|-----|")
        for name, iou in pixel["class_ious_4"].items():
            v = f"{iou:.4f}" if iou is not None else "N/A"
            lines.append(f"| {name} | {v} |")
        lines.append("")
        lines.append("#### Per-Distance-Band mIoU (4-class)")
        lines.append("")
        lines.append("| Band | mIoU |")
        lines.append("|------|------|")
        for bn, d in pixel["per_distance_band_4class"].items():
            lines.append(f"| {bn} | {d['miou']:.4f} |")
        lines.append("")

    if point:
        lines.append("### Point-Level mIoU")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| mIoU (5-class) | {point['miou_5class']:.4f} |")
        lines.append(f"| mIoU (4-class) | {point['miou_4class']:.4f} |")
        lines.append(f"| Scans | {point['n_scans']} |")
        lines.append(f"| Avg latency | {point['latency_ms']['mean']:.1f} ms/scan |")
        lines.append("")
        lines.append("#### Per-Class IoU (4-class)")
        lines.append("")
        lines.append("| Class | IoU |")
        lines.append("|-------|-----|")
        for name, iou in point["class_ious_4"].items():
            v = f"{iou:.4f}" if iou is not None else "N/A"
            lines.append(f"| {name} | {v} |")
        lines.append("")

    mem = data.get("memory", {})
    lines.append("### Memory")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Peak RSS | {mem.get('peak_rss_mb', 0):.1f} MB |")
    if mem.get("gpu_peak_mb") is not None:
        lines.append(f"| GPU Allocated | {mem['gpu_peak_mb']:.1f} MB |")
    if mem.get("gpu_reserved_mb") is not None:
        lines.append(f"| GPU Reserved | {mem['gpu_reserved_mb']:.1f} MB |")
    lines.append("")
    return "\n".join(lines)


def inject(readme_path, results_dir):
    readme_path = Path(readme_path)
    results_dir = Path(results_dir)

    json_path = find_latest_json(results_dir)
    if json_path is None:
        print(f"No eval_*.json found in {results_dir}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    if MARKER_START in readme_text and MARKER_END in readme_text:
        print(f"Markers already present in {readme_path}; skipping (hand-curated Results section).")
        print(f"  Reference: {json_path.name}")
        return

    table_md = format_results(data)
    new_text = f"{readme_text}\n{MARKER_START}{table_md}\n{MARKER_END}\n"
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Injected {json_path.name} into {readme_path}")


if __name__ == "__main__":
    from src.common.config import repo_root
    inject(repo_root() / "README.md", repo_root() / "results")
