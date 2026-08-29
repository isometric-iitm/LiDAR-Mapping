import json
from pathlib import Path

import numpy as np
import pytest

sys_path = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.insert(0, sys_path)
sys.path.insert(0, str(Path(sys_path) / "scripts"))

from train import miou_from_confusion, update_confusion, CLASS_NAMES_5, CLASS_NAMES_4
from eval.run_evaluation import (
    _safe_floats,
    _latency_stats,
    _resolve_checkpoint,
    _resolve_seq_dir,
    DISTANCE_BANDS,
    BAND_NAMES,
    evaluate_pixel_level,
    evaluate_point_level,
    plot_confusion_matrices,
    plot_distance_bands,
    plot_class_ious,
    generate_summary,
)


def test_safe_floats():
    assert _safe_floats([1.0, float("nan"), 0.5]) == [1.0, None, 0.5]
    assert _safe_floats([]) == []


def test_latency_stats_empty():
    s = _latency_stats([])
    assert s["mean"] == 0.0 and s["std"] == 0.0


def test_latency_stats_values():
    s = _latency_stats([10.0, 20.0, 30.0])
    assert abs(s["mean"] - 20.0) < 1e-6
    assert s["min"] == 10.0
    assert s["max"] == 30.0


def test_miou_from_confusion_perfect():
    cm = np.eye(5, dtype=np.int64) * 100
    miou, ci = miou_from_confusion(cm)
    assert abs(miou - 1.0) < 1e-6
    assert all(abs(c - 1.0) < 1e-6 for c in ci)


def test_miou_from_confusion_zero():
    cm = np.zeros((5, 5), dtype=np.int64)
    miou, ci = miou_from_confusion(cm)
    assert miou == 0.0


def test_update_confusion_ignore():
    cm = np.zeros((3, 3), dtype=np.int64)
    preds = np.array([0, 1, 2, 0])
    targets = np.array([0, 1, 255, 255])
    update_confusion(cm, preds, targets)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm.sum() == 2


def test_resolve_checkpoint_env(monkeypatch):
    monkeypatch.setenv("PC2D_CHECKPOINT", str(Path("/tmp/test_ckpt.pt")))
    cfg = {"model": {}, "checkpoint": {"dir": "checkpoints", "best_name": "best_miou.pt"}}
    p = _resolve_checkpoint(cfg)
    assert p.name == "test_ckpt.pt"


def test_resolve_checkpoint_from_model(monkeypatch):
    monkeypatch.delenv("PC2D_CHECKPOINT", raising=False)
    cfg = {"model": {"checkpoint": "checkpoints/best.pt"}, "checkpoint": {}}
    p = _resolve_checkpoint(cfg)
    assert p.name == "best.pt"


def test_resolve_checkpoint_from_dir():
    cfg = {"model": {}, "checkpoint": {"dir": "checkpoints", "best_name": "best_miou.pt"}}
    p = _resolve_checkpoint(cfg)
    assert p.name == "best_miou.pt"


def test_resolve_seq_dir_cli():
    cfg = {"source": {}, "data": {}}
    p = _resolve_seq_dir(cfg, cli_override=str(Path("/data/seq08")))
    assert p.name == "seq08"


def test_resolve_seq_dir_env(monkeypatch):
    monkeypatch.setenv("PC2D_SEQ_DIR", str(Path("/data/seq08")))
    cfg = {"source": {}, "data": {}}
    p = _resolve_seq_dir(cfg)
    assert p.name == "seq08"


def test_resolve_seq_dir_from_source():
    cfg = {"source": {"seq_dir": "data/sequences/08"}, "data": {}}
    p = _resolve_seq_dir(cfg)
    assert p.name == "08"


def test_resolve_seq_dir_from_raw_root():
    cfg = {"source": {}, "data": {"raw_root": "data/sequences", "val_seqs": [8]}}
    p = _resolve_seq_dir(cfg)
    assert p.name == "08"


def test_resolve_seq_dir_none(monkeypatch):
    monkeypatch.delenv("PC2D_SEQ_DIR", raising=False)
    cfg = {"source": {}, "data": {}}
    assert _resolve_seq_dir(cfg) is None


def test_distance_bands_structure():
    assert len(DISTANCE_BANDS) == 5
    assert BAND_NAMES == ["0-5m", "5-10m", "10-20m", "20-40m", "40-80m"]


def test_evaluate_pixel_level_synthetic(tmp_path):
    import torch
    from src.models.unet import RangeImageUNet

    model = RangeImageUNet(in_channels=5, num_classes=5, base_channels=8,
                           use_groupnorm=True, groups=2)
    device = torch.device("cpu")

    ri = torch.randn(3, 5, 64, 2048)
    li = torch.randint(0, 5, (3, 64, 2048), dtype=torch.int64)
    ri[:, 0] = ri[:, 0].clamp(0, 1)

    class FakeLoader:
        def __iter__(self):
            yield ri, li

    result = evaluate_pixel_level(
        model, FakeLoader(), device, num_classes=5, use_amp=False,
        max_range=80.0, limit=1,
    )

    assert result["n_samples"] == 3
    assert 0.0 <= result["miou_5class"] <= 1.0
    assert 0.0 <= result["miou_4class"] <= 1.0
    assert len(result["confusion_5class"]) == 5
    assert len(result["confusion_4class"]) == 4
    assert "latency_ms" in result
    assert all(bn in result["per_distance_band_4class"] for bn in BAND_NAMES)


def test_plot_confusion_matrices(tmp_path):
    cm_5 = np.eye(5, dtype=np.int64) * 100
    cm_4 = np.eye(4, dtype=np.int64) * 100
    plot_confusion_matrices(cm_5, cm_4, tmp_path)
    assert (tmp_path / "confusion_matrices.png").exists()


def test_plot_distance_bands(tmp_path):
    pixel_bands = {bn: {"miou": 0.5} for bn in BAND_NAMES}
    plot_distance_bands(pixel_bands, None, tmp_path)
    assert (tmp_path / "per_distance_band.png").exists()


def test_plot_distance_bands_both(tmp_path):
    pixel_bands = {bn: {"miou": 0.5} for bn in BAND_NAMES}
    point_bands = {bn: {"miou": 0.4} for bn in BAND_NAMES}
    plot_distance_bands(pixel_bands, point_bands, tmp_path)
    assert (tmp_path / "per_distance_band.png").exists()


def test_plot_class_ious(tmp_path):
    pixel_ci = {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2}
    plot_class_ious(pixel_ci, None, tmp_path)
    assert (tmp_path / "per_class_iou.png").exists()


def test_plot_class_ious_both(tmp_path):
    pixel_ci = {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2}
    point_ci = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    plot_class_ious(pixel_ci, point_ci, tmp_path)
    assert (tmp_path / "per_class_iou.png").exists()


def test_generate_summary(tmp_path):
    results = {
        "timestamp": "2026-01-01T00:00:00",
        "device": "cpu",
        "precision": "fp32",
        "pixel": {
            "n_samples": 100,
            "miou_5class": 0.5,
            "miou_4class": 0.6,
            "class_ious_4": {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2},
            "per_distance_band_4class": {bn: {"miou": 0.5} for bn in BAND_NAMES},
            "latency_ms": {"mean": 10.0, "std": 2.0},
        },
        "point": None,
        "memory": {"peak_rss_mb": 500.0, "gpu_peak_mb": None},
    }
    generate_summary(results, tmp_path)
    summary = (tmp_path / "eval_summary.md").read_text(encoding="utf-8")
    assert "Pixel-Level" in summary
    assert "0.5000" in summary


def test_generate_summary_with_point(tmp_path):
    results = {
        "timestamp": "2026-01-01T00:00:00",
        "device": "cuda",
        "precision": "fp16",
        "pixel": {
            "n_samples": 100,
            "miou_5class": 0.5,
            "miou_4class": 0.6,
            "class_ious_4": {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2},
            "per_distance_band_4class": {bn: {"miou": 0.5} for bn in BAND_NAMES},
            "latency_ms": {"mean": 10.0, "std": 2.0},
        },
        "point": {
            "n_scans": 50,
            "miou_5class": 0.4,
            "miou_4class": 0.5,
            "class_ious_4": {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1},
            "per_distance_band_4class": {bn: {"miou": 0.4} for bn in BAND_NAMES},
            "latency_ms": {"mean": 50.0, "std": 5.0},
        },
        "memory": {"peak_rss_mb": 500.0, "gpu_peak_mb": 2000.0},
    }
    generate_summary(results, tmp_path)
    summary = (tmp_path / "eval_summary.md").read_text(encoding="utf-8")
    assert "Point-Level" in summary
    assert "GPU Peak" in summary


def test_inject_results(tmp_path):
    from eval.inject_results import inject, MARKER_START, MARKER_END

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    data = {
        "timestamp": "2026-01-01T00:00:00",
        "device": "cpu",
        "precision": "fp32",
        "pixel": {
            "n_samples": 10,
            "miou_5class": 0.5,
            "miou_4class": 0.6,
            "class_ious_4": {"drivable": 0.5, "terrain": 0.4, "obstacle": 0.3, "dynamic": 0.2},
            "per_distance_band_4class": {bn: {"miou": 0.5} for bn in BAND_NAMES},
            "latency_ms": {"mean": 10.0, "std": 2.0},
        },
        "point": None,
        "memory": {"peak_rss_mb": 100.0, "gpu_peak_mb": None},
    }
    (results_dir / "eval_20260101_000000.json").write_text(
        json.dumps(data), encoding="utf-8")

    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Title\n", encoding="utf-8")

    inject(readme_path, results_dir)

    readme_text = readme_path.read_text(encoding="utf-8")
    assert MARKER_START in readme_text
    assert MARKER_END in readme_text
    assert "0.6000" in readme_text


def test_inject_results_updates_existing(tmp_path):
    from eval.inject_results import inject, MARKER_START, MARKER_END

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    data = {
        "timestamp": "2026-01-01T00:00:00",
        "device": "cpu",
        "precision": "fp32",
        "pixel": {
            "n_samples": 10,
            "miou_5class": 0.7,
            "miou_4class": 0.8,
            "class_ious_4": {"drivable": 0.8, "terrain": 0.7, "obstacle": 0.6, "dynamic": 0.5},
            "per_distance_band_4class": {bn: {"miou": 0.7} for bn in BAND_NAMES},
            "latency_ms": {"mean": 8.0, "std": 1.0},
        },
        "point": None,
        "memory": {"peak_rss_mb": 100.0, "gpu_peak_mb": None},
    }
    (results_dir / "eval_20260101_000000.json").write_text(
        json.dumps(data), encoding="utf-8")

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"# Title\n{MARKER_START}\nold content\n{MARKER_END}\n",
        encoding="utf-8")

    inject(readme_path, results_dir)

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "0.8000" in readme_text
    assert "old content" not in readme_text
