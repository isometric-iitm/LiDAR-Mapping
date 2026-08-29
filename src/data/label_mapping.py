import numpy as np
import torch
from pathlib import Path
import yaml


def load_class_mapping(config_path: str | Path | None = None) -> dict[int, int]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "classes.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {int(k): int(v) for k, v in cfg["semantic_kitti_5class"].items()}


def remap_labels(raw_labels: np.ndarray, mapping: dict[int, int] | None = None) -> np.ndarray:
    if mapping is None:
        mapping = load_class_mapping()
    mapped = np.full_like(raw_labels, fill_value=255, dtype=np.uint8)
    for raw_id, class_id in mapping.items():
        mapped[raw_labels == raw_id] = class_id
    return mapped


_BIN_LUT: np.ndarray | None = None


def load_bin_mapping(config_path: str | Path | None = None) -> dict[int, int]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "classes.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {int(k): int(v) for k, v in cfg["bin_5_to_4"].items()}


def bin_lut(mapping: dict[int, int] | None = None) -> np.ndarray:
    """256-entry uint8 lookup table for 5->4 class remap (built once)."""
    global _BIN_LUT
    if _BIN_LUT is None:
        if mapping is None:
            mapping = load_bin_mapping()
        lut = np.full((256,), 255, dtype=np.uint8)
        for c5, c4 in mapping.items():
            lut[int(c5) & 0xFF] = int(c4)
        _BIN_LUT = lut
    return _BIN_LUT


def bin_5_to_4(labels: np.ndarray, mapping: dict[int, int] | None = None) -> np.ndarray:
    lut = bin_lut(mapping)
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8)
    return lut[labels]


def bin_5_to_4_torch(labels: torch.Tensor, mapping: dict[int, int] | None = None) -> torch.Tensor:
    if mapping is None:
        mapping = load_bin_mapping()
    lut = torch.full((256,), fill_value=255, dtype=labels.dtype)
    for c5, c4 in mapping.items():
        lut[c5] = c4
    return lut[labels.long()]


def compute_class_weights(label_dir: Path, mapping: dict[int, int] | None = None, num_classes: int = 5) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    label_files = sorted(label_dir.glob("*.label"))
    for lf in label_files:
        raw = np.fromfile(str(lf), dtype=np.uint32)
        semantic = (raw & 0xFFFF).astype(np.uint8)
        mapped = remap_labels(semantic, mapping)
        for c in range(num_classes):
            counts[c] += np.sum(mapped == c)
    total = counts.sum()
    if total == 0:
        return np.ones(num_classes, dtype=np.float32)
    freq = np.clip(counts / total, 1e-6, 1.0)
    weights = 1.0 / np.sqrt(freq)
    median = np.median(weights)
    weights = weights / median
    weights = np.clip(weights, 0.1, 5.0)
    return weights.astype(np.float32)
