import numpy as np
from pathlib import Path
import yaml


def load_class_mapping(config_path: str | Path | None = None) -> dict[int, int]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "classes.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {int(k): int(v) for k, v in cfg["full_semantic_kitti_mapping"].items()}


def remap_labels(raw_labels: np.ndarray, mapping: dict[int, int] | None = None) -> np.ndarray:
    if mapping is None:
        mapping = load_class_mapping()
    mapped = np.full_like(raw_labels, fill_value=255, dtype=np.uint8)
    for raw_id, class_id in mapping.items():
        mapped[raw_labels == raw_id] = class_id
    return mapped


def compute_class_weights(label_dir: Path, mapping: dict[int, int] | None = None, num_classes: int = 4) -> np.ndarray:
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
    freq = counts / total
    freq = np.clip(freq, 1e-6, 1.0)
    weights = 1.0 / freq
    median = np.median(weights)
    weights = weights / median
    return weights.astype(np.float32)
