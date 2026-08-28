#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import tarfile
import io
import struct

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.label_mapping import remap_labels, load_class_mapping
from src.data.projection import compute_projection


def read_scan(bin_path: Path) -> np.ndarray:
    scan = np.fromfile(str(bin_path), dtype=np.float32)
    return scan.reshape(-1, 4)


def read_labels(label_path: Path) -> np.ndarray:
    raw = np.fromfile(str(label_path), dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint8)


def process_sequence(
    seq_dir: Path,
    mapping: dict[int, int],
    output_dir: Path,
    shard_size: int = 200,
    max_range: float = 80.0,
    h: int = 64,
    w: int = 2048,
    fov_top_deg: float = 2.0,
    fov_bottom_deg: float = -24.8,
    start_shard_idx: int = 0,
) -> tuple[int, list, int]:
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"

    if not velodyne_dir.exists() or not labels_dir.exists():
        print(f"  [SKIP] {seq_dir}: missing velodyne/ or labels/")
        return 0, [], start_shard_idx

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts = [0, 0, 0, 0]
    count = 0
    shard_idx = start_shard_idx
    shard_data = []

    for bin_path in bin_files:
        label_path = labels_dir / (bin_path.stem + ".label")
        if not label_path.exists():
            continue

        points = read_scan(bin_path)
        raw_labels = read_labels(label_path)
        if len(points) != len(raw_labels):
            continue

        mapped_labels = remap_labels(raw_labels, mapping)
        proj, ranges = compute_projection(points, h=h, w=w, fov_top_deg=fov_top_deg, fov_bottom_deg=fov_bottom_deg)

        mask = ranges <= max_range
        points = points[mask].astype(np.float32)
        mapped_labels = mapped_labels[mask].astype(np.int64)
        proj = proj[mask].astype(np.int32)

        for c in range(4):
            class_counts[c] += int(np.sum(mapped_labels == c))

        points_bytes = points.tobytes()
        labels_bytes = mapped_labels.tobytes()
        proj_bytes = proj.tobytes()
        n = len(points)

        header = struct.pack("<III", n, *proj.shape)
        proj_with_header = header + proj_bytes

        shard_data.append((count, points_bytes, labels_bytes, proj_with_header))
        count += 1

        if len(shard_data) >= shard_size:
            _write_shard(output_dir / f"shard-{shard_idx:06d}.tar", shard_data)
            shard_idx += 1
            shard_data = []
            print(f"  {seq_dir.name}: shard {shard_idx} written ({count} scans so far)")

    if shard_data:
        _write_shard(output_dir / f"shard-{shard_idx:06d}.tar", shard_data)
        shard_idx += 1

    print(f"  {seq_dir.name}: {count} scans -> {shard_idx - start_shard_idx} shards (shards {start_shard_idx}-{shard_idx-1})")
    return count, class_counts, shard_idx


def _write_shard(shard_path: Path, samples: list):
    with tarfile.open(str(shard_path), "w") as tar:
        for idx, points_bytes, labels_bytes, proj_bytes in samples:
            for name, data in [
                (f"{idx:06d}.bin", points_bytes),
                (f"{idx:06d}.label", labels_bytes),
                (f"{idx:06d}.proj", proj_bytes),
            ]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description="Preprocess SemanticKITTI → WebDataset shards")
    parser.add_argument("--raw-root", type=str, default="F:/sih/dataset/sequences")
    parser.add_argument("--processed-root", type=str, default="F:/sih/processed")
    parser.add_argument("--train-seqs", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--val-seqs", type=int, nargs="+", default=[8])
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    mapping = load_class_mapping(
        Path(__file__).resolve().parent.parent / "config" / "classes.yaml"
    )

    meta_path = processed_root / "meta.json"
    if meta_path.exists() and not args.force:
        print(f"Processed data exists at {processed_root}. Use --force to reprocess.")
        return

    total_class_counts = [0, 0, 0, 0]
    total_train = 0
    total_val = 0

    print("Processing training sequences...")
    train_dir = processed_root / "train"
    train_shard_idx = 0
    for seq_id in args.train_seqs:
        seq_dir = raw_root / f"{seq_id:02d}"
        n, counts, train_shard_idx = process_sequence(
            seq_dir, mapping, train_dir, args.shard_size, args.max_range,
            start_shard_idx=train_shard_idx,
        )
        total_train += n
        for i in range(4):
            total_class_counts[i] += counts[i] if counts else 0

    print("\nProcessing validation sequences...")
    val_dir = processed_root / "val"
    val_shard_idx = 0
    for seq_id in args.val_seqs:
        seq_dir = raw_root / f"{seq_id:02d}"
        n, counts, val_shard_idx = process_sequence(
            seq_dir, mapping, val_dir, args.shard_size, args.max_range,
            start_shard_idx=val_shard_idx,
        )
        total_val += n
        for i in range(4):
            total_class_counts[i] += counts[i] if counts else 0

    arr = np.array(total_class_counts, dtype=np.float64)
    freq = np.clip(arr / arr.sum(), 1e-6, 1.0)
    weights = (1.0 / freq / np.median(1.0 / freq)).astype(np.float32).tolist()

    meta = {
        "n_train": total_train,
        "n_val": total_val,
        "class_counts": total_class_counts,
        "class_weights": weights,
        "train_seqs": args.train_seqs,
        "val_seqs": args.val_seqs,
        "shard_size": args.shard_size,
        "max_range": args.max_range,
        "num_classes": 4,
        "format": "raw_bin",
    }

    processed_root.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {total_train} train, {total_val} val scans")
    print(f"Class counts: {total_class_counts}")
    print(f"Class weights: {weights}")


if __name__ == "__main__":
    main()
