import torch
import numpy as np
import struct
import tarfile
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator


def _decode_bin(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(-1, 4).copy()


def _decode_label(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int64).copy()


def _decode_proj(data: bytes) -> np.ndarray:
    n = struct.unpack("<III", data[:12])[0]
    return np.frombuffer(data[12:], dtype=np.int32).reshape(n, 2).copy()


def _iter_tar(tar_path: str | Path) -> Iterator[dict]:
    groups: dict[int, dict] = {}
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            stem, ext = Path(name).stem, Path(name).suffix.lstrip(".")
            key = int(stem)
            if key not in groups:
                groups[key] = {}
            f = tar.extractfile(member)
            if f is None:
                continue
            groups[key][ext] = f.read()

    for key in sorted(groups.keys()):
        g = groups[key]
        if "bin" in g and "label" in g and "proj" in g:
            yield {
                "points": _decode_bin(g["bin"]),
                "labels": _decode_label(g["label"]),
                "proj": _decode_proj(g["proj"]),
            }


class ShardDataset(IterableDataset):
    def __init__(self, shard_dir: Path, shuffle_shards: bool = True, shuffle_samples: int = 200):
        self.shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards in {shard_dir}")
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples

    def __iter__(self):
        import random

        paths = list(self.shard_paths)
        if self.shuffle_shards:
            random.shuffle(paths)

        buf = []
        for path in paths:
            for sample in _iter_tar(path):
                buf.append(sample)
                if self.shuffle_samples > 0 and len(buf) >= self.shuffle_samples:
                    random.shuffle(buf)
                    yield buf.pop(0)

        if buf:
            random.shuffle(buf)
            yield from buf

    def __len__(self):
        return -1


def _collate_fn(batch: list[dict]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    points_list = []
    labels_list = []
    proj_list = []
    for s in batch:
        points_list.append(torch.from_numpy(s["points"].astype(np.float32)))
        labels_list.append(torch.from_numpy(s["labels"].astype(np.int64)))
        proj_list.append(torch.from_numpy(s["proj"].astype(np.int32)))
    return points_list, labels_list, proj_list


def create_train_loader(
    processed_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle_buffer: int = 200,
) -> DataLoader:
    ds = ShardDataset(
        Path(processed_root) / "train",
        shuffle_shards=True,
        shuffle_samples=shuffle_buffer,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )


def create_val_loader(
    processed_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
) -> DataLoader:
    ds = ShardDataset(
        Path(processed_root) / "val",
        shuffle_shards=False,
        shuffle_samples=0,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
