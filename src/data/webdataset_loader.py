import torch
import numpy as np
import tarfile
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator

RI_SHAPE = (5, 64, 2048)
LI_SHAPE = (64, 2048)


def _iter_tar(tar_path: str | Path) -> Iterator[dict]:
    groups: dict[int, dict] = {}
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            ext = Path(member.name).suffix.lstrip(".")
            try:
                key = int(stem)
            except ValueError:
                continue
            if key not in groups:
                groups[key] = {}
            f = tar.extractfile(member)
            if f is None:
                continue
            groups[key][ext] = f.read()

            if "ri" in groups[key] and "li" in groups[key]:
                ri = np.frombuffer(groups[key]["ri"], dtype=np.float16).reshape(*RI_SHAPE).copy()
                li = np.frombuffer(groups[key]["li"], dtype=np.uint8).reshape(*LI_SHAPE).copy()
                del groups[key]
                yield {"ri": ri, "li": li}

    for key in sorted(groups.keys()):
        g = groups[key]
        if "ri" in g and "li" in g:
            ri = np.frombuffer(g["ri"], dtype=np.float16).reshape(*RI_SHAPE).copy()
            li = np.frombuffer(g["li"], dtype=np.uint8).reshape(*LI_SHAPE).copy()
            yield {"ri": ri, "li": li}


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


def _collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    ri_list = []
    li_list = []
    for s in batch:
        ri_list.append(torch.from_numpy(s["ri"].astype(np.float32)))
        li_list.append(torch.from_numpy(s["li"].astype(np.int64)))
    return torch.stack(ri_list), torch.stack(li_list)


def create_train_loader(
    processed_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
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
        pin_memory=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def create_val_loader(
    processed_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
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
        pin_memory=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
