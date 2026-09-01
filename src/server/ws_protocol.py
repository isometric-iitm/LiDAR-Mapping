import struct
import zlib
from pathlib import Path

import numpy as np
import yaml


def load_classes(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "classes.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def class_colors(num_binned: bool = True) -> dict[str, str]:
    cfg = load_classes()
    block = cfg["binned_classes"] if num_binned else cfg["classes"]
    return {str(k): v["color"] for k, v in block.items()}


def grid_meta_message(grid) -> dict:
    colors = class_colors(num_binned=True)
    return {
        "type": "grid_meta",
        "r_min": grid.r_min,
        "r_max": grid.r_max,
        "r_transition": grid.r_transition,
        "alpha": grid.alpha,
        "dr_0": grid.dr_0,
        "n_rings": grid.n_rings,
        "phase1_rings": grid.phase1_rings,
        "n_theta": grid.n_theta,
        "n_classes": grid.n_classes,
        "class_colors": colors,
    }


# Binary WS framing: header "<IHHQQiiiii" (44 bytes):
#   magic(0x50433244) code(1=snap,2=delta,3=cloud) version frame epoch n_a n_b seq total yaw_cd
# Row record: 28 bytes (6 f32 [i,j,z_mean,z_max,occ,trav] + 1 u8 cls + 3 pad).
#   snapshot: n_a rows + n_a cls bytes
#   delta: n_a rows + n_a cls bytes + n_b freed rows (2 f32 each)
#   cloud: n_a xyz f32 + n_a cls bytes
_MAGIC = 0x50433244
_VERSION = 3
K_SNAPSHOT = 1
K_DELTA = 2
K_CLOUD = 3
_HEAD = struct.Struct("<IHHQQiiiii")
_HEAD_LEN = _HEAD.size          # 44
_ROW_F32 = 6 * 4                # i,j,z_mean,z_max,occ,trav = 24 bytes
_ROW_PAD = 3                    # pad cls tail out to 4-byte alignment (24+1+3=28)
_ROW_TOTAL = _ROW_F32 + 1 + _ROW_PAD  # 28 bytes per grid row record
_PAD3 = b"\x00" * _ROW_PAD
CHUNK_CELLS = 8000


def _head(code: int, frame: int, epoch: int, n_a: int, n_b: int, seq: int, total: int,
          yaw_cd: int = 0) -> bytes:
    return _HEAD.pack(_MAGIC, code, _VERSION, int(frame), int(epoch),
                      int(n_a), int(n_b), int(seq), int(total), int(yaw_cd))


def _row_bytes(rows6: np.ndarray, cls: np.ndarray) -> bytes:
    n = cls.shape[0]
    return (rows6.astype("<f4", copy=False).tobytes()
            + cls.astype(np.uint8, copy=False).tobytes()
            + _PAD3 * n)


def _itr(n: int, chunk: int) -> int:
    return max(1, (n + chunk - 1) // chunk)


def iter_snapshot_frames(frame: int, epoch: int, rows6: np.ndarray, cls: np.ndarray,
                         chunk: int = CHUNK_CELLS, yaw_cd: int = 0):
    total = _itr(rows6.shape[0], chunk)
    for p in range(total):
        s = p * chunk
        e = min(rows6.shape[0], s + chunk)
        yield (_head(K_SNAPSHOT, frame, epoch, e - s, 0, p, total, yaw_cd)
               + _row_bytes(rows6[s:e], cls[s:e]))


def iter_delta_frames(frame: int, epoch: int, rows6: np.ndarray, cls: np.ndarray,
                      freed: np.ndarray, chunk: int = CHUNK_CELLS, yaw_cd: int = 0):
    total = _itr(rows6.shape[0], chunk)
    for p in range(total):
        s = p * chunk
        e = min(rows6.shape[0], s + chunk)
        fb = b""
        n_freed = 0
        if p == total - 1 and freed.shape[0]:
            fb = freed.astype("<f4", copy=False).tobytes()
            n_freed = freed.shape[0]
        yield (_head(K_DELTA, frame, epoch, e - s, n_freed, p, total, yaw_cd)
               + _row_bytes(rows6[s:e], cls[s:e]) + fb)


def iter_cloud_frames(frame: int, epoch: int, xyz: np.ndarray, cls: np.ndarray,
                      chunk: int = 30000, yaw_cd: int = 0):
    n = xyz.shape[0]
    total = _itr(n, chunk)
    for p in range(total):
        s = p * chunk
        e = min(n, s + chunk)
        body = (xyz[s:e].reshape(-1).astype("<f4", copy=False).tobytes()
                + cls[s:e].astype(np.uint8, copy=False).tobytes())
        yield (_head(K_CLOUD, frame, epoch, e - s, 0, p, total, yaw_cd) + body)


# Compression (optional, toggled via config)
_COMPRESS_LEVEL = 1  # zlib fastest (1-9); level 1 is ~50 MB/s compression, ~200 MB/s decompress
_COMPRESS_THRESHOLD = 512  # don't bother compressing payloads under this many bytes


def _raw_deflate(body: bytes) -> bytes:
    """Compress with raw DEFLATE (no zlib header/trailer), matching the browser's
    DecompressionStream('deflate-raw') so the dashboard can inflate it natively."""
    cobj = zlib.compressobj(_COMPRESS_LEVEL, zlib.DEFLATED, -15)
    return cobj.compress(body) + cobj.flush()


def _maybe_compress(body: bytes, enabled: bool = False) -> bytes:
    """Optionally compress a binary frame body with raw DEFLATE (level 1, fast).
    Returns compressed bytes with a 5-byte prefix: b'Z' + uint32 original size.
    Returns original body if compression is disabled or payload too small."""
    if not enabled or len(body) < _COMPRESS_THRESHOLD:
        return body
    compressed = _raw_deflate(body)
    if len(compressed) >= len(body):
        return body  # no benefit
    return b"Z" + struct.pack("<I", len(body)) + compressed


def decompress_if_needed(data: bytes) -> bytes:
    """Decompress a binary frame body if it has the 'Z' prefix (raw DEFLATE)."""
    if len(data) > 5 and data[0:1] == b"Z":
        orig_size = struct.unpack("<I", data[1:5])[0]
        return zlib.decompress(data[5:], wbits=-15)
    return data


# JSON variants (HTTP endpoints / legacy)
def snapshot_message(grid) -> dict:
    snap = grid.compute_snapshot()
    cells = np.concatenate([snap["rows"], snap["cls"].reshape(-1, 1).astype(np.float32)], axis=1).tolist()
    return {"type": "snapshot", "frame": snap["frame"], "cells": cells}


def delta_message(grid) -> dict:
    d = grid.compute_delta()
    return {"type": "delta", "frame": d["frame"], "updated": d["rows"].tolist(), "freed": d["freed"].tolist()}


def stats_message(stats: dict) -> dict:
    return {"type": "stats", **stats}