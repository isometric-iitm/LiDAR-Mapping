#!/usr/bin/env python3
"""Pack a demo KITTI sequence slice into chunked browser assets + manifest.

Produces, under $PC2D_DEMO_OUT (default dashboard/public/demo/ — the
dashboard's static asset root, served same-origin at /demo/):
  sequence/chunk_NNN.bin, manifest.json (see module docstring for formats).

  sequence/chunk_NNN.bin   -- header + per-frame index + gzip'd points
  manifest.json            -- chunk index (sha256/bytes/frames), composed poses
                              (pose_k @ Tr), per-frame yaw_cd, grid constants,
                              class colors, fps

Points are stored float16 (x,y,z,intensity) by default (~0.85 MB/frame gzipped
vs ~1.5 MB fp32; max coordinate rounding ~3 cm at 80 m, ground-percentile
delta < 0.5 mm). Pass --fp32 to store full float32 instead.

Output goes DIRECTLY to dashboard/public/demo/ (the dashboard's static asset
root, served same-origin at /demo/ by both next dev and Cloudflare Pages) —
no intermediate staging dir, no copy step.

Chunk layout (little-endian):
  bytes 0..3    magic "PC2D" (0x50433244)
  u16  version  = 1
  u16  n_frames = frames in this chunk
  u16  point_width = 4 floats per point
  u16  point_dtype = 16 (float16) | 32 (float32)
  then n_frames * (u64 offset, u32 length) index entries (offsets into payload
  region AFTER the index; each entry is an independent gzip member of the raw
  float array), then the concatenated payload.

Frames are gzip members so the worker can inflate each one independently with
DecompressionStream("gzip") or pako.

Config (env): PC2D_SEQ_DIR, PC2D_DEMO_FRAMES (default 300),
PC2D_DEMO_CHUNK_FRAMES (default 20), PC2D_DEMO_START (default 0),
PC2D_DEMO_OUT (default dashboard/public/demo).
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402


def load_config(name: str) -> dict:
    with open(ROOT / "config" / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def _load_env_file():
    """Parse pc2d/.env into os.environ if present (never overrides existing vars)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass


def find_seq_dir(explicit: str | None) -> Path:
    import os
    _load_env_file()
    if explicit:
        return Path(explicit)
    env = os.getenv("PC2D_SEQ_DIR", "")
    if env:
        p = Path(env)
        if p.exists():
            return p
    p = ROOT / "data" / "sequences" / "08"
    if p.exists():
        return p
    raise FileNotFoundError("No sequence dir found; pass --seq-dir or set $PC2D_SEQ_DIR")


def load_pose_mats(seq_dir: Path, n_frames: int, start: int) -> np.ndarray | None:
    """Exact port of src/server/app.py:_load_pose_mats: M_k @ Tr (float32)."""
    poses_path = seq_dir / "poses.txt"
    if not poses_path.exists():
        return None
    Tr = np.eye(4, dtype=np.float64)
    calib_path = seq_dir / "calib.txt"
    if calib_path.exists():
        for line in calib_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("Tr"):
                vals = np.fromstring(line.split(":", 1)[1].strip(), sep=" ", dtype=np.float64)
                Tr[:3, :4] = vals[:12].reshape(3, 4)
                break
    poses = np.loadtxt(poses_path, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 12:
        return None
    M = np.zeros((poses.shape[0], 4, 4), dtype=np.float32)
    M[:, :3, :4] = poses[:, :12].reshape(-1, 3, 4)
    M[:, 3, 3] = 1.0
    composed = (M @ Tr.astype(np.float32)).astype(np.float32)
    # clamp + repeat so every demo frame has a pose even if poses.txt is shorter
    if len(composed) < n_frames:
        pad = np.repeat(composed[-1:], n_frames - len(composed), axis=0)
        composed = np.concatenate([composed, pad], axis=0)
    return composed[start:start + n_frames]


def yaw_cd_from_pose(pose: np.ndarray) -> int:
    """Ego forward yaw in centi-degrees (app.py:_yaw_cd): atan2(R[0,0], -R[2,0])."""
    R = pose[:3, :3]
    fx, fz = float(R[0, 0]), float(R[2, 0])
    return int(round(math.degrees(math.atan2(fx, -fz)) * 100))


def main() -> int:
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=str, default=None)
    ap.add_argument("--frames", type=int, default=None, help="Total demo frames (default $PC2D_DEMO_FRAMES or 300)")
    ap.add_argument("--chunk-frames", type=int, default=None, help="Frames per chunk (default $PC2D_DEMO_CHUNK_FRAMES or 20)")
    ap.add_argument("--start", type=int, default=None, help="First sequence frame index (default $PC2D_DEMO_START or 0)")
    ap.add_argument("--out", type=str, default=None, help="Output dir (default $PC2D_DEMO_OUT or dashboard/public/demo)")
    ap.add_argument("--seq-id", type=str, default=None)
    ap.add_argument("--fp32", action="store_true", help="Store float32 points instead of float16")
    args = ap.parse_args()

    seq_dir = find_seq_dir(args.seq_dir)
    n_total = int(args.frames or os.getenv("PC2D_DEMO_FRAMES", 300))
    chunk_frames = int(args.chunk_frames or os.getenv("PC2D_DEMO_CHUNK_FRAMES", 20))
    start = int(args.start if args.start is not None else os.getenv("PC2D_DEMO_START", 0))
    out_dir = Path(args.out or os.getenv("PC2D_DEMO_OUT", ROOT / "dashboard" / "public" / "demo"))
    seq_id = args.seq_id or seq_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence").mkdir(exist_ok=True)

    bins = sorted((seq_dir / "velodyne").glob("*.bin"))
    if not bins:
        print(f"[pack-demo] no .bin files under {seq_dir / 'velodyne'}")
        return 1
    n_avail = min(n_total, len(bins) - start)
    if n_avail < n_total:
        print(f"[pack-demo] requested {n_total} frames but only {n_avail} available from index {start}")
    n_total = n_avail
    n_chunks = math.ceil(n_total / chunk_frames)

    print(f"[pack-demo] seq={seq_id} dir={seq_dir} frames={n_total} (start={start}) chunks={n_chunks} x {chunk_frames}")
    print(f"[pack-demo] out={out_dir}")

    grid_cfg = load_config("grid")
    classes_cfg = load_config("classes")
    g = grid_cfg["grid"]
    n_classes = int(g["n_classes"])
    binned = classes_cfg["binned_classes"]

    pose_mats = load_pose_mats(seq_dir, n_total, start)
    if pose_mats is None:
        print("[pack-demo] WARNING: poses.txt missing; cloud falls back to sweep-only mode")
        poses_flat = None
        yaws = [0] * n_total
    else:
        poses_flat = pose_mats.reshape(-1, 16).astype(np.float32).tolist()  # row-major 4x4 per frame
        yaws = [yaw_cd_from_pose(pose_mats[k]) for k in range(n_total)]

    t_all = time.time()
    chunks_meta = []
    total_bytes = 0
    for c in range(n_chunks):
        c_start = c * chunk_frames
        c_len = min(chunk_frames, n_total - c_start)
        members: list[bytes] = []
        for k in range(c_len):
            fi = start + c_start + k
            pts = np.fromfile(str(bins[fi]), dtype=np.float32).reshape(-1, 4)
            store = pts.astype(np.float16 if not args.fp32 else np.float32)
            raw = store.tobytes()
            members.append(gzip.compress(raw, compresslevel=6))
        # index + payload
        index = np.zeros(c_len, dtype=[("offset", "<u8"), ("length", "<u4")])
        header_len = 8 + index.nbytes
        off = 0
        for k, m in enumerate(members):
            index[k] = (off, len(m))
            off += len(m)
        buf = io.BytesIO()
        buf.write(b"PC2D")
        buf.write(np.uint16(1).tobytes())          # version
        buf.write(np.uint16(c_len).tobytes())      # n_frames
        buf.write(np.uint16(4).tobytes())          # point_width (floats per point)
        buf.write(np.uint16(16 if not args.fp32 else 32).tobytes())  # point dtype bits
        buf.write(index.tobytes())
        for m in members:
            buf.write(m)
        data = buf.getvalue()
        p = out_dir / "sequence" / f"chunk_{c:03d}.bin"
        p.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        chunks_meta.append({
            "file": f"sequence/chunk_{c:03d}.bin",
            "sha256": sha,
            "bytes": len(data),
            "frame_start": c_start,
            "frames": c_len,
        })
        total_bytes += len(data)
        print(f"[pack-demo] chunk {c:02d}/{n_chunks}: {c_len} frames, {len(data) / 1e6:.2f} MB -> {p.name}")

    manifest = {
        "version": 1,
        "seq_id": seq_id,
        "seq_len": n_total,
        "fps": 10.0,
        "start_idx": start,
        "chunk_frames": chunk_frames,
        "n_chunks": n_chunks,
        "point_dtype": "float16" if not args.fp32 else "float32",
        "chunks": chunks_meta,
        "poses": poses_flat,          # [n][16] row-major f32 (world_T_velo), null if absent
        "yaws_cd": yaws,              # per-frame ego yaw, centi-degrees
        "grid": {
            "r_min": float(g["r_min"]), "r_max": float(g["r_max"]),
            "r_transition": float(g["r_transition"]), "dr_0": float(g["dr_0"]),
            "alpha": float(g["alpha"]), "n_theta": int(g["n_theta"]),
            "z_min": float(g["z_min"]), "z_max": float(g["z_max"]),
            "n_classes": n_classes,
            "occupancy_gain": float(g["occupancy_gain"]), "occ_threshold": float(g["occ_threshold"]),
            "phase1_rings": round((float(g["r_transition"]) - float(g["r_min"])) / float(g["dr_0"])),
            "n_rings": None,  # filled below
            "class_colors": {str(cid): meta["color"] for cid, meta in binned.items()},
        },
        "trav": {
            "enabled": bool(grid_cfg["traversability"]["enabled"]),
            "weights": [float(x) for x in grid_cfg["traversability"]["weights"]],
            "z_diff_thresh": float(grid_cfg["traversability"]["z_diff_thresh"]),
            "slope_thresh": float(grid_cfg["traversability"]["slope_thresh"]),
            "class_scores": [float(x) for x in grid_cfg["traversability"]["class_scores"]],
        },
        "cloud": {
            "points_max": 30000,
            "history_frames": 30,
        },
        "model": {
            "file": "models/unet_web_fp16_sm.onnx",
            "in_channels": 5,
            "num_classes": 5,
            "h": 64,
            "w": 2048,
            "fov_top_deg": 2.0,
            "fov_bottom_deg": -24.8,
            "max_range": 80.0,
            "bin_5_to_4": [0, 1, 2, 3, 3],
        },
        "total_bytes": total_bytes,
    }

    # resolve ring geometry exactly like logpolar_grid.py (phase-2 count via closed form)
    gr = manifest["grid"]
    phase1 = gr["phase1_rings"]
    phase2_span = gr["r_max"] - (gr["r_min"] + phase1 * gr["dr_0"])
    if phase2_span > 0 and gr["alpha"] > 1.0:
        n2 = math.log(1.0 + (phase2_span * (gr["alpha"] - 1.0) / gr["dr_0"])) / math.log(gr["alpha"])
        phase2 = max(1, int(round(n2)))
    else:
        phase2 = 0
    gr["phase2_rings"] = phase2
    gr["n_rings"] = phase1 + phase2

    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[pack-demo] manifest -> {mpath}")
    print(f"[pack-demo] total sequence bytes: {total_bytes / 1e6:.2f} MB across {n_chunks} chunks")
    print(f"[pack-demo] done in {time.time() - t_all:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
