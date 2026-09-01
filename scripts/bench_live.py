#!/usr/bin/env python3
"""Wall-clock benchmark of the LIVE grid path (no GPU, no sensor data).

This mirrors what src/server/app.py's ingest loop does each frame in precise
mode: update the grid from a synthetic scan, compute_delta, (simulate) send it,
commit_delta — plus a periodic snapshot every `--snapshot-interval` frames.
Unlike bench_grid.py (which reports the grid-internal per-stage timings), this
times the whole call sequence from the caller's perspective, wall-clock, which
is what actually gates the live 30fps dashboard stream.

Usage:
    uv run python scripts/bench_live.py [--frames 300] [--n 60000] [--snap-interval 5]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import GridConfig
from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config


def _synthetic_scan(ego_x: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    r = np.sqrt(np.random.uniform(0.25, 95.0**2, n))
    az = np.random.uniform(0, 2 * np.pi, n)
    el = np.random.uniform(-0.4, 0.12, n)
    x = ego_x + r * np.cos(az)
    y = r * np.sin(az)
    z = 0.25 + np.sin(r * 0.3) * 0.15 + el * r
    pts = np.stack([x, y, z, np.full(n, 0.5)], axis=1).astype(np.float32)
    cls = rng.integers(0, 4, n).astype(np.uint8)
    return pts, cls


def main():
    ap = argparse.ArgumentParser(description="live-grid wall-clock bench")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--n", type=int, default=60_000, help="points per synthetic scan")
    ap.add_argument("--snap-interval", type=int, default=5)
    args = ap.parse_args()

    base = load_grid_config()
    cfg = GridConfig.from_dict(base)
    grid = LogPolarGrid(cfg)

    frames = [_synthetic_scan(ego_x=k * 0.8, n=args.n) for k in range(args.frames)]

    # warm up numba JIT / torch fallback so the timed loops below are steady-state
    grid.update(frames[0][0], frames[0][1])
    d = grid.compute_delta()
    grid.commit_delta(d)
    grid = LogPolarGrid(cfg)

    # variant A: delta-only every frame
    t0 = time.perf_counter()
    total_delta = 0.0
    total_commit = 0.0
    for k in range(args.frames):
        grid.update(frames[k][0], frames[k][1])
        s = time.perf_counter()
        d = grid.compute_delta()
        total_delta += (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        grid.commit_delta(d)
        total_commit += (time.perf_counter() - s) * 1000
    wall_ms = (time.perf_counter() - t0) * 1000.0 / args.frames
    print("\n=== live loop (delta-only, every frame) ===")
    print(f"  wall: {wall_ms:.2f} ms/frame  ({1000 / max(wall_ms, 1e-9):.1f} fps worst-ceiling)")
    print(f"  compute_delta: {total_delta / args.frames:.3f} ms/frame  "
          f"commit_delta: {total_commit / args.frames:.3f} ms/frame")

    # variant B: with periodic full snapshots at the default cadence
    grid = LogPolarGrid(cfg)
    t0 = time.perf_counter()
    total_snap = 0.0
    n_snaps = 0
    for k in range(args.frames):
        grid.update(frames[k][0], frames[k][1])
        d = grid.compute_delta()
        grid.commit_delta(d)
        if k % args.snap_interval == args.snap_interval - 1:
            s = time.perf_counter()
            sn = grid.compute_snapshot()
            grid.commit_snapshot()
            total_snap += (time.perf_counter() - s) * 1000
            n_snaps += 1
    wall_ms = (time.perf_counter() - t0) * 1000.0 / args.frames
    print(f"\n=== live loop (with snapshot every {args.snap_interval} frames) ===")
    print(f"  wall: {wall_ms:.2f} ms/frame avg")
    print(f"  snapshot: {total_snap / max(n_snaps, 1):.2f} ms/snapshot  ({n_snaps} snapshots)")

    sn = grid.compute_snapshot()
    mr = grid.memory_report()
    print(f"\n=== grid ===")
    print(f"  rendered: {int(sn['mask'].sum()):,} / {grid.n_cells:,} cells")
    print(f"  grid: {mr['grid_kb'] / 1024:.1f} MB  uniform-equivalent: {mr['uniform_mb']:.1f} MB  "
          f"compression: {mr['compression_ratio']:.0f}x")


if __name__ == "__main__":
    main()