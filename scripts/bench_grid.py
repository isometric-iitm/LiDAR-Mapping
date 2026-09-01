#!/usr/bin/env python3
"""Synthetic grid throughput benchmark (no GPU, no sensor data needed).

Builds a fixed replay of ~N ego-centric Velodyne-like scans (120k points each),
pre-generates them ONCE, then streams them into LogPolarGrid.update() in both
modes (precise = sensor-direct; decay = exponential occupancy + traversability)
to report ms/frame, FPS, and the per-stage breakdown the grid timestamps.

Usage:
    uv run python scripts/bench_grid.py [--frames 300] [--n-classes 4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import GridConfig
from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.grid_engine import jit_reduce

_STAGES = ("polar_ms", "occ_ms", "reduce_ms", "cls_ms", "state_ms")


def _synthetic_scan(ego_x: float, n: int = 120_000) -> tuple[np.ndarray, np.ndarray]:
    """Velodyne-64-like ring: azimuth-blended points around the ego position."""
    rng = np.random.default_rng(0)
    r = np.sqrt(np.random.uniform(0.25, 95.0**2, n))
    az = np.random.uniform(0, 2 * np.pi, n)
    el = np.random.uniform(-0.4, 0.12, n)  # small vertical spread (road + clutter)
    x = ego_x + r * np.cos(az)
    y = r * np.sin(az)
    z = 0.25 + np.sin(r * 0.3) * 0.15 + el * r  # gentle terrain undulation
    pts = np.stack([x, y, z, np.full(n, 0.5)], axis=1).astype(np.float32)
    cls = rng.integers(0, 4, n).astype(np.uint8)  # uniform-ish 4-class label stream
    return pts, cls


def _run_mode(cfg: GridConfig, frames: list[np.ndarray], labels: list[np.ndarray], n_frames: int) -> dict:
    grid = LogPolarGrid(cfg)
    stage_acc = {k: 0.0 for k in _STAGES}
    t0 = time.perf_counter()
    for i in range(n_frames):
        grid.update(frames[i], labels[i])
        st = grid._timings
        for k in stage_acc:
            stage_acc[k] += st.get(k, 0.0)
    wall = (time.perf_counter() - t0) * 1000.0 / n_frames
    rendered = grid.rendered()
    return {
        "wall_ms": wall,
        "frames": n_frames,
        "n_cells": grid.n_cells,
        "rendered": int(rendered.sum()),
        "stages": {k: float(v / n_frames) for k, v in stage_acc.items()},
        "jit": "numba" if jit_reduce._NUMBA_OK else "numpy",
    }


def main():
    ap = argparse.ArgumentParser(description="synthetic LogPolarGrid throughput bench")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--n", type=int, default=120_000, help="points per synthetic scan")
    args = ap.parse_args()

    base = load_grid_config()

    # pre-generate the scan replay once (deterministic), then reuse in both modes
    frames, labels = [], []
    print(f"generating {args.frames} synthetic scans ({args.n:,} pts each)...")
    for k in range(args.frames):
        pts, cls = _synthetic_scan(ego_x=k * 0.8, n=args.n)
        frames.append(pts)
        labels.append(cls)

    print(f"\n=== grid layout (config/grid.yaml) ===")
    g = LogPolarGrid(GridConfig.from_dict(base))
    print(f"rings={g.n_rings} ({g.phase1_rings} phase-1 + {g.phase2_rings} phase-2)  "
          f"sectors={g.n_theta}  cells={g.n_cells:,}  ring@100m={g.ring_widths[-1]:.3f}m")

    results = {}
    for name, tweak in (("precise", {"decay": {"enabled": False}}), ("decay", {"decay": {"enabled": True}})):
        # shallow-merge the tweak into the base config dict (grid/trav untouched)
        cfg_dict = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
        for sec, vals in tweak.items():
            cfg_dict.setdefault(sec, {}).update(vals)
        cfg = GridConfig.from_dict(cfg_dict)
        print(f"\n=== mode: {name} ===")
        res = _run_mode(cfg, frames, labels, args.frames)
        results[name] = res
        print(f"  wall: {res['wall_ms']:.2f} ms/frame  ({1000 / max(res['wall_ms'], 1e-9):.1f} fps)")
        print(f"  rendered cells: {res['rendered']:,} / {res['n_cells']:,}  reduce backend: {res['jit']}")
        print("  stage breakdown:", ", ".join(f"{k}={v:.2f}" for k, v in res["stages"].items()))
        mr = g.memory_report()
        print(f"  grid: {mr['grid_kb'] / 1024:.1f} MB  "
              f"uniform-equivalent: {mr['uniform_mb']:.1f} MB  "
              f"compression: {mr['compression_ratio']:.0f}x")

    spend = 1000.0 / results["decay"]["wall_ms"] if results["decay"]["wall_ms"] else 0
    print(f"\n=== summary ===")
    print(f"  precise {results['precise']['wall_ms']:.2f} ms/frame ({1000 / results['precise']['wall_ms']:.1f} fps)")
    print(f"  decay   {results['decay']['wall_ms']:.2f} ms/frame ({spend:.1f} fps)")


if __name__ == "__main__":
    main()