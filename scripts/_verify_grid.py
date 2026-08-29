import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.models.predict import Segmenter
from src.data.label_mapping import bin_5_to_4

cfg = load_grid_config()
grid = LogPolarGrid(cfg)
print("=== memory report ===")
print(grid.memory_report())

# synthetic: ring of points
print("\n=== synthetic ring test ===")
for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    for rad in [1.0, 5.0, 20.0, 60.0, 95.0]:
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        pts = np.array([[x, y, 0.1, 0.5]] * 4, dtype=np.float32)
        cls = np.array([0, 1, 2, 3], dtype=np.uint8)  # vary class a bit
        grid.update(pts, cls)
snap = grid.snapshot()
print(f"rendered cells: {len(snap['cells'])}, frame={snap['frame']}")
# check ring mapping correctness: point at 1.0 m should be ring ~0
print("ring_index(0.999)==", grid.ring_index(np.array([0.999])).item(), "(expect ~0)")
print("ring_index(100.0)==", grid.ring_index(np.array([100.0])).item(), f"(expect <= {grid.n_rings - 1})")

# real segmented scan streaming simulation
print("\n=== real seq-08 streaming (60 frames, speed fast) ===")
seg = Segmenter("F:/sih/checkpoints/best_miou.pt")
bins = sorted(Path("F:/sih/dataset/sequences/08/velodyne").glob("*.bin"))
grid2 = LogPolarGrid(cfg)
t0 = time.time()
sizes = []
for k in range(60):
    pts = np.fromfile(str(bins[k]), dtype=np.float32).reshape(-1, 4)
    cls5, _ = seg.segment(pts)
    cls4 = bin_5_to_4(cls5)
    grid2.update(pts, cls4)
    d = grid2.delta()
    sizes.append(len(d["updated"]) + len(d["freed"]))
dt = time.time() - t0
print(f"60 frames in {dt:.1f}s ({60 / dt:.0f} fps)")
print(f"delta cells/frame: mean={np.mean(sizes):.0f} max={np.max(sizes)}")
mr = grid2.memory_report()
print("memory:", {k: mr[k] for k in ("n_cells", "grid_kb", "uniform_mb", "compression_ratio")})
snap2 = grid2.snapshot()
print(f"snapshot rendered cells: {len(snap2['cells'])}")
print("sample cell:", snap2["cells"][0][:7])