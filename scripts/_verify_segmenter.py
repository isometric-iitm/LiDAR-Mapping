import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict import Segmenter
from src.data.label_mapping import bin_5_to_4

seq_dir = Path("F:/sih/dataset/sequences/08/velodyne")
bins = sorted(seq_dir.glob("*.bin"))
print(f"found {len(bins)} scans")

seg = Segmenter("F:/sih/checkpoints/best_miou.pt")

for idx in [0, 100, 1000]:
    pts = np.fromfile(str(bins[idx]), dtype=np.float32).reshape(-1, 4)
    t0 = time.perf_counter()
    cls, timings = seg.segment(pts)
    dt = (time.perf_counter() - t0) * 1000
    vals, counts = np.unique(cls, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    binned = bin_5_to_4(cls)
    bvals, bcounts = np.unique(binned, return_counts=True)
    print(f"scan {idx}: N={len(pts)} total={dt:.1f}ms (proj={timings['project']:.1f} fwd={timings['forward']:.1f})")
    print(f"  5-class hist: {hist}")
    print(f"  4-class hist: {dict(zip(map(int, bvals), map(int, bcounts)))}")