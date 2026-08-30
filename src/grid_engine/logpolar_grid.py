import math
from pathlib import Path

import numpy as np
import yaml

UNKNOWN = 255


def load_grid_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "grid.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


class LogPolarGrid:
    """Variable-resolution 2.5D grid engine (PROJECT_SPEC §9).

    Cells are (ring, sector) pairs with geometrically growing ring width:
      ring i covers [r_min·α^i, r_min·α^(i+1)).
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = load_grid_config()
        g = cfg["grid"]
        self.r_min = g["r_min"]
        self.r_max = g["r_max"]
        self.dr_0 = g["dr_0"]
        self.alpha = g["alpha"]
        self.n_theta = int(g["n_theta"])
        self.z_min = g["z_min"]
        self.z_max = g["z_max"]
        self.n_classes = int(g["n_classes"])

        d = cfg["decay"]
        self.tau_free = d["tau_free"]
        self.frame_hz = d["frame_hz"]
        self.occ_gain = d["occupancy_gain"]
        self.occ_threshold = d["occ_threshold"]
        self.dyn_threshold = d["dyn_threshold"]
        self.dyn_change_boost = d["dyn_change_boost"]
        self.dyn_ring_k = int(d.get("dyn_ring_k", 5))

        m = cfg["memory"]
        self.uniform_side = m["uniform_cell_guess"]
        self.uniform_dx = m["uniform_cell_size"]

        # derived ring geometry
        cumulative = 0.0
        i = 0
        while cumulative < (self.r_max - self.r_min):
            cumulative += self.dr_0 * (self.alpha ** i)
            i += 1
        self.n_rings = i
        self.n_cells = self.n_rings * self.n_theta
        self.ring_widths = self.dr_0 * (self.alpha ** np.arange(self.n_rings))
        self._ring_edges = np.concatenate([[self.r_min], self.r_min + np.cumsum(self.ring_widths)])
        # ring i covers [_ring_edges[i], _ring_edges[i+1]); width dr_0 * alpha^i (grows 5cm -> ~50cm)

        # state (struct-of-arrays)
        self.z_mean = np.zeros(self.n_cells, dtype=np.float32)
        self._z_min_state = np.full(self.n_cells, np.inf, dtype=np.float32)
        self._z_max_state = np.full(self.n_cells, -np.inf, dtype=np.float32)
        self.cls_hist = np.zeros((self.n_cells, self.n_classes), dtype=np.float32)
        self.dominant_class = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        self.occupancy = np.zeros(self.n_cells, dtype=np.float32)
        self.dynamic_score = np.zeros(self.n_cells, dtype=np.float32)
        self.last_update = np.zeros(self.n_cells, dtype=np.int64)
        self.frame = 0

        # dynamic detection ring buffer: K frames of occupancy state
        # (replaces single-flip boost with K-frame lookback)
        self._occ_ringbuf = np.zeros((self.dyn_ring_k, self.n_cells), dtype=bool)
        self._ring_ptr = 0

        # traversability config
        trav_cfg = cfg.get("traversability", {})
        self.trav_enabled = trav_cfg.get("enabled", True)
        self.trav_weights = trav_cfg.get("weights", [0.3, 0.3, 0.3, 0.1])
        self.trav_class_scores = trav_cfg.get("class_scores", [1.0, 0.6, 0.2, 0.1])
        self.trav_z_diff_thresh = trav_cfg.get("z_diff_thresh", 0.12)
        self.trav_slope_thresh = trav_cfg.get("slope_thresh", 0.15)
        self.traversability = np.zeros(self.n_cells, dtype=np.float32)

        # last-sent display values (change detection for incremental deltas)
        self._sent_zmax = np.zeros(self.n_cells, dtype=np.float32)
        self._sent_cls = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        # road elevation in the ego frame: heights are reported relative to it
        # so the grid and the point cloud share a common "ground = 0" reference
        self.ground_z = 0.0

        # per-frame scratch
        self._z_min_f = np.full(self.n_cells, np.inf, dtype=np.float32)
        self._z_max_f = np.full(self.n_cells, -np.inf, dtype=np.float32)
        self._z_sum_f = np.zeros(self.n_cells, dtype=np.float64)
        self._count_f = np.zeros(self.n_cells, dtype=np.int64)
        self._cls_count_f = np.zeros((self.n_cells, self.n_classes), dtype=np.int32)
        self._prev_rendered = np.zeros(self.n_cells, dtype=bool)
        self._touched = np.zeros(self.n_cells, dtype=bool)

    # --- geometry ---
    def ring_index(self, r: np.ndarray) -> np.ndarray:
        """Ring index via searchsorted over the cumulative-sum ring boundaries."""
        r = np.clip(r, self.r_min, self._ring_edges[-1] - 1e-9)
        return np.searchsorted(self._ring_edges, r, side="right") - 1

    def sector_index(self, theta: np.ndarray) -> np.ndarray:
        return (np.floor((theta + np.pi) / (2.0 * np.pi / self.n_theta)).astype(np.int64)) % self.n_theta

    def cell_ids_from_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        i = self.ring_index(r)
        j = self.sector_index(theta)
        valid = (r >= self.r_min) & (r <= self.r_max)
        i = np.clip(i, 0, self.n_rings - 1)
        return i * self.n_theta + j, valid

    # --- frame update ---
    def update(self, points: np.ndarray, per_class: np.ndarray) -> int:
        """Points: [N, >=3] float32 (x,y,z,...). per_class: [N] uint8 in [0, n_classes)."""
        n = points.shape[0]
        r = np.hypot(points[:, 0], points[:, 1])
        theta = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2]

        # per-frame road reference: low z within the near field is dominated by
        # the road surface, so its 20th percentile tracks the local ground level
        near = (r > 1.5) & (r < 15.0) & (z > -8.0) & (z < 4.0)
        zn = z[near]
        if zn.size >= 128:
            self.ground_z = float(np.percentile(zn, 20))

        keep = (r >= self.r_min) & (r <= self.r_max) & (z >= self.z_min) & (z <= self.z_max)
        if not np.any(keep):
            return 0

        rk, tk, zk, pck = r[keep], theta[keep], z[keep], per_class[keep]
        i = np.clip(self.ring_index(rk), 0, self.n_rings - 1)
        j = self.sector_index(tk)
        cells = i * self.n_theta + j

        # decay all cells a step
        decay = math.exp(-(1.0 / self.frame_hz) / self.tau_free)
        self.occupancy *= decay
        self.dynamic_score *= decay

        # grouped scatter-reduce: one stable sort + vectorized min/max/sum,
        # replacing 5 unbuffered np.*.at calls (~10x faster, SIMD-friendly)
        n = cells.shape[0]
        order = np.argsort(cells, kind="stable")
        csort = cells[order]
        zsort = zk[order]
        neq = np.concatenate(([True], csort[1:] != csort[:-1]))
        firsts = np.flatnonzero(neq)
        uniq = csort[firsts]
        ends = np.concatenate((firsts[1:], [n]))
        seg_count = ends - firsts

        zmin_c = np.minimum.reduceat(zsort, firsts)
        zmax_c = np.maximum.reduceat(zsort, firsts)
        zsum_c = np.add.reduceat(zsort.astype(np.float64), firsts)

        cls_key = csort.astype(np.int64) * self.n_classes + pck[order].astype(np.int64)
        cls_counts = np.bincount(cls_key, minlength=self.n_cells * self.n_classes).reshape(self.n_cells, self.n_classes)

        # stash per-frame scratch (scatter only the hit cells)
        self._z_min_f[uniq] = zmin_c.astype(np.float32)
        self._z_max_f[uniq] = zmax_c.astype(np.float32)
        self._z_sum_f[uniq] = zsum_c
        self._count_f[uniq] = seg_count
        self._cls_count_f[uniq] = cls_counts[uniq]
        self._touched[uniq] = True

        # occupancy up on hit cells
        occ_hit = self.occupancy[uniq]
        self.occupancy[uniq] = occ_hit + (1.0 - occ_hit) * self.occ_gain

        # class histogram EMA + dominant class
        frac = cls_counts[uniq].astype(np.float32) / seg_count[:, None].astype(np.float32)
        self.cls_hist[uniq] = self.cls_hist[uniq] * 0.6 + frac * 0.4
        self.dominant_class[uniq] = np.argmax(self.cls_hist[uniq], axis=1).astype(np.uint8)

        # z-stat update (persistent)
        self._z_min_state[uniq] = np.minimum(self._z_min_state[uniq], zmin_c.astype(np.float32))
        self._z_max_state[uniq] = np.maximum(self._z_max_state[uniq], zmax_c.astype(np.float32))
        new_mean = zsum_c / seg_count
        self.z_mean[uniq] = self.z_mean[uniq] * 0.7 + new_mean.astype(np.float32) * 0.3

        # dynamics: K-frame ring buffer occupancy change detection
        # (replaces single-flip with lookback over dyn_ring_k frames)
        now_occ = self.occupancy > self.occ_threshold
        k = self.dyn_ring_k
        old_occ = self._occ_ringbuf[self._ring_ptr % k]
        flipped = now_occ[uniq] & ~old_occ[uniq]
        self.dynamic_score[uniq] += self.dyn_change_boost * flipped.astype(np.float32)
        np.clip(self.dynamic_score, 0.0, 1.0, out=self.dynamic_score)
        self._occ_ringbuf[self._ring_ptr % k] = now_occ
        self._ring_ptr += 1

        # bookkeeping
        self.last_update[uniq] = self.frame
        self.frame += 1

        # traversability: per-cell drivability score (0-1)
        if self.trav_enabled:
            from src.grid_engine.traversability import compute_traversability
            self.traversability = compute_traversability(
                self._z_min_state, self._z_max_state, self.z_mean,
                self.dominant_class, self.occupancy,
                self.n_rings, self.n_theta, self.ring_widths,
                weights=self.trav_weights,
                class_scores=self.trav_class_scores,
                z_diff_thresh=self.trav_z_diff_thresh,
                slope_thresh=self.trav_slope_thresh,
            )

        n_hit = len(uniq)
        self._clear_scratch()
        return n_hit

    def _clear_scratch(self):
        self._z_min_f[:] = np.inf
        self._z_max_f[:] = -np.inf
        self._z_sum_f[:] = 0.0
        self._count_f[:] = 0
        self._cls_count_f[:] = 0
        self._touched[:] = False

    # --- extraction ---
    def rendered(self) -> np.ndarray:
        return self.occupancy > self.occ_threshold

    def _rows(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Client rows (k,7) float32 [i, j, z_mean, z_max, occ, dyn, trav] + cls u8,
        with z stats rebased onto the ego ground so height reads "above road"."""
        i = idx // self.n_theta
        j = idx % self.n_theta
        g = self.ground_z
        rows = np.stack([
            i.astype(np.float32), j.astype(np.float32),
            self.z_mean[idx] - g, self._z_max_state[idx] - g,
            self.occupancy[idx], self.dynamic_score[idx],
            self.traversability[idx],
        ], axis=1)
        return rows, self.dominant_class[idx]

    def _mark_sent(self, idx: np.ndarray):
        if idx.size == 0:
            return
        self._sent_zmax[idx] = self._z_max_state[idx]
        self._sent_cls[idx] = self.dominant_class[idx]

    def snapshot(self) -> dict:
        """Full snapshot of rendered cells.

        Returns {frame, rows: (k,6) float32 [i, j, z_mean, z_max, occ, dyn],
        cls: (k,) uint8} — raw arrays, packed to binary by the protocol layer.
        Marks every cell as sent so subsequent deltas are purely incremental.
        """
        mask = self.rendered()
        idx = np.nonzero(mask)[0]
        if idx.size:
            rows, cls = self._rows(idx)
        else:
            rows = np.zeros((0, 6), dtype=np.float32)
            cls = np.zeros(0, dtype=np.uint8)
        self._prev_rendered = mask
        self._mark_sent(idx)
        clear = np.nonzero(~mask)[0]
        self._sent_zmax[clear] = 0.0
        self._sent_cls[clear] = UNKNOWN
        return {"frame": self.frame, "rows": rows, "cls": cls}

    def delta(self) -> dict:
        """Changed cells since last call.

        Returns {frame, rows: (k,6) float32 [i, j, z_mean, z_max, occ, dyn],
        cls: (k,) uint8, freed: (m,2) float32 [i, j]} — raw arrays.

        Includes: cells that toggled rendered state (added/freed) AND cells
        that stayed rendered but whose displayed height (z_max, >= 5 cm) or
        dominant class changed. Without the latter, already-rendered cells
        would be frozen at their first-seen height until the next full
        snapshot. Occupancy alpha is deliberately excluded from change
        detection (it climbs/fades in a way that would re-send every rendered
        cell each frame); alpha is refreshed when a cell is (re)added and at
        every snapshot.
        """
        mask = self.rendered()
        added = mask & ~self._prev_rendered
        freed = self._prev_rendered & ~mask
        changed = (mask & ~added) & (
            (np.abs(self._z_max_state - self._sent_zmax) >= 0.05)
            | (self.dominant_class != self._sent_cls)
        )
        idx = np.nonzero(added | changed)[0]
        if idx.size:
            rows, cls = self._rows(idx)
        else:
            rows = np.zeros((0, 6), dtype=np.float32)
            cls = np.zeros(0, dtype=np.uint8)
        self._prev_rendered = mask
        self._mark_sent(idx)

        idx_free = np.nonzero(freed)[0]
        if idx_free.size:
            freed_rows = np.stack([
                (idx_free // self.n_theta).astype(np.float32),
                (idx_free % self.n_theta).astype(np.float32),
            ], axis=1)
        else:
            freed_rows = np.zeros((0, 2), dtype=np.float32)
        self._sent_zmax[idx_free] = 0.0
        self._sent_cls[idx_free] = UNKNOWN
        return {
            "frame": self.frame,
            "rows": rows,
            "cls": cls,
            "freed": freed_rows,
        }

    def reset(self):
        self.__init__()

    # --- memory accountant (PROJECT_SPEC §9.2 / acceptance criterion) ---
    @property
    def bytes_per_cell(self) -> float:
        return 3 * 4 + self.n_classes * 4 + 1 + 4 + 4 + 8

    def memory_report(self) -> dict:
        grid_bytes = self.n_cells * self.bytes_per_cell
        uniform_cells = (self.uniform_side / self.uniform_dx) ** 2
        uniform_bytes = uniform_cells * self.bytes_per_cell
        return {
            "n_rings": self.n_rings,
            "n_theta": self.n_theta,
            "n_cells": self.n_cells,
            "ring_width_near_m": float(self.ring_widths[0]),
            "ring_width_far_m": float(self.ring_widths[-1]),
            "grid_bytes": int(grid_bytes),
            "grid_kb": round(grid_bytes / 1024.0, 1),
            "uniform_cells": int(uniform_cells),
            "uniform_mb": round(uniform_bytes / 1e6, 1),
            "compression_ratio": round(uniform_cells / self.n_cells, 1),
        }