import math
import time
from pathlib import Path

import numpy as np
import yaml

from src.grid_engine.traversability import compute_traversability

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
        self.decay_enabled = bool(d.get("enabled", False))
        self.tau_free = d["tau_free"]
        self.frame_hz = d["frame_hz"]
        self.occ_gain = d["occupancy_gain"]
        self.occ_threshold = d["occ_threshold"]

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
        # ring i covers [_ring_edges[i], _ring_edges[i+1]); width dr_0 * alpha^i (grows 5cm -> ~50cm at 100 m)

        # state (struct-of-arrays) — AP0 precise: no temporal smoothing
        self.z_mean = np.zeros(self.n_cells, dtype=np.float32)
        self._z_min_state = np.full(self.n_cells, np.inf, dtype=np.float32)
        self._z_max_state = np.full(self.n_cells, -np.inf, dtype=np.float32)
        self.cls_hist = np.zeros((self.n_cells, self.n_classes), dtype=np.float32)
        self.dominant_class = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        self.occupancy = np.zeros(self.n_cells, dtype=np.float32)
        # kept for wire compat (row 6/7) but always 0 in precise mode — no DK
        self.dynamic_score = np.zeros(self.n_cells, dtype=np.float32)
        self.last_update = np.zeros(self.n_cells, dtype=np.int64)
        self.frame = 0

        # traversability config
        trav_cfg = cfg.get("traversability", {})
        self.trav_enabled = trav_cfg.get("enabled", True)
        self.trav_weights = trav_cfg.get("weights", [0.25, 0.25, 0.35, 0.15])
        self.trav_class_scores = trav_cfg.get("class_scores", [1.0, 0.6, 0.2, 0.1])
        self.trav_z_diff_thresh = trav_cfg.get("z_diff_thresh", 0.5)
        self.trav_slope_thresh = trav_cfg.get("slope_thresh", 0.4)
        self.traversability = np.zeros(self.n_cells, dtype=np.float32)
        self._trav_dirty = False

        # last-sent display values (change detection for incremental deltas)
        self._sent_zmax = np.zeros(self.n_cells, dtype=np.float32)
        self._sent_cls = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        # road elevation in the ego frame: heights are reported relative to it
        # so the grid and the point cloud share a common "ground = 0" reference
        self.ground_z = 0.0

        # per-frame scratch (no DK — per-frame sensor truth)
        self._z_min_f = np.full(self.n_cells, np.inf, dtype=np.float32)
        self._z_max_f = np.full(self.n_cells, -np.inf, dtype=np.float32)
        self._z_sum_f = np.zeros(self.n_cells, dtype=np.float64)
        self._count_f = np.zeros(self.n_cells, dtype=np.int64)
        self._cls_count_f = np.zeros((self.n_cells, self.n_classes), dtype=np.int32)
        self._prev_rendered = np.zeros(self.n_cells, dtype=bool)
        self._touched = np.zeros(self.n_cells, dtype=bool)
        self._ever_seen = np.zeros(self.n_cells, dtype=bool)

        # per-frame timing breakdown (reset each update() call)
        self._timings: dict[str, float] = {}
        self._last_n_hit = 0
        self._last_n_rendered = 0
        self._last_n_freed = 0
        self._last_n_changed = 0
        self._last_payload_bytes = 0

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
        t_start = time.perf_counter()
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
            self._timings = {"total_ms": (time.perf_counter() - t_start) * 1000}
            return 0

        rk, tk, zk, pck = r[keep], theta[keep], z[keep], per_class[keep]
        i = np.clip(self.ring_index(rk), 0, self.n_rings - 1)
        j = self.sector_index(tk)
        cells = i * self.n_theta + j
        t_polar = time.perf_counter()

        # occupancy handling: precise mode (decay disabled) = per-frame sensor view,
        # no temporal persistence; otherwise exponential decay.
        pre_rendered = self.occupancy > self.occ_threshold
        if self.decay_enabled:
            decay = math.exp(-(1.0 / self.frame_hz) / self.tau_free)
            self.occupancy *= decay
        t_occ = time.perf_counter()

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
        t_reduce = time.perf_counter()

        # class counts. Decay mode needs the full-grid bincount (for EMA over
        # all cells); precise mode only needs per-hit majority, computed from
        # the already-sorted (by cell) class labels via per-segment bincounts.
        cls_counts = None
        if self.decay_enabled:
            cls_key = csort.astype(np.int64) * self.n_classes + pck[order].astype(np.int64)
            cls_counts = np.bincount(cls_key, minlength=self.n_cells * self.n_classes).reshape(self.n_cells, self.n_classes)
            # stash per-frame scratch (scatter only the hit cells)
            self._z_min_f[uniq] = zmin_c.astype(np.float32)
            self._z_max_f[uniq] = zmax_c.astype(np.float32)
            self._z_sum_f[uniq] = zsum_c
            self._count_f[uniq] = seg_count
            self._cls_count_f[uniq] = cls_counts[uniq]
            self._touched[uniq] = True
        else:
            # precise: per-hit majority vote (rows aligned to `uniq`).
            # Fully vectorized: map each point to its unique-cell group index via
            # searchsorted, then one bincount gives per-(cell,class) counts.
            # Equivalent to the per-segment majority but ~10-20x faster (C-level).
            seg_labels = pck[order]
            local = np.searchsorted(uniq, csort)
            cls_counts = np.bincount(
                local * self.n_classes + seg_labels.astype(np.int64),
                minlength=uniq.shape[0] * self.n_classes,
            ).reshape(uniq.shape[0], self.n_classes)
        t_cls = time.perf_counter()

        if self.decay_enabled:
            # decaying mode: occupancy EMA, class EMA, z running min/max/EMA
            occ_hit = self.occupancy[uniq]
            self.occupancy[uniq] = occ_hit + (1.0 - occ_hit) * self.occ_gain
            frac = cls_counts[uniq].astype(np.float32) / seg_count[:, None].astype(np.float32)
            self.cls_hist[uniq] = self.cls_hist[uniq] * 0.6 + frac * 0.4
            self.dominant_class[uniq] = np.argmax(self.cls_hist[uniq], axis=1).astype(np.uint8)
            new_mean = zsum_c / seg_count
            first_seen = ~self._ever_seen[uniq]
            self.z_mean[uniq] = np.where(
                first_seen,
                new_mean.astype(np.float32),
                self.z_mean[uniq] * 0.7 + new_mean.astype(np.float32) * 0.3,
            )
            fs_idx = np.flatnonzero(first_seen)
            if fs_idx.size:
                fsu = uniq[fs_idx]
                self._z_min_state[fsu] = zmin_c[fs_idx].astype(np.float32)
                self._z_max_state[fsu] = zmax_c[fs_idx].astype(np.float32)
            rs_idx = np.flatnonzero(~first_seen)
            if rs_idx.size:
                rsu = uniq[rs_idx]
                self._z_min_state[rsu] = np.minimum(self._z_min_state[rsu], zmin_c[rs_idx].astype(np.float32))
                self._z_max_state[rsu] = np.maximum(self._z_max_state[rsu], zmax_c[rs_idx].astype(np.float32))
            self._ever_seen[uniq] = True
        else:
            # precise mode: strictly per-frame — no history, no window, no DK.
            # Occupancy is binary this scan only; anything not hit this frame is free.
            # This eliminates all temporal ghosts / stale height.
            self.occupancy[uniq] = float(self.occ_gain)
            # instant free: any previously rendered cell not hit this scan goes free now
            is_hit = np.zeros(self.n_cells, dtype=bool)
            is_hit[uniq] = True
            to_free = pre_rendered & ~is_hit
            if np.any(to_free):
                self.occupancy[to_free] = 0.0
            # class: per-frame majority (no history)
            self.dominant_class[uniq] = np.argmax(
                cls_counts.astype(np.float32) / seg_count[:, None].astype(np.float32), axis=1
            ).astype(np.uint8)
            self.cls_hist[uniq] = 0
            # one-hot the current dominant for debug consistency
            self.cls_hist[uniq, self.dominant_class[uniq]] = 1.0
            # z: per-frame directly (no running min/max)
            new_mean = zsum_c / seg_count
            self.z_mean[uniq] = new_mean.astype(np.float32)
            self._z_min_state[uniq] = zmin_c.astype(np.float32)
            self._z_max_state[uniq] = zmax_c.astype(np.float32)
            self._ever_seen[uniq] = True
            # trav: per-hit compute using absolute height above ground + class.
            # Intra-cell z_diff is ~0 for single-point precise cells, so use height.
            height_u = zmax_c.astype(np.float32) - float(self.ground_z)
            # height 0-0.3m => drivable flat, 0.3-1.0m => curb/caution, >1.0m => blocked
            z_score_u = np.clip(1.0 - np.maximum(height_u, 0.0) / (self.trav_z_diff_thresh * 2.5), 0, 1)
            cls_u = self.dominant_class[uniq]
            lut = np.zeros(256, dtype=np.float32)
            for ci, sc in enumerate(self.trav_class_scores):
                if ci < 256:
                    lut[ci] = sc
            cls_score_u = lut[cls_u]
            # estimate slope from per-cell height vs neighbor ground (skip full O(n) slope);
            # use height itself as proxy: tall => steep => slope_score low
            slope_score_u = np.clip(1.0 - np.maximum(height_u - 0.15, 0.0) / self.trav_slope_thresh, 0, 1)
            occ_score_u = np.clip(float(self.occ_gain), 0, 1)
            w_z, w_s, w_c, w_o = self.trav_weights
            trav_u = (w_z * z_score_u + w_s * slope_score_u + w_c * cls_score_u + w_o * occ_score_u) / (w_z + w_s + w_c + w_o)
            self.traversability[uniq] = trav_u.astype(np.float32)
            # cells freed this frame must have their ever_seen cleared so next
            # hit is treated as first-seen (already handled by occupancy free,
            # but also need to keep freed clearing block below effective)

        # freed by decay — stale world content has scrolled past this polar cell;
        # clear height / class so the next hit starts fresh (prevents radial
        # ghost terrain and empty holes where dynamic left but stale _z_max kept
        # height inflated or cls_hist kept the blue tint).
        post_rendered = self.occupancy > self.occ_threshold
        newly_freed = pre_rendered & ~post_rendered
        if np.any(newly_freed):
            self._z_min_state[newly_freed] = np.inf
            self._z_max_state[newly_freed] = -np.inf
            self.z_mean[newly_freed] = 0
            self._ever_seen[newly_freed] = False
            self.cls_hist[newly_freed] = 0
            self.dominant_class[newly_freed] = UNKNOWN
            self.traversability[newly_freed] = 0

        # bookkeeping
        self.last_update[uniq] = self.frame
        self.frame += 1
        # In precise (sensor-direct) mode traversability is not computed every frame
        # to avoid the O(n_cells) full-grid cost that was dropping 30fps -> 10fps.
        # Trav view will still read per-cell trav as 0 (no shading) until decay mode.
        if self.decay_enabled:
            self._trav_dirty = True

        # traversability: per-cell drivability score (0-1) — only in decay mode
        if self.decay_enabled and self.trav_enabled and self._trav_dirty:
            self.traversability = compute_traversability(
                self._z_min_state, self._z_max_state, self.z_mean,
                self.dominant_class, self.occupancy,
                self.n_rings, self.n_theta, self.ring_widths,
                weights=self.trav_weights,
                class_scores=self.trav_class_scores,
                z_diff_thresh=self.trav_z_diff_thresh,
                slope_thresh=self.trav_slope_thresh,
            )
            self._trav_dirty = False
        t_state = time.perf_counter()

        n_hit = len(uniq)
        self._last_n_hit = n_hit
        # Skip _clear_scratch in precise mode: scratch arrays are only used in decay mode
        if self.decay_enabled:
            self._clear_scratch()
        t_end = time.perf_counter()

        self._timings = {
            "polar_ms": (t_polar - t_start) * 1000,
            "occ_ms": (t_occ - t_polar) * 1000,
            "reduce_ms": (t_reduce - t_occ) * 1000,
            "cls_ms": (t_cls - t_reduce) * 1000,
            "state_ms": (t_state - t_cls) * 1000,
            "total_ms": (t_end - t_start) * 1000,
            "n_hit": n_hit,
            "n_input": int(points.shape[0]),
            "n_rendered": int(np.count_nonzero(self.occupancy > self.occ_threshold)),
        }
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

        Returns {frame, rows: (k,7) float32 [i, j, z_mean, z_max, occ, dyn, trav],
        cls: (k,) uint8} — raw arrays, packed to binary by the protocol layer.
        Marks every cell as sent so subsequent deltas are purely incremental.
        """
        t0 = time.perf_counter()
        mask = self.rendered()
        t_mask = time.perf_counter()
        idx = np.nonzero(mask)[0]
        if idx.size:
            rows, cls = self._rows(idx)
        else:
            rows = np.zeros((0, 7), dtype=np.float32)
            cls = np.zeros(0, dtype=np.uint8)
        t_rows = time.perf_counter()
        self._prev_rendered = mask
        self._mark_sent(idx)
        clear = np.nonzero(~mask)[0]
        self._sent_zmax[clear] = 0.0
        self._sent_cls[clear] = UNKNOWN
        self._last_n_rendered = int(idx.size)
        self._last_n_freed = 0
        self._last_n_changed = int(idx.size)
        t_end = time.perf_counter()
        self._timings["snapshot_ms"] = (t_end - t0) * 1000
        self._timings["snapshot_mask_ms"] = (t_mask - t0) * 1000
        self._timings["snapshot_rows_ms"] = (t_rows - t_mask) * 1000
        return {"frame": self.frame, "rows": rows, "cls": cls}

    def delta(self) -> dict:
        """Changed cells since last call.

        Returns {frame, rows: (k,7) float32 [i, j, z_mean, z_max, occ, dyn, trav],
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
        t0 = time.perf_counter()
        mask = self.rendered()
        added = mask & ~self._prev_rendered
        freed = self._prev_rendered & ~mask
        changed = (mask & ~added) & (
            (np.abs(self._z_max_state - self._sent_zmax) >= 0.05)
            | (self.dominant_class != self._sent_cls)
        )
        t_detect = time.perf_counter()
        idx = np.nonzero(added | changed)[0]
        if idx.size:
            rows, cls = self._rows(idx)
        else:
            rows = np.zeros((0, 7), dtype=np.float32)
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
        t_end = time.perf_counter()
        self._last_n_rendered = int(np.count_nonzero(mask))
        self._last_n_freed = int(idx_free.size)
        self._last_n_changed = int(idx.size)
        self._timings["delta_ms"] = (t_end - t0) * 1000
        self._timings["delta_detect_ms"] = (t_detect - t0) * 1000
        return {
            "frame": self.frame,
            "rows": rows,
            "cls": cls,
            "freed": freed_rows,
        }

    def reset(self):
        self.z_mean[:] = 0
        self._z_min_state[:] = np.inf
        self._z_max_state[:] = -np.inf
        self.cls_hist[:] = 0
        self.dominant_class[:] = UNKNOWN
        self.occupancy[:] = 0
        self.dynamic_score[:] = 0
        self.last_update[:] = 0
        self.frame = 0
        self._ever_seen[:] = False
        self.traversability[:] = 0
        self._trav_dirty = False
        self._sent_zmax[:] = 0
        self._sent_cls[:] = UNKNOWN
        self.ground_z = 0.0
        self._prev_rendered[:] = False
        self._clear_scratch()

    # --- memory accountant (PROJECT_SPEC §9.2 / acceptance criterion) ---
    @property
    def bytes_per_cell(self) -> float:
        return 3 * 4 + self.n_classes * 4 + 1 + 4 + 4 + 4 + 8

    def memory_report(self) -> dict:
        rendered = int(np.count_nonzero(self.occupancy > self.occ_threshold))
        grid_bytes = self.n_cells * self.bytes_per_cell
        rendered_bytes = rendered * self.bytes_per_cell
        uniform_cells = (self.uniform_side / self.uniform_dx) ** 2
        uniform_bytes = uniform_cells * self.bytes_per_cell
        capacity_ratio = round(uniform_cells / self.n_cells, 1)
        return {
            "n_rings": self.n_rings,
            "n_theta": self.n_theta,
            "n_cells": self.n_cells,
            "rendered_cells": rendered,
            "ring_width_near_m": float(self.ring_widths[0]),
            "ring_width_far_m": float(self.ring_widths[-1]),
            "grid_bytes": int(grid_bytes),
            "grid_kb": round(grid_bytes / 1024.0, 1),
            "rendered_bytes": int(rendered_bytes),
            "rendered_kb": round(rendered_bytes / 1024.0, 1),
            "uniform_cells": int(uniform_cells),
            "uniform_mb": round(uniform_bytes / 1e6, 1),
            "compression_ratio": round(uniform_cells / max(1, rendered), 1) if rendered else capacity_ratio,
            "capacity_compression": capacity_ratio,
        }