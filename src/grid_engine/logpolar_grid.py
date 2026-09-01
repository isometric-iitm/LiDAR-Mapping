import math
import time
from pathlib import Path

import numpy as np
import yaml

from src.common.config import GridConfig, as_grid_config
from src.grid_engine import jit_reduce

UNKNOWN = 255

# Optional numba acceleration for the per-cell scatter-reduce in update().
# Defaults on; any runtime failure flips it off and the pure-numpy path is
# used for the rest of the process, so a compile/link hiccup never breaks a run.
_jit_enabled = jit_reduce._NUMBA_OK


def _set_jit_enabled(v: bool):
    global _jit_enabled
    _jit_enabled = bool(v)


def load_grid_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "grid.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_grid_cfg(path: str | Path | None = None) -> GridConfig:
    """Typed grid config (GridConfig) from config/grid.yaml."""
    return GridConfig.from_dict(load_grid_config(path))


class LogPolarGrid:
    """Variable-resolution 2.5D grid engine (PROJECT_SPEC §9).

    Two-phase ring geometry (PS: "5cm within 10m, 50cm up to 100m"):
      Phase 1 (r_min → r_transition): uniform dr_0 rings (alpha = 1.0).
      Phase 2 (r_transition → r_max): geometric growth dr_0 · α^i.
    Cells are (ring, sector) pairs; sectors are uniform across all rings.

    Semantics are strictly per-frame (precise mode, PROJECT_SPEC §9.2 AP0):
      - occupancy is binary this scan only (hit = occ_gain, else 0 → freed)
      - class is the per-frame majority vote of this scan's points
      - z stats (mean, max) are this scan's points rebased onto the road
      - traversability is computed per-hit from height + majority class
    There is no temporal decay / dynamic score / EMA of any kind: the grid is
    the live sensor view and stale content scrolls out the instant it is no
    longer hit.

    Delta extraction is *pure*: ``compute_delta``/``compute_snapshot`` read the
    grid without mutating any sent-tracking state; ``commit_delta``/
    ``commit_snapshot`` apply the sent-tracking mutation and must be called only
    after the outgoing frame was actually queued. A dropped frame therefore
    leaves state untouched and the next successful delta is recomputed from the
    last *committed* state; no pending/retransmit bookkeeping is needed.
    """

    def __init__(self, cfg: dict | GridConfig | None = None):
        if cfg is None:
            cfg = load_grid_cfg()
        else:
            cfg = as_grid_config(cfg)
        g = cfg.grid
        self.r_min = g.r_min
        self.r_max = g.r_max
        self.dr_0 = g.dr_0
        self.r_transition = g.r_transition
        self.alpha = g.alpha
        self.n_theta = g.n_theta
        self.z_min = g.z_min
        self.z_max = g.z_max
        self.n_classes = g.n_classes
        self.occ_gain = g.occupancy_gain
        self.occ_threshold = g.occ_threshold

        m = cfg.memory
        self.uniform_side = m.uniform_cell_guess
        self.uniform_dx = m.uniform_cell_size

        # derived ring geometry (two-phase)
        # Phase 1: uniform dr_0 rings from r_min to r_transition
        # Phase 2: geometric dr_0 * alpha^i from r_transition to r_max
        self.phase1_rings = round((self.r_transition - self.r_min) / self.dr_0)
        phase2_span = self.r_max - (self.r_min + self.phase1_rings * self.dr_0)
        if phase2_span > 0 and self.alpha > 1.0:
            # closed-form geometric series: sum(dr_0*alpha^i, i=0..n-1) = dr_0*(alpha^n - 1)/(alpha - 1)
            # solve for the integer ring count n that reaches phase2_span
            n2_exact = math.log(1.0 + (phase2_span * (self.alpha - 1.0) / self.dr_0)) / math.log(self.alpha)
            # round (not ceil) so the final ring stays close to the geometric width (~50 cm),
            # then the last ring is clamped to the exact remainder below
            self.phase2_rings = max(1, int(round(n2_exact)))
        else:
            self.phase2_rings = 0

        self.n_rings = self.phase1_rings + self.phase2_rings
        self.n_cells = self.n_rings * self.n_theta

        widths_p2 = None
        if self.phase2_rings > 0:
            widths_p2 = self.dr_0 * (self.alpha ** np.arange(self.phase2_rings))
            # exact cover: clamp the final ring to land precisely on r_max
            # (never near-zero; it stays close to the geometric width)
            covered = np.sum(widths_p2[:-1])
            widths_p2[-1] = phase2_span - covered
            self.ring_widths = np.concatenate([np.full(self.phase1_rings, self.dr_0), widths_p2])
        else:
            self.ring_widths = np.full(self.phase1_rings, self.dr_0)
        self._ring_edges = np.concatenate([[self.r_min], self.r_min + np.cumsum(self.ring_widths)])
        # Phase 1: ring i width = dr_0 (uniform 5 cm within r_transition)
        # Phase 2: ring i width = dr_0 * alpha^(i - phase1_rings) (grows to ~50 cm at r_max)

        # state (struct-of-arrays), strictly per-frame (no temporal state at all)
        self.z_mean = np.zeros(self.n_cells, dtype=np.float32)
        self._z_min_state = np.full(self.n_cells, np.inf, dtype=np.float32)
        self._z_max_state = np.full(self.n_cells, -np.inf, dtype=np.float32)
        self.cls_hist = np.zeros((self.n_cells, self.n_classes), dtype=np.float32)
        self.dominant_class = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        self.occupancy = np.zeros(self.n_cells, dtype=np.float32)
        self.last_update = np.zeros(self.n_cells, dtype=np.int64)
        self.frame = 0

        # traversability config (per-hit scoring in _apply_precise)
        tcfg = cfg.traversability
        self.trav_enabled = tcfg.enabled
        self.trav_weights = tcfg.weights
        self.trav_class_scores = tcfg.class_scores
        self.trav_z_diff_thresh = tcfg.z_diff_thresh
        self.trav_slope_thresh = tcfg.slope_thresh
        self.traversability = np.zeros(self.n_cells, dtype=np.float32)

        # last-sent display values (change detection for incremental deltas).
        # These are the ONLY fields mutated by commit_delta/commit_snapshot; a
        # dropped frame leaves them untouched so the next delta re-derives
        # everything relative to the last actually-sent state.
        self._prev_rendered = np.zeros(self.n_cells, dtype=bool)
        self._sent_zmax = np.zeros(self.n_cells, dtype=np.float32)
        self._sent_cls = np.full(self.n_cells, UNKNOWN, dtype=np.uint8)
        # road elevation in the ego frame: heights are reported relative to it
        # so the grid and the point cloud share a common "ground = 0" reference
        self.ground_z = 0.0

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
        """Points: [N, >=3] float32 (x,y,z,...). per_class: [N] uint8 in [0, n_classes).

        Strictly per-frame: occupancy is this scan only, anything not hit this
        frame is freed instantly, and all displayed values (height, majority
        class, traversability) reflect this scan alone.
        """
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

        # occupancy is this scan only: remember which cells were rendered before
        # this scan so anything no longer hit is freed below.
        pre_rendered = self.occupancy > self.occ_threshold
        t_occ = time.perf_counter()

        # grouped scatter-reduce: one stable sort + vectorized min/max/sum.
        # The per-cell reduce AND per-cell class tally are fused into a single
        # numba pass (jit_reduce.reduce_segments) when available; the pure-numpy
        # path below is the exact fallback, so results are bit-identical either way.
        order = np.argsort(cells, kind="stable")
        csort = cells[order]
        zsort = zk[order].astype(np.float32)
        seg_labels = pck[order]

        cls_counts_seg = None
        if _jit_enabled:
            try:
                uniq, zmin_c, zmax_c, zsum_c, seg_count, cls_counts_seg = jit_reduce.reduce_segments(
                    csort, zsort, seg_labels.astype(np.uint8), self.n_classes
                )
            except Exception:
                _set_jit_enabled(False)
        if cls_counts_seg is None:
            # numpy fallback reduce (bit-identical to the JIT kernel)
            neq = np.concatenate(([True], csort[1:] != csort[:-1]))
            firsts = np.flatnonzero(neq)
            uniq = csort[firsts]
            ends = np.concatenate((firsts[1:], [n]))
            seg_count = ends - firsts
            zmin_c = np.minimum.reduceat(zsort, firsts)
            zmax_c = np.maximum.reduceat(zsort, firsts)
            zsum_c = np.add.reduceat(zsort.astype(np.float64), firsts)
            local = np.searchsorted(uniq, csort)
            cls_counts_seg = np.bincount(
                local * self.n_classes + seg_labels.astype(np.int64),
                minlength=uniq.shape[0] * self.n_classes,
            ).reshape(uniq.shape[0], self.n_classes)
        t_reduce = time.perf_counter()

        # per-frame majority vote + instant free (no history at all)
        self._apply_precise(uniq, zmin_c, zmax_c, zsum_c, seg_count, cls_counts_seg, pre_rendered)
        t_cls = time.perf_counter()

        # clear height/class state for cells that left the rendered mask so the
        # next hit starts fresh (prevents radial ghost terrain / stale tint).
        post_rendered = self.occupancy > self.occ_threshold
        self._clear_freed(pre_rendered, post_rendered)

        # bookkeeping
        self.last_update[uniq] = self.frame
        self.frame += 1
        t_state = time.perf_counter()

        n_hit = uniq.shape[0]
        self._last_n_hit = n_hit
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

    def _apply_precise(self, uniq, zmin_c, zmax_c, zsum_c, seg_count, cls_counts, pre_rendered):
        """Strictly per-frame state update: no history, no window, no DK.
        Occupancy is binary this scan only; anything not hit this frame is free."""
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
        # trav: per-hit compute using absolute height above ground + class.
        # Intra-cell z_diff is ~0 for single-point precise cells, so use height.
        height_u = zmax_c.astype(np.float32) - float(self.ground_z)
        z_score_u = np.clip(1.0 - np.maximum(height_u, 0.0) / (self.trav_z_diff_thresh * 2.5), 0, 1)
        cls_u = self.dominant_class[uniq]
        lut = np.zeros(256, dtype=np.float32)
        for ci, sc in enumerate(self.trav_class_scores):
            if ci < 256:
                lut[ci] = sc
        cls_score_u = lut[cls_u]
        # estimate slope from per-cell height vs neighbor ground (skip full O(n) slope)
        slope_score_u = np.clip(1.0 - np.maximum(height_u - 0.15, 0.0) / self.trav_slope_thresh, 0, 1)
        occ_score_u = np.clip(float(self.occ_gain), 0, 1)
        if self.trav_enabled:
            w_z, w_s, w_c, w_o = self.trav_weights
            trav_u = (w_z * z_score_u + w_s * slope_score_u + w_c * cls_score_u + w_o * occ_score_u) / (w_z + w_s + w_c + w_o)
            self.traversability[uniq] = trav_u.astype(np.float32)

    def _clear_freed(self, pre_rendered: np.ndarray, post_rendered: np.ndarray):
        """Reset height / class state for cells that just dropped out of the
        rendered mask so the next hit starts fresh."""
        newly_freed = pre_rendered & ~post_rendered
        if not np.any(newly_freed):
            return
        self._z_min_state[newly_freed] = np.inf
        self._z_max_state[newly_freed] = -np.inf
        self.z_mean[newly_freed] = 0
        self.cls_hist[newly_freed] = 0
        self.dominant_class[newly_freed] = UNKNOWN
        self.traversability[newly_freed] = 0

    # --- extraction (pure: compute_* never mutates sent-tracking state) ---
    def rendered(self) -> np.ndarray:
        return self.occupancy > self.occ_threshold

    def _rows(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Client rows (k,6) float32 [i, j, z_mean, z_max, occ, trav] + cls u8,
        with z stats rebased onto the ego ground so height reads "above road"."""
        i = idx // self.n_theta
        j = idx % self.n_theta
        g = self.ground_z
        rows = np.stack([
            i.astype(np.float32), j.astype(np.float32),
            self.z_mean[idx] - g, self._z_max_state[idx] - g,
            self.occupancy[idx], self.traversability[idx],
        ], axis=1)
        return rows, self.dominant_class[idx]

    def _mark_sent(self, idx: np.ndarray):
        if idx.size == 0:
            return
        self._sent_zmax[idx] = self._z_max_state[idx]
        self._sent_cls[idx] = self.dominant_class[idx]

    def compute_snapshot(self) -> dict:
        """Full snapshot of currently-rendered cells (no mutation).

        Returns {frame, rows: (k,6) float32 [i, j, z_mean, z_max, occ, trav],
        cls: (k,) uint8, mask: (n_cells,) bool}; raw arrays, packed to binary
        by the protocol layer. Call ``commit_snapshot`` once the frame is sent.
        """
        t0 = time.perf_counter()
        mask = self.rendered()
        t_mask = time.perf_counter()
        idx = np.nonzero(mask)[0]
        if idx.size:
            rows, cls = self._rows(idx)
        else:
            rows = np.zeros((0, 6), dtype=np.float32)
            cls = np.zeros(0, dtype=np.uint8)
        t_rows = time.perf_counter()
        self._timings["snapshot_ms"] = (time.perf_counter() - t0) * 1000
        self._timings["snapshot_mask_ms"] = (t_mask - t0) * 1000
        self._timings["snapshot_rows_ms"] = (t_rows - t_mask) * 1000
        return {"frame": self.frame, "rows": rows, "cls": cls, "mask": mask}

    def commit_snapshot(self):
        """Mark every rendered cell (and only those) as sent. Call after the
        snapshot frame was actually queued, never before."""
        t0 = time.perf_counter()
        mask = self.rendered()
        self._prev_rendered = mask
        idx = np.nonzero(mask)[0]
        self._mark_sent(idx)
        clear = np.nonzero(~mask)[0]
        self._sent_zmax[clear] = 0.0
        self._sent_cls[clear] = UNKNOWN
        self._last_n_rendered = int(idx.size)
        self._last_n_freed = 0
        self._last_n_changed = int(idx.size)
        self._timings["commit_ms"] = (time.perf_counter() - t0) * 1000

    def compute_delta(self) -> dict:
        """Changed cells since the last committed delta/snapshot (no mutation).

        Returns {frame, rows: (k,6) float32 [i, j, z_mean, z_max, occ, trav],
        cls: (k,) uint8, freed: (m,2) float32 [i, j]} plus internal tracking
        arrays (``_idx``/``_idx_free``/``_mask``) consumed by ``commit_delta``.

        Includes: cells that toggled rendered state (added/freed) AND cells
        that stayed rendered but whose displayed height (z_max, >= 5 cm) or
        dominant class changed. Without the latter, already-rendered cells
        would be frozen at their first-seen height until the next full
        snapshot. Occupancy alpha is deliberately excluded from change
        detection (it is binary in precise mode; re-sending every rendered cell
        every frame would defeat incremental delivery); occupancy is refreshed
        when a cell is (re)added and at every snapshot.
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
            rows = np.zeros((0, 6), dtype=np.float32)
            cls = np.zeros(0, dtype=np.uint8)

        idx_free = np.nonzero(freed)[0]
        if idx_free.size:
            freed_rows = np.stack([
                (idx_free // self.n_theta).astype(np.float32),
                (idx_free % self.n_theta).astype(np.float32),
            ], axis=1)
        else:
            freed_rows = np.zeros((0, 2), dtype=np.float32)
        self._timings["delta_ms"] = (time.perf_counter() - t0) * 1000
        self._timings["delta_detect_ms"] = (t_detect - t0) * 1000
        return {
            "frame": self.frame,
            "rows": rows,
            "cls": cls,
            "freed": freed_rows,
            "_idx": idx,
            "_idx_free": idx_free,
            "_mask": mask,
        }

    def commit_delta(self, delta: dict):
        """Advance sent-tracking to match the given (already-delivered) delta.
        Call after the delta frame was actually queued, never before. `delta`
        must be the dict returned by a prior ``compute_delta``."""
        idx = delta["_idx"]
        idx_free = delta["_idx_free"]
        self._prev_rendered = delta["_mask"]
        self._mark_sent(idx)
        self._sent_zmax[idx_free] = 0.0
        self._sent_cls[idx_free] = UNKNOWN
        self._last_n_rendered = int(np.count_nonzero(self._prev_rendered))
        self._last_n_freed = int(idx_free.size)
        self._last_n_changed = int(idx.size)

    def reset(self):
        self.z_mean[:] = 0
        self._z_min_state[:] = np.inf
        self._z_max_state[:] = -np.inf
        self.cls_hist[:] = 0
        self.dominant_class[:] = UNKNOWN
        self.occupancy[:] = 0
        self.last_update[:] = 0
        self.frame = 0
        self.traversability[:] = 0
        self._sent_zmax[:] = 0
        self._sent_cls[:] = UNKNOWN
        self.ground_z = 0.0
        self._prev_rendered[:] = False

    # --- memory accountant (PROJECT_SPEC §9.2 / acceptance criterion) ---
    @property
    def bytes_per_cell(self) -> float:
        return 2 * 4 + self.n_classes * 4 + 1 + 4 + 4 + 8

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