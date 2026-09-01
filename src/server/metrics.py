"""Per-frame timing windows and live stats aggregation for the pipeline.

Pure bookkeeping: the pipeline thread feeds ``push_frame(...)`` per rendered
frame and the wrapper logs a detailed perf line every N frames / on slow frames.
"""

import numpy as np

MAX_WINDOW = 100
PERF_LOG_EVERY = 30
PERF_LOG_SLOW_MS = 400.0


class FrameMetrics:
    def __init__(self):
        self._frame_time_window: list[float] = []
        self._stage_window: dict[str, list[float]] = {
            "seg": [], "grid": [], "pack": [], "cloud": [],
        }
        self._project_window: list[float] = []
        self._forward_window: list[float] = []
        self._perf_n = 0
        self._perf_sum = {"seg": 0.0, "grid": 0.0, "pack": 0.0, "proc": 0.0}

    def reset(self):
        self._frame_time_window.clear()
        for buf in self._stage_window.values():
            buf.clear()
        self._project_window.clear()
        self._forward_window.clear()
        self._perf_n = 0
        self._perf_sum = {"seg": 0.0, "grid": 0.0, "pack": 0.0, "proc": 0.0}

    def push_frame(self, seg_total: float, grid_ms: float, pack_ms: float,
                   cloud_ms: float, project_ms: float, forward_ms: float, proc_ms: float):
        self._stage_window["seg"].append(seg_total)
        self._stage_window["grid"].append(grid_ms)
        self._stage_window["pack"].append(pack_ms)
        self._stage_window["cloud"].append(cloud_ms)
        self._project_window.append(project_ms)
        self._forward_window.append(forward_ms)
        self._frame_time_window.append(proc_ms)
        for buf in (self._stage_window["seg"], self._stage_window["grid"],
                    self._stage_window["pack"], self._stage_window["cloud"],
                    self._project_window, self._forward_window,
                    self._frame_time_window):
            if len(buf) > MAX_WINDOW:
                buf.pop(0)

    def compute_stats(self, mem: dict, frame: int, seq_pos: int, seq_len: int,
                      frames_emitted: int, frames_dropped: int) -> dict:
        win = self._frame_time_window
        avg_ms = float(np.mean(win)) if win else 0.0
        latency_p50 = float(np.percentile(win, 50)) if win else 0.0
        latency_p95 = float(np.percentile(win, 95)) if win else 0.0
        stages = {k: round(float(np.mean(v)), 1) if v else 0.0 for k, v in self._stage_window.items()}
        return {
            "frame": frame,
            "fps": 1000.0 / max(avg_ms, 1e-3),
            "latency_ms_p50": round(latency_p50, 1),
            "latency_ms_p95": round(latency_p95, 1),
            "seg_ms": stages["seg"],
            "grid_ms": stages["grid"],
            "pack_ms": stages["pack"],
            "cloud_ms": stages["cloud"],
            "project_ms": round(float(np.mean(self._project_window)), 1) if self._project_window else 0.0,
            "forward_ms": round(float(np.mean(self._forward_window)), 1) if self._forward_window else 0.0,
            "grid_mem_kb": mem.get("rendered_kb", mem["grid_kb"]),
            "uniform_equiv_mb": mem["uniform_mb"],
            "compression_ratio": mem["compression_ratio"],
            "capacity_compression": mem.get("capacity_compression", mem["compression_ratio"]),
            "rendered_cells": mem.get("rendered_cells", 0),
            "n_cells": mem["n_cells"],
            "seq_pos": seq_pos,
            "seq_len": seq_len,
            "frames_emitted": frames_emitted,
            "frames_dropped": frames_dropped,
        }

    def maybe_log_perf(self, grid, seg_total: float, disk_ms: float, update_ms: float,
                       snap_ms: float, pack_ms: float, proc_ms: float, is_snap: bool,
                       n_rows: int, n_freed: int, n_chunks: int, wire_bytes: int,
                       cloud_ms: float, queue_size: int):
        """Accumulate per-stage sums; every PERF_LOG_EVERY frames (or on a slow
        frame) emit one detailed perf line. Returns True if a line was printed."""
        self._perf_n += 1
        self._perf_sum["seg"] += seg_total
        self._perf_sum["grid"] += update_ms
        self._perf_sum["pack"] += pack_ms
        self._perf_sum["proc"] += proc_ms
        if self._perf_n < PERF_LOG_EVERY and proc_ms <= PERF_LOG_SLOW_MS:
            return False
        n = self._perf_n
        avg = {k: v / n for k, v in self._perf_sum.items()}
        gt = grid._timings
        tag = "SNAP" if is_snap else "DELTA"
        print(
            f"[perf] f={grid.frame:5d} {tag:4s} | "
            f"proc={avg['proc']:.1f}ms seg={avg['seg']:.1f} "
            f"sync={seg_total:.1f} disk={disk_ms:.1f} "
            f"update={update_ms:.1f} snap/delta={snap_ms:.1f} "
            f"pack={avg['pack']:.1f} cloud={cloud_ms:.1f} | "
            f"rows={n_rows:5d} freed={n_freed:4d} chunks={n_chunks} "
            f"wire={wire_bytes/1024:.0f}KB | "
            f"rendered={grid._last_n_rendered:6d} "
            f"hit={grid._last_n_hit:5d} "
            f"n_cells={grid.n_cells} | "
            f"polar={gt.get('polar_ms',0):.1f} "
            f"reduce={gt.get('reduce_ms',0):.1f} "
            f"cls={gt.get('cls_ms',0):.1f} "
            f"state={gt.get('state_ms',0):.1f} | "
            f"queue={queue_size}"
        )
        self._perf_n = 0
        self._perf_sum = {"seg": 0.0, "grid": 0.0, "pack": 0.0, "proc": 0.0}
        return True