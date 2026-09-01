import asyncio
import json
import math
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.data.label_mapping import bin_5_to_4
from src.data.replayer import SemanticKITTIReplayer
from src.grid_engine.logpolar_grid import LogPolarGrid
from src.models.predict import Segmenter
from src.server import pending as pending_state
from src.server import ws_protocol
from src.server.broadcaster import Broadcaster
from src.server.metrics import FrameMetrics


class Pipeline:
    """Two-stage streaming pipeline: SEG thread (GPU segmenter) runs ahead of the
    GRID thread (CPU grid update + pack/emit), hiding the GPU's device→host sync
    behind the previous frame's grid work. A single-slot mailbox preserves strict
    frame ordering with at most 1 frame of look-ahead.     Falls back to a serial
    single-thread path when ``pipeline.stages: single`` is set in config."""

    # When at least this many frames drop consecutively (or the pending
    # retransmit volume reaches PENDING_THRESHOLD), emit a full snapshot instead
    # of a large merged delta so the client resyncs cleanly (no transient map
    # bloat / layout jank). Tuned well above the normal snapshot interval so the
    # periodic snapshot path is unaffected.
    CATCHUP_DROP_THRESHOLD = 6
    CATCHUP_PENDING_THRESHOLD = 40_000

    def __init__(self, cfg: dict):
        from src.common.config import resolve_path
        self.cfg = cfg
        src = cfg["source"]
        mdl = cfg["model"]
        # seq_dir and checkpoint may be relative to the repo root â€” resolve them
        # so we don't depend on the process CWD.
        seq_dir = resolve_path(src["seq_dir"])
        self.seq_base_dir = seq_dir.parent
        self.seq_id = seq_dir.name
        checkpoint = resolve_path(mdl["checkpoint"])
        self.replayer = SemanticKITTIReplayer(
            seq_dir, playback_speed=src["playback_speed"], loop=src["loop"]
        )
        self.segmenter = Segmenter(
            checkpoint,
            h=mdl["h"],
            w=mdl["w"],
            device=mdl.get("device", "auto"),
            precision=mdl.get("precision", "fp16"),
        )
        self.grid = LogPolarGrid()
        self.grid_lock = threading.Lock()
        # Larger queue; precise mode drops intermediate snapshots (coalesce) so
        # 10Hz snapshots don't overflow when dashboard is briefly slow. Sized well
        # above the typical per-frame chunk count (~7) so short disk/I/O bursts
        # are absorbed instead of dropping frames; any frames that DO drop are now
        # re-transmitted losslessly via _pending_upsert/_pending_free.
        self.broadcaster = Broadcaster(
            maxsize=128, wire_compress=bool(self.cfg["server"].get("wire_compress", False))
        )
        self.pause_event = threading.Event()
        # Flat cell indices (i*n_theta+j) whose `freed` was computed by delta()
        # but whose outgoing frame was dropped by the queue (overflow). Merged
        # into the next successfully-emitted delta so the client never loses a
        # free (the source of ghost cells). Cleared on snapshot / seek.
        self._pending_free: set = set()
        # Flat cell indices (i*n_theta+j) whose `added`/`changed` rows were
        # computed by delta() but whose outgoing frame was dropped by the queue.
        # Unlike `_pending_free` (only frees), there is no server-side guard for
        # lost upserts, so a dropped delta leaves the client permanently missing
        # those cells until the next full snapshot. Merged into the next
        # successfully-emitted delta, mirroring `_pending_free`. Cleared on a
        # successful snapshot / seek.
        self._pending_upsert: set = set()
        # Consecutive frames dropped by the broadcast queue without a single
        # successful send. When this (or the pending retransmit volume) grows
        # large, a full snapshot is emitted instead of a giant merged delta so
        # the client resyncs cleanly (no transient map bloat / layout jank).
        self._consecutive_drops = 0
        # start paused: the dashboard must explicitly press Play (auto-play
        # off on page load). Seek previews still render one frame while paused.
        self.speed = float(src["playback_speed"])
        self.running = False
        self._thread = None
        self._last_snapshot_frame = 0
        self._last_stats_frame = 0
        self.metrics = FrameMetrics()
        self._mem = None
        self.cloud_on = False
        self.cloud_max = int(self.cfg["server"].get("cloud_points_max", 30000))
        self._snap_iv = int(self.cfg["server"].get("snapshot_interval_frames", 5))
        self._stats_iv = int(self.cfg["server"].get("stats_interval_frames", 10))

        # Two-stage pipelining: the GPU segmenter runs on its own thread ahead
        # of the CPU grid/pack/emit stage, so the segmenter's device->host sync
        # (.cpu()) overlaps the previous frame's grid work instead of stalling
        # the single pipeline thread. `stages: auto` (default) enables it;
        # `stages: single` restores the exact previous serial behavior.
        self._stages = str(self.cfg.get("pipeline", {}).get("stages", "auto")).lower()
        # bounded look-ahead of segmented frames: maxsize 2 keeps at most ~1
        # frame of lead so seek latency stays small and ordering is preserved.
        self._prefetch: queue.Queue = queue.Queue(maxsize=2)
        self._seg_thread = None

        # rolling world-frame cloud history (for the ego-anchored accumulation view)
        self.cloud_history_frames = int(self.cfg["server"].get("cloud_history_frames", 30))
        self._cloud_hist: deque = deque(maxlen=self.cloud_history_frames)
        self._pose_mats = _load_pose_mats(seq_dir)
        self.epoch = 0  # bumped on seek so clients can drop stale in-flight frames
        self._seek_lock = threading.Lock()
        self._replayer_lock = threading.Lock()
        self._preview = False  # process exactly one frame on the next loop pass

        # readiness: set after Pipeline fully initialized + first frame emitted
        self.ready = threading.Event()
        self._first_frame_emitted = False
        self.ready.set()

    @property
    def messages(self) -> queue.Queue:
        """Back-compat accessor for the broadcast queue (Broadcaster.messages)."""
        return self.broadcaster.messages

    def start(self):
        self.running = True
        self.replayer.playback_speed = self.speed
        if self._stages == "auto":
            self._thread = threading.Thread(target=self._grid_loop, daemon=True, name="grid")
            self._seg_thread = threading.Thread(target=self._seg_loop, daemon=True, name="seg")
            self._thread.start()
            self._seg_thread.start()
        else:
            self._thread = threading.Thread(target=self._single_loop, daemon=True, name="pipeline")
            self._thread.start()

    def stop(self):
        self.running = False
        if self._seg_thread:
            self._seg_thread.join(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)
        self.replayer.close()

    def set_speed(self, speed: float):
        self.speed = max(0.05, float(speed))
        self.replayer.playback_speed = self.speed

    def pause(self, paused: bool):
        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def seek(self, idx: int):
        """Jump the replayer to an absolute frame index and reset all derived
        state (grid map, accumulation history, message queue) so the dashboard
        rebuilds cleanly from the new position.

        Playback state is left untouched (paused stays paused). The next loop
        pass renders exactly one preview frame so the new location is visible
        immediately, then playback resumes only if it was already running.
        """
        n = len(self.replayer)
        if n == 0:
            return
        idx = max(0, min(n - 1, int(idx)))
        with self.grid_lock:
            self.grid.reset()
            self._last_snapshot_frame = self.grid.frame - self._snap_iv  # force snapshot next frame
            self._last_stats_frame = self.grid.frame - self._stats_iv    # force stats too
            self._mem = None
        self.replayer.seek(idx)
        self.broadcaster.clear()
        self._drain_prefetch()
        self._cloud_hist.clear()
        self.metrics.reset()
        self.epoch += 1
        self._pending_free.clear()
        self._pending_upsert.clear()
        self._consecutive_drops = 0
        with self._seek_lock:
            self._preview = True

    def switch_sequence(self, seq_id: str):
        """Swap the replayer to a different SemanticKITTI sequence.

        Similar to seek() but also rebuilds the replayer and pose mats for the
        new sequence directory. The pipeline is paused during the swap so the
        background thread doesn't read a half-initialised replayer.
        """
        new_dir = self.seq_base_dir / seq_id
        if not (new_dir / "velodyne").exists():
            print(f"[pipeline] switch_sequence: {new_dir}/velodyne not found, ignoring")
            return
        was_paused = not self.pause_event.is_set()
        self.pause_event.clear()
        time.sleep(0.05)
        old_replayer = self.replayer
        with self._replayer_lock:
            with self.grid_lock:
                self.grid.reset()
                self._last_snapshot_frame = self.grid.frame - self._snap_iv
                self._last_stats_frame = self.grid.frame - self._stats_iv
                self._mem = None
            self.replayer = SemanticKITTIReplayer(
                new_dir,
                playback_speed=self.speed,
                loop=self.cfg["source"]["loop"],
            )
            self._pose_mats = _load_pose_mats(new_dir)
            self.seq_id = seq_id
        old_replayer.close()
        self._cloud_hist.clear()
        self.metrics.reset()
        self.broadcaster.frames_emitted = 0
        self.broadcaster.frames_dropped = 0
        self._pending_free.clear()
        self._pending_upsert.clear()
        self._consecutive_drops = 0
        self.broadcaster.clear()
        self._drain_prefetch()
        self.epoch += 1
        with self._seek_lock:
            self._preview = True
        if not was_paused:
            self.pause_event.set()

    def _take_preview(self) -> bool:
        with self._seek_lock:
            v = self._preview
            self._preview = False
            return v

    # ------------------------------------------------------------------
    # Two-stage pipeline (stages: auto)
    #
    #   SEG thread: paced replayer read (disk) -> GPU segment -> mailbox
    #   GRID thread: mailbox -> bin_5_to_4 -> grid.update -> pack -> emit
    #
    # The single-slot handoff guarantees strict frame ordering while letting the
    # GPU segmenter run up to ~1 frame ahead, hiding its device->host .cpu()
    # sync behind the CPU grid work of the previous frame.
    # ------------------------------------------------------------------

    def _single_loop(self):
        stats_iv = self.cfg["server"]["stats_interval_frames"]
        snap_iv = self.cfg["server"]["snapshot_interval_frames"]
        while self.running:
            try:
                self._single_step(stats_iv, snap_iv)
            except Exception as e:  # surface errors, keep stream alive
                import traceback
                print("[pipeline] error:", e)
                traceback.print_exc()
                time.sleep(0.1)

    def _single_render(self, frame, stats_iv: int, snap_iv: int, disk_ms: float = 0.0):
        """Serial path: segment (GPU) then grid/emit inline on this thread."""
        cls5, seg_t = self.segmenter.segment(frame.points)
        cls4 = bin_5_to_4(cls5)
        self._process_frame_seg(frame, cls4, seg_t, stats_iv, snap_iv, disk_ms=disk_ms)

    def _single_step(self, stats_iv: int, snap_iv: int):
        # A seek preview must render even while paused: check before blocking.
        if self._take_preview():
            with self._replayer_lock:
                frame = self.replayer.get(timeout=2.0)
            if frame is not None:
                self._single_render(frame, stats_iv, snap_iv)
            return
        if not self.pause_event.wait(timeout=0.25):
            return  # paused: retry at the top of the loop (still sees seeks)
        td = time.perf_counter()
        with self._replayer_lock:
            frame = self.replayer.get(timeout=2.0)
        disk_ms = (time.perf_counter() - td) * 1000.0
        if frame is None:
            time.sleep(0.05)
            return
        self._single_render(frame, stats_iv, snap_iv, disk_ms=disk_ms)

    # ------------------------------------------------------------------
    # Two-stage pipeline (stages: auto)
    #
    #   SEG thread: paced replayer read (disk) -> GPU segment -> mailbox
    #   GRID thread: mailbox -> bin_5_to_4 -> grid.update -> pack -> emit
    #
    # The single-slot handoff guarantees strict frame ordering while letting the
    # GPU segmenter run up to ~1 frame ahead, hiding its device->host .cpu()
    # sync behind the CPU grid work of the previous frame.
    # ------------------------------------------------------------------

    def _seg_loop(self):
        while self.running:
            try:
                epoch = self.epoch
                frame = None
                if self._take_preview():
                    # seek preview renders even while paused
                    with self._replayer_lock:
                        frame = self.replayer.get(timeout=2.0)
                    disk_ms = 0.0
                elif self.pause_event.wait(timeout=0.25):
                    td = time.perf_counter()
                    with self._replayer_lock:
                        frame = self.replayer.get(timeout=2.0)
                    disk_ms = (time.perf_counter() - td) * 1000.0
                else:
                    continue  # paused, no preview: retry at top
                if frame is None:
                    continue
                cls5, seg_t = self.segmenter.segment(frame.points)
                # Blocking put: if the grid thread is behind, we wait here (natural
                # backpressure) rather than piling up stale frames.
                self._prefetch.put((epoch, frame, cls5, seg_t, disk_ms))
            except Exception as e:
                import traceback
                print("[seg] error:", e)
                traceback.print_exc()
                time.sleep(0.1)

    def _grid_loop(self):
        stats_iv = self.cfg["server"]["stats_interval_frames"]
        snap_iv = self.cfg["server"]["snapshot_interval_frames"]
        while self.running:
            try:
                epoch, frame, cls5, seg_t, disk_ms = self._prefetch.get(timeout=0.5)
                if epoch != self.epoch:
                    # stale frame from before a seek/switch — discard
                    continue
                cls4 = bin_5_to_4(cls5)
                self._process_frame_seg(frame, cls4, seg_t, stats_iv, snap_iv, disk_ms=disk_ms)
            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                print("[grid] error:", e)
                traceback.print_exc()
                time.sleep(0.05)

    def _process_frame_seg(self, frame, cls4, seg_timings: dict, stats_iv: int, snap_iv: int,
                           disk_ms: float = 0.0):
        t0 = time.perf_counter()
        t_seg = time.perf_counter()
        stats_msg = None
        with self.grid_lock:
            self.grid.update(frame.points, cls4)
            t_update = time.perf_counter()
            # Both precise and decay modes respect snapshot_interval_frames:
            # send full snapshot periodically, deltas in between.
            # This avoids O(n_cells) snapshot cost every frame in precise mode.
            # When a burst of frames has been dropped (large retransmit backlog),
            # a full snapshot is emitted instead of a giant merged delta so the
            # client resyncs cleanly (no transient map bloat / layout jank).
            if self._should_snapshot(snap_iv):
                snap = self.grid.snapshot()
                is_snap = True
                self._last_snapshot_frame = self.grid.frame
                self._consecutive_drops = 0
            else:
                snap = self.grid.delta()
                is_snap = False
                # Merge frees that were computed but dropped earlier (queue
                # overflow) into this delta so the client never loses a free
                # (the source of ghost cells). Doing it under the lock keeps
                # the rendered-mask read consistent with delta().
                snap["freed"] = self._merge_pending_free(snap["freed"])
                # Merge re-send upserts computed but dropped earlier into this
                # delta, mirroring _merge_pending_free, so no added/changed cell
                # is lost to the client (the other half of the ghost story).
                snap["rows"], snap["cls"] = self._merge_pending_upsert(snap["rows"], snap["cls"])
            t_snap = time.perf_counter()
            if self.grid.frame - self._last_stats_frame >= stats_iv:
                # Recompute every interval so rendered KB / compression reflect live scene (was cached static)
                self._mem = self.grid.memory_report()
                stats = self.metrics.compute_stats(
                    self._mem, self.grid.frame, self.replayer.i, len(self.replayer),
                    self.broadcaster.frames_emitted, self.broadcaster.frames_dropped)
                self._last_stats_frame = self.grid.frame
                stats_msg = ws_protocol.stats_message(stats)
        t_grid = time.perf_counter()

        # binary grid frames (40-100x smaller than JSON, no .tolist() tax)
        yaw_cd = self._yaw_cd(frame.idx)
        n_rows = snap["rows"].shape[0]
        if is_snap:
            # A snapshot is authoritative: the client fully resyncs, so any
            # previously-pending frees AND upserts are subsumed. Clear them once
            # we actually manage to send it (dropped snapshots keep them).
            freed = np.zeros((0, 2), dtype=np.float32)
            frames = list(ws_protocol.iter_snapshot_frames(
                snap["frame"], self.epoch, snap["rows"], snap["cls"], yaw_cd=yaw_cd))
            sent = self.broadcaster.emit_frame(frames, "binary")
            if sent:
                self._pending_free.clear()
                self._pending_upsert.clear()
                self._consecutive_drops = 0
            else:
                self._consecutive_drops += 1
        else:
            # snap["freed"] was already merged with pending frees (under lock).
            freed = snap["freed"]
            frames = list(ws_protocol.iter_delta_frames(
                snap["frame"], self.epoch, snap["rows"], snap["cls"], freed, yaw_cd=yaw_cd))
            sent = self.broadcaster.emit_frame(frames, "binary")
            if sent:
                self._consecutive_drops = 0
            else:
                # Frame dropped wholesale — preserve its upserts (new + any
                # already-merged pending) and its frees for the next successful
                # delta (cells are still never-sent / not-rendered), so neither
                # missing cells nor ghost cells are introduced by the drop.
                self._consecutive_drops += 1
                self._pending_upsert |= self._rows_to_flat(snap["rows"])
                if freed.shape[0]:
                    self._pending_free |= self._free_to_flat(freed)
        n_freed = freed.shape[0]
        n_chunks = len(frames)
        wire_bytes = sum(len(b) for b in frames)
        if not self._first_frame_emitted:
            self._first_frame_emitted = True
            self.broadcaster.emit({"type": "status", "state": "ready", "msg": "Streaming", "seq_id": self.seq_id}, "text")
        t_pack = time.perf_counter()

        # cloud only streams while a cloud view is active
        cloud_ms = 0.0
        if self.cloud_on:
            tc = time.perf_counter()
            ego = self._accum_stream(frame.points, cls4, frame.idx)
            # rebase onto the grid's road reference so the cloud ground sits on
            # the grid floor instead of ~sensor-height below it
            cxyz = np.round(ego[0], 2)
            cxyz[:, 2] -= float(self.grid.ground_z)
            for b in ws_protocol.iter_cloud_frames(self.grid.frame, self.epoch, cxyz, ego[1], yaw_cd=yaw_cd):
                self.broadcaster.emit(b, "binary")
            cloud_ms = (time.perf_counter() - tc) * 1000.0

        proc_ms = (time.perf_counter() - t0) * 1000.0
        grid_ms = (t_grid - t_seg) * 1000.0
        update_ms = self.grid._timings.get("total_ms", 0.0)
        snap_ms = self.grid._timings.get("snapshot_ms", self.grid._timings.get("delta_ms", 0.0))
        pack_ms = (t_pack - t_grid) * 1000.0
        self.metrics.push_frame(
            seg_timings.get("total", 0.0), grid_ms, pack_ms, cloud_ms,
            seg_timings.get("project", 0.0), seg_timings.get("forward", 0.0), proc_ms)
        self.metrics.maybe_log_perf(
            self.grid, seg_timings.get('total', 0.0), disk_ms, update_ms, snap_ms, pack_ms,
            proc_ms, is_snap, n_rows, n_freed, n_chunks, wire_bytes, cloud_ms,
            self.broadcaster.qsize)

        if stats_msg is not None:
            stats_msg["epoch"] = self.epoch
            self.broadcaster.emit(stats_msg, "text")

    def _yaw_cd(self, idx: int) -> int:
        """Ego forward yaw in centi-degrees for heading-up UI (0 if unknown).

        The velodyne forward is +X; in world terms R[:, 0] is that direction,
        projected on the XZ ground plane. We return the angle (radians->deg)
        that rotates that forward onto screen-up (-Z in the top-down view),
        i.e. atan2(fx, -fz), so a default straight-ahead car faces "up".
        """
        if self._pose_mats is None:
            return 0
        pose = self._pose_mats[idx % len(self._pose_mats)]
        R = pose[:3, :3]
        fx, fz = float(R[0, 0]), float(R[2, 0])
        return int(round(math.degrees(math.atan2(fx, -fz)) * 100))

    def _accum_stream(self, points: np.ndarray, cls4: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Rolling world-anchored point cloud re-projected into the CURRENT ego
        frame, downsampled to <= cloud_max points.

        With seq poses available this gives a densely accumulated, correctly
        aligned full-area cloud (vs. a tiny single-sweep patch). Without poses
        it degrades gracefully to the current sweep only.
        """
        if self._pose_mats is not None:
            k = idx % len(self._pose_mats)
            pose = self._pose_mats[k]
            R, t = pose[:3, :3], pose[:3, 3]
            new_world = (points[:, :3].astype(np.float64) @ R.T.astype(np.float64) + t).astype(np.float32)
            self._cloud_hist.append((new_world.copy(), cls4.astype(np.uint8)))
            total = sum(len(w) for w, _ in self._cloud_hist)
            stride = max(1, int(np.ceil(total / self.cloud_max)))
            parts = [(w[::stride], c[::stride]) for w, c in self._cloud_hist]
            world = np.concatenate([w for w, _ in parts])
            cls = np.concatenate([c for _, c in parts])
            # re-anchor the whole accumulation to the current ego frame
            ego = ((world.astype(np.float64) - t) @ R.astype(np.float64)).astype(np.float32)
            if ego.shape[0] > self.cloud_max:
                ego = ego[: self.cloud_max]
                cls = cls[: self.cloud_max]
            return ego, cls
        # fallback: current sweep only
        n = points.shape[0]
        stride = max(1, int(np.ceil(n / self.cloud_max)))
        return points[::stride, :3].astype(np.float32), cls4[::stride].astype(np.uint8)

    def _drain_prefetch(self):
        """Discard any in-flight segmented frames (stale after a seek/switch)."""
        while True:
            try:
                self._prefetch.get_nowait()
            except queue.Empty:
                break

    def _free_to_flat(self, freed: np.ndarray) -> set:
        """Convert freed rows [i,j] into a set of flat cell indices (i*n_theta+j)."""
        return pending_state.free_to_flat(freed, self.grid.n_theta)

    def _rows_to_flat(self, rows: np.ndarray) -> set:
        """Convert upsert rows [i,j,...] into a set of flat cell indices."""
        return pending_state.rows_to_flat(rows, self.grid.n_theta)

    def _should_snapshot(self, snap_iv: int) -> bool:
        """True when the frame should be sent as a full (authoritative) snapshot
        instead of an incremental delta: either the periodic cadence is due, or a
        burst of dropped frames / large pending retransmit backlog would
        otherwise produce a giant merged delta (the source of transient client
        map bloat)."""
        return pending_state.should_snapshot(
            self.grid.frame, self._last_snapshot_frame, snap_iv,
            self._consecutive_drops, self._pending_free, self._pending_upsert,
            self.CATCHUP_DROP_THRESHOLD, self.CATCHUP_PENDING_THRESHOLD)

    def _merge_pending_upsert(self, rows: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Merge re-send upsert rows from dropped deltas into the current delta.
        Caller must hold self.grid_lock (reads rendered()/_rows). Delegates to
        src.server.pending; see that module for the full contract."""
        return pending_state.merge_pending_upsert(
            self._pending_upsert, rows, cls, self.grid.rendered(), self.grid._rows)

    def _merge_pending_free(self, freed: np.ndarray) -> np.ndarray:
        """Merge this frame's freed rows with any pending frees from dropped
        frames. Caller must hold self.grid_lock. Delegates to src.server.pending."""
        return pending_state.merge_pending_free(
            self._pending_free, freed, ~self.grid.rendered(), self.grid.n_theta)


def _load_pose_mats(seq_dir: str | Path) -> np.ndarray | None:
    """Compose per-frame 4x4 world transforms: pose_k @ velodyne->cam0.

    SemanticKITTI poses.txt stores the camera-0 odometry pose; calib.txt holds
    the constant velodyne->camera0 rigid transform, so world = P_k @ Tr @ pv.
    Returns (N,4,4) float32 or None when poses are unavailable.
    """
    seq_dir = Path(seq_dir)
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
    return M @ Tr.astype(np.float32)


def load_pipeline_config() -> dict:
    from src.common.config import load_config
    return load_config("pipeline")


def load_history(ckpt_dir: str | Path) -> list[dict]:
    p = Path(ckpt_dir) / "history.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


async def _fan_out(connections: set, item):
    """Send one queued broadcast item (('B', bytes) | ('T', dict)) to every live ws."""
    is_bin = item[0] == "B"
    text = None if is_bin else json.dumps(item[1])
    raw = item[1] if is_bin else None
    dead = []
    for ws in list(connections):
        if ws.application_state.name == "DISCONNECTED":
            dead.append(ws)
            continue
        try:
            if is_bin:
                await ws.send_bytes(raw)
            else:
                await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.discard(ws)


async def _broadcast_loop(connections: set, messages: queue.Queue):
    """Drain the pipeline's broadcast queue and fan each item out to live ws."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(None, messages.get)
        except asyncio.CancelledError:
            break
        await _fan_out(connections, item)


async def _control_ack(pipeline: "Pipeline", ws, action: str, **extra):
    await ws.send_text(json.dumps({
        "type": "control_ack", "action": action,
        "frame": pipeline.grid.frame, **extra}))


async def _handle_control_message(pipeline: "Pipeline", ws, data: dict) -> bool:
    """Apply one ws 'control' command. Returns False when the message is not a control."""
    if data.get("type") != "control":
        return False
    action = data.get("action")
    if action == "pause":
        pipeline.pause(True)
        print(f"[ws] pause ack frame={pipeline.grid.frame}")
        await _control_ack(pipeline, ws, "pause")
    elif action == "play":
        pipeline.pause(False)
        print(f"[ws] play ack frame={pipeline.grid.frame}")
        await _control_ack(pipeline, ws, "play")
        if not pipeline._first_frame_emitted:
            await ws.send_text(json.dumps(
                {"type": "status", "state": "buffering", "msg": "Processing first frame\u2026", "seq_id": pipeline.seq_id}))
    elif action == "speed":
        pipeline.set_speed(data.get("value", 1.0))
    elif action == "cloud":
        pipeline.cloud_on = bool(data.get("value", False))
        print(f"[server] cloud stream {'ON' if pipeline.cloud_on else 'OFF'}")
    elif action == "seek":
        pipeline.seek(data.get("value", 0))
        await _control_ack(pipeline, ws, "seek", idx=pipeline.replayer.i, epoch=pipeline.epoch)
    elif action == "switch_sequence":
        seq_id = data.get("value", "")
        print(f"[ws] switch_sequence -> {seq_id}")
        pipeline.switch_sequence(seq_id)
        await _control_ack(pipeline, ws, "switch_sequence",
                           seq_id=pipeline.seq_id, seq_len=len(pipeline.replayer),
                           epoch=pipeline.epoch)
    return True


def create_app(cfg: dict | None = None) -> FastAPI:
    if cfg is None:
        cfg = load_pipeline_config()
    pipeline = Pipeline(cfg)
    connections: set[WebSocket] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.start()
        broadcaster = asyncio.create_task(_broadcast_loop(connections, pipeline.messages))
        try:
            yield
        finally:
            broadcaster.cancel()
            pipeline.stop()

    app = FastAPI(title="pc2d live mapping", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        from src.common.config import resolve_path
        ckpt = resolve_path(cfg["model"]["checkpoint"])
        return {"status": "ok", "n_rings": pipeline.grid.n_rings,
                "n_theta": pipeline.grid.n_theta,
                "frame": pipeline.grid.frame,
                "device": str(pipeline.segmenter.device),
                "precision": pipeline.segmenter.precision,
                "checkpoint_exists": Path(str(ckpt)).is_file(),
                "model_ready": pipeline.ready.is_set(),
                "first_frame_emitted": pipeline._first_frame_emitted}

    @app.get("/ready")
    async def ready():
        if pipeline.ready.is_set():
            return {"status": "ready"}
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "loading", "model_ready": False})

    @app.get("/snapshot")
    async def snapshot():
        with pipeline.grid_lock:
            return ws_protocol.snapshot_message(pipeline.grid)

    @app.get("/metrics/history")
    async def metrics_history():
        from src.common.config import resolve_path
        return load_history(resolve_path(cfg["server"]["ckpt_dir"]))

    @app.get("/metrics/memory")
    async def metrics_memory():
        return pipeline.grid.memory_report()

    @app.get("/metrics/eval")
    async def metrics_eval():
        from src.common.config import repo_root
        results_dir = repo_root() / "results"
        jsons = sorted(results_dir.glob("eval_*.json"))
        if not jsons:
            return {"error": "no eval results found"}
        with open(jsons[-1], encoding="utf-8") as f:
            return json.load(f)

    @app.get("/sequences")
    async def list_sequences():
        base = pipeline.seq_base_dir
        seqs = []
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / "velodyne").is_dir():
                    n_bins = len(list((d / "velodyne").glob("*.bin")))
                    has_poses = (d / "poses.txt").is_file()
                    has_labels = (d / "labels").is_dir()
                    seqs.append({
                        "id": d.name,
                        "frames": n_bins,
                        "has_poses": has_poses,
                        "has_labels": has_labels,
                    })
        return {"current": pipeline.seq_id, "sequences": seqs}

    @app.websocket("/ws/map")
    async def ws_map(ws: WebSocket):
        await ws.accept()
        connections.add(ws)
        meta = ws_protocol.grid_meta_message(pipeline.grid)
        meta["seq_id"] = pipeline.seq_id
        meta["seq_len"] = len(pipeline.replayer)
        await ws.send_text(json.dumps(meta))
        if pipeline.ready.is_set():
            await ws.send_text(json.dumps(
                {"type": "status", "state": "ready", "msg": "Ready", "seq_id": pipeline.seq_id}))
        else:
            await ws.send_text(json.dumps(
                {"type": "status", "state": "loading", "msg": "Loading model\u2026", "seq_id": pipeline.seq_id}))
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if await _handle_control_message(pipeline, ws, data):
                    continue
                if data.get("type") == "request_snapshot":
                    with pipeline.grid_lock:
                        snap = pipeline.grid.snapshot()
                    for b in ws_protocol.iter_snapshot_frames(
                            snap["frame"], pipeline.epoch, snap["rows"], snap["cls"],
                            yaw_cd=pipeline._yaw_cd(pipeline.replayer.i)):
                        await ws.send_bytes(b)
        except WebSocketDisconnect:
            pass
        finally:
            connections.discard(ws)

    return app


# module-level import safety
_ = ws_protocol
