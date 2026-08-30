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
from src.server import ws_protocol


class Pipeline:
    """Background thread: replayer -> segmenter -> grid. Produces WS message dicts."""

    def __init__(self, cfg: dict):
        from src.common.config import resolve_path
        self.cfg = cfg
        src = cfg["source"]
        mdl = cfg["model"]
        # seq_dir and checkpoint may be relative to the repo root — resolve them
        # so we don't depend on the process CWD.
        seq_dir = resolve_path(src["seq_dir"])
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
        # 10Hz snapshots don't overflow when dashboard is briefly slow.
        self.messages: queue.Queue = queue.Queue(maxsize=32)
        self.pause_event = threading.Event()
        self._frames_emitted = 0
        self._frames_dropped = 0
        # start paused: the dashboard must explicitly press Play (auto-play
        # off on page load). Seek previews still render one frame while paused.
        self.speed = float(src["playback_speed"])
        self.running = False
        self._thread = None
        self._last_snapshot_frame = 0
        self._last_stats_frame = 0
        self._frame_time_window = []
        self._stage_window = {"seg": [], "grid": [], "pack": [], "cloud": []}
        self._project_window = []
        self._forward_window = []
        self._mem = None
        self._perf_n = 0
        self._perf_sum = {"seg": 0.0, "grid": 0.0, "pack": 0.0, "proc": 0.0}
        self.cloud_on = False
        self.cloud_max = int(self.cfg["server"].get("cloud_points_max", 30000))
        self._snap_iv = int(self.cfg["server"].get("snapshot_interval_frames", 20))
        self._stats_iv = int(self.cfg["server"].get("stats_interval_frames", 10))

        # rolling world-frame cloud history (for the ego-anchored accumulation view)
        self.cloud_history_frames = int(self.cfg["server"].get("cloud_history_frames", 30))
        self._cloud_hist: deque = deque(maxlen=self.cloud_history_frames)
        self._pose_mats = _load_pose_mats(src["seq_dir"])
        self.epoch = 0  # bumped on seek so clients can drop stale in-flight frames
        self._seek_lock = threading.Lock()
        self._preview = False  # process exactly one frame on the next loop pass

    def start(self):
        self.running = True
        self.replayer.playback_speed = self.speed
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline")
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

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
        while True:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break
        self._cloud_hist.clear()
        self._frame_time_window.clear()
        for buf in self._stage_window.values():
            buf.clear()
        self.epoch += 1
        with self._seek_lock:
            self._preview = True

    def _take_preview(self) -> bool:
        with self._seek_lock:
            v = self._preview
            self._preview = False
            return v

    def _emit(self, payload, kind: str = "text"):
        """Queue a broadcast payload. kind='text' -> JSON dict, 'binary' -> bytes frame."""
        item = ("T", payload) if kind == "text" else ("B", payload)
        # In precise mode every frame is a full snapshot (large). Instead of
        # dropping just the oldest entry and still overflowing next put, coalesce:
        # keep only the latest binary frame so we never lag (text stats/ack preserved).
        is_binary = (kind != "text")
        try:
            self.messages.put_nowait(item)
            self._frames_emitted += 1
        except queue.Full:
            if is_binary:
                # Drain all queued binary frames, keep only text (stats/ack) if any
                kept: list = []
                while True:
                    try:
                        q = self.messages.get_nowait()
                        if q[0] == "T":
                            kept.append(q)
                        else:
                            self._frames_dropped += 1
                    except queue.Empty:
                        break
                for k in kept:
                    try:
                        self.messages.put_nowait(k)
                    except queue.Full:
                        break
            else:
                self._drop_oldest()
                self._frames_dropped += 1
            try:
                self.messages.put_nowait(item)
                self._frames_emitted += 1
            except queue.Full:
                self._frames_dropped += 1
                pass

    def _loop(self):
        stats_iv = self.cfg["server"]["stats_interval_frames"]
        snap_iv = self.cfg["server"]["snapshot_interval_frames"]
        while self.running:
            try:
                self._loop_step(stats_iv, snap_iv)
            except Exception as e:  # surface errors, keep stream alive
                import traceback
                print("[pipeline] error:", e)
                traceback.print_exc()
                time.sleep(0.1)

    def _loop_step(self, stats_iv: int, snap_iv: int):
        # A seek preview must render even while paused: check before blocking.
        if self._take_preview():
            frame = self.replayer.get(timeout=2.0)
            if frame is not None:
                self._process_frame(frame, stats_iv, snap_iv)
            return
        if not self.pause_event.wait(timeout=0.25):
            return  # paused: retry at the top of the loop (still sees seeks)
        frame = self.replayer.get(timeout=2.0)
        if frame is None:
            time.sleep(0.05)
            return
        self._process_frame(frame, stats_iv, snap_iv)

    def _process_frame(self, frame, stats_iv: int, snap_iv: int):
        t0 = time.perf_counter()
        cls5, timings = self.segmenter.segment(frame.points)
        t_seg = time.perf_counter()
        cls4 = bin_5_to_4(cls5)
        stats_msg = None
        with self.grid_lock:
            self.grid.update(frame.points, cls4)
            # precise (no-decay) mode: every frame is a full snapshot so client never holds stale cells
            # until the next incremental snapshot — eliminates the "older stays until new comes" ghost.
            if not self.grid.decay_enabled:
                snap = self.grid.snapshot()
                is_snap = True
                self._last_snapshot_frame = self.grid.frame
            elif self.grid.frame - self._last_snapshot_frame >= snap_iv:
                snap = self.grid.snapshot()
                is_snap = True
                self._last_snapshot_frame = self.grid.frame
            else:
                snap = self.grid.delta()
                is_snap = False
            if self.grid.frame - self._last_stats_frame >= stats_iv:
                # Recompute every interval so rendered KB / compression reflect live scene (was cached static)
                self._mem = self.grid.memory_report()
                stats = self._compute_stats(self._mem)
                self._last_stats_frame = self.grid.frame
                stats_msg = ws_protocol.stats_message(stats)
        t_grid = time.perf_counter()

        # binary grid frames (40-100x smaller than JSON, no .tolist() tax)
        yaw_cd = self._yaw_cd(frame.idx)
        frames = (ws_protocol.iter_snapshot_frames(snap["frame"], self.epoch, snap["rows"], snap["cls"], yaw_cd=yaw_cd)
                  if is_snap else
                  ws_protocol.iter_delta_frames(snap["frame"], self.epoch, snap["rows"], snap["cls"], snap["freed"], yaw_cd=yaw_cd))
        for b in frames:
            self._emit(b, "binary")
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
                self._emit(b, "binary")
            cloud_ms = (time.perf_counter() - tc) * 1000.0

        proc_ms = (time.perf_counter() - t0) * 1000.0
        grid_ms = (t_grid - t_seg) * 1000.0
        pack_ms = (t_pack - t_grid) * 1000.0
        self._stage_window["seg"].append(timings.get("total", 0.0))
        self._stage_window["grid"].append(grid_ms)
        self._stage_window["pack"].append(pack_ms)
        self._stage_window["cloud"].append(cloud_ms)
        self._project_window.append(timings.get("project", 0.0))
        self._forward_window.append(timings.get("forward", 0.0))
        self._frame_time_window.append(proc_ms)
        for buf in (self._stage_window["seg"], self._stage_window["grid"],
                    self._stage_window["pack"], self._stage_window["cloud"],
                    self._project_window, self._forward_window,
                    self._frame_time_window):
            if len(buf) > 100:
                buf.pop(0)

        # detailed perf log every 30 frames (and on any slow frame)
        self._perf_n += 1
        self._perf_sum["seg"] += timings.get('total', 0.0)
        self._perf_sum["grid"] += grid_ms
        self._perf_sum["pack"] += pack_ms
        self._perf_sum["proc"] += proc_ms
        if self._perf_n >= 30 or proc_ms > 400.0:
            n = self._perf_n
            avg = {k: v / n for k, v in self._perf_sum.items()}
            qd = self.messages.qsize()
            try:
                sr = snap["rows"]
                tot = int(sr.shape[0])
                if tot:
                    js = sr[:, 1]  # sector j
                    behind = int(np.count_nonzero((js < 90) | (js > 630)))
                    front = tot - behind
                else:
                    behind = front = 0
                    tot = 0
            except:
                tot = behind = front = -1
            print(f"[perf] avg{n} seg={avg['seg']:.1f}ms grid={avg['grid']:.1f}ms pack={avg['pack']:.1f}ms proc={avg['proc']:.1f}ms q={qd} drop={self._frames_dropped} rendered={tot} front={front} behind={behind}")
            self._perf_n = 0
            self._perf_sum = {"seg": 0.0, "grid": 0.0, "pack": 0.0, "proc": 0.0}
        elif proc_ms > 400.0:
            print(f"[slow] frame {self.grid.frame}: seg={timings.get('total', 0.0):.0f}ms "
                  f"grid={grid_ms:.1f}ms pack={pack_ms:.1f}ms cloud={cloud_ms:.1f}ms "
                  f"n_cells={self._mem['n_cells'] if self._mem else 0}")

        if stats_msg is not None:
            stats_msg["epoch"] = self.epoch
            self._emit(stats_msg, "text")

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

    def _drop_oldest(self):
        try:
            self.messages.get_nowait()
        except queue.Empty:
            pass

    def _compute_stats(self, mem: dict) -> dict:
        win = self._frame_time_window
        avg_ms = float(np.mean(win)) if win else 0.0
        latency_p50 = float(np.percentile(win, 50)) if win else 0.0
        latency_p95 = float(np.percentile(win, 95)) if win else 0.0
        stages = {k: round(float(np.mean(v)), 1) if v else 0.0 for k, v in self._stage_window.items()}
        return {
            "frame": self.grid.frame,
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
            "seq_pos": self.replayer.i,
            "seq_len": len(self.replayer),
            "frames_emitted": self._frames_emitted,
            "frames_dropped": self._frames_dropped,
        }


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


def create_app(cfg: dict | None = None) -> FastAPI:
    if cfg is None:
        cfg = load_pipeline_config()
    pipeline = Pipeline(cfg)
    connections: set[WebSocket] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.start()
        broadcaster = asyncio.create_task(_broadcast_loop())
        try:
            yield
        finally:
            broadcaster.cancel()
            pipeline.stop()

    async def _broadcast_loop():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, pipeline.messages.get)
            except asyncio.CancelledError:
                break
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
                "checkpoint_exists": Path(str(ckpt)).is_file()}

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

    @app.websocket("/ws/map")
    async def ws_map(ws: WebSocket):
        await ws.accept()
        connections.add(ws)
        meta = ws_protocol.grid_meta_message(pipeline.grid)
        await ws.send_text(json.dumps(meta))
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "control":
                    action = data.get("action")
                    if action == "pause":
                        pipeline.pause(True)
                        print(f"[ws] pause ack frame={pipeline.grid.frame}")
                        await ws.send_text(json.dumps(
                            {"type": "control_ack", "action": "pause", "frame": pipeline.grid.frame}))
                    elif action == "play":
                        pipeline.pause(False)
                        print(f"[ws] play ack frame={pipeline.grid.frame}")
                        await ws.send_text(json.dumps(
                            {"type": "control_ack", "action": "play", "frame": pipeline.grid.frame}))
                    elif action == "speed":
                        pipeline.set_speed(data.get("value", 1.0))
                    elif action == "cloud":
                        pipeline.cloud_on = bool(data.get("value", False))
                        print(f"[server] cloud stream {'ON' if pipeline.cloud_on else 'OFF'}")
                    elif action == "seek":
                        pipeline.seek(data.get("value", 0))
                        await ws.send_text(json.dumps({
                            "type": "control_ack", "action": "seek",
                            "frame": pipeline.grid.frame, "idx": pipeline.replayer.i,
                            "epoch": pipeline.epoch}))
                elif data.get("type") == "request_snapshot":
                    with pipeline.grid_lock:
                        snap = pipeline.grid.snapshot()
                    for b in ws_protocol.iter_snapshot_frames(snap["frame"], pipeline.epoch, snap["rows"], snap["cls"],
                                              yaw_cd=pipeline._yaw_cd(pipeline.replayer.i)):
                        await ws.send_bytes(b)
        except WebSocketDisconnect:
            pass
        finally:
            connections.discard(ws)

    return app


# module-level import safety
_ = ws_protocol