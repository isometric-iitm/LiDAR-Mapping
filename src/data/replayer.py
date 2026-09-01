import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    timestamp: float
    points: np.ndarray  # N x 4 float32 (x, y, z, remission), sensor frame
    seq_id: str
    idx: int


DEFAULT_PREFETCH_WINDOW = 32


class SemanticKITTIReplayer:
    """Paces a SemanticKITTI velodyne sequence at its native ~10 Hz cadence.

    A background thread prefetches a bounded window of upcoming velodyne frames
    into memory (the ``prefetch_window``). ``get()`` still paces the stream at
    the native cadence and hands frames to the consumer one-by-one in real time
    ; the prefetch only hides the disk read latency so a slow disk ("disk=NNms"
    spikes) never starves the stream or causes frame drops. Frames are never
    delivered early; they just sit in the buffer until ``get()`` releases them.
    """

    def __init__(
        self,
        seq_dir: str | Path,
        playback_speed: float = 1.0,
        loop: bool = False,
        native_hz: float = 10.0,
        start_idx: int = 0,
        prefetch_window: int = DEFAULT_PREFETCH_WINDOW,
    ):
        self.seq_dir = Path(seq_dir)
        self.seq_id = self.seq_dir.name
        self.velodyne_dir = self.seq_dir / "velodyne"
        self.speed = max(0.05, playback_speed)
        self.loop = loop
        self.native_hz = native_hz
        self.period = 1.0 / native_hz
        self.bin_paths = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.bin_paths:
            raise FileNotFoundError(f"No .bin files in {self.velodyne_dir}")
        self.prefetch_window = int(prefetch_window)
        self.i = start_idx
        self._t_next = 0.0
        self._started = False

        # prefetch state (guarded by _cv)
        self._cv = threading.Condition()
        self._buf: deque = deque()
        self._next_load = start_idx
        self._eof = False
        self._stop = False
        self._pf = threading.Thread(target=self._prefetch_loop, daemon=True,
                                    name="replayer-prefetch")
        self._pf.start()
        with self._cv:
            self._prime_locked()

    # io
    def _read_disk(self, idx: int) -> Frame:
        pts = np.fromfile(str(self.bin_paths[idx]), dtype=np.float32).reshape(-1, 4)
        return Frame(timestamp=idx / self.native_hz, points=pts, seq_id=self.seq_id, idx=idx)

    def read(self, idx: int) -> tuple[Frame, bool]:
        """Read frame at absolute index idx (no pacing, bypasses prefetch).
        Returns (frame, ok). Used for seek previews / direct indexed reads."""
        if idx >= len(self.bin_paths):
            return None, False
        return self._read_disk(idx), True

    # pacing
    def _reset_time(self):
        self._t_next = time.perf_counter()
        self._started = False

    def get(self, timeout: float = 10.0) -> Frame | None:
        """Wall-clock paced frame fetch. Returns None on stop / timeout / EOF.

        Pacing is enforced exactly as before; the only difference is that the
        frame data comes from the prefetch buffer instead of a synchronous read,
        so a slow disk can't stall playback below the native cadence.
        """
        if not self._started:
            self._reset_time()
            self._started = True

        if self._t_next <= time.perf_counter():
            # advance one period and deliver immediately (already due)
            self._t_next += self.period / self.speed
            return self._pop_next()
        delay = self._t_next - time.perf_counter()
        if delay > timeout:
            return None
        time.sleep(max(0.0, delay))
        self._t_next += self.period / self.speed
        return self._pop_next()

    def _pop_next(self) -> Frame | None:
        """Deliver the next paced frame. Prefers the (already-loaded) prefetch
        buffer; falls back to a direct synchronous read only when the buffer is
        momentarily empty (cold start / end). Never blocks the consumer."""
        n = len(self.bin_paths)
        with self._cv:
            if self._buf:
                f = self._buf.popleft()
                self.i = f.idx + 1
                self._cv.notify_all()  # let prefetch refill the freed slot
                return f

        # buffer empty -> direct read (with loop wrap + pacing reset)
        if self.i >= n:
            if not self.loop:
                return None
            self.i = 0
            self._reset_time()
        f = self._read_disk(self.i)
        self.i += 1
        with self._cv:
            # realign the prefetch cursor so we never double-deliver a frame
            self._buf.clear()
            self._next_load = self.i
            self._eof = False
            self._cv.notify_all()
        return f

    # prefetch
    def _prime_locked(self):
        """(Re)start prefetch from the current cursor. Caller holds self._cv."""
        self._buf.clear()
        self._next_load = self.i
        self._eof = False
        self._cv.notify_all()

    def _prefetch_loop(self):
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._stop
                    or (len(self._buf) < self.prefetch_window and not self._eof)
                )
                if self._stop:
                    return
                n = len(self.bin_paths)
                while len(self._buf) < self.prefetch_window:
                    if self._next_load >= n:
                        if not self.loop:
                            self._eof = True
                            break
                        self._next_load = 0
                    try:
                        f = self._read_disk(self._next_load)
                    except Exception:
                        self._eof = True
                        break
                    self._next_load += 1
                    self._buf.append(f)
                    if self._stop:
                        return
            # loop: full buffer (wait for a consume) or eof; re-check predicate

    # control
    def restart(self):
        self.i = 0
        with self._cv:
            self._prime_locked()
        self._started = False

    def seek(self, idx: int):
        """Jump to an absolute frame index (next get() returns immediately)."""
        n = len(self.bin_paths)
        if n == 0:
            return
        self.i = max(0, min(n - 1, int(idx)))
        with self._cv:
            self._prime_locked()
        self._started = False

    def close(self):
        """Stop the prefetch thread (idempotent)."""
        with self._cv:
            if self._stop:
                return
            self._stop = True
            self._cv.notify_all()
        self._pf.join(timeout=2.0)

    def __len__(self):
        return len(self.bin_paths)
