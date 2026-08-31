import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    timestamp: float
    points: np.ndarray  # N x 4 float32 (x, y, z, remission), sensor frame
    seq_id: str
    idx: int


class SemanticKITTIReplayer:
    """Paces a SemanticKITTI velodyne sequence at its native ~10 Hz cadence."""

    def __init__(
        self,
        seq_dir: str | Path,
        playback_speed: float = 1.0,
        loop: bool = False,
        native_hz: float = 10.0,
        start_idx: int = 0,
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
        self.i = start_idx
        self._t_next = 0.0
        self._started = False

    def _reset_time(self):
        self._t_next = time.perf_counter()
        self._started = False

    def read(self, idx: int) -> tuple[Frame, bool]:
        """Read frame at absolute index idx (no pacing). Returns (frame, ok)."""
        if idx >= len(self.bin_paths):
            return None, False
        pts = np.fromfile(str(self.bin_paths[idx]), dtype=np.float32).reshape(-1, 4)
        return Frame(timestamp=idx / self.native_hz, points=pts, seq_id=self.seq_id, idx=idx), True

    def get(self, timeout: float = 10.0) -> Frame | None:
        """Wall-clock paced frame fetch. Returns None on stop / timeout."""
        if not self._started:
            self._reset_time()
            self._started = True

        # advance pacing time by one native period already passed?
        while self._t_next <= time.perf_counter():
            self._t_next += self.period / self.speed
            frame, ok = self.read(self.i)
            if not ok:
                if not self.loop:
                    return None
                self.i = 0
                self._reset_time()
                frame, ok = self.read(self.i)
                if not ok:
                    return None
            self.i += 1
            return frame

        # wait until pacing time
        delay = self._t_next - time.perf_counter()
        if delay > timeout:
            return None
        time.sleep(max(0.0, delay))
        self._t_next += self.period / self.speed
        frame, ok = self.read(self.i)
        if not ok:
            if not self.loop:
                return None
            self.i = 0
            self._reset_time()
            frame, ok = self.read(self.i)
            if not ok:
                return None
        self.i += 1
        return frame

    def restart(self):
        self.i = 0
        self._started = False

    def seek(self, idx: int):
        """Jump to an absolute frame index (next get() returns immediately)."""
        n = len(self.bin_paths)
        if n == 0:
            return
        self.i = max(0, min(n - 1, int(idx)))
        self._started = False

    def __len__(self):
        return len(self.bin_paths)