"""Tests for the replayer's bounded multi-frame prefetch buffer.

The prefetch loads a window of upcoming frames into memory ahead of time but
``get()`` must still pace delivery one frame at a time at the native cadence —
frames are never handed to the consumer early. The buffer size must stay bounded
under a slow consumer, and seek/restart must invalidate stale buffered frames.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from src.data.replayer import SemanticKITTIReplayer


@pytest.fixture
def synth_seq(tmp_path):
    vel = tmp_path / "velodyne"
    vel.mkdir()
    rng = np.random.default_rng(42)
    for i in range(8):
        rng.standard_normal((50, 4)).astype(np.float32).tofile(str(vel / f"{i:06d}.bin"))
    return tmp_path


class TestPrefetchBuffer:
    def test_prefetches_ahead_and_delivers_in_order(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False, prefetch_window=4)
        try:
            time.sleep(0.3)  # let the thread warm the window
            with rep._cv:
                assert len(rep._buf) == 4  # fully warmed
            idxs = [rep.get(timeout=1.0).idx for _ in range(8)]
            assert idxs == list(range(8))
        finally:
            rep.close()

    def test_buffer_stays_bounded_under_slow_consumer(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=10.0, loop=False, prefetch_window=3)
        try:
            time.sleep(0.3)
            maxbuf = 0
            for _ in range(5):  # stay < total frames; non-loop would EOF otherwise
                f = rep.get(timeout=2.0)
                assert f is not None
                with rep._cv:
                    maxbuf = max(maxbuf, len(rep._buf))
                time.sleep(0.02)  # consumer slower than the prefetch can fill
            assert maxbuf <= 3
        finally:
            rep.close()

    def test_natural_cadence_is_not_paced_ahead(self, synth_seq):
        """Frames must be delivered at the native cadence, not as fast as the
        buffer allows. At speed 1.0 / 20 Hz (~50ms period) consuming 5 frames
        must take at least ~4 x 50ms — delivery is wall-clock paced."""
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=1.0, loop=False, native_hz=20.0, prefetch_window=4)
        try:
            time.sleep(0.3)
            t0 = time.perf_counter()
            for _ in range(5):
                rep.get(timeout=2.0)
            dt = time.perf_counter() - t0
            assert dt >= 4 * (1.0 / 20.0) * 0.8  # ~4 intervals, generous slack
        finally:
            rep.close()

    def test_seek_invalidates_stale_buffer(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False, prefetch_window=4)
        try:
            rep.seek(3)
            time.sleep(0.3)
            first = rep.get(timeout=1.0)
            assert first is not None and first.idx == 3
        finally:
            rep.close()

    def test_loop_wraps_and_never_ends(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=True, prefetch_window=3)
        try:
            # consume past the end and around the loop; must never return None
            seen = [rep.get(timeout=1.0).idx for _ in range(12)]
            assert len(seen) == 12
        finally:
            rep.close()

    def test_non_loop_returns_none_at_end(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False, prefetch_window=4)
        try:
            for _ in range(8):
                assert rep.get(timeout=1.0) is not None
            n = len(rep.bin_paths)
            assert rep.i >= n
            time.sleep(0.3)
            assert rep.get(timeout=0.1) is None
        finally:
            rep.close()