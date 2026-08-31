import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.replayer import SemanticKITTIReplayer


@pytest.fixture
def synth_seq(tmp_path):
    velodyne_dir = tmp_path / "velodyne"
    velodyne_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(5):
        pts = rng.standard_normal((100, 4)).astype(np.float32)
        pts.tofile(str(velodyne_dir / f"{i:06d}.bin"))
    return tmp_path


class TestReplayerSeek:
    def test_seek_updates_index(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False)
        assert len(rep) == 5
        rep.seek(3)
        assert rep.i == 3

    def test_seek_clamps_to_valid_range(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False)
        rep.seek(100)
        assert rep.i == 4
        rep.seek(-5)
        assert rep.i == 0

    def test_read_returns_correct_frame(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False)
        frame, ok = rep.read(2)
        assert ok
        assert frame.idx == 2
        assert frame.points.shape[1] == 4

    def test_read_beyond_end_returns_false(self, synth_seq):
        rep = SemanticKITTIReplayer(synth_seq, playback_speed=100.0, loop=False)
        _, ok = rep.read(10)
        assert not ok


class TestPipelineSeekReset:
    def test_seek_resets_grid_and_bumps_epoch(self, synth_seq):
        import os
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        os.environ["PC2D_DEVICE"] = "cpu"
        os.environ["PC2D_PRECISION"] = "fp32"

        from src.common.config import load_config, reset_cache
        from src.server.app import Pipeline

        reset_cache()
        cfg = load_config("pipeline")
        cfg["source"]["seq_dir"] = str(synth_seq)
        cfg["source"]["playback_speed"] = 100.0
        pl = Pipeline(cfg)
        assert pl.segmenter.device.type == "cpu"
        assert pl.epoch == 0

        pl.pause(False)
        import time
        time.sleep(2)
        pl.pause(True)
        time.sleep(0.5)

        frame_before = pl.grid.frame
        epoch_before = pl.epoch

        pl.seek(0)
        time.sleep(0.5)

        assert pl.grid.frame == 0
        assert pl.epoch == epoch_before + 1

        while not pl.messages.empty():
            pl.messages.get_nowait()

        pl.pause(True)
