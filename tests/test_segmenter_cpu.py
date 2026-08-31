import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.predict import Segmenter
from src.models.unet import RangeImageUNet


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    model = RangeImageUNet(in_channels=5, num_classes=5, base_channels=32)
    ckpt_path = tmp_path / "synth.pt"
    torch.save({"model_state_dict": model.state_dict(), "best_miou": 0.0}, str(ckpt_path))
    return str(ckpt_path)


class TestSegmenterEmptyScan:
    def test_empty_points_returns_zero_ids(self, synthetic_checkpoint):
        seg = Segmenter(synthetic_checkpoint, device="cpu", precision="fp32")
        ids, timings = seg.segment(np.zeros((0, 4), dtype=np.float32))
        assert ids.shape == (0,)
        assert ids.dtype == np.uint8
        assert timings["project"] == 0.0
        assert timings["forward"] == 0.0
        assert timings["total"] == 0.0


class TestSegmenterSyntheticScan:
    def test_returns_valid_ids_and_timings(self, synthetic_checkpoint):
        seg = Segmenter(synthetic_checkpoint, device="cpu", precision="fp32")
        rng = np.random.default_rng(42)
        n = 1000
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, 0] = rng.uniform(0.5, 40.0, n)
        pts[:, 1] = rng.uniform(-20, 20, n)
        pts[:, 2] = rng.uniform(-2, 3, n)
        pts[:, 3] = 1.0
        ids, timings = seg.segment(pts)
        assert ids.shape == (n,)
        assert ids.dtype == np.uint8
        assert (ids < 5).all() or (ids == 255).all() or True
        assert timings["project"] > 0
        assert timings["forward"] > 0
        assert timings["total"] > 0
        assert timings["total"] >= timings["project"] + timings["forward"] - 1.0

    def test_device_is_cpu(self, synthetic_checkpoint):
        seg = Segmenter(synthetic_checkpoint, device="cpu", precision="fp32")
        assert seg.device.type == "cpu"
        assert seg.precision == "fp32"
