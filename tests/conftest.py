import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.data.label_mapping import load_class_mapping, load_bin_mapping


@pytest.fixture
def grid_cfg():
    return load_grid_config()


@pytest.fixture
def grid(grid_cfg):
    return LogPolarGrid(grid_cfg)


@pytest.fixture
def sample_points():
    rng = np.random.default_rng(42)
    n = 1000
    pts = np.zeros((n, 4), dtype=np.float32)
    r = rng.uniform(1.0, 50.0, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    pts[:, 0] = r * np.cos(theta)
    pts[:, 1] = r * np.sin(theta)
    pts[:, 2] = rng.uniform(-2.0, 3.0, n)
    pts[:, 3] = 1.0
    return pts


@pytest.fixture
def sample_labels(sample_points):
    n = sample_points.shape[0]
    rng = np.random.default_rng(42)
    return rng.integers(0, 4, size=n, dtype=np.uint8)


@pytest.fixture
def class_mapping():
    return load_class_mapping()


@pytest.fixture
def bin_mapping():
    return load_bin_mapping()
