import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.grid_engine.traversability import compute_traversability, traversability_tier


@pytest.fixture
def grid():
    return LogPolarGrid(load_grid_config())


def _populate_flat(grid, cls=0):
    rng = np.random.default_rng(42)
    n = 500
    pts = np.zeros((n, 4), dtype=np.float32)
    r = rng.uniform(1.0, 50.0, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    pts[:, 0] = r * np.cos(theta)
    pts[:, 1] = r * np.sin(theta)
    pts[:, 2] = 0.5
    pts[:, 3] = 1.0
    labels = np.full(n, cls, dtype=np.uint8)
    grid.update(pts, labels)


class TestComputeTraversability:
    def test_flat_drivable_is_high(self, grid):
        _populate_flat(grid, cls=0)
        trav = compute_traversability(
            grid._z_min_state, grid._z_max_state, grid.z_mean,
            grid.dominant_class, grid.occupancy,
            grid.n_rings, grid.n_theta, grid.ring_widths,
        )
        rendered = grid.rendered()
        assert trav[rendered].mean() > 0.7

    def test_steep_is_low(self, grid):
        rng = np.random.default_rng(42)
        n = 500
        pts = np.zeros((n, 4), dtype=np.float32)
        r = rng.uniform(1.0, 50.0, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        pts[:, 0] = r * np.cos(theta)
        pts[:, 1] = r * np.sin(theta)
        pts[:, 2] = rng.uniform(0, 5.0, n)
        pts[:, 3] = 1.0
        labels = np.full(n, 2, dtype=np.uint8)
        grid.update(pts, labels)
        trav = compute_traversability(
            grid._z_min_state, grid._z_max_state, grid.z_mean,
            grid.dominant_class, grid.occupancy,
            grid.n_rings, grid.n_theta, grid.ring_widths,
        )
        rendered = grid.rendered()
        assert trav[rendered].mean() < 0.6

    def test_class_scores(self, grid):
        _populate_flat(grid, cls=0)
        trav_d = compute_traversability(
            grid._z_min_state, grid._z_max_state, grid.z_mean,
            grid.dominant_class, grid.occupancy,
            grid.n_rings, grid.n_theta, grid.ring_widths,
            class_scores=[1.0, 0.6, 0.2, 0.1],
        )
        grid2 = LogPolarGrid(load_grid_config())
        _populate_flat(grid2, cls=2)
        trav_s = compute_traversability(
            grid2._z_min_state, grid2._z_max_state, grid2.z_mean,
            grid2.dominant_class, grid2.occupancy,
            grid2.n_rings, grid2.n_theta, grid2.ring_widths,
            class_scores=[1.0, 0.6, 0.2, 0.1],
        )
        r1 = grid.rendered()
        r2 = grid2.rendered()
        assert trav_d[r1].mean() > trav_s[r2].mean()

    def test_output_range(self, grid):
        _populate_flat(grid)
        trav = compute_traversability(
            grid._z_min_state, grid._z_max_state, grid.z_mean,
            grid.dominant_class, grid.occupancy,
            grid.n_rings, grid.n_theta, grid.ring_widths,
        )
        assert trav.min() >= 0.0
        assert trav.max() <= 1.0


class TestTraversabilityTier:
    def test_tiers(self):
        trav = np.array([0.8, 0.6, 0.3, 0.0, 1.0], dtype=np.float32)
        tier = traversability_tier(trav)
        assert tier[0] == 0  # drivable
        assert tier[1] == 1  # caution
        assert tier[2] == 2  # blocked
        assert tier[3] == 2  # blocked
        assert tier[4] == 0  # drivable

    def test_grid_integration(self, grid):
        _populate_flat(grid, cls=0)
        assert grid.traversability is not None
        assert grid.traversability.dtype == np.float32
        rendered = grid.rendered()
        assert grid.traversability[rendered].mean() > 0.5
