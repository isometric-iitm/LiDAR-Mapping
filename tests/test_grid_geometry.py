import math

import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config


class TestRingIndex:
    def test_against_cumulative_sum_boundaries(self, grid):
        edges = grid._ring_edges
        for i in range(min(10, grid.n_rings)):
            r_lo = edges[i] + 1e-6
            r_hi = edges[i + 1] - 1e-6
            assert grid.ring_index(np.array([r_lo]))[0] == i
            assert grid.ring_index(np.array([r_hi]))[0] == i

    def test_first_ring(self, grid):
        assert grid.ring_index(np.array([grid.r_min + 0.001]))[0] == 0

    def test_last_ring(self, grid):
        r_near_max = grid._ring_edges[-2] + 1e-6
        idx = grid.ring_index(np.array([r_near_max]))[0]
        assert idx == grid.n_rings - 1

    def test_multiple_values(self, grid):
        r = np.array([0.51, 1.0, 10.0, 50.0, 99.0])
        idx = grid.ring_index(r)
        assert idx.shape == r.shape
        assert (idx >= 0).all()
        assert (idx < grid.n_rings).all()

    def test_hand_computed_ring0_width(self, grid):
        assert grid.ring_widths[0] == pytest.approx(grid.dr_0, rel=1e-10)
        assert grid.ring_widths[1] == pytest.approx(grid.dr_0 * grid.alpha, rel=1e-10)


class TestSectorIndex:
    def test_boundary_minus_pi(self, grid):
        assert grid.sector_index(np.array([-np.pi]))[0] == 0

    def test_boundary_pi(self, grid):
        assert grid.sector_index(np.array([np.pi - 1e-6]))[0] == grid.n_theta - 1

    def test_zero_angle(self, grid):
        assert grid.sector_index(np.array([0.0]))[0] == grid.n_theta // 2

    def test_wrap_around(self, grid):
        idx_minus = grid.sector_index(np.array([-np.pi + 1e-6]))[0]
        idx_plus = grid.sector_index(np.array([np.pi - 1e-6]))[0]
        assert idx_minus == 0
        assert idx_plus == grid.n_theta - 1

    def test_all_valid(self, grid):
        theta = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
        idx = grid.sector_index(theta)
        assert (idx >= 0).all()
        assert (idx < grid.n_theta).all()


class TestNRingsAndNCells:
    def test_n_rings_at_least_90(self, grid):
        assert grid.n_rings >= 90

    def test_n_cells_equals_rings_times_theta(self, grid):
        assert grid.n_cells == grid.n_rings * grid.n_theta

    def test_n_cells_around_65k(self, grid):
        assert 50000 < grid.n_cells < 100000

    def test_derived_from_config(self, grid_cfg):
        g = LogPolarGrid(grid_cfg)
        cumulative = 0.0
        i = 0
        while cumulative < (g.r_max - g.r_min):
            cumulative += g.dr_0 * (g.alpha ** i)
            i += 1
        assert g.n_rings == i


class TestMemoryReport:
    def test_grid_kb_under_5000(self, grid):
        mem = grid.memory_report()
        assert mem["grid_kb"] < 5000

    def test_compression_ratio_over_100(self, grid):
        mem = grid.memory_report()
        assert mem["compression_ratio"] > 100

    def test_ring_width_near(self, grid):
        mem = grid.memory_report()
        assert mem["ring_width_near_m"] == pytest.approx(0.05, rel=1e-6)

    def test_ring_width_far_reasonable(self, grid):
        mem = grid.memory_report()
        assert 0.5 < mem["ring_width_far_m"] < 10.0

    def test_n_cells_matches(self, grid):
        mem = grid.memory_report()
        assert mem["n_cells"] == grid.n_cells


class TestCellIdsFromXY:
    def test_round_trip_known_point(self, grid):
        r = 10.0
        theta = 0.0
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        cell_id, valid = grid.cell_ids_from_xy(np.array([x]), np.array([y]))
        assert valid[0]
        ring = cell_id[0] // grid.n_theta
        sector = cell_id[0] % grid.n_theta
        assert ring == grid.ring_index(np.array([r]))[0]
        assert sector == grid.sector_index(np.array([theta]))[0]

    def test_multiple_points(self, grid, sample_points):
        ids, valid = grid.cell_ids_from_xy(sample_points[:, 0], sample_points[:, 1])
        assert ids.shape == (sample_points.shape[0],)
        assert valid.shape == (sample_points.shape[0],)
        assert (ids[valid] >= 0).all()
        assert (ids[valid] < grid.n_cells).all()

    def test_point_below_r_min_invalid(self, grid):
        x = np.array([0.3])
        y = np.array([0.0])
        _, valid = grid.cell_ids_from_xy(x, y)
        assert not valid[0]

    def test_point_beyond_r_max_invalid(self, grid):
        x = np.array([101.0])
        y = np.array([0.0])
        _, valid = grid.cell_ids_from_xy(x, y)
        assert not valid[0]
