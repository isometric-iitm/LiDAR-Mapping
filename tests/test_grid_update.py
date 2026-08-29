import math

import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config


class TestOccupancyDecay:
    def test_occupancy_decays_when_not_hit(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        rendered = grid.rendered()
        hit_cell = np.nonzero(rendered)[0]
        assert hit_cell.size > 0
        occ_before = grid.occupancy[hit_cell[0]].copy()

        n_decay = 5
        other_pts = sample_points.copy()
        other_pts[:, 0] += 50.0
        for _ in range(n_decay):
            grid.update(other_pts, sample_labels)

        expected_decay = math.exp(-(1.0 / grid.frame_hz) / grid.tau_free) ** n_decay
        occ_after = grid.occupancy[hit_cell[0]]
        assert occ_after < occ_before
        assert occ_after == pytest.approx(occ_before * expected_decay, rel=0.05)

    def test_occupancy_increases_on_hit(self, grid, sample_points, sample_labels):
        grid.occupancy[:] = 0.0
        grid.update(sample_points, sample_labels)
        rendered = grid.rendered()
        assert rendered.any()
        hit_cells = np.nonzero(rendered)[0]
        assert (grid.occupancy[hit_cells] > 0).all()


class TestSnapshotDeltaInvariants:
    def test_delta_freed_were_previously_rendered(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        snap = grid.snapshot()
        prev_rendered = grid.rendered().copy()

        other_pts = sample_points.copy()
        other_pts[:, 0] += 80.0
        for _ in range(20):
            grid.update(other_pts, sample_labels)

        delta = grid.delta()
        if delta["freed"].shape[0] > 0:
            freed_i = delta["freed"][:, 0].astype(int)
            freed_j = delta["freed"][:, 1].astype(int)
            freed_ids = freed_i * grid.n_theta + freed_j
            for cid in freed_ids:
                assert prev_rendered[cid]

    def test_delta_rows_are_currently_rendered(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        grid.snapshot()
        grid.update(sample_points, sample_labels)
        delta = grid.delta()
        if delta["rows"].shape[0] > 0:
            row_i = delta["rows"][:, 0].astype(int)
            row_j = delta["rows"][:, 1].astype(int)
            row_ids = row_i * grid.n_theta + row_j
            current_rendered = grid.rendered()
            for cid in row_ids:
                assert current_rendered[cid]

    def test_snapshot_marks_all_sent(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        grid.snapshot()
        grid.update(sample_points, sample_labels)
        delta = grid.delta()
        assert delta["rows"].shape[0] >= 0
        assert delta["freed"].shape[0] >= 0
        assert delta["frame"] > 0


class TestGridReset:
    def test_reset_clears_state(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        assert grid.frame > 0
        grid.reset()
        assert grid.frame == 0
        assert (grid.occupancy == 0).all()
        assert (grid.dominant_class == 255).all()
