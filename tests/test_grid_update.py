import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config


def _freed_ids(delta, grid):
    if delta["freed"].shape[0] == 0:
        return set()
    fi = delta["freed"][:, 0].astype(int)
    fj = delta["freed"][:, 1].astype(int)
    return set((fi * grid.n_theta + fj).tolist())


def _row_ids(delta, grid):
    if delta["rows"].shape[0] == 0:
        return set()
    ri = delta["rows"][:, 0].astype(int)
    rj = delta["rows"][:, 1].astype(int)
    return set((ri * grid.n_theta + rj).tolist())


class TestOccupancyPrecise:
    def test_occupancy_goes_free_instantly_when_not_hit(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        rendered = grid.rendered()
        hit_cell = np.nonzero(rendered)[0]
        assert hit_cell.size > 0

        other_pts = sample_points.copy()
        other_pts[:, 0] += 50.0
        grid.update(other_pts, sample_labels)

        # precise mode: instant free when not re-hit, no decay math at all
        assert not grid.rendered()[hit_cell[0]]
        assert grid.occupancy[hit_cell[0]] == pytest.approx(0.0, abs=1e-6)

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
        snap = grid.compute_snapshot()
        grid.commit_snapshot()
        prev_rendered = grid.rendered().copy()

        other_pts = sample_points.copy()
        other_pts[:, 0] += 80.0
        for _ in range(20):
            grid.update(other_pts, sample_labels)

        delta = grid.compute_delta()
        for cid in _freed_ids(delta, grid):
            assert prev_rendered[cid]

    def test_delta_rows_are_currently_rendered(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        snap = grid.compute_snapshot()
        grid.commit_snapshot()
        grid.update(sample_points, sample_labels)
        delta = grid.compute_delta()
        current_rendered = grid.rendered()
        for cid in _row_ids(delta, grid):
            assert current_rendered[cid]

    def test_snapshot_marks_all_sent(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        snap = grid.compute_snapshot()
        grid.commit_snapshot()
        grid.update(sample_points, sample_labels)
        delta = grid.compute_delta()
        # nothing freed (same points), and only height/class changed rows re-sent
        assert delta["freed"].shape[0] == 0
        assert delta["frame"] == 2

    def test_compute_is_pure_repeatable(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        grid.update(sample_points, sample_labels)
        d1 = grid.compute_delta()
        d2 = grid.compute_delta()
        # no mutation until commit: identical results on repeat, and sent-state untouched
        assert np.array_equal(d1["rows"], d2["rows"])
        assert np.array_equal(d1["freed"], d2["freed"])
        assert (grid._sent_zmax == 0).all()
        assert (grid._prev_rendered == False).all()

    def test_dropped_delta_is_recomputed_from_last_commit(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        snap = grid.compute_snapshot()
        grid.commit_snapshot()
        committed = grid.rendered().copy()

        # a "dropped" frame: computed but committed never happened
        other_pts = sample_points.copy()
        other_pts[:, 0] += 80.0
        grid.update(other_pts, sample_labels)
        grid.compute_delta()

        # yet another update; the next delta must still carry every change since
        # the last *committed* state (no pending bookkeeping required)
        moved2 = sample_points.copy()
        moved2[:, 0] -= 40.0
        grid.update(moved2, sample_labels)
        next_delta = grid.compute_delta()

        # every cell that left the committed rendered mask must be freed now
        left_mask = grid._prev_rendered & ~grid.rendered()
        assert _freed_ids(next_delta, grid) == set(np.nonzero(left_mask)[0].tolist())
        # every cell that is rendered now but wasn't at the last commit must be
        # (re)added now, even though it was silently present during the drop
        entered_mask = grid.rendered() & ~grid._prev_rendered
        assert _row_ids(next_delta, grid) >= set(np.nonzero(entered_mask)[0].tolist())
        assert next_delta["frame"] == 3


class TestGridReset:
    def test_reset_clears_state(self, grid, sample_points, sample_labels):
        grid.update(sample_points, sample_labels)
        assert grid.frame > 0
        grid.reset()
        assert grid.frame == 0
        assert (grid.occupancy == 0).all()
        assert (grid.dominant_class == 255).all()
        assert (grid._prev_rendered == False).all()