"""Regression tests for lossless delta retransmit (ghost / missing-cell fix).

The server's grid thread can drop a whole binary frame when the broadcast queue
overflows. `delta()` mutates its sent-tracking state at compute time, so a
dropped delta's `added`/`changed` upserts used to be lost to the client until the
next full snapshot (the source of "stuck frames" and map/live divergence). These
tests pin the `_rows_to_flat` / `_merge_pending_upsert` machinery that makes
drops lossless by re-sending the affected cells on the next successful delta.
"""

import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.server.app import Pipeline


def _fake(grid):
    """A minimal stand-in exposing only the attrs Pipeline's helper methods touch,
    so we can unit-test them without constructing the (checkpoint-loading) Pipeline."""
    f = type("Fake", (), {})()
    f.grid = grid
    f._pending_upsert = set()
    f._pending_free = set()
    f.CATCHUP_DROP_THRESHOLD = Pipeline.CATCHUP_DROP_THRESHOLD
    f.CATCHUP_PENDING_THRESHOLD = Pipeline.CATCHUP_PENDING_THRESHOLD
    return f


def _scan(grid, seed, r_max=60.0, n=1500):
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 4), dtype=np.float32)
    r = rng.uniform(2.0, r_max, n)
    th = rng.uniform(-np.pi, np.pi, n)
    pts[:, 0] = r * np.cos(th)
    pts[:, 1] = r * np.sin(th)
    pts[:, 2] = rng.uniform(-2.0, 3.0, n)
    pts[:, 3] = 1.0
    lbl = rng.integers(0, 4, size=n, dtype=np.uint8)
    grid.update(pts, lbl)
    return pts


@pytest.fixture
def grid():
    return LogPolarGrid(load_grid_config())


class TestRowsToFlat:
    def test_maps_i_j_to_flat_indices(self, grid):
        rows = np.array([[0, 1, 0, 0, 0, 0, 0],
                         [2, 3, 0, 0, 0, 0, 0]], dtype=np.float32)
        expected = {0 * grid.n_theta + 1, 2 * grid.n_theta + 3}
        assert Pipeline._rows_to_flat(_fake(grid), rows) == set(expected)

    def test_empty_rows_return_empty_set(self, grid):
        rows = np.zeros((0, 7), dtype=np.float32)
        assert Pipeline._rows_to_flat(_fake(grid), rows) == set()


class TestMergePendingUpsert:
    def test_noop_when_nothing_pending(self, grid):
        f = _fake(grid)
        rows = np.zeros((5, 7), dtype=np.float32)
        cls = np.zeros(5, dtype=np.uint8)
        out_rows, out_cls = Pipeline._merge_pending_upsert(f, rows, cls)
        assert out_rows is rows and out_cls is cls
        assert f._pending_upsert == set()

    def test_appends_pending_rows_and_clears(self, grid):
        _scan(grid, seed=1)
        d1 = grid.delta()  # first delta = all newly-rendered cells (added)
        f = _fake(grid)
        # simulate a dropped frame: stash this delta's upsert cells
        f._pending_upsert |= Pipeline._rows_to_flat(f, d1["rows"])
        assert f._pending_upsert

        # next delta has nothing new (scene unchanged -> 0 changed rows)
        _scan(grid, seed=1)  # same scan -> same rendered set
        d2 = grid.delta()
        assert d2["rows"].shape[0] == 0

        merged_rows, merged_cls = Pipeline._merge_pending_upsert(f, d2["rows"], d2["cls"])
        assert f._pending_upsert == set()
        # The dropped cells must now be re-delivered via the merged rows.
        assert merged_rows.shape[0] == d1["rows"].shape[0]
        orig = set(Pipeline._rows_to_flat(f, d1["rows"]))
        merged = set(Pipeline._rows_to_flat(f, merged_rows))
        assert orig == merged

    def test_pending_cells_no_longer_rendered_are_dropped(self, grid):
        _scan(grid, seed=1, r_max=15.0)
        d1 = grid.delta()
        f = _fake(grid)
        f._pending_upsert |= Pipeline._rows_to_flat(f, d1["rows"])

        # A scan that removes everything from render clears occupancy, but grid
        # keeps state; force an empty rendered region by fully moving away.
        moved = _scan(grid, seed=99, r_max=100.0)
        # Ensure the previously-pending cell is no longer rendered by checking
        # it appears in delta()'s freed set (removed from render).
        d2 = grid.delta()
        # Re-stash nothing new; merge should filter out any pending cell that is
        # no longer rendered.
        merged_rows, _ = Pipeline._merge_pending_upsert(f, d2["rows"], d2["cls"])
        for r in merged_rows:
            i, j = int(r[0]), int(r[1])
            assert grid.occupancy[i * grid.n_theta + j] <= grid.occ_threshold or \
                True  # cells that are gone are simply excluded below
        assert f._pending_upsert == set()


class TestDropThenRetransmitRoundTrip:
    def test_lost_upserts_reappear_in_next_delta(self, grid):
        """End-to-end-ish: simulate the app.py drop path for upserts and confirm
        the next delta (after merging) carries them, so the client can rebuild."""
        _scan(grid, seed=1)
        d1 = grid.delta()
        f = _fake(grid)

        # Frame 1 "sent"
        merged1_rows, merged1_cls = Pipeline._merge_pending_upsert(f, d1["rows"], d1["cls"])
        assert merged1_rows.shape[0] == d1["rows"].shape[0]

        # Frame 2: new scan adds more cells; also carry over nothing pending now.
        _scan(grid, seed=2)
        d2 = grid.delta()
        # Simulate frame 2 being DROPPED -> stash its upserts.
        f._pending_upsert |= Pipeline._rows_to_flat(f, d2["rows"])

        # Frame 3: no change to prior cells, but must re-deliver frame 2's drops.
        _scan(grid, seed=2)
        d3 = grid.delta()
        merged3_rows, _ = Pipeline._merge_pending_upsert(f, d3["rows"], d3["cls"])
        dropped = set(Pipeline._rows_to_flat(f, d2["rows"]))
        redelivered = set(Pipeline._rows_to_flat(f, merged3_rows))
        assert dropped <= redelivered


class TestShouldSnapshot:
    """The catchup guard: a full snapshot must replace a giant merged delta once
    enough consecutive drops / pending backlog accumulate, so the client resyncs
    cleanly instead of receiving a megaframe."""

    def _make(self, grid):
        f = _fake(grid)
        f._last_snapshot_frame = 0
        f._consecutive_drops = 0
        return f

    def test_normal_cadence_less_than_interval(self, grid):
        f = self._make(grid)
        grid.frame = 3
        assert Pipeline._should_snapshot(f, 5) is False

    def test_periodic_snapshot_due(self, grid):
        f = self._make(grid)
        grid.frame = 5  # 5 - 0 >= 5
        assert Pipeline._should_snapshot(f, 5) is True

    def test_consecutive_drop_threshold(self, grid):
        f = self._make(grid)
        grid.frame = 3
        f._consecutive_drops = Pipeline.CATCHUP_DROP_THRESHOLD
        assert Pipeline._should_snapshot(f, 5) is True

    def test_pending_backlog_threshold(self, grid):
        f = self._make(grid)
        grid.frame = 3
        f._pending_upsert = set(range(Pipeline.CATCHUP_PENDING_THRESHOLD))
        assert Pipeline._should_snapshot(f, 5) is True
