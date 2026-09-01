"""Regression tests for lossless pure-delta delivery (ghost / missing-cell fix).

The old design mutated sent-tracking at delta()-compute time, so a whole frame
dropped by the broadcast queue permanently lost its upserts to the client until
the next full snapshot. The new design separates pure computation
(``compute_delta``/``compute_snapshot`) from sent-state mutation
(``commit_delta``/``commit_snapshot``); commit only happens after a frame is
actually queued, so a drop is never lossy and needs no pending/retransmit state.
"""

import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.server.app import Pipeline


def _fake(grid):
    f = type("Fake", (), {})()
    f.grid = grid
    f._last_snapshot_frame = 0
    f._consecutive_drops = 0
    f.CATCHUP_DROP_THRESHOLD = Pipeline.CATCHUP_DROP_THRESHOLD
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


class TestShouldSnapshot:
    """The catchup guard: once enough consecutive frames drop, a full snapshot
    replaces the (now giant) delta so the client resyncs cheaply."""

    def test_normal_cadence_less_than_interval(self, grid):
        f = _fake(grid)
        grid.frame = 3
        assert Pipeline._should_snapshot(f, 5) is False

    def test_periodic_snapshot_due(self, grid):
        f = _fake(grid)
        grid.frame = 5  # 5 - 0 >= 5
        assert Pipeline._should_snapshot(f, 5) is True

    def test_consecutive_drop_threshold(self, grid):
        f = _fake(grid)
        grid.frame = 3
        f._consecutive_drops = Pipeline.CATCHUP_DROP_THRESHOLD
        assert Pipeline._should_snapshot(f, 5) is True


class TestLosslessDrops:
    def test_compute_without_commit_is_pure(self, grid):
        _scan(grid, seed=1)
        d1 = grid.compute_delta()
        d2 = grid.compute_delta()
        assert np.array_equal(d1["rows"], d2["rows"])
        assert np.array_equal(d1["freed"], d2["freed"])
        # sent-tracking untouched by computation alone
        assert not grid._prev_rendered.any()
        assert (grid._sent_zmax == 0).all()

    def test_dropped_delta_is_recomputed_from_last_commit(self, grid):
        _scan(grid, seed=1)
        grid.commit_snapshot()

        # frame computed but dropped (no commit)
        _scan(grid, seed=2)
        grid.compute_delta()

        # the next frame is a fresh scene; the next delta must cover every
        # change since the last committed state, including the dropped frame's
        _scan(grid, seed=3, r_max=80.0)
        d = grid.compute_delta()
        freed_flat = {(int(i), int(j)) for i, j in d["freed"]}
        rows_flat = {(int(r[0]), int(r[1])) for r in d["rows"]}
        current = grid.rendered()
        for cid in np.nonzero(current & ~grid._prev_rendered)[0]:
            assert (cid // grid.n_theta, cid % grid.n_theta) in rows_flat
        for cid in np.nonzero(grid._prev_rendered & ~current)[0]:
            assert (cid // grid.n_theta, cid % grid.n_theta) in freed_flat

    def test_commit_only_after_delivery_matches_client_state(self, grid):
        """Simulate the app's broadcast loop: compute, maybe commit, apply on the
        client only for committed frames. The client's rendered set must converge
        to the grid's rendered set (no ghosts, no missing cells)."""
        rng = np.random.default_rng(0)
        render = {}  # flat cell id -> (z_max, cls) as the client sees it
        snap_iv = 6
        last_snap = 0
        for t in range(30):
            _scan(grid, seed=int(rng.integers(0, 9999)))
            f = _fake(grid)
            f._last_snapshot_frame = last_snap
            if Pipeline._should_snapshot(f, snap_iv):
                computed = grid.compute_snapshot()
                is_snap = True
            else:
                computed = grid.compute_delta()
                is_snap = False

            if t % 7 == 0:
                # dropped: no commit, client unchanged
                continue

            rows = computed["rows"]
            cls = computed["cls"]
            if is_snap:
                # a snapshot is authoritative: the client fully resyncs (the
                # server lists every rendered cell and no freed rows)
                render.clear()
            else:
                for i, j in computed["freed"]:
                    render.pop(int(i) * grid.n_theta + int(j), None)
            for k, r in enumerate(rows):
                i, j = int(r[0]), int(r[1])
                render[i * grid.n_theta + j] = (r[3], cls[k])
            if is_snap:
                grid.commit_snapshot()
                last_snap = grid.frame
            else:
                grid.commit_delta(computed)

        # final forced snapshot so every rendered cell is delivered
        grid.compute_snapshot()
        target = set(np.nonzero(grid.rendered())[0].tolist())
        grid.commit_snapshot()
        assert set(render) == target