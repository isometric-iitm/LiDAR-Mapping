import numpy as np
import pytest

from src.grid_engine.jit_reduce import numpy_reduce_segments, reduce_segments


def _assert_same(uniq_a, zmin_a, zmax_a, zsum_a, cnt_a, clsc_a, uniq_b, zmin_b, zmax_b, zsum_b, cnt_b, clsc_b):
    np.testing.assert_array_equal(uniq_a, uniq_b)
    np.testing.assert_array_equal(cnt_a, cnt_b)
    np.testing.assert_allclose(zmin_a, zmin_b, rtol=1e-6)
    np.testing.assert_allclose(zmax_a, zmax_b, rtol=1e-6)
    np.testing.assert_allclose(zsum_a, zsum_b, rtol=1e-9)
    np.testing.assert_array_equal(clsc_a, clsc_b)


def test_reduce_matches_numpy_random():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(1, 5000))
        n_cls = 4
        cells = rng.integers(0, 300, size=n)
        order = np.argsort(cells, kind="stable")
        cells = cells[order]
        z = rng.uniform(-3, 40, size=n).astype(np.float32)
        cls = rng.integers(0, n_cls, size=n).astype(np.uint8)
        got = reduce_segments(cells, z, cls, n_cls)
        ref = numpy_reduce_segments(cells, z, cls, n_cls)
        _assert_same(*got, *ref)


def test_reduce_single_point():
    cells = np.array([7], dtype=np.int64)
    z = np.array([1.5], dtype=np.float32)
    cls = np.array([2], dtype=np.uint8)
    got = reduce_segments(cells, z, cls, 4)
    ref = numpy_reduce_segments(cells, z, cls, 4)
    _assert_same(*got, *ref)


def test_reduce_single_segment_many_points():
    cells = np.zeros(100, dtype=np.int64)
    z = np.arange(100, dtype=np.float32)
    cls = (np.arange(100) % 4).astype(np.uint8)
    got = reduce_segments(cells, z, cls, 4)
    ref = numpy_reduce_segments(cells, z, cls, 4)
    _assert_same(*got, *ref)


def test_reduce_class_tally_correct():
    cells = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    z = np.array([1, 2, 3, 10, 20, 5], dtype=np.float32)
    cls = np.array([3, 3, 1, 0, 2, 2], dtype=np.uint8)
    uniq, zmin, zmax, zsum, cnt, clsc = reduce_segments(cells, z, cls, 4)
    np.testing.assert_array_equal(uniq, [0, 1, 2])
    np.testing.assert_array_equal(cnt, [3, 2, 1])
    np.testing.assert_array_equal(clsc, [[0, 1, 0, 2], [1, 0, 1, 0], [0, 0, 1, 0]])
    np.testing.assert_allclose(zmin, [1, 10, 5])
    np.testing.assert_allclose(zmax, [3, 20, 5])
    np.testing.assert_allclose(zsum, [6, 30, 5])
