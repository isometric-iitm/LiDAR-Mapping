"""numba-accelerated per-cell scatter-reduce for LogPolarGrid.update().

This replaces a chain of numpy passes (stable argsort -> reduceat for
min/max/sum, plus a bincount for per-segment class counts) with ONE fused,
single-threaded pass over the cell-sorted points. It is numerically identical
to the vectorized numpy path it supersedes:

  - cells : flat cell indices (i*n_theta+j), sorted ascending
  - z     : heights aligned with `cells`
  - cls   : class labels aligned with `cells`

Outputs (first `n_uniq` rows of each preallocated buffer are valid):

  uniq        int64   the unique cells, ascending (== reduceat 'firsts')
  n_cell      int64   points per unique cell            (== seg_count)
  zmin/zmax   float32 per-cell min/max z                (== reduceat)
  zsum        float64 per-cell running sum of z         (== reduceat)
  counts      int32   per-(cell,class) tallies          (== bincount)

Equivalence is asserted by tests/test_jit_reduce.py against the pure numpy
implementation, so a numba compile/link failure can never silently corrupt the
grid state.

.. note:: numba is an OPTIONAL accelerator. If import or compilation fails the
   caller falls back to the existing numpy vectorized path; the grid is never
   left in a half-updated state (all buffers are fully written before use).
"""

import numpy as np

try:
    from numba import njit

    _NUMBA_OK = True
except Exception:  # pragma: no cover - numba optional
    njit = None
    _NUMBA_OK = False


if _NUMBA_OK:

    @njit(cache=True, nogil=True)
    def _reduce_segments(cells, z, cls, uniq, zmin, zmax, zsum, n_cell, counts, n_cls):
        """One fused pass over cell-sorted points. Returns the number of unique
        cells `u+1`. `cells` MUST be non-empty."""
        cur = cells[0]
        uniq[0] = cur
        zmin[0] = z[0]
        zmax[0] = z[0]
        zsum[0] = z[0]
        n_cell[0] = 1
        counts[0 * n_cls + cls[0]] = 1
        u = 0
        for k in range(1, cells.shape[0]):
            c = cells[k]
            if c != cur:
                u += 1
                cur = c
                uniq[u] = c
                zmin[u] = z[k]
                zmax[u] = z[k]
                zsum[u] = z[k]
                n_cell[u] = 1
                counts[u * n_cls + cls[k]] = 1
            else:
                zk = z[k]
                if zk < zmin[u]:
                    zmin[u] = zk
                if zk > zmax[u]:
                    zmax[u] = zk
                zsum[u] += zk
                n_cell[u] += 1
                counts[u * n_cls + cls[k]] += 1
        return u + 1


def reduce_segments(cells, z, cls, n_classes):
    """JIT-accelerated drop-in for the numpy scatter-reduce in update().

    Returns (uniq, zmin, zmax, zsum, seg_count, cls_counts) where
      uniq        int64[m] unique cells (ascending)
      zmin/zmax   float32[m]
      zsum        float64[m]
      seg_count   int64[m]
      cls_counts  int32[m, n_classes]
    Raises any numba error so the caller can fall back to numpy.
    """
    n = cells.shape[0]
    uniq = np.empty(n, dtype=np.int64)
    zmin = np.empty(n, dtype=np.float32)
    zmax = np.empty(n, dtype=np.float32)
    zsum = np.empty(n, dtype=np.float64)
    n_cell = np.empty(n, dtype=np.int64)
    counts = np.zeros(n * n_classes, dtype=np.int32)

    # Allocate bufffers, then run the fused kernel.
    n_uniq = _reduce_segments(cells, z, cls, uniq, zmin, zmax, zsum, n_cell, counts, n_classes)
    n_uniq = int(n_uniq)

    # Kernel wrote strictly-increasing `uniq` in its filled prefix; return
    # views limited to the valid prefix to match numpy semantics.
    return (
        uniq[:n_uniq].copy(),
        zmin[:n_uniq].copy(),
        zmax[:n_uniq].copy(),
        zsum[:n_uniq].copy(),
        n_cell[:n_uniq].copy(),
        counts[: n_uniq * n_classes].reshape(n_uniq, n_classes),
    )


# ---- pure-numpy reference (used as fallback + in tests) ----
def numpy_reduce_segments(cells, z, cls, n_classes):
    """Vectorized reference identical to the JIT kernel's output."""
    n = cells.shape[0]
    order = np.argsort(cells, kind="stable")
    csort = cells[order]
    zsort = z[order]
    neq = np.concatenate(([True], csort[1:] != csort[:-1]))
    firsts = np.flatnonzero(neq)
    uniq = csort[firsts]
    ends = np.concatenate((firsts[1:], [n]))
    seg_count = ends - firsts
    zmin_c = np.minimum.reduceat(zsort, firsts)
    zmax_c = np.maximum.reduceat(zsort, firsts)
    zsum_c = np.add.reduceat(zsort.astype(np.float64), firsts)
    cls_c = cls[order]
    local = np.searchsorted(uniq, csort)
    counts = np.bincount(
        local * n_classes + cls_c.astype(np.int64),
        minlength=uniq.shape[0] * n_classes,
    ).reshape(uniq.shape[0], n_classes)
    return uniq, zmin_c, zmax_c, zsum_c, seg_count, counts
