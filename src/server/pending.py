"""Lossless delta retransmit (ghost / missing-cell fix).

The grid thread can drop a whole binary frame when the broadcast queue overflows.
``delta()`` mutates its sent-tracking state at compute time, so a dropped delta's
``added``/``changed`` upserts used to be lost to the client until the next full
snapshot. This module owns the pending-free / pending-upsert bookkeeping (as
flat cell index sets) and the merge logic that makes drops lossless, plus the
catchup guard that substitutes a full snapshot once drops accumulate.
"""

import numpy as np


def rows_to_flat(rows: np.ndarray, n_theta: int) -> set:
    """Convert upsert rows [i,j,...] into a set of flat cell indices (i*n_theta+j)."""
    if rows.shape[0] == 0:
        return set()
    i = rows[:, 0].astype(np.int64)
    j = rows[:, 1].astype(np.int64)
    return set((i * n_theta + j).tolist())


def free_to_flat(freed: np.ndarray, n_theta: int) -> set:
    """Convert freed rows [i,j] into a set of flat cell indices (i*n_theta+j)."""
    if freed.size == 0:
        return set()
    i = freed[:, 0].astype(np.int64)
    j = freed[:, 1].astype(np.int64)
    return set((i * n_theta + j).tolist())


def should_snapshot(frame: int, last_snapshot_frame: int, snap_iv: int,
                    consecutive_drops: int, pending_free: set, pending_upsert: set,
                    catchup_drop_threshold: int, catchup_pending_threshold: int) -> bool:
    """True when the frame should be sent as a full (authoritative) snapshot
    instead of an incremental delta: either the periodic cadence is due, or a
    burst of dropped frames / large pending retransmit backlog would otherwise
    produce a giant merged delta (the source of transient client map bloat)."""
    if frame - last_snapshot_frame >= snap_iv:
        return True
    backlog = len(pending_free) + len(pending_upsert)
    return (
        consecutive_drops >= catchup_drop_threshold
        or backlog >= catchup_pending_threshold
    )


def merge_pending_upsert(pending_upsert: set, rows: np.ndarray, cls: np.ndarray,
                         rendered: np.ndarray, read_rows) -> tuple[np.ndarray, np.ndarray]:
    """Merge re-send upsert rows from deltas that were dropped by the queue into
    the current delta's upsert rows, so added/changed cells are never lost to the
    client (the counterpart of merge_pending_free, which only covers frees).

    Pending cells that are no longer rendered are dropped: their removal was
    already conveyed by a delta() `freed`. Rows are re-read fresh from the grid
    so the client always gets current state. The pending set is cleared after
    merging; if the merged frame is itself dropped, the caller re-stashes the
    flat indices."""
    if not pending_upsert:
        return rows, cls
    items = list(pending_upsert)
    pending_upsert.clear()
    alive_idx = np.asarray([f for f in items if rendered[f]], dtype=np.int64)
    if alive_idx.size == 0:
        return rows, cls
    a_rows, a_cls = read_rows(alive_idx)
    if rows.shape[0]:
        return np.concatenate([rows, a_rows], axis=0), np.concatenate([cls, a_cls], axis=0)
    return a_rows, a_cls


def merge_pending_free(pending_free: set, freed: np.ndarray,
                       not_rendered: np.ndarray, n_theta: int) -> np.ndarray:
    """Merge this frame's freed rows with any pending frees from frames that were
    dropped by the queue. Pending frees are filtered against the current rendered
    mask so a cell that was freed then re-added is NOT freed again (which would
    wrongly remove the fresh cell). The pending set is cleared after merging."""
    if not pending_free:
        return freed
    items = list(pending_free)
    pending_free.clear()
    alive = [f for f in items if not_rendered[f]]
    if not alive:
        return freed
    flat = np.asarray(alive, dtype=np.int64)
    pend = np.stack([
        (flat // n_theta).astype(np.float32),
        (flat % n_theta).astype(np.float32),
    ], axis=1)
    if freed.size:
        return np.concatenate([pend, freed], axis=0)
    return pend