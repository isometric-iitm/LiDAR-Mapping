import numpy as np


def compute_traversability(
    z_min: np.ndarray,
    z_max: np.ndarray,
    z_mean: np.ndarray,
    dominant_class: np.ndarray,
    occupancy: np.ndarray,
    n_rings: int,
    n_theta: int,
    ring_widths: np.ndarray,
    weights: list[float] | None = None,
    class_scores: list[float] | None = None,
    z_diff_thresh: float = 0.12,
    slope_thresh: float = 0.15,
) -> np.ndarray:
    n_cells = n_rings * n_theta
    if weights is None:
        weights = [0.3, 0.3, 0.3, 0.1]
    if class_scores is None:
        class_scores = [1.0, 0.6, 0.2, 0.1]

    w_z, w_slope, w_cls, w_occ = weights

    z_diff = z_max - z_min
    z_score = np.clip(1.0 - z_diff / (z_diff_thresh * 3), 0, 1)

    slope = _neighbor_slope(z_mean, n_rings, n_theta, ring_widths)
    slope_score = np.clip(1.0 - slope / slope_thresh, 0, 1)

    cls_lut = np.zeros(256, dtype=np.float32)
    for i, s in enumerate(class_scores):
        if i < 256:
            cls_lut[i] = s
    cls_score = cls_lut[dominant_class]

    occ_score = np.clip(occupancy, 0, 1)

    total_w = w_z + w_slope + w_cls + w_occ
    trav = (w_z * z_score + w_slope * slope_score + w_cls * cls_score + w_occ * occ_score) / total_w
    return trav.astype(np.float32)


def _neighbor_slope(
    z_mean: np.ndarray,
    n_rings: int,
    n_theta: int,
    ring_widths: np.ndarray,
) -> np.ndarray:
    grid = z_mean.reshape(n_rings, n_theta)
    padded = np.pad(grid, ((1, 1), (1, 1)), mode="edge")
    max_diff = np.zeros_like(grid)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            neighbor = padded[1 + di : n_rings + 1 + di, 1 + dj : n_theta + 1 + dj]
            max_diff = np.maximum(max_diff, np.abs(grid - neighbor))

    ring_idx = np.arange(n_rings).reshape(-1, 1)
    widths = ring_widths[ring_idx]
    arc_widths = widths * (2 * np.pi / n_theta)
    approx_dist = np.maximum(widths, arc_widths)
    approx_dist = np.maximum(approx_dist, 0.05)
    slope = max_diff / approx_dist
    return slope.reshape(-1)


def traversability_tier(trav: np.ndarray) -> np.ndarray:
    tier = np.zeros(len(trav), dtype=np.uint8)
    tier[trav >= 0.7] = 0  # drivable
    tier[(trav >= 0.4) & (trav < 0.7)] = 1  # caution
    tier[trav < 0.4] = 2  # blocked
    return tier
