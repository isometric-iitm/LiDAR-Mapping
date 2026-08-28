import numpy as np
import torch


def compute_projection(
    points: np.ndarray,
    h: int = 64,
    w: int = 2048,
    fov_top_deg: float = 2.0,
    fov_bottom_deg: float = -24.8,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.clip(r, 1e-6, None)

    fov_top = np.deg2rad(fov_top_deg)
    fov_bottom = np.deg2rad(fov_bottom_deg)
    fov = fov_top - fov_bottom

    row = (1.0 - (np.arcsin(z / r) - fov_bottom) / fov) * (h - 1)
    col = 0.5 * (np.arctan2(y, x) / np.pi + 1.0) * w

    row = np.clip(np.floor(row).astype(np.int32), 0, h - 1)
    col = np.clip(np.floor(col).astype(np.int32), 0, w - 1)

    proj = np.stack([row, col], axis=-1).astype(np.int32)
    return proj, r


def build_range_image(
    points: np.ndarray,
    proj: np.ndarray,
    remissions: np.ndarray,
    ranges: np.ndarray,
    h: int = 64,
    w: int = 2048,
    max_range: float = 80.0,
) -> np.ndarray:
    img = np.zeros((h, w, 5), dtype=np.float32)
    row, col = proj[:, 0], proj[:, 1]

    order = np.argsort(-ranges)
    ranges_sorted = ranges[order]
    points_sorted = points[order]
    rem_sorted = remissions[order]
    row_sorted = row[order]
    col_sorted = col[order]

    img[row_sorted, col_sorted, 0] = ranges_sorted / max_range
    img[row_sorted, col_sorted, 1] = points_sorted[:, 0] / max_range
    img[row_sorted, col_sorted, 2] = points_sorted[:, 1] / max_range
    img[row_sorted, col_sorted, 3] = points_sorted[:, 2] / max_range
    img[row_sorted, col_sorted, 4] = rem_sorted

    return img.transpose(2, 0, 1)


def knn_project_back(
    pixel_logits: torch.Tensor,
    proj: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    b, c, h, w = pixel_logits.shape
    n = proj.shape[1]

    proj_h = proj[:, :, 0]
    proj_w = proj[:, :, 1]

    soft = torch.softmax(pixel_logits, dim=1)
    pixel_probs = soft.permute(0, 2, 3, 1).contiguous()

    point_probs = torch.zeros(b, n, c, device=pixel_logits.device, dtype=pixel_logits.dtype)

    for di in range(-1, 2):
        for dj in range(-1, 2):
            ri = (proj_h + di).clamp(0, h - 1)
            rj = (proj_w + dj).clamp(0, w - 1)
            ri = ri.unsqueeze(-1).expand(-1, -1, c)
            rj = rj.unsqueeze(-1).expand(-1, -1, c)
            gathered = torch.gather(
                pixel_probs.reshape(b, -1, c),
                1,
                (ri * w + rj).reshape(b, -1, c),
            )
            point_probs += gathered

    point_probs = point_probs / 9.0
    return point_probs


def build_label_image(
    labels: np.ndarray,
    proj: np.ndarray,
    ranges: np.ndarray,
    h: int = 64,
    w: int = 2048,
    ignore_index: int = 255,
) -> np.ndarray:
    img = np.full((h, w), fill_value=ignore_index, dtype=np.int64)
    row, col = proj[:, 0], proj[:, 1]
    order = np.argsort(-ranges)
    img[row[order], col[order]] = labels[order]
    return img
