import numpy as np
import torch
from src.data.projection import compute_projection, build_range_image, build_label_image


class RangeImageAugment:
    def __init__(self, cfg: dict, h: int = 64, w: int = 2048,
                 fov_top_deg: float = 2.0, fov_bottom_deg: float = -24.8,
                 max_range: float = 80.0):
        self.h = h
        self.w = w
        self.fov_top_deg = fov_top_deg
        self.fov_bottom_deg = fov_bottom_deg
        self.max_range = max_range
        self.flip_h = cfg.get("random_flip_h", True)
        self.flip_h_prob = cfg.get("random_flip_h_prob", 0.5)
        self.rot_z = cfg.get("random_rotation_z", True)
        self.rot_z_range = cfg.get("random_rotation_z_range", [-0.7854, 0.7854])
        self.intensity_jitter = cfg.get("intensity_jitter", True)
        self.intensity_jitter_prob = cfg.get("intensity_jitter_prob", 0.3)
        self.intensity_jitter_std = cfg.get("intensity_jitter_std", 0.05)
        self.cutout = cfg.get("cutout", True)
        self.cutout_prob = cfg.get("cutout_prob", 0.3)
        self.cutout_h = cfg.get("cutout_h", 16)
        self.cutout_w = cfg.get("cutout_w", 256)

    def __call__(
        self,
        points: np.ndarray,
        proj: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        needs_reproject = False

        if self.rot_z and np.random.random() < 1.0:
            angle = np.random.uniform(*self.rot_z_range)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points[:, :2] = points[:, :2] @ rot.T
            needs_reproject = True

        if self.flip_h and np.random.random() < self.flip_h_prob:
            points[:, 1] = -points[:, 1]
            needs_reproject = True

        if needs_reproject:
            proj, ranges = compute_projection(
                points, h=self.h, w=self.w,
                fov_top_deg=self.fov_top_deg, fov_bottom_deg=self.fov_bottom_deg,
            )
        else:
            ranges = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))

        remissions = points[:, 3] if points.shape[1] > 3 else np.zeros(len(points))
        range_img = build_range_image(
            points, proj, remissions, ranges,
            h=self.h, w=self.w, max_range=self.max_range,
        )
        label_img = build_label_image(labels, proj, ranges, h=self.h, w=self.w)

        if self.intensity_jitter and np.random.random() < self.intensity_jitter_prob:
            jitter = np.random.normal(1.0, self.intensity_jitter_std)
            range_img[4] = np.clip(range_img[4] * jitter, 0.0, 1.0)

        if self.cutout and np.random.random() < self.cutout_prob:
            ri_h, ri_w = range_img.shape[1], range_img.shape[2]
            ch = np.random.randint(0, max(ri_h - self.cutout_h, 1))
            cw = np.random.randint(0, max(ri_w - self.cutout_w, 1))
            range_img[:, ch:ch + self.cutout_h, cw:cw + self.cutout_w] = 0.0

        return range_img, label_img, proj
