import time
from pathlib import Path

import numpy as np
import torch

from src.data.projection import (
    build_range_image,
    build_range_image_gpu,
    compute_projection,
    project_points_gpu,
)
from src.models.unet import RangeImageUNet


class Segmenter:
    """Loads a trained RangeImageUNet and segments raw KITTI .bin scans into per-point class ids."""

    def __init__(
        self,
        checkpoint: str | Path,
        in_channels: int = 5,
        num_classes: int = 5,
        base_channels: int = 32,
        use_groupnorm: bool = True,
        groups: int = 8,
        h: int = 64,
        w: int = 2048,
        fov_top_deg: float = 2.0,
        fov_bottom_deg: float = -24.8,
        max_range: float = 80.0,
        device: str | None = None,
    ):
        self.h = h
        self.w = w
        self.fov_top_deg = fov_top_deg
        self.fov_bottom_deg = fov_bottom_deg
        self.max_range = max_range
        self.num_classes = num_classes

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = RangeImageUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_groupnorm=use_groupnorm,
            groups=groups,
        )
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        print(f"[Segmenter] loaded {checkpoint} (best_miou={ckpt.get('best_miou', 'n/a'):.4f}) on {self.device}")

    @torch.inference_mode()
    def segment(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (per_point_class_ids [N] uint8 in [0,num_classes), timings dict)."""
        t = {}
        if points.shape[0] == 0 or points.shape[1] < 4:
            return np.zeros(0, dtype=np.uint8), {"project": 0.0, "forward": 0.0, "total": 0.0}

        t0 = time.perf_counter()
        if self.device.type == "cuda":
            # all projection math on GPU — no CPU sort / fancy-index / PCIe hitch
            pts = torch.from_numpy(np.ascontiguousarray(points[:, :4])).to(self.device)
            row, col, r = project_points_gpu(pts, h=self.h, w=self.w,
                                             fov_top_deg=self.fov_top_deg,
                                             fov_bottom_deg=self.fov_bottom_deg)
            ri = build_range_image_gpu(pts, row, col, r, h=self.h, w=self.w,
                                       max_range=self.max_range)
            proj_t = torch.stack((row, col), dim=1).unsqueeze(0)
        else:
            proj, ranges = compute_projection(points, h=self.h, w=self.w,
                                              fov_top_deg=self.fov_top_deg,
                                              fov_bottom_deg=self.fov_bottom_deg)
            ri = build_range_image(points, proj, points[:, 3], ranges, h=self.h, w=self.w,
                                   max_range=self.max_range)
            ri = torch.from_numpy(np.ascontiguousarray(ri))
            proj_t = torch.from_numpy(proj).long().unsqueeze(0)
        ri_t = ri.unsqueeze(0)
        t1 = time.perf_counter()
        t["project"] = (t1 - t0) * 1000.0

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            logits = self.model(ri_t)
            probs = torch.softmax(logits.float(), dim=1)
            # 3x3 neighbour gather -> per-point probabilities (knn_project_back)
            point_probs = self._knn_probs(probs, proj_t, 3)

        t2 = time.perf_counter()
        t["forward"] = (t2 - t1) * 1000.0

        per_point = torch.argmax(point_probs, dim=1).to(torch.uint8).cpu().numpy()
        t["total"] = (t2 - t0) * 1000.0
        return per_point, t

    @staticmethod
    def _knn_probs(pixel_probs: torch.Tensor, proj: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Mean of the (k,k) neighbour probabilities per point — single gather,
        batch free (segment() feeds batch 1)."""
        _, c, h, w = pixel_probs.shape
        n = proj.shape[1]
        pp = pixel_probs.permute(0, 2, 3, 1).reshape(-1, c)  # (h*w, c) contiguous
        half = k // 2
        off = torch.arange(-half, half + 1, device=proj.device, dtype=torch.long)
        rows = proj[:, :, 0].clamp(0, h - 1)
        cols = proj[:, :, 1].clamp(0, w - 1)
        rinds = (rows[:, :, None] + off[None, None, :]).clamp(0, h - 1).reshape(n, -1)
        cinds = (cols[:, :, None] + off[None, None, :]).clamp(0, w - 1).reshape(n, -1)
        flat = rinds * w + cinds  # (n, k*k)
        gathered = pp[flat]       # (n, k*k, c) — one big gather, no per-offset allocs
        return gathered.mean(dim=1)


def load_segmenter(checkpoint: str | Path, **kwargs) -> Segmenter:
    return Segmenter(checkpoint, **kwargs)