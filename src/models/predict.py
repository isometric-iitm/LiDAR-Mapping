import os
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
        precision: str | None = None,
    ):
        self.h = h
        self.w = w
        self.fov_top_deg = fov_top_deg
        self.fov_bottom_deg = fov_bottom_deg
        self.max_range = max_range
        self.num_classes = num_classes

        from src.common.config import resolve_device, resolve_precision
        self.precision = resolve_precision(precision)
        self.device = torch.device(resolve_device(device))

        self.model = RangeImageUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_groupnorm=use_groupnorm,
            groups=groups,
        )
        from src.common.config import resolve_path
        # Env override (highest priority) lets a machine point the segmenter at
        # its own checkpoint without touching config; falls back to the passed arg.
        checkpoint = os.getenv("PC2D_CHECKPOINT", "") or checkpoint
        ckpt_path = resolve_path(checkpoint)
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        # fp16: cast weights to half on CUDA (halves memory + uses tensor-core math).
        if self.device.type == "cuda" and self.precision == "fp16":
            self.model.half()

        # torch.compile: only on CUDA and torch >= 2.0.
        if self.device.type == "cuda" and torch.__version__.split(".")[0] == "2":
            self.model = torch.compile(self.model)

        print(f"[Segmenter] loaded {ckpt_path} (best_miou={ckpt.get('best_miou', 'n/a'):.4f}) on {self.device} precision={self.precision}")

        # Lazy-allocated pinned host buffer for async (non_blocking) H2D copies,
        # so the CPU thread never stalls submitting the point cloud to the GPU.
        self._pin = None  # np.ndarray (n,4) float32, pinned via torch.zeros(..., pin_memory=True)

    def _pin_buffer(self, n: int) -> np.ndarray:
        """Return a pinned float32 (n,4) host buffer large enough for n points."""
        if self._pin is None or self._pin.shape[0] < n:
            self._pin = torch.zeros((n, 4), dtype=torch.float32, pin_memory=True).numpy()
        return self._pin

    @torch.inference_mode()
    def segment(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (per_point_class_ids [N] uint8 in [0,num_classes), timings dict).

        Timings sub-fields (ms): project (H2D + projection + range image),
        forward (UNet + softmax + knn gather), sync (argmax + device->host copy).
        On CUDA the H2D copy is asynchronous (non_blocking) so submitting it does
        not stall the CPU; the only blocking device sync is the final .cpu().
        """
        t = {"project": 0.0, "forward": 0.0, "sync": 0.0, "total": 0.0}
        if points.shape[0] == 0 or points.shape[1] < 4:
            return np.zeros(0, dtype=np.uint8), {"project": 0.0, "forward": 0.0, "sync": 0.0, "total": 0.0}

        t0 = time.perf_counter()
        if self.device.type == "cuda":
            # Async H2D from a pinned buffer: submitting the copy does not block
            # the CPU; only the final .cpu() below synchronizes with the GPU stream.
            pin = self._pin_buffer(points.shape[0])
            pin_view = pin[: points.shape[0]]
            np.copyto(pin_view, points[:, :4])
            pts = torch.from_numpy(pin_view).to(self.device, non_blocking=True)
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
        t1 = time.perf_counter()
        t["project"] = (t1 - t0) * 1000.0

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.precision == "fp16")):
            logits = self.model(ri.unsqueeze(0))
            probs = torch.softmax(logits.float(), dim=1)
            # 3x3 neighbour gather -> per-point probabilities (knn_project_back)
            point_probs = self._knn_probs(probs, proj_t, 3)
        t2 = time.perf_counter()
        t["forward"] = (t2 - t1) * 1000.0

        # Fold argmax -> uint8 in one op, then a single device->host sync copy.
        per_point = torch.argmax(point_probs, dim=1, keepdim=False).to(torch.uint8).cpu().numpy()
        t3 = time.perf_counter()
        t["sync"] = (t3 - t2) * 1000.0
        t["total"] = (t3 - t0) * 1000.0
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