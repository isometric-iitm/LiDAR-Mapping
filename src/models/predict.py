import os
import time
from pathlib import Path

import numpy as np
import torch

from src.data.projection import make_projector
from src.models.unet import RangeImageUNet

try:
    from src.models import knn_triton
except Exception:  # pragma: no cover - triton absent/non-CUDA machine
    knn_triton = None


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
        from src.common.config import resolve_device, resolve_precision
        self.precision = resolve_precision(precision)
        self.device = torch.device(resolve_device(device))
        self.h, self.w = h, w
        self.projector = make_projector(
            self.device, h, w, fov_top_deg, fov_bottom_deg, max_range)

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

        self._compile_failed = False

        if self._should_compile(self.device):
            plain = self.model
            for mode in ("reduce-overhead", "default"):
                self.model = torch.compile(plain, mode=mode, dynamic=False)
                if self._warmup(in_channels):
                    break
                print(f"[Segmenter] torch.compile mode={mode!r} warmup failed")
                self.model = plain
            else:
                print("[Segmenter] torch.compile failed for all modes, falling back to eager")
                self.model = plain

        print(f"[Segmenter] loaded {ckpt_path} (best_miou={ckpt.get('best_miou', 'n/a'):.4f}) on {self.device} precision={self.precision}")

    @staticmethod
    def _should_compile(device: torch.device) -> bool:
        """torch.compile is a CUDA-only win here; also requires torch >= 2.0."""
        if device.type != "cuda":
            return False
        try:
            major = int(torch.__version__.split(".")[0])
            minor = int(torch.__version__.split(".")[1])
        except (IndexError, ValueError):
            return False
        return (major, minor) >= (2, 0)

    def _warmup(self, in_channels: int) -> bool:
        """One dummy forward so torch.compile finishes graph capture before the
        live stream's first frame (otherwise frame 1 pays the whole compile
        cost). Returns False if the forward raised (compile backout)."""
        dtype = torch.half if (self.device.type == "cuda" and self.precision == "fp16") else torch.float32
        dummy = torch.zeros(1, in_channels, self.h, self.w, device=self.device, dtype=dtype)
        try:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.precision == "fp16")):
                    self.model(dummy)
        except Exception:
            return False
        return True

    @torch.inference_mode()
    def segment(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (per_point_class_ids [N] uint8 in [0,num_classes), timings dict).

        Timings sub-fields (ms): project (projection + range image via the device
        Projector), forward (UNet + softmax + knn gather), sync (argmax + device->host
        copy). On CUDA the H2D copy is asynchronous (non_blocking) so submitting it
        does not stall the CPU; the only blocking device sync is the final .cpu().
        """
        t = {"project": 0.0, "forward": 0.0, "sync": 0.0, "total": 0.0}
        if points.shape[0] == 0 or points.shape[1] < 4:
            return np.zeros(0, dtype=np.uint8), {"project": 0.0, "forward": 0.0, "sync": 0.0, "total": 0.0}

        t0 = time.perf_counter()
        ri, proj_t = self.projector.project(points)
        t1 = time.perf_counter()
        t["project"] = (t1 - t0) * 1000.0

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.precision == "fp16")):
            try:
                logits = self.model(ri.unsqueeze(0))
            except Exception:
                # torch.compile (especially cudagraphs) can raise at runtime on
                # some GPUs even after a successful warmup. Fall back to eager
                # for the rest of the process lifetime.
                if not self._compile_failed and hasattr(self.model, "_orig_mod"):
                    print("[Segmenter] compiled model runtime error, falling back to eager")
                    self.model = self.model._orig_mod
                    self._compile_failed = True
                    logits = self.model(ri.unsqueeze(0))
                else:
                    raise
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
        """Mean of the (k,k) neighbour probabilities per point.

        On CUDA float32 with k == 3 the work is delegated to a single fused
        Triton pass (no index materialization); everything else uses the torch
        gather-mean below. Bit-compatible within float tolerance (same 9
        neighbours, same mean)."""
        _, c, h, w = pixel_probs.shape
        n = proj.shape[1]
        if (
            k == 3
            and knn_triton is not None
            and proj.device.type == "cuda"
            and pixel_probs.dtype == torch.float32
            and pixel_probs.is_contiguous()
        ):
            try:
                return knn_triton.triton_knn3(pixel_probs, proj, n)
            except Exception:
                pass  # fall through to the torch path
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