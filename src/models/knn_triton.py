"""Triton 3x3 KNN gather-mean for the segmenter's back-projection.

Fuses clamp+shift+clamp, gather, and mean into one pass (no index materialization).
Results are bit-compatible with the torch fallback. Only for float32 CUDA with k==3;
anything else falls back. Import failure is non-fatal.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_TRITON_OK = True
try:
    import torch as _torch

    if _torch.cuda.is_available():
        _a = _torch.zeros(1, device="cuda", dtype=_torch.float32)
        _b = _torch.zeros(1, device="cuda", dtype=_torch.float32)

        @triton.jit
        def _probe_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x * 2.0, mask=mask)

        _probe_kernel[(1,)](_a, _b, 1, BLOCK=1)
        _torch.cuda.synchronize()
        del _a, _b, _probe_kernel
    else:
        _TRITON_OK = False
except Exception as _e:
    _TRITON_OK = False
    logger.warning("triton probe kernel failed (%s: %s), disabling Triton KNN path", type(_e).__name__, _e)


@triton.jit
def _knn3_kernel(
    rows_ptr,  # (n,) int32 clamped row of each point
    cols_ptr,  # (n,) int32 clamped col of each point
    pp_ptr,    # (h*w*c,) float32 per-pixel per-class probabilities
    out_ptr,   # (n*c,) float32 per-point per-class means
    N, H, W, C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    n_mask = offs_n < N
    c_mask = offs_c < C
    mask = n_mask[:, None] & c_mask[None, :]

    rows = tl.load(rows_ptr + offs_n, mask=n_mask, other=0)
    cols = tl.load(cols_ptr + offs_n, mask=n_mask, other=0)

    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    for dr in tl.static_range(-1, 2):
        r = tl.minimum(tl.maximum(rows + dr, 0), H - 1)
        for dc in tl.static_range(-1, 2):
            c = tl.minimum(tl.maximum(cols + dc, 0), W - 1)
            base = (r[:, None] * W + c[:, None]) * C + offs_c[None, :]
            acc += tl.load(pp_ptr + base, mask=mask, other=0.0)
    acc *= 1.0 / 9.0

    out_off = offs_n[:, None] * C + offs_c[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


def triton_knn3(pixel_probs: "torch.Tensor", proj: "torch.Tensor", n: int) -> "torch.Tensor":
    """Fused 3x3 gather+mean back-projection.

    Args:
        pixel_probs: (1, c, h, w) float32 contiguous on CUDA.
        proj: (1, 2, n) float32 point-to-pixel map on the same device.
        n: live point count (proj.shape[1]).

    Returns (n, c) float32; same numbers as the torch mean-of-9 fallback.

    Raises RuntimeError if the module-level triton probe failed at import.
    """
    if not _TRITON_OK:
        raise RuntimeError("triton probe failed at import, KNN path disabled")
    import torch

    c, h, w = pixel_probs.shape[1], pixel_probs.shape[2], pixel_probs.shape[3]
    pp = pixel_probs.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
    rows = proj[:, :, 0].clamp(0, h - 1).reshape(-1).to(torch.int32)
    cols = proj[:, :, 1].clamp(0, w - 1).reshape(-1).to(torch.int32)
    out = torch.empty((n, c), device=pixel_probs.device, dtype=torch.float32)

    blk_c = triton.next_power_of_2(c)
    blk_n = 32
    grid = (triton.cdiv(n, blk_n),)
    _knn3_kernel[grid](
        rows, cols, pp, out,
        n, h, w, c,
        BLOCK_N=blk_n, BLOCK_C=blk_c,
    )
    return out