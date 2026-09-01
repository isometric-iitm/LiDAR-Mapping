#!/usr/bin/env python3
"""Export the trained RangeImageUNet to ONNX for CPU inference (onnxruntime).

Usage:
    uv run python scripts/export_onnx.py [--checkpoint PATH] [--out PATH] [--h 64] [--w 2048]

The export runs on a plain fp32 CPU model (no torch.compile / fp16), so the
produced graph is runtime-agnostic: onnxruntime can execute it on CPU without
GPU or triton. The default output is checkpoints/best_miou.onnx (gitignored
via *.onnx).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# torch.onnx prints UTF-8 glyphs that can crash a cp1252 Windows console; make stdio non-fatal.
if hasattr(sys.stdout, "reconfigure") and sys.stdout is not None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure") and sys.stderr is not None:
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import resolve_path
from src.models.unet import RangeImageUNet


def main():
    ap = argparse.ArgumentParser(description="Export RangeImageUNet to ONNX")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_miou.pt",
                    help="Path to the .pt checkpoint (supports $PC2D_CHECKPOINT)")
    ap.add_argument("--out", type=str, default="checkpoints/best_miou.onnx",
                    help="Output .onnx path (default: checkpoints/best_miou.onnx)")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=2048)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--in-channels", type=int, default=5)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--base-channels", type=int, default=32)
    args = ap.parse_args()

    ckpt_path = resolve_path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "config" in ckpt and ckpt["config"].get("model"):
        m = ckpt["config"]["model"]
        args.num_classes = m.get("num_classes", args.num_classes)
        args.in_channels = m.get("in_channels", args.in_channels)
        args.base_channels = m.get("base_channels", args.base_channels)
    print(f"[onnx] checkpoint: {ckpt_path}  best_miou={ckpt.get('best_miou', 'n/a'):.4f}")

    model = RangeImageUNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, args.in_channels, args.h, args.w)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[onnx] exporting to {out_path} (opset {args.opset}, input {tuple(dummy.shape)}) ...")
    t0 = time.time()
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["range_image"],
        output_names=["logits"],
        dynamic_axes=None,          # static shape (1,C,H,W) -> max CPU throughput
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[onnx] exported in {time.time() - t0:.1f}s")

    # smoke test with onnxruntime (CPU)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(out_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"range_image": dummy.numpy()})[0]
    arg = out.argmax(axis=1)
    n_classes = out.shape[1]
    print(f"[onnx] ort output {tuple(out.shape)} n_classes={n_classes} "
          f"unique_argmax={len(np.unique(arg))} (expected=={n_classes})")
    assert out.shape == (1, args.num_classes, args.h, args.w), "unexpected output shape"
    print(f"[onnx] OK -> {out_path}")


if __name__ == "__main__":
    main()
