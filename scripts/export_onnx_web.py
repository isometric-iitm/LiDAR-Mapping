#!/usr/bin/env python3
"""Export the trained RangeImageUNet to a browser-ready fp16 ONNX for ORT-Web.

Standalone companion to scripts/export_onnx.py (which stays the fp32 CPU
reference exporter). This script:

  1. loads the checkpoint (honours $PC2D_CHECKPOINT like the rest of the repo),
  2. exports a static-shape (1, 5, 64, 2048) opset-17 fp32 graph,
  3. appends a Softmax node on the fp32 output (computed on the GPU by
     ORT-Web; saves ~655k JS Math.exp calls per frame in the browser),
  4. writes dashboard/public/demo/models/unet_web_fp16_sm.onnx (~24 MB, under
     the Cloudflare Pages 25 MiB per-file limit; the _sm suffix invalidates
     stale IndexedDB caches),
  5. smoke-tests with onnxruntime (CPU) and runs a numerical parity gate:
     torch fp32 vs ONNX fp16 argmax agreement on a REAL KITTI range image
     (frame 000000 of $PC2D_SEQ_DIR) must be >= 99.5% of labelled pixels,
  6. optionally emits the per-point prediction fixture used by the browser
     engine parity test (dashboard/src/lib/engine/__fixtures__/frame0_preds.json).

Usage:
    uv run python scripts/export_onnx_web.py
        [--checkpoint PATH] [--seq-dir PATH] [--frames 0] [--skip-parity]
        [--fixture-out PATH] [--out PATH]

No repo files are modified besides writing the output artifacts.
"""
import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.models.unet import RangeImageUNet  # noqa: E402


def load_config(name: str) -> dict:
    with open(ROOT / "config" / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def _load_env_file():
    """Parse pc2d/.env into os.environ if present (never overrides existing vars).

    Mirrors src.common.config._load_env so $PC2D_CHECKPOINT / $PC2D_SEQ_DIR
    from the local .env work without importing the config module.
    """
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass


def find_checkpoint(explicit: str | None) -> Path:
    import os
    _load_env_file()
    from src.common.config import resolve_path
    if explicit:
        return resolve_path(explicit)
    env = os.getenv("PC2D_CHECKPOINT", "")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for cand in ("checkpoints/best_miou.pt",):
        p = ROOT / cand
        if p.exists():
            return p
    raise FileNotFoundError("No checkpoint found; pass --checkpoint or set $PC2D_CHECKPOINT")


def find_seq_dir(explicit: str | None) -> Path:
    import os
    _load_env_file()
    if explicit:
        return Path(explicit)
    env = os.getenv("PC2D_SEQ_DIR", "")
    if env:
        p = Path(env)
        if p.exists():
            return p
    p = ROOT / "data" / "sequences" / "08"
    if p.exists():
        return p
    raise FileNotFoundError("No sequence dir found; pass --seq-dir or set $PC2D_SEQ_DIR")


def build_range_image(points: np.ndarray, h: int, w: int, fov_top_deg: float,
                      fov_bottom_deg: float, max_range: float):
    """Exact port-forward of src/data/projection.py:compute_projection + build_range_image.

    Same math (row scale h-1, col scale w, floor+clamp, nearest-wins via
    descending-range write order), so the exported graph is fed exactly what
    the browser projector will feed it.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.clip(r, 1e-6, None)

    fov_top = np.deg2rad(fov_top_deg)
    fov_bottom = np.deg2rad(fov_bottom_deg)
    fov = fov_top - fov_bottom

    row = (1.0 - (np.arcsin(z / r) - fov_bottom) / fov) * (h - 1)
    col = 0.5 * (np.arctan2(y, x) / np.pi + 1.0) * w
    row = np.clip(np.floor(row).astype(np.int32), 0, h - 1)
    col = np.clip(np.floor(col).astype(np.int32), 0, w - 1)

    img = np.zeros((h, w, 5), dtype=np.float32)
    order = np.argsort(-r)  # nearest last-write wins
    rs, cs = row[order], col[order]
    ps = points[order]
    img[rs, cs, 0] = r[order] / max_range
    img[rs, cs, 1] = ps[:, 0] / max_range
    img[rs, cs, 2] = ps[:, 1] / max_range
    img[rs, cs, 3] = ps[:, 2] / max_range
    img[rs, cs, 4] = ps[:, 3]
    ri = img.transpose(2, 0, 1)  # (5, h, w)
    proj = np.stack([row, col], axis=-1).astype(np.int32)  # (N, 2)
    return ri, proj


def knn_probs_from_probs(pixel_probs: torch.Tensor, proj: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean of the (k,k) neighbour PROBABILITIES per point; input is already
    softmaxed (the ONNX graph ends in Softmax). Same gather as knn_probs_torch."""
    _, c, h, w = pixel_probs.shape
    n = proj.shape[0]
    pp = pixel_probs.permute(0, 2, 3, 1).reshape(-1, c)  # (h*w, c)
    rows = proj[:, 0].clamp(0, h - 1)
    cols = proj[:, 1].clamp(0, w - 1)
    off = torch.arange(-1, 2)
    rinds = (rows[:, None] + off[None, :]).clamp(0, h - 1).reshape(n, -1)
    cinds = (cols[:, None] + off[None, :]).clamp(0, w - 1).reshape(n, -1)
    flat = rinds * w + cinds  # (n, 9)
    return pp[flat].mean(dim=1)  # (n, c)


def np2t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--seq-dir", type=str, default=None, help="Sequence dir for the parity scan")
    ap.add_argument("--frames", type=int, default=1, help="How many real frames to parity-test (default 1)")
    ap.add_argument("--skip-parity", action="store_true", help="Only export + shape smoke test")
    ap.add_argument("--fixture-out", type=str, default=None,
                    help="Write the per-point binned-4 prediction fixture for frame 0 here")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--min-agreement", type=float, default=0.995)
    args = ap.parse_args()

    out_default = ROOT / "dashboard" / "public" / "demo" / "models" / "unet_web_fp16_sm.onnx"
    out_path = Path(args.out) if args.out else out_default
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = find_checkpoint(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    h, w = int(mcfg["h"]), int(mcfg["w"])
    in_ch, base_ch, n_cls = int(mcfg["in_channels"]), int(mcfg["base_channels"]), int(mcfg["num_classes"])
    fov_top, fov_bottom = float(mcfg["fov_top_deg"]), float(mcfg["fov_bottom_deg"])
    max_range = float(ckpt["config"]["train"].get("max_range", 80.0))
    print(f"[export-web] checkpoint: {ckpt_path} (best_miou={ckpt.get('best_miou', 0):.4f})")
    print(f"[export-web] model: in={in_ch} cls={n_cls} base={base_ch} input=({h},{w}) fov=({fov_top},{fov_bottom}) max_range={max_range}")

    # 1) fp32 static-shape export (opset 17, same contract as export_onnx.py)
    model = RangeImageUNet(in_channels=in_ch, num_classes=n_cls, base_channels=base_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(1, in_ch, h, w, dtype=torch.float32)

    fp32_tmp = out_path.with_suffix(".fp32.tmp.onnx")
    print(f"[export-web] exporting fp32 graph (opset 17, static (1,{in_ch},{h},{w})) ...")
    t0 = time.time()
    torch.onnx.export(
        model,
        dummy,
        str(fp32_tmp),
        input_names=["range_image"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export-web] fp32 export done in {time.time() - t0:.1f}s ({fp32_tmp.stat().st_size / 1e6:.1f} MB)")

    # 2) fp16 conversion (fp32 I/O preserved) + trailing Softmax
    import onnx
    from onnx import helper as oh, TensorProto as TP
    from onnxruntime.transformers.float16 import convert_float_to_float16
    print("[export-web] converting weights to fp16 (keep_io_types=True) + appending Softmax...")
    m32 = onnx.load(str(fp32_tmp))
    m16 = convert_float_to_float16(m32, keep_io_types=True)
    # Append Softmax(axis=1) so the browser worker reads probabilities directly
    # (ORT computes it on the WebGPU EP; the JS per-pixel softmax disappears).
    graph = m16.graph
    sm_out = graph.output[0].name + "_probs"
    softmax = oh.make_node(
        "Softmax", inputs=[graph.output[0].name], outputs=[sm_out], name="softmax_web", axis=1)
    graph.node.append(softmax)
    graph.output[0].name = sm_out
    onnx.save_model(m16, str(out_path.with_suffix(".sm.tmp.onnx")))
    # Atomic replace: Windows can fail writing over an existing large file handle.
    import os
    os.replace(out_path.with_suffix(".sm.tmp.onnx"), out_path)
    size = out_path.stat().st_size
    print(f"[export-web] wrote {out_path} ({size / 1e6:.2f} MB = {size / 1048576:.2f} MiB)")
    if size > 25 * 1048576:
        print(f"[export-web] WARNING: {size / 1048576:.1f} MiB exceeds the 25 MiB Cloudflare Pages per-file limit")

    # 3) ORT smoke test (CPU EP): output shape + finite probabilities + rows sum to 1
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    (in_meta,) = sess.get_inputs()
    (out_meta,) = sess.get_outputs()
    print(f"[export-web] ORT inputs: {in_meta.name} {in_meta.type} {in_meta.shape} | outputs: {out_meta.name} {out_meta.type} {out_meta.shape}")
    probs = sess.run(["logits_probs"], {"range_image": dummy.numpy()})[0]

    assert probs.shape == (1, n_cls, h, w), f"unexpected output shape {probs.shape}"
    assert np.isfinite(probs).all(), "non-finite probs"
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), "softmax rows do not sum to 1"
    print(f"[export-web] ORT smoke OK: probs {probs.shape} finite, rows sum to 1, argmax classes={np.unique(probs.argmax(1)).tolist()}")

    # 4) Numerical parity gate on real KITTI scans (torch path applies its own softmax)
    if not args.skip_parity:
        seq_dir = find_seq_dir(args.seq_dir)
        bins = sorted((seq_dir / "velodyne").glob("*.bin"))
        if not bins:
            print("[export-web] no .bin files found; skipping parity")
            return 0
        n_frames = min(args.frames, len(bins))
        agreement_all = []
        for fi in range(n_frames):
            pts = np.fromfile(str(bins[fi]), dtype=np.float32).reshape(-1, 4)
            ri, proj = build_range_image(pts, h, w, fov_top, fov_bottom, max_range)
            ri_t = torch.from_numpy(ri).unsqueeze(0)  # (1,5,h,w)
            with torch.inference_mode():
                torch_logits = model(ri_t)
                tpp = knn_probs_from_probs(torch.softmax(torch_logits, dim=1), torch.from_numpy(proj), 3)
                t_pred = tpp.argmax(dim=1).numpy()
            o_probs = sess.run(["logits_probs"], {"range_image": ri.astype(np.float32)[None]})[0]
            opp = knn_probs_from_probs(np2t(o_probs), torch.from_numpy(proj), 3)
            o_pred = opp.argmax(dim=1).numpy()
            agree = float((t_pred == o_pred).mean())
            agreement_all.append(agree)
            delta = float(np.abs(torch_logits.numpy() - np.log(np.maximum(o_probs, 1e-9))).mean())
            print(f"[export-web] parity frame {fi}: n={len(pts)} argmax agreement={agree * 100:.3f}% mean|dlogit|={delta:.5f}")
            if fi == 0 and args.fixture_out:
                # 5-class -> binned-4 LUT (mirrors dashboard engine postprocess)
                lut = np.array([0, 1, 2, 3, 3], dtype=np.uint8)
                fixture = {
                    "n": int(len(t_pred)),
                    "pred5": t_pred.astype(np.uint8).tolist(),
                    "pred4": lut[t_pred.astype(np.uint8)].tolist(),
                    "agreement": agree,
                }
                fix_path = Path(args.fixture_out)
                fix_path.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(fixture).encode()
                with gzip.open(fix_path, "wt", encoding="utf-8") as f:
                    json.dump(fixture, f)
                print(f"[export-web] fixture -> {fix_path} ({len(payload) / 1e6:.2f} MB json, {fix_path.stat().st_size / 1e6:.2f} MB gz)")
        worst = min(agreement_all)
        if worst < args.min_agreement:
            print(f"[export-web] FAIL: parity {worst * 100:.3f}% < {args.min_agreement * 100:.1f}% required")
            return 1
        print(f"[export-web] parity gate PASSED (worst {worst * 100:.3f}% >= {args.min_agreement * 100:.1f}%)")

    fp32_tmp.unlink(missing_ok=True)
    for junk in out_path.parent.glob("*.tmp.onnx*"):
        junk.unlink(missing_ok=True)
    print(f"[export-web] DONE -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
