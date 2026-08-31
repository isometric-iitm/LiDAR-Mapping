"""Launch the PC2D live demo: FastAPI grid server + Next.js dashboard.

Usage:
    uv run python scripts/run_live_demo.py [--no-dashboard] [--cpu]

Starts uvicorn on 127.0.0.1:8000 and `next dev` on 127.0.0.1:3000.
Ctrl+C stops both.

--cpu forces CPU inference at 0.5x playback for GPUs-less judge laptops.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_npm() -> str:
    name = "npm.cmd" if sys.platform == "win32" else "npm"
    resolved = shutil.which(name)
    if resolved:
        return resolved
    return name  # let subprocess raise a clear error if truly missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dashboard", action="store_true", help="only start the FastAPI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU inference + playback_speed 0.5 for judge laptops without a GPU (2-4 Hz vs GPU 10-15 Hz)")
    args = ap.parse_args()

    # Env vars are inherited by the uvicorn subprocess, which applies them over
    # the pipeline config (PC2D_DEVICE -> model.device, PC2D_PLAYBACK_SPEED ->
    # source.playback_speed) -> smooth demo on CPU hardware.
    if args.cpu:
        os.environ["PC2D_DEVICE"] = "cpu"
        os.environ["PC2D_PRECISION"] = "fp32"
        os.environ["PC2D_PLAYBACK_SPEED"] = "0.5"
        print("[pc2d] CPU mode: device=cpu, precision=fp32, playback_speed=0.5 (expected 2-4 Hz; GPU is 10-15 Hz)")

    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.server.app:create_app",
        "--factory", "--host", args.host, "--port", str(args.port),
    ]
    procs = []
    try:
        print(f"[pc2d] starting FastAPI grid server on http://{args.host}:{args.port} ...")
        srv = subprocess.Popen(server_cmd, cwd=ROOT)
        procs.append(srv)

        if not args.no_dashboard:
            print("[pc2d] starting Next.js dashboard on http://localhost:3000 ...")
            dash = subprocess.Popen(
                [find_npm(), "run", "dev"],
                cwd=ROOT / "dashboard",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            procs.append(dash)

        print()
        print("  Live map:      http://localhost:3000")
        print("  Training page: http://localhost:3000/training")
        print("  API docs:      http://127.0.0.1:8000/docs")
        print("  WebSocket:     ws://127.0.0.1:8000/ws/map")
        print()
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[pc2d] shutting down ...")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()