#!/usr/bin/env python3
"""Download the trained checkpoint (best_miou.pt) from a GitHub Release asset.

The model is shipped as a Release asset (not committed, not Git LFS, see
checkpoints/README.md). Judges fetch it with:

    uv run python scripts/download_checkpoint.py

Defaults to the Release asset URL for isometric-iitm/LiDAR-Mapping (tag v1.0.0)
and writes checkpoints/best_miou.pt (resolved relative to the pc2d/ repo root,
so this works from any working directory). Pass --url/--out/--force to override.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

# Default Release asset URL for isometric-iitm/LiDAR-Mapping, tag v1.0.0.
# Change _TAG/_REPO here (or pass --url) when a newer checkpoint is published.
_REPO = "isometric-iitm/LiDAR-Mapping"
_TAG = "v1.0.0"
DEFAULT_URL = f"https://github.com/{_REPO}/releases/download/{_TAG}/best_miou.pt"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded / total_size * 100.0)
    done = int(50 * downloaded / total_size)
    bar = "#" * done + "-" * (50 - done)
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  ({downloaded / 1e6:8.1f} / {total_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="Direct download URL of the Release asset (default: remote v1.0.0 asset URL)")
    ap.add_argument("--out", default="checkpoints/best_miou.pt",
                    help="Output path, relative to pc2d/ repo root (default: checkpoints/best_miou.pt)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the destination even if it already exists")
    args = ap.parse_args()

    if not args.url:
        ap.error("DEFAULT_URL is empty; set it in the script after publishing the Release")

    out = repo_root() / args.out
    if out.exists() and not args.force:
        print(f"[download_checkpoint] {out} already exists ({out.stat().st_size / 1e6:.1f} MB). Use --force to re-download.")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_checkpoint] downloading {args.url}")
    print(f"  -> {out}")
    req = urllib.request.Request(args.url, headers={"User-Agent": "pc2d/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            read = 0
            with open(out, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    _progress(read, 1, total)
            sys.stdout.write("\n")
    except Exception as e:
        out.unlink(missing_ok=True)
        print(f"\n[download_checkpoint] FAILED: {e}")
        return 1

    print(f"[download_checkpoint] done ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
