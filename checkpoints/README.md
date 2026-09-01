# Checkpoints

| File | Size | Tracked in git? | Notes |
|------|------|-----------------|-------|
| `best_miou.pt` | 143 MB | No (gitignored via `*.pt`) | Trained range-image UNet; used by the live pipeline (`config/pipeline.yaml` → `model.checkpoint`) and eval |
| `history.jsonl` | 42 KB | Yes | Training metrics log; powers the `/training` dashboard (`TrainingCurves.tsx`) |

## Why is `best_miou.pt` not committed?

It is a **143 MB binary**, too large to track in plain git. Adding binary blobs of this size bloats the repository and slows every clone.

## Delivery: GitHub Release (not Git LFS)

Checkpoints are delivered as a **GitHub Release asset**, deliberately **not** via Git LFS.

Rationale: models can grow well beyond 143 MB. Git LFS has free bandwidth quotas that larger models can exceed and adds a mandatory client-side dependency for every collaborator. A Release asset has no per-download quota at these sizes and requires no LFS setup; anyone can fetch it with a plain HTTPS request.

### How judges get the model (one command)

```
uv run python scripts/download_checkpoint.py
```

This writes `checkpoints/best_miou.pt` (same file judges would need either way).
`download_checkpoint.py` already defaults to the **live** Release asset URL for repo
`isometric-iitm/LiDAR-Mapping`, tag `v1.0.0`:
`https://github.com/isometric-iitm/LiDAR-Mapping/releases/download/v1.0.0/best_miou.pt`
Pass `--url` to override. See `scripts/download_checkpoint.py --help` for options (`--url`, `--out`, `--force`).

> Current model: **v1.0.0** published (`best_miou.pt`, mIoU 80.9% 4-class pixel-level on full val set).

### How to publish a NEWER checkpoint

When a better model is trained, publish it under the same flow (bump the tag):

1. Log in once (interactive, browser/device flow):
   ```
   gh auth login
   ```
2. Create a Release with the new `.pt` as an asset:
   ```
   gh release create v1.1.0 checkpoints/best_miou.pt --title "v1.1.0" --notes "Improved checkpoint (mIoU <X>%)"
   ```
   (The same works in the GitHub web UI under **Releases → Create a new release**.)
3. Update the tag in `scripts/download_checkpoint.py` (`_TAG = "v1.1.0"`).

> Do **not** run `git add checkpoints/best_miou.pt`; the `*.pt` rule in `.gitignore` keeps it out, and it should stay out. Only `history.jsonl` and this `README.md` are committed.
