# Adaptive Variable Resolution 2.5D LiDAR Mapping

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Tests 128 passing](https://img.shields.io/badge/tests-128%20passing-brightgreen.svg)](#testing)

Real-time semantic segmentation of LiDAR point clouds projected onto a **variable-resolution log-polar 2.5D grid**, streamed over binary WebSocket to an interactive 3D dashboard. Built on SemanticKITTI, powered by a range-image UNet.

![PC2D Dashboard Demo](docs/assets/dashboard.gif)

---

<!-- RESULTS_START -->
## Results

### Pixel-level segmentation

| | GPU (RTX 3060 Ti, fp16) | CPU (i5-11400F, fp32) |
|---|:---:|:---:|
| **mIoU 4-class** | **80.9%** | **82.6%** |
| **mIoU 5-class** | 67.8% | 66.6% |
| Latency | 7.8 ms/batch | ~442 ms/batch |
| Throughput | ~128 Hz | ~2.3 Hz |

### Point-level segmentation

| | GPU (RTX 3060 Ti, fp16) | CPU (i5-11400F, fp32) |
|---|:---:|:---:|
| **mIoU 4-class** | **75.8%** | **65.3%** |
| **mIoU 5-class** | 63.3% | 52.6% |
| Latency | 59.6 ms/scan | ~243 ms/scan |
| Throughput | ~19 Hz | ~4.1 Hz |

### Per-class IoU (4-class, GPU)

| Class | Pixel | Point |
|-------|:-----:|:-----:|
| Drivable | 84.6% | 81.1% |
| Terrain / Non-drivable | 85.4% | 82.4% |
| Static Obstacle | 68.5% | 65.7% |
| Dynamic Object | 85.2% | 73.9% |

### Per-distance-band mIoU (4-class, GPU pixel-level)

| Band | mIoU | Note |
|------|:----:|------|
| 0-5 m | 80.7% | Near-field - 5 cm cells |
| 5-10 m | 82.7% | |
| 10-20 m | 79.2% | |
| 20-40 m | 68.6% | |
| 40-80 m | 55.7% | Far-field - ~50 cm cells |

<details>
<summary>Per-distance-band mIoU tables (4-class &amp; 5-class)</summary>

**4-class**

| Band | Pixel | Point |
|------|:-----:|:-----:|
| 0-5 m | 80.7% | 74.7% |
| 5-10 m | 82.7% | 77.7% |
| 10-20 m | 79.2% | 74.6% |
| 20-40 m | 68.6% | 62.8% |
| 40-80 m | 55.7% | 50.3% |

**5-class**

| Band | Pixel | Point |
|------|:-----:|:-----:|
| 0-5 m | 72.0% | 61.8% |
| 5-10 m | 70.3% | 62.4% |
| 10-20 m | 65.7% | 56.1% |
| 20-40 m | 56.3% | 44.3% |
| 40-80 m | 38.4% | 31.7% |

![Confusion matrices](results/confusion_matrices.png)

</details>

### Memory: ~34x compression

| Grid type | Cell size | Cell count | Memory |
|-----------|-----------|------------|--------|
| Uniform 5 cm | 0.05 m x 0.05 m | ~16 million | ~784 MB |
| **Log-polar (ours)** | 5 cm -> 50 cm | ~469K | **~22 MB** |

<!-- RESULTS_END -->

### Test hardware

| | GPU Bench | CPU Bench |
|---|---|---|
| **CPU** | Intel i5-11400F @ 2.60 GHz | Same |
| **GPU** | NVIDIA RTX 3060 Ti (8 GB VRAM) | - |
| **RAM** | 16 GB DDR4 | Same |
| **OS** | Windows 11 | Same |
| **Python** | 3.14 | Same |
| **PyTorch** | 2.11.0+cu128 | Same |



## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/isometric-iitm/LiDAR-Mapping.git
cd LiDAR-Mapping/pc2d
uv sync

# 2. Download the trained checkpoint (~143 MB)
uv run python scripts/download_checkpoint.py

# 3. Set your data path (SemanticKITTI sequence 08)
cp .env.example .env
# Edit .env: PC2D_SEQ_DIR=F:/sih/dataset/sequences/08

# 4. Launch the backend (Terminal 1)
uv run python -m uvicorn src.server.app:create_app --factory --host 127.0.0.1 --port 8000

# 5. Launch the dashboard (Terminal 2)
cd dashboard
npm run dev
```

Open **http://localhost:3000** for the real-time dashboard (Training: **/training**, Evaluation: **/eval**). API docs are at **http://127.0.0.1:8000/docs**.

**No GPU?** Force CPU inference in the backend terminal before starting it:
```powershell
# PowerShell
$env:PC2D_DEVICE="cpu"; $env:PC2D_PRECISION="fp32"; $env:PC2D_PLAYBACK_SPEED="0.5"
```
Then run the backend `uvicorn` command from step 4.

---

## Architecture

```mermaid
flowchart LR
    subgraph Input
        BIN["Raw .bin scan"]
    end

    subgraph Segmenter
        PROJ["Spherical Projection"]
        RI["Range Image (5x64x2048)"]
        UNET["RangeImageUNet"]
        KNN["KNN Back-projection"]
        PPL["Per-point labels"]
    end

    subgraph Grid Engine
        BIN4["5->4 class bin"]
        LP["LogPolarGrid"]
        SNAP["Snapshot / Delta"]
    end

    subgraph Server
        WS["Binary WebSocket"]
        CTRL["Control (pause/seek/speed)"]
    end

    subgraph Dashboard
        MAP["3D Map (InstancedMesh)"]
        CLOUD["Point Cloud Overlay"]
        PANEL["Metrics + Timeline"]
    end

    BIN --> PROJ --> RI --> UNET --> KNN --> PPL
    PPL --> BIN4 --> LP --> SNAP --> WS
    CTRL --> WS
    WS --> MAP
    WS --> CLOUD
    WS --> PANEL
```

### Pipeline stages

| Stage | What | GPU path | CPU path |
|-------|------|----------|----------|
| Projection | Points -> (row, col, range) | `project_points_gpu` (all torch on-device) | `compute_projection` (numpy) |
| Range image | Scatter into 5x64x2048 tensor | `build_range_image_gpu` (nearest-wins dedup) | `build_range_image` (far-wins overwrite) |
| Segmentation | UNet forward pass | fp16 autocast + `torch.compile` | fp32, no compile |
| Back-projection | Pixel logits -> per-point probs | `_knn_probs` (3x3 gather) | same (tensor ops) |
| Grid update | Per-cell reduce + class majority | numba JIT fused kernel (`jit_reduce`), fallback to numpy `reduceat`+`bincount` | same |
| Pipeline | Thread orchestration | Two-stage (SEG thread -> GRID thread), hides GPU `.cpu()` sync behind CPU grid work | `stages: single` fallback |
| Streaming | Grid cells -> binary frames | `ws_protocol` (44-byte header, 28-byte rows) | same |

---

## Project Structure

```
pc2d/
├── config/                   # YAML configs (env overrides via PC2D_* vars)
│   ├── pipeline.yaml         #   live streaming pipeline
│   ├── train_range_image.yaml#   training / preprocessing
│   ├── grid.yaml             #   log-polar grid geometry + occupancy knobs
│   └── classes.yaml          #   semantic_kitti_5class + bin_5_to_4 mapping
├── src/
│   ├── common/config.py      # load_config, resolve_path, resolve_device/precision
│   ├── data/
│   │   ├── projection.py     # CPU + GPU spherical projection, range-image build
│   │   ├── label_mapping.py  # remap_labels, bin_5_to_4, compute_class_weights
│   │   ├── augmentation.py   # flip-H, intensity jitter, cutout
│   │   ├── webdataset_loader.py  # tar-shard IterableDataset + DataLoader
│   │   └── replayer.py       # wall-clock-paced SemanticKITTI sequence reader
│   ├── models/
│   │   ├── unet.py           # RangeImageUNet (5-level encoder-decoder)
│   │   ├── lovasz.py         # CombinedLoss (CE + Lovasz-Softmax)
│   │   └── predict.py        # Segmenter (end-to-end: project -> forward -> KNN)
│   ├── grid_engine/
│   │   ├── logpolar_grid.py  # LogPolarGrid (~469K cells, ~22 MB, two-phase, pure-delta, JIT-accelerated reduce)
│   │   └── jit_reduce.py     # numba fused per-cell scatter-reduce (min/max/sum/count/class)
│   └── server/
│       ├── app.py            # FastAPI + two-stage Pipeline (SEG thread + GRID thread) + WS endpoints
│       └── ws_protocol.py   # Binary framing (magic 0x50433244, v3)
├── eval/
│   ├── run_evaluation.py     # pixel + point mIoU, per-distance-band, latency
│   └── inject_results.py     # auto-inject eval results into this README
├── scripts/
│   ├── train.py              # train the UNet on precomputed shards
│   ├── preprocess_to_shards.py  # raw .bin -> tar shards (ri + li)
│   ├── download_checkpoint.py# fetch best_miou.pt from GitHub Release
│   └── export_onnx.py        # export UNet to ONNX (opset 17)
├── tests/                    # 128 tests (see Testing section)
├── results/                  # eval outputs (JSON, PNG, MD)
├── checkpoints/              # best_miou.pt (gitignored), history.jsonl
├── dashboard/                # Next.js 16 real-time dashboard (see dashboard/README.md)
└── .env.example              # documents all PC2D_* environment variables
```

---

## Variable-Resolution Grid

The core innovation: a **two-phase** ring geometry. Phase 1 keeps uniform **5 cm** rings from the sensor out to 10 m, exactly matching the PS requirement of "5cm cells within 10m radius". Phase 2 then grows each ring's width geometrically to **~50 cm** out to 100 m, achieving dramatic memory savings while preserving fine detail where it matters most. The seam at 10 m is invisible (Phase 2's first ring is also 5 cm).

### Ring geometry

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dr_0` | 0.05 m | Ring base width (5 cm) |
| `r_transition` | 10.0 m | Phase 1 -> Phase 2 boundary (uniform -> geometric) |
| `alpha` | 1.005 | Phase 2 geometric growth factor per ring |
| `r_min` | 0.5 m | Inner dead zone |
| `r_max` | 100 m | Outer cutoff |
| `n_theta` | 720 | Angular sectors (0.5deg each) |
| Phase 1 rings | 190 | Uniform 5 cm rings (0.5 -> 10 m) |
| Phase 2 rings | 462 | Geometric 5 cm -> ~50 cm (10 -> 100 m) |
| Resulting rings | ~652 | Phase 1 + Phase 2 |
| Resulting cells | ~469K | rings x sectors |

### Per-frame occupancy (precise sensor mode only)

The grid is strictly per-frame; occupancy is this scan's sensor truth, nothing
more. There is no temporal decay/EMA: any cell not hit this scan goes free the
instant it leaves the sensor view, so stale cells never linger (no ghosts on the
dashboard).

| Feature | Mechanism | Parameter |
|---------|-----------|-----------|
| Occupancy on hit | binary hit = `occ_gain` for this scan's cells | `occupancy_gain = 1.0` |
| Enter/leave render | `occ > occ_threshold` | `occ_threshold = 0.2` |
| Freeing | anything not hit this scan -> `occ = 0` immediately | - |
| Ground reference | 20th percentile of z in 1.5-15 m | auto-tracked per frame |

---

## Label Mapping

```mermaid
flowchart LR
    subgraph Raw["SemanticKITTI (34 classes)"]
        R0["road, parking, sidewalk"]
        R1["terrain, vegetation"]
        R2["building, fence, pole..."]
        R3["car, truck, bus..."]
        R4["pedestrian, rider..."]
        RX["(unmapped)"]
    end

    subgraph Train["5-class (train)"]
        T0["0: drivable"]
        T1["1: terrain_nondrivable"]
        T2["2: static_obstacle"]
        T3["3: dynamic_vehicle"]
        T4["4: dynamic_pedestrian"]
        T255["255: ignore"]
    end

    subgraph Eval["4-class (display/eval)"]
        E0["0: drivable"]
        E1["1: terrain_nondrivable"]
        E2["2: static_obstacle"]
        E3["3: dynamic_object"]
        E255["255: ignore"]
    end

    R0 --> T0 --> E0
    R1 --> T1 --> E1
    R2 --> T2 --> E2
    R3 --> T3 --> E3
    R4 --> T4 --> E3
    RX --> T255 --> E255
```

Training uses 5-class with inverse-sqrt-frequency class weights. Grid display and evaluation use the binned 4-class scheme (merging vehicle + pedestrian -> dynamic_object).

---

## Configuration

All paths and model parameters come from **config YAML + environment overrides**. No hardcoded absolute paths.

### Precedence (highest wins)

```
CLI flag  >  PC2D_* env var  >  config YAML  >  default
```

### Environment variables

| Variable | Config target | Type | Description |
|----------|--------------|------|-------------|
| `PC2D_SEQ_DIR` | `source.seq_dir` | path | Raw SemanticKITTI sequence directory |
| `PC2D_CHECKPOINT` | `model.checkpoint` | path | Trained model checkpoint (.pt) |
| `PC2D_CKPT_DIR` | `server.ckpt_dir`, `checkpoint.dir` | path | Checkpoint + history directory |
| `PC2D_DEVICE` | `model.device` | `auto\|cuda\|cpu` | Compute device |
| `PC2D_PRECISION` | `model.precision` | `fp32\|fp16` | Model precision |
| `PC2D_PLAYBACK_SPEED` | `source.playback_speed` | float | Playback speed multiplier |
| `PC2D_RAW_ROOT` | `data.raw_root` | path | Root of raw sequences |
| `PC2D_PROCESSED_ROOT` | `data.processed_root` | path | Root of preprocessed shards |
| `PC2D_CORS_ORIGINS` | `dashboard.cors_origins` | str (comma-sep) | Extra allowed CORS origins for the API (Cloudflare Pages `*.pages.dev` always allowed) |

All optional. Unset vars fall back to relative defaults in YAML, resolved against the `pc2d/` repo root. Copy `.env.example` to `.env` for your machine.

---

## Device & Precision

| Device | Precision | Behavior |
|--------|-----------|----------|
| `auto` | - | Resolves to CUDA if available, else CPU |
| `cuda` | `fp16` | Model `.half()`, AMP autocast, `torch.compile`, GradScaler |
| `cuda` | `fp32` | Full precision, no AMP, no compile |
| `cpu` | any | No `.half()`, no AMP, no GradScaler, no compile |

---

## Evaluation

```bash
# Full eval (pixel + point level) on GPU
uv run python eval/run_evaluation.py --device cuda

# CPU-only eval
uv run python eval/run_evaluation.py --device cpu

# Pixel-level only (uses preprocessed shards)
uv run python eval/run_evaluation.py --pixel-only --limit 100

# Point-level only (uses raw .bin + .label files)
uv run python eval/run_evaluation.py --point-only --seq-dir /path/to/sequences/08

# Inject latest results into README
uv run python eval/inject_results.py
```

Outputs written to `results/`:

| File | Description |
|------|-------------|
| `eval_{timestamp}.json` | Full results (confusion matrices, per-class IoU, per-distance-band) |
| `confusion_matrices.png` | Row-normalized 5-class and 4-class confusion matrices |
| `per_class_iou.png` | Per-class IoU bar chart |
| `per_distance_band.png` | mIoU by distance band (pixel + point) |
| `eval_summary.md` | Human-readable Markdown summary |

---

## Scripts Reference

| Script | Purpose | Key args |
|--------|---------|----------|
| `scripts/train.py` | Train RangeImageUNet | `--config`, `--resume`, `--device`, `--precision`, `--processed-root` |
| `scripts/preprocess_to_shards.py` | Raw .bin -> tar shards | `--raw-root`, `--processed-root`, `--train-seqs`, `--val-seqs`, `--force` |
| `scripts/download_checkpoint.py` | Fetch best_miou.pt from Release | `--url`, `--out`, `--force` |
| `scripts/export_onnx.py` | Export UNet to ONNX | `--checkpoint`, `--out`, `--opset 17` |
| `eval/run_evaluation.py` | Full eval harness | `--device`, `--precision`, `--pixel-only`, `--point-only`, `--limit` |
| `eval/inject_results.py` | Inject eval into README | (reads `results/`, writes `README.md`) |

---

## WebSocket Protocol

Real-time data uses a compact binary framing protocol (40-100x smaller than JSON):

### Header (44 bytes)

| Field | Type | Description |
|-------|------|-------------|
| `magic` | uint32 | `0x50433244` ("PC2D") |
| `code` | uint16 | 1=snapshot, 2=delta, 3=cloud |
| `version` | uint16 | 2 |
| `frame` | uint64 | Grid frame counter |
| `epoch` | uint64 | Bumped on seek (stale-frame filter) |
| `n_a` | int32 | Active/updated rows |
| `n_b` | int32 | Freed rows (delta only) |
| `seq` | int32 | Chunk sequence index |
| `total` | int32 | Total chunks in this frame |
| `yaw_cd` | int32 | Ego yaw in centi-degrees |

### Body

| Message | Layout |
|---------|--------|
| Snapshot | `n_a` x 28 bytes (i, j, z_mean, z_max, occ, dyn as float32 + cls as uint8 + 3 pad) |
| Delta | same + `n_b` x 8 bytes (freed i, j as float32) |
| Cloud | `n_a` x 12 bytes (x, y, z as float32) + `n_a` x uint8 (cls) |

Large frames split at 8,000 cells / 30,000 cloud points. The epoch field lets the client discard in-flight stale data after a seek.

---

## Testing

```bash
uv run pytest
```

128 tests across 13 files, zero dependency on external data paths:

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test_grid_geometry.py` | 26 | Ring/sector indexing, cell counts, memory report, round-trip xy->cell |
| `test_projection.py` | 10 | CPU/GPU parity, nearest-wins, KNN back-projection, numba JIT parity, edge cases |
| `test_label_mapping.py` | 9 | remap_labels, bin_5_to_4, compute_class_weights |
| `test_grid_update.py` | 8 | Per-frame occupancy (precise), snapshot/delta compute/commit invariants, dropped-delta recompute, reset |
| `test_ws_protocol.py` | 8 | Binary header magic/codes, 28-byte row layout (v3), raw-deflate wire compression round-trip |
| `test_jit_reduce.py` | 4 | numba fused reduce vs numpy equivalence, single-point/segment/class tallies |
| `test_segmenter_cpu.py` | 3 | Empty scan, synthetic scan, CPU device |
| `test_e2e_replay.py` | 5 | Replayer seek, pipeline seek-reset + epoch guard |
| `test_ws_reliability.py` | 6 | Lossless pure-delta drops, catchup snapshot threshold, client-state convergence |
| `test_evaluation.py` | 25 | Eval helpers, path resolution, synthetic pixel eval, plots, inject |
| `test_traversability.py` | 6 | Flat drivable, steep blocked, class scores, tiers, grid integration |
| `test_harness.py` | 12 | Fixture smoke tests, projection, grid geometry |
| `test_replayer_prefetch.py` | 6 | Bounded prefetch buffer, natural cadence, seek invalidation, loop wrap |

---

## Checkpoint Delivery

Model weights are **not committed to git** (143 MB binary). Instead they're delivered via GitHub Release:

```bash
uv run python scripts/download_checkpoint.py
```

This fetches `best_miou.pt` from [v1.0.0](https://github.com/isometric-iitm/LiDAR-Mapping/releases/tag/v1.0.0) (143 MB, mIoU 80.9% 4-class pixel-level on full val set). See `checkpoints/README.md` for publishing newer checkpoints.

---

## Dashboard

A Next.js 16 + Three.js real-time 3D dashboard with interactive map, point cloud overlays, playback controls, and training curves. See **[dashboard/README.md](dashboard/README.md)** for full documentation.

| 2.5D Grid | Segmented Cloud | Compare (Grid + Raw Cloud) | Traversability |
|-----------|-----------------|----------------------------|----------------|
| ![2.5D Grid](docs/assets/grid_2.5d.png) | ![Segmented Cloud](docs/assets/segmented_cloud.png) | ![Compare View](docs/assets/compare_view.png) | ![Traverse View](docs/assets/traverse.png) |

*Full animated demo at the top of this README.*

---

## Credits

- **Dataset**: [SemanticKITTI](http://semantic-kitti.org/) - Geiger et al., IV 2012 / Behley et al., ICCV 2019
- **Framework**: [PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/), [Next.js](https://nextjs.org/), [Three.js](https://threejs.org/)
- **Loss**: Lovasz-Softmax - Berman et al., CVPR 2018
