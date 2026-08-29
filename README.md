# Adaptive Variable Resolution 2.5D LiDAR Mapping

<!-- RESULTS_START -->
## Evaluation Results

**Device**: cpu | **Precision**: fp16 | **Date**: 2026-08-29

### Pixel-Level mIoU

| Metric | Value |
|--------|-------|
| mIoU (5-class) | 0.6661 |
| mIoU (4-class) | 0.8258 |
| Samples | 50 |
| Avg latency | 442.1 ms/batch |

#### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8831 |
| terrain_nondrivable | 0.8506 |
| static_obstacle | 0.7200 |
| dynamic_object | 0.8495 |

#### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.7747 |
| 5-10m | 0.8523 |
| 10-20m | 0.8327 |
| 20-40m | 0.7282 |
| 40-80m | 0.5977 |

### Point-Level mIoU

| Metric | Value |
|--------|-------|
| mIoU (5-class) | 0.5256 |
| mIoU (4-class) | 0.6533 |
| Scans | 50 |
| Avg latency | 242.9 ms/scan |

#### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8903 |
| terrain_nondrivable | 0.8567 |
| static_obstacle | 0.4641 |
| dynamic_object | 0.4022 |

### Memory

| Metric | Value |
|--------|-------|
| Peak RSS | 5876.7 MB |

<!-- RESULTS_END -->
