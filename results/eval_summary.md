# Evaluation Summary

- **Timestamp**: 2026-09-01T16:52:50.819736+00:00
- **Device**: cuda | **Precision**: fp16

## Pixel-Level

- **Samples**: 4071
- **mIoU (5-class)**: 0.6779
- **mIoU (4-class)**: 0.8092

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8458 |
| terrain_nondrivable | 0.8539 |
| static_obstacle | 0.6847 |
| dynamic_object | 0.8523 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.8065 |
| 5-10m | 0.8269 |
| 10-20m | 0.7916 |
| 20-40m | 0.6864 |
| 40-80m | 0.5573 |

### Latency: 7.8 ms/batch (mean 8.3 +/- 3.2, p90 10.3)

## Point-Level

- **Scans**: 4071
- **mIoU (5-class)**: 0.6331
- **mIoU (4-class)**: 0.7584

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8117 |
| terrain_nondrivable | 0.8252 |
| static_obstacle | 0.6582 |
| dynamic_object | 0.7384 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.7462 |
| 5-10m | 0.7773 |
| 10-20m | 0.7474 |
| 20-40m | 0.6297 |
| 40-80m | 0.5033 |

### Latency: 59.6 ms/scan (mean 58.6 +/- 9.0, p90 63.6)

## Memory

- **Peak RSS**: 6838.2 MB
- **GPU Allocated**: 307.3 MB
- **GPU Reserved**: 328.0 MB
