# Evaluation Summary

- **Timestamp**: 2026-08-31T15:45:16.585816+00:00
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

### Latency: 10.1 +/- 69.9 ms/batch

## Point-Level

- **Scans**: 4071
- **mIoU (5-class)**: 0.6326
- **mIoU (4-class)**: 0.7577

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8113 |
| terrain_nondrivable | 0.8242 |
| static_obstacle | 0.6565 |
| dynamic_object | 0.7389 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.7467 |
| 5-10m | 0.7766 |
| 10-20m | 0.7465 |
| 20-40m | 0.6282 |
| 40-80m | 0.5035 |

### Latency: 54.0 +/- 14.9 ms/scan

## Memory

- **Peak RSS**: 5876.8 MB
- **GPU Peak**: 323.5 MB
