# Evaluation Summary

- **Timestamp**: 2026-08-29T23:40:41.677077+00:00
- **Device**: cuda | **Precision**: fp16

## Pixel-Level

- **Samples**: 50
- **mIoU (5-class)**: 0.6843
- **mIoU (4-class)**: 0.8467

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.9000 |
| terrain_nondrivable | 0.8864 |
| static_obstacle | 0.7012 |
| dynamic_object | 0.8993 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.8092 |
| 5-10m | 0.8529 |
| 10-20m | 0.8384 |
| 20-40m | 0.7219 |
| 40-80m | 0.5818 |

### Latency: 1108.3 +/- 7661.8 ms/batch

## Point-Level

- **Scans**: 50
- **mIoU (5-class)**: 0.5256
- **mIoU (4-class)**: 0.6533

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8904 |
| terrain_nondrivable | 0.8568 |
| static_obstacle | 0.4641 |
| dynamic_object | 0.4020 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.6686 |
| 5-10m | 0.6160 |
| 10-20m | 0.6325 |
| 20-40m | 0.5661 |
| 40-80m | 0.4795 |

### Latency: 24.9 +/- 36.2 ms/scan

## Memory

- **Peak RSS**: 5876.7 MB
- **GPU Peak**: 323.5 MB
