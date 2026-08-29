# Evaluation Summary

- **Timestamp**: 2026-08-29T22:50:47.838434+00:00
- **Device**: cpu | **Precision**: fp16

## Pixel-Level

- **Samples**: 50
- **mIoU (5-class)**: 0.6661
- **mIoU (4-class)**: 0.8258

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8831 |
| terrain_nondrivable | 0.8506 |
| static_obstacle | 0.7200 |
| dynamic_object | 0.8495 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.7747 |
| 5-10m | 0.8523 |
| 10-20m | 0.8327 |
| 20-40m | 0.7282 |
| 40-80m | 0.5977 |

### Latency: 442.1 +/- 1579.8 ms/batch

## Point-Level

- **Scans**: 50
- **mIoU (5-class)**: 0.5256
- **mIoU (4-class)**: 0.6533

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.8903 |
| terrain_nondrivable | 0.8567 |
| static_obstacle | 0.4641 |
| dynamic_object | 0.4022 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.6685 |
| 5-10m | 0.6159 |
| 10-20m | 0.6325 |
| 20-40m | 0.5660 |
| 40-80m | 0.4794 |

### Latency: 242.9 +/- 7.2 ms/scan

## Memory

- **Peak RSS**: 5876.7 MB
