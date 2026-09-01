# Evaluation Summary

- **Timestamp**: 2026-09-01T12:26:05.747174+00:00
- **Device**: cuda | **Precision**: fp16

## Pixel-Level

- **Samples**: 5
- **mIoU (5-class)**: 0.7211
- **mIoU (4-class)**: 0.8409

### Per-Class IoU (4-class)

| Class | IoU |
|-------|-----|
| drivable | 0.9133 |
| terrain_nondrivable | 0.8503 |
| static_obstacle | 0.6838 |
| dynamic_object | 0.9163 |

### Per-Distance-Band mIoU (4-class)

| Band | mIoU |
|------|------|
| 0-5m | 0.8478 |
| 5-10m | 0.8404 |
| 10-20m | 0.7753 |
| 20-40m | 0.7216 |
| 40-80m | 0.5234 |

### Latency: 2924.4 +/- 5695.8 ms/batch

## Point-Level: Skipped (no raw data)

## Memory

- **Peak RSS**: 5876.3 MB
- **GPU Peak**: 323.5 MB
