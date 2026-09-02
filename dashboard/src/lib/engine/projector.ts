/**
 * Spherical range-image projector - TypeScript port of src/data/projection.py.
 *
 * Produces the exact input contract the exported ONNX model expects:
 *   image: CHW Float32Array (5, h, w) = [r/max_range, x/max_range, y/max_range,
 *          z/max_range, remission-raw], zero-filled, nearest point per pixel
 *   proj:  Int32Array (n, 2) of (row, col) per point for the 3x3 kNN gather
 *
 * Port notes (must stay identical to Python):
 *   - row scale is (h-1), col scale is w
 *   - floor-then-clamp to [0, h-1] / [0, w-1]
 *   - r = max(||p||, 1e-6)
 *   - nearest-wins per pixel (Python: descending-range argsort write order)
 *
 * Buffers are double-buffered (ping-pong): the engine pipeline starts frame
 * k's GPU inference from one slot while frame k-1's CPU stages still read the
 * other, so no stage ever overwrites data a pending frame depends on. The
 * slot also carries the per-point polar values (theta, planar radius) the
 * grid needs - computed here once instead of recomputed by the grid.
 */

export interface ProjectorOptions {
  h: number;
  w: number;
  fovTopDeg: number;
  fovBottomDeg: number;
  maxRange: number;
}

export interface ProjectorSlot {
  /** (maxPoints, 4) xyzi - filled by the frame reader, read by project/grid */
  points: Float32Array;
  /** CHW (5, h, w) model input image */
  image: Float32Array;
  /** (maxPoints, 2) (row, col) per point */
  proj: Int32Array;
  /** (maxPoints,) atan2(y, x) per point - reused by the grid */
  thetas: Float64Array;
  /** (maxPoints,) hypot(x, y) per point - reused by the grid */
  planarRs: Float64Array;
}

export class RangeProjector {
  readonly h: number;
  readonly w: number;
  readonly maxRange: number;
  private readonly fovTop: number;
  private readonly fovBottom: number;
  private readonly fov: number;
  private readonly slots: [ProjectorSlot, ProjectorSlot];
  private slotIdx = 0;
  /** Scratch: nearest range per pixel for the nearest-wins rule. */
  private readonly minRange: Float32Array;

  constructor(opts: ProjectorOptions, maxPoints: number) {
    this.h = opts.h;
    this.w = opts.w;
    this.maxRange = opts.maxRange;
    this.fovTop = (opts.fovTopDeg * Math.PI) / 180;
    this.fovBottom = (opts.fovBottomDeg * Math.PI) / 180;
    this.fov = this.fovTop - this.fovBottom;
    const mk = (): ProjectorSlot => ({
      points: new Float32Array(maxPoints * 4),
      image: new Float32Array(5 * this.h * this.w),
      proj: new Int32Array(maxPoints * 2),
      thetas: new Float64Array(maxPoints),
      planarRs: new Float64Array(maxPoints),
    });
    this.slots = [mk(), mk()];
    this.minRange = new Float32Array(this.h * this.w);
  }

  /** Flip to the next buffer set and return it. Call once per frame. */
  beginFrame(): ProjectorSlot {
    const s = this.slots[this.slotIdx];
    this.slotIdx ^= 1;
    return s;
  }

  /**
   * Project `slot.points` (n*4 flat xyzi) into the range image.
   * Writes image/proj and records thetas/planarRs for the grid.
   */
  project(slot: ProjectorSlot, n: number): void {
    const { h, w, minRange, maxRange } = this;
    const { points, image, proj, thetas, planarRs } = slot;
    const fovBottom = this.fovBottom;
    const fov = this.fov;
    const hw = h * w;
    image.fill(0);
    minRange.fill(Infinity);

    const chRange = 0 * hw;
    const chX = 1 * hw;
    const chY = 2 * hw;
    const chZ = 3 * hw;
    const chRem = 4 * hw;

    for (let k = 0; k < n; k++) {
      const x = points[k * 4];
      const y = points[k * 4 + 1];
      const z = points[k * 4 + 2];
      const rem = points[k * 4 + 3];
      let r = Math.sqrt(x * x + y * y + z * z);
      if (r < 1e-6) r = 1e-6;

      const rowf = (1 - (Math.asin(z / r) - fovBottom) / fov) * (h - 1);
      const colf = 0.5 * (Math.atan2(y, x) / Math.PI + 1) * w;
      let row = Math.floor(rowf);
      if (row < 0) row = 0;
      else if (row > h - 1) row = h - 1;
      let col = Math.floor(colf);
      if (col < 0) col = 0;
      else if (col > w - 1) col = w - 1;

      proj[k * 2] = row;
      proj[k * 2 + 1] = col;
      thetas[k] = Math.atan2(y, x);
      planarRs[k] = Math.hypot(x, y);

      const cell = row * w + col;
      if (r < minRange[cell]) {
        minRange[cell] = r;
        image[chRange + cell] = r / maxRange;
        image[chX + cell] = x / maxRange;
        image[chY + cell] = y / maxRange;
        image[chZ + cell] = z / maxRange;
        image[chRem + cell] = rem;
      }
    }
  }
}
