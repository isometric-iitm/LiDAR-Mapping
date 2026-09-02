/**
 * Log-polar variable-resolution 2.5D grid — TypeScript port of
 * src/grid_engine/logpolar_grid.py (LogPolarGrid).
 *
 * Semantics (strictly per-frame, "precise mode"):
 *   - occupancy is binary this scan only (hit = occ_gain, else 0 -> freed)
 *   - class is the per-frame majority vote of this scan's points
 *   - z stats are this scan's points rebased onto the per-frame ground
 *   - traversability is per-hit from height + majority class
 *   - no temporal state anywhere; stale content frees instantly
 *
 * Delta extraction is pure: computeDelta/computeSnapshot never mutate the
 * sent-tracking state; commit* applies it only after the frame was actually
 * delivered (here: posted to the main thread).
 */

export interface GridParams {
  rMin: number;
  rMax: number;
  dr0: number;
  rTransition: number;
  alpha: number;
  nTheta: number;
  zMin: number;
  zMax: number;
  nClasses: number;
  occGain: number;
  occThreshold: number;
}

export interface TravParams {
  enabled: boolean;
  weights: [number, number, number, number];
  zDiffThresh: number;
  slopeThresh: number;
  classScores: number[];
}

export const UNKNOWN = 255;

/** Cell row as delivered to the dashboard: [i, j, zMean, zMax, cls, occ, trav]. */
export type CellRow = [number, number, number, number, number, number, number];

export interface GridDelta {
  isSnap: boolean;
  frame: number;
  rows: CellRow[];
  freed: [number, number][];
}

export class LogPolarGrid {
  readonly params: GridParams;
  readonly trav: TravParams;
  readonly nRings: number;
  readonly phase1Rings: number;
  readonly phase2Rings: number;
  readonly nCells: number;
  readonly ringWidths: Float64Array;
  readonly ringEdges: Float64Array;
  frame = 0;
  groundZ = 0;

  // state (struct-of-arrays)
  private readonly zMean: Float32Array;
  private readonly zMinState: Float32Array;
  private readonly zMaxState: Float32Array;
  private readonly dominant: Uint8Array;
  private readonly occupancy: Float32Array;
  private readonly traversability: Float32Array;
  // sent-tracking (mutated only by commit*)
  private prevRendered: Uint8Array;
  private readonly sentZmax: Float32Array;
  private readonly sentCls: Uint8Array;

  // per-frame scratch (reused)
  private readonly nearZ: Float32Array;
  private readonly cellOf: Int32Array;
  private readonly order: Int32Array;
  private readonly clsCount: Int32Array; // per-unique-cell C counts
  private readonly segZ: Float32Array; // sorted z of kept points
  private readonly segCls: Uint8Array;

  constructor(params: GridParams, trav: TravParams, maxPoints: number) {
    this.params = params;
    this.trav = trav;
    const { rMin, rMax, dr0, rTransition, alpha, nTheta } = params;

    // two-phase ring geometry (mirrors logpolar_grid.py __init__)
    this.phase1Rings = Math.round((rTransition - rMin) / dr0);
    const phase2Span = rMax - (rMin + this.phase1Rings * dr0);
    if (phase2Span > 0 && alpha > 1) {
      const n2Exact = Math.log(1 + (phase2Span * (alpha - 1) / dr0)) / Math.log(alpha);
      this.phase2Rings = Math.max(1, Math.round(n2Exact));
    } else {
      this.phase2Rings = 0;
    }
    this.nRings = this.phase1Rings + this.phase2Rings;
    this.nCells = this.nRings * nTheta;

    this.ringWidths = new Float64Array(this.nRings);
    for (let i = 0; i < this.nRings; i++) {
      if (i < this.phase1Rings) {
        this.ringWidths[i] = dr0;
      } else {
        this.ringWidths[i] = dr0 * Math.pow(alpha, i - this.phase1Rings);
      }
    }
    if (this.phase2Rings > 0) {
      // clamp final ring so the outer edge lands exactly on rMax
      let sumPrev = 0;
      for (let i = 0; i < this.nRings - 1; i++) sumPrev += this.ringWidths[i];
      this.ringWidths[this.nRings - 1] = phase2Span - sumPrev;
    }
    this.ringEdges = new Float64Array(this.nRings + 1);
    this.ringEdges[0] = rMin;
    let cum = 0;
    for (let i = 0; i < this.nRings; i++) {
      cum += this.ringWidths[i];
      this.ringEdges[i + 1] = rMin + cum;
    }
    this.ringEdges[this.nRings] = rMax;

    this.zMean = new Float32Array(this.nCells);
    this.zMinState = new Float32Array(this.nCells).fill(Infinity);
    this.zMaxState = new Float32Array(this.nCells).fill(-Infinity);
    this.dominant = new Uint8Array(this.nCells).fill(UNKNOWN);
    this.occupancy = new Float32Array(this.nCells);
    this.traversability = new Float32Array(this.nCells);
    this.prevRendered = new Uint8Array(this.nCells);
    this.sentZmax = new Float32Array(this.nCells);
    this.sentCls = new Uint8Array(this.nCells).fill(UNKNOWN);

    const maxKeep = Math.max(1024, maxPoints);
    this.nearZ = new Float32Array(Math.min(maxKeep, 1 << 16));
    this.cellOf = new Int32Array(maxKeep);
    this.order = new Int32Array(maxKeep);
    this.clsCount = new Int32Array(maxKeep * params.nClasses);
    this.segZ = new Float32Array(maxKeep);
    this.segCls = new Uint8Array(maxKeep);
  }

  /** Ring index via binary search over ringEdges (searchsorted right - 1). */
  ringIndex(r: number): number {
    const e = this.ringEdges;
    let lo = 0;
    let hi = e.length; // n_rings + 1
    // find first index where e[idx] > r  (side="right")
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (e[mid] <= r) lo = mid + 1;
      else hi = mid;
    }
    let i = lo - 1;
    if (i < 0) i = 0;
    if (i > this.nRings - 1) i = this.nRings - 1;
    return i;
  }

  /** rendered mask: occupancy > threshold. */
  private renderedInto(mask: Uint8Array): void {
    const occ = this.occupancy;
    const t = this.params.occThreshold;
    for (let c = 0; c < this.nCells; c++) mask[c] = occ[c] > t ? 1 : 0;
  }

  /**
   * Frame update — port of update(points, per_class) + _apply_precise +
   * _clear_freed. Returns the number of hit cells (0 if nothing kept).
   */
  update(points: Float32Array, perClass: Uint8Array, n: number): number {
    const { rMin, rMax, zMin, zMax, nTheta, nClasses, occGain } = this.params;
    const { cellOf, order, segZ, segCls } = this;

    // polar coords + per-frame ground reference (20th percentile of near z)
    let nearCount = 0;
    const nearZ = this.nearZ;
    for (let k = 0; k < n; k++) {
      const x = points[k * 4];
      const y = points[k * 4 + 1];
      const z = points[k * 4 + 2];
      const r = Math.hypot(x, y);
      if (r > 1.5 && r < 15 && z > -8 && z < 4 && nearCount < nearZ.length) {
        nearZ[nearCount++] = z;
      }
    }
    if (nearCount >= 128) {
      // numpy.percentile(zn, 20) == linear interpolation on sorted values
      this.groundZ = percentile20(nearZ, nearCount);
    }

    // keep mask + cell ids
    let kept = 0;
    for (let k = 0; k < n; k++) {
      const x = points[k * 4];
      const y = points[k * 4 + 1];
      const z = points[k * 4 + 2];
      const r = Math.hypot(x, y);
      if (r < rMin || r > rMax || z < zMin || z > zMax) continue;
      const theta = Math.atan2(y, x);
      const ri = this.ringIndex(r);
      let sj = Math.floor((theta + Math.PI) / ((2 * Math.PI) / nTheta));
      sj %= nTheta;
      cellOf[kept] = ri * nTheta + sj;
      segZ[kept] = z;
      segCls[kept] = perClass[k];
      kept++;
    }
    if (kept === 0) {
      // Nothing valid this scan: everything previously rendered frees instantly.
      const preRendered = new Uint8Array(this.nCells);
      this.renderedInto(preRendered);
      for (let c = 0; c < this.nCells; c++) {
        if (preRendered[c]) {
          this.occupancy[c] = 0;
          this.zMinState[c] = Infinity;
          this.zMaxState[c] = -Infinity;
          this.zMean[c] = 0;
          this.dominant[c] = UNKNOWN;
          this.traversability[c] = 0;
        }
      }
      this.frame++;
      return 0;
    }

    const preRendered = new Uint8Array(this.nCells);
    this.renderedInto(preRendered);

    // stable sort kept indices by cell id (mirrors argsort(cells, kind="stable"))
    const idx = order.subarray(0, kept);
    for (let k = 0; k < kept; k++) idx[k] = k;
    sortIndicesByCell(idx, cellOf, kept);

    // segment reduce: run boundaries of equal cell ids
    let nUniq = 0;
    const uniqStart: number[] = []; // start offsets in sorted arrays
    for (let s = 0; s < kept; ) {
      const cell = cellOf[idx[s]];
      let e = s + 1;
      while (e < kept && cellOf[idx[e]] === cell) e++;
      uniqStart.push(s);
      nUniq++;
      s = e;
    }

    // per-unique-cell stats
    const clsCount = this.clsCount;
    clsCount.fill(0, 0, nUniq * nClasses);
    const uniqCells: number[] = new Array(nUniq);
    const zminU = new Float32Array(nUniq);
    const zmaxU = new Float32Array(nUniq);
    const zsumU = new Float64Array(nUniq);
    const cntU = new Int32Array(nUniq);
    const domU = new Uint8Array(nUniq);

    const wZ = this.trav.weights[0];
    const wS = this.trav.weights[1];
    const wC = this.trav.weights[2];
    const wO = this.trav.weights[3];
    const wSum = wZ + wS + wC + wO;
    const zDen = this.trav.zDiffThresh * 2.5;
    const sDen = this.trav.slopeThresh;
    const clsScores = this.trav.classScores;

    for (let u = 0; u < nUniq; u++) {
      const s = uniqStart[u];
      const e = u + 1 < nUniq ? uniqStart[u + 1] : kept;
      const cell = cellOf[idx[s]];
      uniqCells[u] = cell;
      let zmin = Infinity;
      let zmax = -Infinity;
      let zsum = 0;
      const cBase = u * nClasses;
      for (let q = s; q < e; q++) {
        const oi = idx[q];
        const z = segZ[oi];
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
        zsum += z;
        const c = segCls[oi];
        if (c < nClasses) clsCount[cBase + c]++;
      }
      const cnt = e - s;
      cntU[u] = cnt;
      zminU[u] = zmin;
      zmaxU[u] = zmax;
      zsumU[u] = zsum;

      // majority class
      let bestC = 0;
      let bestN = -1;
      for (let c = 0; c < nClasses; c++) {
        const v = clsCount[cBase + c];
        if (v > bestN) {
          bestN = v;
          bestC = c;
        }
      }
      domU[u] = bestC;
    }

    // apply precise state for hit cells
    const isHit = new Uint8Array(this.nCells);
    for (let u = 0; u < nUniq; u++) {
      const cell = uniqCells[u];
      this.occupancy[cell] = occGain;
      isHit[cell] = 1;
      this.dominant[cell] = domU[u];
      this.zMean[cell] = zsumU[u] / cntU[u];
      this.zMinState[cell] = zminU[u];
      this.zMaxState[cell] = zmaxU[u];

      if (this.trav.enabled) {
        const heightU = zmaxU[u] - this.groundZ;
        const hPos = heightU > 0 ? heightU : 0;
        let zScore = 1 - hPos / zDen;
        if (zScore < 0) zScore = 0;
        else if (zScore > 1) zScore = 1;
        let slopeScore = 1 - Math.max(heightU - 0.15, 0) / sDen;
        if (slopeScore < 0) slopeScore = 0;
        else if (slopeScore > 1) slopeScore = 1;
        const clsScore = domU[u] < clsScores.length ? clsScores[domU[u]] : 0;
        let occScore = occGain;
        if (occScore < 0) occScore = 0;
        else if (occScore > 1) occScore = 1;
        this.traversability[cell] = (wZ * zScore + wS * slopeScore + wC * clsScore + wO * occScore) / wSum;
      }
    }

    // instant free: previously rendered cells not hit this scan
    let anyFree = false;
    for (let c = 0; c < this.nCells; c++) {
      if (preRendered[c] && !isHit[c]) {
        this.occupancy[c] = 0;
        anyFree = true;
      }
    }
    // clear height/class state for cells that left the rendered mask
    if (anyFree) {
      for (let c = 0; c < this.nCells; c++) {
        if (preRendered[c] && !isHit[c]) {
          this.zMinState[c] = Infinity;
          this.zMaxState[c] = -Infinity;
          this.zMean[c] = 0;
          this.dominant[c] = UNKNOWN;
          this.traversability[c] = 0;
        }
      }
    }

    this.frame++;
    return nUniq;
  }

  /** Build the dashboard rows for the given flat cell ids. */
  private rowsFor(ids: Int32Array | number[], count: number): CellRow[] {
    const nTheta = this.params.nTheta;
    const g = this.groundZ;
    const rows: CellRow[] = new Array(count);
    for (let k = 0; k < count; k++) {
      const cell = ids[k];
      const i = Math.floor(cell / nTheta);
      const j = cell % nTheta;
      rows[k] = [i, j, this.zMean[cell] - g, this.zMaxState[cell] - g, this.dominant[cell], this.occupancy[cell], this.traversability[cell]];
    }
    return rows;
  }

  /** Pure snapshot of all rendered cells (no sent-tracking mutation). */
  computeSnapshot(): GridDelta {
    const mask = new Uint8Array(this.nCells);
    this.renderedInto(mask);
    const ids: number[] = [];
    for (let c = 0; c < this.nCells; c++) if (mask[c]) ids.push(c);
    return { isSnap: true, frame: this.frame, rows: this.rowsFor(ids, ids.length), freed: [] };
  }

  /**
   * Pure delta since the last commit: added cells, freed cells, and cells whose
   * z_max moved >= 5 cm or whose dominant class changed.
   */
  computeDelta(): GridDelta {
    const mask = new Uint8Array(this.nCells);
    this.renderedInto(mask);
    const ids: number[] = [];
    const freed: [number, number][] = [];
    const nTheta = this.params.nTheta;
    for (let c = 0; c < this.nCells; c++) {
      const m = mask[c];
      const prev = this.prevRendered[c];
      if (m && !prev) {
        ids.push(c); // added
      } else if (m) {
        const dz = Math.abs(this.zMaxState[c] - this.sentZmax[c]);
        if (dz >= 0.05 || this.dominant[c] !== this.sentCls[c]) ids.push(c); // changed
      } else if (prev) {
        freed.push([Math.floor(c / nTheta), c % nTheta]);
      }
    }
    return { isSnap: false, frame: this.frame, rows: this.rowsFor(ids, ids.length), freed };
  }

  /** Apply sent-tracking after the frame was actually delivered. */
  commit(delta: GridDelta): void {
    if (delta.isSnap) {
      const mask = new Uint8Array(this.nCells);
      this.renderedInto(mask);
      this.prevRendered = mask;
      for (let c = 0; c < this.nCells; c++) {
        if (mask[c]) {
          this.sentZmax[c] = this.zMaxState[c];
          this.sentCls[c] = this.dominant[c];
        } else {
          this.sentZmax[c] = 0;
          this.sentCls[c] = UNKNOWN;
        }
      }
    } else {
      const mask = new Uint8Array(this.nCells);
      this.renderedInto(mask);
      this.prevRendered = mask;
      for (const row of delta.rows) {
        const cell = row[0] * this.params.nTheta + row[1];
        this.sentZmax[cell] = this.zMaxState[cell];
        this.sentCls[cell] = this.dominant[cell];
      }
      const nTheta = this.params.nTheta;
      for (const [i, j] of delta.freed) {
        const cell = i * nTheta + j;
        this.sentZmax[cell] = 0;
        this.sentCls[cell] = UNKNOWN;
      }
    }
  }

  /** Full reset (seek / restart). */
  reset(): void {
    this.zMean.fill(0);
    this.zMinState.fill(Infinity);
    this.zMaxState.fill(-Infinity);
    this.dominant.fill(UNKNOWN);
    this.occupancy.fill(0);
    this.traversability.fill(0);
    this.prevRendered = new Uint8Array(this.nCells);
    this.sentZmax.fill(0);
    this.sentCls.fill(UNKNOWN);
    this.frame = 0;
    this.groundZ = 0;
  }

  /** Rendered-cell count (occupancy above threshold). */
  renderedCount(): number {
    let c = 0;
    const t = this.params.occThreshold;
    for (let i = 0; i < this.nCells; i++) if (this.occupancy[i] > t) c++;
    return c;
  }

  get bytesPerCell(): number {
    return 2 * 4 + this.params.nClasses * 4 + 1 + 4 + 4 + 8;
  }

  memoryReport(): { gridKb: number; renderedCells: number; nCells: number; compressionRatio: number; uniformMb: number } {
    const rendered = this.renderedCount();
    // uniform equivalent (grid.yaml memory block)
    const uniformCells = Math.pow(200 / 0.05, 2);
    const uniformBytes = uniformCells * this.bytesPerCell;
    return {
      gridKb: Math.round((rendered * this.bytesPerCell) / 1024 * 10) / 10,
      renderedCells: rendered,
      nCells: this.nCells,
      compressionRatio: rendered > 0 ? Math.round((uniformCells / rendered) * 10) / 10 : Math.round((uniformCells / this.nCells) * 10) / 10,
      uniformMb: Math.round(uniformBytes / 1e6 * 10) / 10,
    };
  }
}

/** Stable index sort by cell id (Array.prototype.sort is stable per ES2019+,
 *  matching numpy's kind="stable" argsort semantics). */
function sortIndicesByCell(idx: Int32Array, cellOf: Int32Array, kept: number): void {
  const arr = Array.from(idx.subarray(0, kept));
  arr.sort((a, b) => cellOf[a] - cellOf[b]);
  for (let k = 0; k < kept; k++) idx[k] = arr[k];
}

function percentile20(values: Float32Array, n: number): number {
  const sorted = Array.from(values.subarray(0, n));
  sorted.sort((a, b) => a - b);
  if (n === 1) return sorted[0];
  const rank = (n - 1) * 0.2;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}
