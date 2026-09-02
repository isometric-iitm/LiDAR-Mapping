/**
 * Log-polar variable-resolution 2.5D grid — TypeScript port of
 * src/grid_engine/logpolar_grid.py (LogPolarGrid), optimized for the browser:
 *
 *  - Counting sort by cell id (stable, O(n + n_cells)) replaces the O(n log n)
 *    comparator sort; stability preserves within-cell point order, so the
 *    segment reduce (zmin/zmax/zsum) matches numpy reduceat semantics.
 *  - The rendered-cell set is maintained incrementally as a compact list +
 *    membership flags, so delta extraction and instant-free walk ~52k live
 *    cells instead of scanning all 469,440 cells five times per frame.
 *
 * Semantics (strictly per-frame, "precise mode" — identical to Python):
 *   - occupancy is binary this scan only (hit = occGain, else 0 -> freed)
 *   - class is the per-frame majority vote of this scan's points
 *   - z stats are this scan's points rebased onto the per-frame ground
 *   - traversability is per-hit from height + majority class
 *   - no temporal state anywhere; stale content frees instantly
 *
 * Delta extraction stays pure: computeDelta/computeSnapshot never mutate the
 * sent-tracking state; commit() applies it only after the frame was
 * actually delivered.
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

/** Binary-packed patch. `rows`/`cls`/`freed` are the transferable transport
 *  buffers; the `up*`/`freedIds` fields are worker-side scratch that
 *  commit() reads AFTER the transport buffers were transferred (detached). */
export interface PackedPatch {
  isSnap: boolean;
  frame: number;
  /** (k,6) f32: i, j, zMean, zMax, occ, trav — transferable */
  rows: Float32Array;
  /** (k,) u8: cls — transferable */
  cls: Uint8Array;
  /** (m,2) f32: freed [i,j] pairs — transferable */
  freed: Float32Array;
  /** worker-side: upsert cell ids (NOT transferred) */
  upIds: Int32Array;
  upCount: number;
  /** worker-side: freed cell ids (NOT transferred) */
  freedIds: Int32Array;
  freedCount: number;
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

  // per-cell state (struct-of-arrays)
  private readonly zMean: Float32Array;
  private readonly zMaxState: Float32Array;
  private readonly dominant: Uint8Array;
  private readonly occupancy: Float32Array;
  private readonly traversability: Float32Array;

  // rendered-cell tracking: compact list + membership flags (no full scans)
  private readonly renderedList: Int32Array; // flat cell ids (capacity nCells)
  private renderedCount = 0;
  private readonly inRendered: Uint8Array; // 1 if cell is in renderedList

  // sent-tracking (mutated only by commit*)
  private readonly sentZmax: Float32Array;
  private readonly sentCls: Uint8Array;
  private readonly wasSent: Uint8Array; // 1 if cell was in the last committed mask
  private prevSentList: Int32Array; // the committed mask as a compact list
  private prevSentCount = 0;

  // per-frame scratch (allocated once)
  private readonly nearZ: Float32Array;
  private readonly cellOf: Int32Array;
  private readonly zOf: Float32Array;
  private readonly clsOf: Uint8Array;
  private readonly hist: Int32Array; // counting-sort histogram (also hit-test)
  private readonly cursor: Int32Array; // counting-sort scatter cursor
  private readonly sortedIdx: Int32Array;
  // per-unique-cell reduce outputs (unique cells <= kept points)
  private readonly uCell: Int32Array;
  private readonly uZmin: Float32Array;
  private readonly uZmax: Float32Array;
  private readonly uZsum: Float64Array;
  private readonly uCnt: Int32Array;
  private readonly uClsCounts: Int32Array; // (cap * nClasses)
  // delta-path scratch
  private readonly freeStamp: Uint32Array; // freed-membership stamp (no clearing)
  private freeStampValue = 0;

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
      this.ringWidths[i] = i < this.phase1Rings ? dr0 : dr0 * Math.pow(alpha, i - this.phase1Rings);
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
    this.zMaxState = new Float32Array(this.nCells).fill(-Infinity);
    this.dominant = new Uint8Array(this.nCells).fill(UNKNOWN);
    this.occupancy = new Float32Array(this.nCells);
    this.traversability = new Float32Array(this.nCells);

    this.renderedList = new Int32Array(this.nCells);
    this.inRendered = new Uint8Array(this.nCells);
    this.sentZmax = new Float32Array(this.nCells);
    this.sentCls = new Uint8Array(this.nCells).fill(UNKNOWN);
    this.wasSent = new Uint8Array(this.nCells);
    this.prevSentList = new Int32Array(1024);
    this.freeStamp = new Uint32Array(this.nCells);

    const cap = Math.max(1024, maxPoints);
    this.nearZ = new Float32Array(cap);
    this.cellOf = new Int32Array(cap);
    this.zOf = new Float32Array(cap);
    this.clsOf = new Uint8Array(cap);
    this.hist = new Int32Array(this.nCells);
    this.cursor = new Int32Array(this.nCells);
    this.sortedIdx = new Int32Array(cap);
    // unique cells are bounded by kept points (several points may share a cell)
    const uCap = Math.max(4096, cap);
    this.uCell = new Int32Array(uCap);
    this.uZmin = new Float32Array(uCap);
    this.uZmax = new Float32Array(uCap);
    this.uZsum = new Float64Array(uCap);
    this.uCnt = new Int32Array(uCap);
    this.uClsCounts = new Int32Array(uCap * params.nClasses);
  }

  /** Ring index via binary search over ringEdges (searchsorted right - 1). */
  ringIndex(r: number): number {
    const e = this.ringEdges;
    let lo = 0;
    let hi = e.length; // n_rings + 1
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

  /**
   * Frame update — port of update(points, per_class) + _apply_precise +
   * _clear_freed, with counting sort + incremental rendered-list maintenance.
   * thetas/planarRs come from the projector slot (computed once, reused here).
   */
  update(points: Float32Array, perClass: Uint8Array, n: number, thetas: Float64Array, planarRs: Float64Array): number {
    const { rMin, rMax, zMin, zMax, nTheta, nClasses, occGain } = this.params;

    // ---- per-frame ground reference (20th percentile of near z) ----
    let nearCount = 0;
    const nearZ = this.nearZ;
    for (let k = 0; k < n; k++) {
      const z = points[k * 4 + 2];
      const r2 = planarRs[k] * planarRs[k]; // 1.5^2=2.25, 15^2=225 (sqrt-free)
      if (r2 > 2.25 && r2 < 225 && z > -8 && z < 4 && nearCount < nearZ.length) {
        nearZ[nearCount++] = z;
      }
    }
    if (nearCount >= 128) this.groundZ = percentile20(nearZ, nearCount);

    // ---- cell ids (keep mask; theta/r reused from the projector pass) ----
    let kept = 0;
    const cellOf = this.cellOf;
    const zOf = this.zOf;
    const clsOf = this.clsOf;
    for (let k = 0; k < n; k++) {
      const r = planarRs[k];
      const z = points[k * 4 + 2];
      if (r < rMin || r > rMax || z < zMin || z > zMax) continue;
      const ri = this.ringIndex(r);
      let sj = Math.floor((thetas[k] + Math.PI) / ((2 * Math.PI) / nTheta));
      sj %= nTheta;
      cellOf[kept] = ri * nTheta + sj;
      zOf[kept] = z;
      clsOf[kept] = perClass[k];
      kept++;
    }
    if (kept === 0) {
      // Nothing valid this scan: everything previously rendered frees instantly.
      this.freeAllRendered();
      this.frame++;
      return 0;
    }

    // ---- counting sort by cell id (stable; histogram doubles as hit-test) ----
    const hist = this.hist;
    hist.fill(0, 0, this.nCells);
    for (let k = 0; k < kept; k++) hist[cellOf[k]]++;
    const cursor = this.cursor;
    // exclusive prefix sum -> scatter offsets (stable sort)
    cursor[0] = 0;
    for (let c = 1; c < this.nCells; c++) cursor[c] = cursor[c - 1] + hist[c - 1];
    const sorted = this.sortedIdx;
    for (let k = 0; k < kept; k++) {
      const cell = cellOf[k];
      sorted[cursor[cell]++] = k;
    }

    // ---- segment reduce over unique cells ----
    const C = nClasses;
    const uCell = this.uCell;
    const uZmin = this.uZmin;
    const uZmax = this.uZmax;
    const uZsum = this.uZsum;
    const uCnt = this.uCnt;
    const uClsCounts = this.uClsCounts;
    let nUniq = 0;
    for (let s = 0; s < kept; ) {
      const cell = cellOf[sorted[s]];
      let e = s + 1;
      while (e < kept && cellOf[sorted[e]] === cell) e++;
      let zmin = Infinity;
      let zmax = -Infinity;
      let zsum = 0;
      const cBase = nUniq * C;
      for (let c = 0; c < C; c++) uClsCounts[cBase + c] = 0;
      for (let q = s; q < e; q++) {
        const oi = sorted[q];
        const z = zOf[oi];
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
        zsum += z;
        const c = clsOf[oi];
        if (c < C) uClsCounts[cBase + c]++;
      }
      uCell[nUniq] = cell;
      uZmin[nUniq] = zmin;
      uZmax[nUniq] = zmax;
      uZsum[nUniq] = zsum;
      uCnt[nUniq] = e - s;
      nUniq++;
      s = e;
    }

    // ---- apply per-frame state (hits) ----
    const gz = this.groundZ;
    const wZ = this.trav.weights[0];
    const wS = this.trav.weights[1];
    const wC = this.trav.weights[2];
    const wO = this.trav.weights[3];
    const wSum = wZ + wS + wC + wO;
    const zDen = this.trav.zDiffThresh * 2.5;
    const sDen = this.trav.slopeThresh;
    const clsScores = this.trav.classScores;
    const dom = this.dominant;

    for (let u = 0; u < nUniq; u++) {
      const cell = uCell[u];
      const cnt = uCnt[u];
      const zmax = uZmax[u];
      this.occupancy[cell] = occGain;
      this.zMean[cell] = uZsum[u] / cnt;
      this.zMaxState[cell] = zmax;
      // majority class
      const cBase = u * C;
      let bestC = 0;
      let bestN = -1;
      for (let c = 0; c < C; c++) {
        const v = uClsCounts[cBase + c];
        if (v > bestN) {
          bestN = v;
          bestC = c;
        }
      }
      dom[cell] = bestC;

      // traversability (mirrors _apply_precise)
      if (this.trav.enabled) {
        const heightU = zmax - gz;
        const hPos = heightU > 0 ? heightU : 0;
        let zScore = 1 - hPos / zDen;
        if (zScore < 0) zScore = 0;
        else if (zScore > 1) zScore = 1;
        let slopeScore = 1 - Math.max(heightU - 0.15, 0) / sDen;
        if (slopeScore < 0) slopeScore = 0;
        else if (slopeScore > 1) slopeScore = 1;
        const clsScore = bestC < clsScores.length ? clsScores[bestC] : 0;
        this.traversability[cell] = (wZ * zScore + wS * slopeScore + wC * clsScore + wO * occGain) / wSum;
      }

      // incremental rendered-list maintenance
      if (!this.inRendered[cell]) {
        this.inRendered[cell] = 1;
        this.renderedList[this.renderedCount++] = cell;
      }
    }

    // ---- instant free: previously rendered cells not hit this scan ----
    // hist[cell] > 0 marks this scan's hits; walk the compact list in place.
    let w = 0;
    for (let q = 0; q < this.renderedCount; q++) {
      const cell = this.renderedList[q];
      if (hist[cell] > 0) {
        this.renderedList[w++] = cell; // still hit: keep
      } else {
        this.inRendered[cell] = 0;
        this.occupancy[cell] = 0;
        this.zMaxState[cell] = -Infinity;
        this.zMean[cell] = 0;
        dom[cell] = UNKNOWN;
        this.traversability[cell] = 0;
      }
    }
    this.renderedCount = w;

    this.frame++;
    return nUniq;
  }

  private freeAllRendered(): void {
    for (let q = 0; q < this.renderedCount; q++) {
      const cell = this.renderedList[q];
      this.inRendered[cell] = 0;
      this.occupancy[cell] = 0;
      this.zMaxState[cell] = -Infinity;
      this.zMean[cell] = 0;
      this.dominant[cell] = UNKNOWN;
      this.traversability[cell] = 0;
    }
    this.renderedCount = 0;
  }

  /** Build the binary-packed patch rows for the given cell ids. */
  private packRows(ids: Int32Array, count: number, rows: Float32Array, cls: Uint8Array): void {
    const nTheta = this.params.nTheta;
    const g = this.groundZ;
    for (let k = 0; k < count; k++) {
      const cell = ids[k];
      const o6 = k * 6;
      rows[o6] = Math.floor(cell / nTheta);
      rows[o6 + 1] = cell % nTheta;
      rows[o6 + 2] = this.zMean[cell] - g;
      rows[o6 + 3] = this.zMaxState[cell] - g;
      rows[o6 + 4] = this.occupancy[cell];
      rows[o6 + 5] = this.traversability[cell];
      cls[k] = this.dominant[cell];
    }
  }

  /** Pure snapshot of all rendered cells as a packed binary patch (no mutation). */
  computeSnapshot(): PackedPatch {
    const rows = new Float32Array(this.renderedCount * 6);
    const cls = new Uint8Array(this.renderedCount);
    this.packRows(this.renderedList, this.renderedCount, rows, cls);
    // commit() for snapshots only reads the rendered list, but keep the ids
    // fields populated for a uniform interface.
    const upIds = new Int32Array(this.renderedCount);
    upIds.set(this.renderedList.subarray(0, this.renderedCount));
    return {
      isSnap: true,
      frame: this.frame,
      rows,
      cls,
      freed: new Float32Array(0),
      upIds,
      upCount: this.renderedCount,
      freedIds: new Int32Array(0),
      freedCount: 0,
    };
  }

  /**
   * Pure delta since the last commit: added cells, freed cells, and cells
   * whose z_max moved >= 5 cm or whose dominant class changed. Packed binary.
   */
  computeDelta(): PackedPatch {
    const list = this.renderedList;
    const live = this.renderedCount;

    // pass 1: count upserts (added or changed) and frees
    let nUps = 0;
    let nFree = 0;
    for (let q = 0; q < live; q++) {
      const cell = list[q];
      if (!this.wasSent[cell]) nUps++; // added
      else if (Math.abs(this.zMaxState[cell] - this.sentZmax[cell]) >= 0.05 || this.dominant[cell] !== this.sentCls[cell]) nUps++; // changed
    }
    const prev = this.prevSentList;
    for (let q = 0; q < this.prevSentCount; q++) {
      if (!this.inRendered[prev[q]]) nFree++;
    }

    // pass 2: collect upsert ids + freed ids (worker-side, survive transfer)
    const upIds = new Int32Array(nUps);
    let k = 0;
    for (let q = 0; q < live; q++) {
      const cell = list[q];
      const added = !this.wasSent[cell];
      const changed =
        !added &&
        (Math.abs(this.zMaxState[cell] - this.sentZmax[cell]) >= 0.05 || this.dominant[cell] !== this.sentCls[cell]);
      if (added || changed) upIds[k++] = cell;
    }
    const rows = new Float32Array(nUps * 6);
    const cls = new Uint8Array(nUps);
    this.packRows(upIds, nUps, rows, cls);

    const freed = new Float32Array(nFree * 2);
    const freedIds = new Int32Array(nFree);
    const nTheta = this.params.nTheta;
    let f = 0;
    let g = 0;
    for (let q = 0; q < this.prevSentCount; q++) {
      const cell = prev[q];
      if (!this.inRendered[cell]) {
        freedIds[g++] = cell;
        freed[f++] = Math.floor(cell / nTheta);
        freed[f++] = cell % nTheta;
      }
    }
    return { isSnap: false, frame: this.frame, rows, cls, freed, upIds, upCount: nUps, freedIds, freedCount: nFree };
  }

  /** Apply sent-tracking after the frame was actually delivered. */
  commit(patch: PackedPatch): void {
    if (patch.isSnap) {
      // 1) clear wasSent for cells that left since the last committed mask
      //    (walk the OLD prev list BEFORE replacing it).
      for (let q = 0; q < this.prevSentCount; q++) {
        const cell = this.prevSentList[q];
        if (!this.inRendered[cell]) this.wasSent[cell] = 0;
      }
      // 2) the new committed mask is exactly the rendered list
      if (this.prevSentList.length < this.renderedCount) {
        this.prevSentList = new Int32Array(Math.max(1024, this.renderedCount * 2));
      }
      this.prevSentList.set(this.renderedList.subarray(0, this.renderedCount));
      this.prevSentCount = this.renderedCount;
      for (let q = 0; q < this.renderedCount; q++) {
        const cell = this.renderedList[q];
        this.wasSent[cell] = 1;
        this.sentZmax[cell] = this.zMaxState[cell];
        this.sentCls[cell] = this.dominant[cell];
      }
      return;
    }

    // delta path: freed cells leave the mask, upsert cells join/update it.
    // Reads the worker-side id lists (transport buffers may be detached).
    // 1) stamp freed membership (O(nFree), no clears)
    this.freeStampValue++;
    const stamp = this.freeStamp;
    const stampV = this.freeStampValue;
    const freedIds = patch.freedIds;
    for (let k = 0; k < patch.freedCount; k++) {
      stamp[freedIds[k]] = stampV;
    }
    // 2) compact the old prev list in place, dropping freed cells
    let w = 0;
    for (let q = 0; q < this.prevSentCount; q++) {
      const cell = this.prevSentList[q];
      if (stamp[cell] === stampV) {
        this.wasSent[cell] = 0; // left the mask
      } else {
        this.prevSentList[w++] = cell;
      }
    }
    const keptPrev = w;
    // 3) append added upsert cells + refresh sent state for all upserts
    const upIds = patch.upIds;
    const nUps = patch.upCount;
    let need = keptPrev;
    for (let k = 0; k < nUps; k++) {
      if (!this.wasSent[upIds[k]]) need++;
    }
    if (this.prevSentList.length < need) {
      const grown = new Int32Array(Math.max(1024, need * 2));
      grown.set(this.prevSentList.subarray(0, keptPrev));
      this.prevSentList = grown;
    }
    for (let k = 0; k < nUps; k++) {
      const cell = upIds[k];
      if (!this.wasSent[cell]) {
        this.wasSent[cell] = 1;
        this.prevSentList[w++] = cell;
      }
      this.sentZmax[cell] = this.zMaxState[cell];
      this.sentCls[cell] = this.dominant[cell];
    }
    this.prevSentCount = w;
  }

  /** Full reset (seek / restart). */
  reset(): void {
    this.zMean.fill(0);
    this.zMaxState.fill(-Infinity);
    this.dominant.fill(UNKNOWN);
    this.occupancy.fill(0);
    this.traversability.fill(0);
    this.renderedCount = 0;
    this.inRendered.fill(0);
    this.wasSent.fill(0);
    this.sentZmax.fill(0);
    this.sentCls.fill(UNKNOWN);
    this.prevSentCount = 0;
    this.frame = 0;
    this.groundZ = 0;
  }

  /** Rendered-cell count. */
  get liveCount(): number {
    return this.renderedCount;
  }

  get bytesPerCell(): number {
    return 2 * 4 + this.params.nClasses * 4 + 1 + 4 + 4 + 8;
  }

  memoryReport(): { gridKb: number; renderedCells: number; nCells: number; compressionRatio: number; uniformMb: number } {
    const rendered = this.renderedCount;
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

/** numpy.percentile(x, 20) with linear interpolation (quickselect nth element). */
function percentile20(values: Float32Array, n: number): number {
  // exact nth_element (quickselect) on a copy; identical result to sort+interp
  const sorted = Array.from(values.subarray(0, n));
  const rank = (n - 1) * 0.2;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return quickselect(sorted, lo);
  const a = quickselect(sorted, lo);
  // for hi = lo + 1, quickselect of hi leaves sorted[hi] as the next smallest
  const b = quickselect(sorted, hi);
  return a + (b - a) * (rank - lo);
}

/** Hoare-partition quickselect: reorders v so v[k] is the k-th smallest. */
function quickselect(v: number[], k: number): number {
  let lo = 0;
  let hi = v.length - 1;
  while (lo < hi) {
    const pivot = v[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (v[i] < pivot) i++;
      while (v[j] > pivot) j--;
      if (i <= j) {
        const t = v[i];
        v[i] = v[j];
        v[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return v[k];
}
