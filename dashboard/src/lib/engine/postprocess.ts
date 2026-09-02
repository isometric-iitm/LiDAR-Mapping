/**
 * Post-processing: logits -> per-point class ids.
 * TypeScript port of src/models/predict.py Segmenter.segment + _knn_probs
 * and the 5->4 binning from src/data/label_mapping.py.
 *
 * Pipeline (exact order, matching Python):
 *   1. per-pixel softmax over C channels (CHW logits)
 *   2. per-point 3x3 clamped neighbourhood gather, mean of 9 probability rows
 *   3. argmax -> uint8 5-class per point
 *   4. LUT bin 5->4 ([0,1,2,3,3]) -> uint8 4-class (grid + cloud colour)
 */

/** 256-entry LUT mirroring bin_lut(): 0..4 mapped, everything else 255. */
export function makeBinLut(bin5to4: number[]): Uint8Array {
  const lut = new Uint8Array(256).fill(255);
  for (let c5 = 0; c5 < bin5to4.length && c5 < 256; c5++) lut[c5] = bin5to4[c5];
  return lut;
}

/**
 * Softmax over C channels at pixel (row, col), then gather the 3x3
 * neighbourhood mean per point and argmax. All in one pass: for each point we
 * visit 9 pixels, and for each pixel we compute its softmax row once and
 * accumulate into the point's 5 (or C) accumulators.
 *
 * `pixelSoftmax` (hw * C floats) is a reusable scratch buffer.
 */
export function gatherPointProbs(
  logits: Float32Array, // CHW (C, h, w)
  proj: Int32Array, // (n, 2) row/col per point
  n: number,
  C: number,
  h: number,
  w: number,
  pixelSoftmax: Float32Array, // scratch (h*w, C)
  pointProbs: Float32Array, // out (n, C)
): void {
  const hw = h * w;
  // 1) softmax per pixel. Track per-pixel max and sum in two passes.
  for (let p = 0; p < hw; p++) {
    let max = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = logits[c * hw + p];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let c = 0; c < C; c++) {
      const e = Math.exp(logits[c * hw + p] - max);
      pixelSoftmax[c * hw + p] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let c = 0; c < C; c++) pixelSoftmax[c * hw + p] *= inv;
  }
  // 2) 3x3 clamped gather + mean of probabilities per point.
  for (let k = 0; k < n; k++) {
    const row = proj[k * 2];
    const col = proj[k * 2 + 1];
    const acc = pointProbs.subarray(k * C, k * C + C);
    acc.fill(0);
    for (let di = -1; di <= 1; di++) {
      let ri = row + di;
      if (ri < 0) ri = 0;
      else if (ri > h - 1) ri = h - 1;
      for (let dj = -1; dj <= 1; dj++) {
        let cj = col + dj;
        if (cj < 0) cj = 0;
        else if (cj > w - 1) cj = w - 1;
        const p = ri * w + cj;
        for (let c = 0; c < C; c++) acc[c] += pixelSoftmax[c * hw + p];
      }
    }
    const inv9 = 1 / 9;
    for (let c = 0; c < C; c++) acc[c] *= inv9;
  }
}

/** argmax over the per-point probability rows -> uint8 class ids. */
export function argmaxClasses(pointProbs: Float32Array, n: number, C: number, out: Uint8Array): Uint8Array {
  for (let k = 0; k < n; k++) {
    const row = k * C;
    let best = 0;
    let bestV = pointProbs[row];
    for (let c = 1; c < C; c++) {
      const v = pointProbs[row + c];
      if (v > bestV) {
        bestV = v;
        best = c;
      }
    }
    out[k] = best;
  }
  return out;
}

/** Apply the bin LUT in place: 5-class -> 4-class per point. */
export function binClasses(cls: Uint8Array, lut: Uint8Array): Uint8Array {
  for (let k = 0; k < cls.length; k++) cls[k] = lut[cls[k]];
  return cls;
}
