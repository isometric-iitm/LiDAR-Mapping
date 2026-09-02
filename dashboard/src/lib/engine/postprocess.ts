/**
 * Post-processing: model output (already Softmax probabilities, computed on
 * the GPU by the fused ONNX graph) -> per-point class ids.
 * Port of Segmenter._knn_probs (k=3) + bin 5->4 from label_mapping.py.
 *
 * Pipeline (exact order, matching Python):
 *   1. per-point 3x3 clamped neighbourhood gather, mean of 9 PROBABILITY rows
 *   2. argmax -> uint8 5-class per point
 *   3. LUT bin 5->4 ([0,1,2,3,3]) -> uint8 4-class (grid + cloud colour)
 */

/** 256-entry LUT mirroring bin_lut(): 0..4 mapped, everything else 255. */
export function makeBinLut(bin5to4: number[]): Uint8Array {
  const lut = new Uint8Array(256).fill(255);
  for (let c5 = 0; c5 < bin5to4.length && c5 < 256; c5++) lut[c5] = bin5to4[c5];
  return lut;
}

/**
 * Fused gather + argmax: for each point, visit the 3x3 clamped neighbourhood,
 * accumulate each pixel's probability row, and argmax the summed row.
 * (argmax of the mean-of-9 equals argmax of the sum-of-9, so the division by
 * 9 is skipped — identical predictions, one pass, no pointProbs buffer.)
 *
 * `pixelProbs` is CHW (C, h, w), already softmaxed by the model.
 */
export function gatherArgmax(
  pixelProbs: Float32Array, // CHW (C, h, w) probabilities
  proj: Int32Array, // (n, 2) row/col per point
  n: number,
  C: number,
  h: number,
  w: number,
  acc: Float32Array, // scratch (C)
  out5: Uint8Array, // out: 5-class per point
): void {
  const hw = h * w;
  for (let k = 0; k < n; k++) {
    const row = proj[k * 2];
    const col = proj[k * 2 + 1];
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
        for (let c = 0; c < C; c++) acc[c] += pixelProbs[c * hw + p];
      }
    }
    let best = 0;
    let bestV = acc[0];
    for (let c = 1; c < C; c++) {
      const v = acc[c];
      if (v > bestV) {
        bestV = v;
        best = c;
      }
    }
    out5[k] = best;
  }
}

/** Apply the bin LUT in place: 5-class -> 4-class per point. */
export function binClasses(cls: Uint8Array, lut: Uint8Array): Uint8Array {
  for (let k = 0; k < cls.length; k++) cls[k] = lut[cls[k]];
  return cls;
}
