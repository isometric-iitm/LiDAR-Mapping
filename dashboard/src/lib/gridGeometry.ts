import type { GridMeta } from "./types";

/** Flat key for a cell: i*j plane packed as i*nTheta+j. Cheaper than string
 *  keys and lets MapScene hash the same ints the server uses for cells. */
export const keyOf = (i: number, j: number, nTheta: number): number => i * nTheta + j;

export const ijOf = (key: number, nTheta: number): [number, number] => [
  Math.floor(key / nTheta),
  key % nTheta,
];

/** Server ring edges -- mirrored from src/grid_engine/logpolar_grid.py:
 *  phase-1 rings are uniform dr_0, phase-2 rings grow by alpha, and the final
 *  ring width is clamped so the outer edge lands exactly on r_max. Clients use
 *  this everywhere a cell's radial base/scale or the hover range is computed. */
export function computeRingEdges(
  meta: Pick<GridMeta, "r_min" | "r_max" | "dr_0" | "alpha" | "n_rings" | "phase1_rings">
): number[] {
  const { r_min, r_max, dr_0, alpha, n_rings, phase1_rings = 0 } = meta;
  const edges: number[] = new Array(n_rings + 1);
  edges[0] = r_min;
  let cum = 0;
  for (let k = 0; k < n_rings; k++) {
    cum += k < phase1_rings ? dr_0 : dr_0 * alpha ** (k - phase1_rings);
    edges[k + 1] = r_min + cum;
  }
  edges[n_rings] = r_max;
  return edges;
}