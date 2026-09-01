"use client";

// Barrel for map scene components. Kept as the public import surface for
// @/components/MapScene (page.tsx) — implementation now lives in
// InstancedCellLayer / CloudLayers / MapDecor, with shared ring-edge + cell-key
// math in @/lib/gridGeometry.
import * as THREE from "three";

export {
  CellLayer,
  type CellInfo,
  type CellGeo,
} from "./InstancedCellLayer";
export { CloudComparisonView, CloudSegView } from "./CloudLayers";
export {
  RangeRings,
  EgoMarker,
  GroundPlane,
  PerfOverlay,
  RING_RADII,
} from "./MapDecor";

export { THREE };