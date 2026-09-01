"use client";

/* Barrel for map scene components. Public import surface for page.tsx;
   implementation in InstancedCellLayer / CloudLayers / MapDecor. */
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