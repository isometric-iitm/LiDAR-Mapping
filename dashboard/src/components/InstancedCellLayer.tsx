"use client";

import { useLayoutEffect, useRef } from "react";
import React from "react";
import * as THREE from "three";
import { CLASS_COLOR } from "@/lib/colors";
import { computeRingEdges, ijOf, keyOf } from "@/lib/gridGeometry";
import type { Cell, GridMeta } from "@/lib/types";
import type { CellMap, CellPatch } from "@/lib/useMapStream";

/* Slot-based incremental instancing. 150k instances (observed live peak ~60k)
   keep the per-frame GPU upload 3.3x smaller than the old 500k high-water mark. */
const MAX_INSTANCES = 150000;
/* Render debug logging: enabled by adding ?griddebug=1 to the URL. */
const GRID_DEBUG = typeof window !== "undefined" && window.location.search.includes("griddebug=1");
const UP_AXIS = new THREE.Vector3(0, 1, 0);

export type CellInfo = {
  i: number;
  j: number;
  zMean: number;
  zMax: number;
  cls: number;
  occ: number;
  trav: number;
  pos: [number, number, number];
};

export type CellGeo = CellInfo & {
  rotY: number;
  scale: [number, number, number];
  color: string;
  alpha: number;
  rgb: number;
};

type SlotCache = {
  edges: number[];
  thetaStep: number;
  nRings: number;
  sec: { th: number; cosA: number; sinA: number; q: THREE.Quaternion }[];
  base: Map<number, { posX: number; posZ: number; scaleX: number; scaleZ: number }>;
  slotOf: Map<number, number>;
  geo: (CellGeo | null)[];
  free: number[];
  next: number;
  hex: Map<number, number>;
};

/* Slot-based incremental instancing. Per-frame cost scales with changed cells
   (~1-3k), not full ~37k rebuild. Freed slots are zero-scaled and recycled. */
function InstancedCells({
  cells,
  meta,
  patch,
  onHover,
  opacity = 1,
  travMode = false,
}: {
  cells: CellMap;
  meta: GridMeta | null;
  patch: CellPatch | null;
  onHover?: (info: CellInfo | null) => void;
  opacity?: number;
  travMode?: boolean;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const cache = useRef<SlotCache | null>(null);
  const tmp = useRef<{
    m: THREE.Matrix4;
    q: THREE.Quaternion;
    s: THREE.Vector3;
    p: THREE.Vector3;
    c: THREE.Color;
  } | null>(null);
  /* Last rendered frame. Frame gaps mean dropped wire frames; reconcile ghost slots cheaply. */
  const lastRenderedFrame = useRef(-1);

  // Throttle stats-driven re-renders: only layout on actual patch changes
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh || !meta || !patch) return;
    // Use DynamicDrawUsage for streaming updates (hint to WebGL)
    if (mesh.instanceMatrix) mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (mesh.instanceColor) mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    if (!tmp.current) {
      tmp.current = {
        m: new THREE.Matrix4(),
        q: new THREE.Quaternion(),
        s: new THREE.Vector3(),
        p: new THREE.Vector3(),
        c: new THREE.Color(),
      };
    }
    const t = tmp.current;

    let c = cache.current;
    if (!c) {
      c = {
        edges: [],
        thetaStep: 0,
        nRings: 0,
        sec: [],
        base: new Map(),
        slotOf: new Map(),
        geo: [],
        free: [],
        next: 0,
        hex: new Map(),
      };
      cache.current = c;
      const tC = new THREE.Color();
      for (const [id, col] of CLASS_COLOR) {
        tC.set(col);
        c.hex.set(id, tC.getHex());
      }
    }

    const clearAll = () => {
      for (const slot of c.slotOf.values()) {
        t.p.set(0, -1000, 0);
        t.s.set(0, 0, 0);
        t.q.identity();
        t.m.compose(t.p, t.q, t.s);
        mesh.setMatrixAt(slot, t.m);
        t.c.setHex(0);
        mesh.setColorAt(slot, t.c);
      }
      c.slotOf = new Map();
      c.geo = [];
      c.free = [];
      c.next = 0;
      c.base = new Map();
      mesh.count = 0;
    };

    const thetaStep = (2 * Math.PI) / meta.n_theta;
    if (c.nRings !== meta.n_rings || Math.abs(c.thetaStep - thetaStep) > 1e-9) {
      clearAll();
      /* Two-phase edges mirror the server: uniform dr_0 then dr_0*alpha^j, clamped to r_max. */
      c.edges = computeRingEdges(meta);
      c.thetaStep = thetaStep;
      c.nRings = meta.n_rings;
      c.sec = [];
      for (let j = 0; j < meta.n_theta; j++) {
        const th = Math.PI + j * thetaStep;
        c.sec.push({
          th,
          cosA: Math.cos(th),
          sinA: Math.sin(th),
          q: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, th, 0)),
        });
      }
    }

    const writeCell = (cell: Cell): boolean => {
      const [i, j, zMean, zMax, cls, occ, trav] = cell;
      if (!Number.isFinite(zMax) || !Number.isFinite(zMean)) return false;
      if (i < 0 || i >= meta.n_rings || j < 0 || j >= meta.n_theta) return false;
      const key = keyOf(i, j, meta.n_theta);
      const prevSlot = c.slotOf.get(key);
      const prev = prevSlot !== undefined ? c.geo[prevSlot] : null;
      // Dirty-skip on the exact display inputs (height, class, occ, trav).
      /* trav drives per-instance colour in trav mode; occ drives alpha. Both must be in dirty key. */
      if (
        prev &&
        prev.zMax === zMax &&
        prev.zMean === zMean &&
        prev.occ === occ &&
        prev.cls === cls &&
        prev.trav === trav
      ) {
        return false;
      }
      let b = c.base.get(key);
      if (!b) {
        const dr = c.edges[i + 1] - c.edges[i];
        const rMid = (c.edges[i] + c.edges[i + 1]) / 2;
        const sec = c.sec[j];
        b = {
          posX: rMid * sec.cosA,
          posZ: rMid * sec.sinA,
          scaleX: Math.max(0.05, Math.min(dr, rMid * c.thetaStep)),
          scaleZ: dr,
        };
        c.base.set(key, b);
      }
      let slot = prevSlot;
      if (slot === undefined) {
        slot = c.free.length ? (c.free.pop() as number) : c.next++;
        if (slot >= MAX_INSTANCES) return false;
        c.slotOf.set(key, slot);
      }
      const secA = c.sec[j];
      const height = Math.min(12, Math.max(0.35, zMax));
      const posY = height / 2;
      /* Skip matrix update if (pos, rot, scale) unchanged (~37k/frame saved in steady scenes). */
      const needMatrix = !prev ||
        prev.pos[0] !== b.posX || prev.pos[1] !== posY || prev.pos[2] !== b.posZ ||
        prev.rotY !== secA.th ||
        prev.scale[0] !== b.scaleX || prev.scale[1] !== height || prev.scale[2] !== b.scaleZ;
      if (needMatrix) {
        t.p.set(b.posX, posY, b.posZ);
        t.q.copy(secA.q);
        t.s.set(b.scaleX, height, b.scaleZ);
        t.m.compose(t.p, t.q, t.s);
        mesh.setMatrixAt(slot, t.m);
        matDirty = true;
        dirty = true;
      }
      let cellColor: number;
      if (travMode) {
        if (trav >= 0.7) cellColor = 0x22c55e;       // green: drivable
        else if (trav >= 0.4) cellColor = 0xf59e0b;   // amber: caution
        else cellColor = 0xef4444;                      // red: blocked
      } else {
        cellColor = c.hex.get(cls) ?? 0xffffff;
      }
      /* Per-band donut shading for legibility: 0-5m:1.0 5-10m:0.9 10-25m:0.78 25-50m:0.62 50m+:0.48 */
      const rMid = (c.edges[i] + c.edges[i + 1]) / 2;
      const bandIntensity = rMid < 5 ? 1.0 : rMid < 10 ? 0.9 : rMid < 25 ? 0.78 : rMid < 50 ? 0.62 : 0.48;
      t.c.setHex(cellColor);
      t.c.multiplyScalar((0.35 + 0.65 * Math.min(1, 0.4 + occ)) * bandIntensity);
      // Only write the per-instance color attribute when the RGB actually changed.
      if (!prev || prev.rgb !== t.c.getHex()) {
        mesh.setColorAt(slot, t.c);
        colDirty = true;
        dirty = true;
      }
      // Reuse the slot's geo object (avoid ~37k allocations/frame -> GC pressure).
      const g = prev ?? ({} as CellGeo);
      g.i = i;
      g.j = j;
      g.zMean = zMean;
      g.zMax = zMax;
      g.cls = cls;
      g.occ = occ;
      g.trav = trav;
      g.pos = [b.posX, posY, b.posZ];
      g.rotY = secA.th;
      g.scale = [b.scaleX, height, b.scaleZ];
      g.color = travMode
        ? (trav >= 0.7 ? "#22c55e" : trav >= 0.4 ? "#f59e0b" : "#ef4444")
        : (CLASS_COLOR.get(cls) ?? "#ffffff");
      g.alpha = Math.min(1, 0.4 + occ);
      g.rgb = t.c.getHex();
      c.geo[slot] = g;
      return true;
    };

    const freeCell = (i: number, j: number): boolean => {
      const key = keyOf(i, j, meta.n_theta);
      const slot = c.slotOf.get(key);
      if (slot === undefined) return false;
      /* Park freed instances far below ground with zero scale. Matrix-only
       * write: the color buffer refreshes when the slot is recycled. */
      t.p.set(0, -1000, 0);
      t.s.set(0, 0, 0);
      t.q.identity();
      t.m.compose(t.p, t.q, t.s);
      mesh.setMatrixAt(slot, t.m);
      c.slotOf.delete(key);
      c.geo[slot] = null;
      c.free.push(slot);
      return true;
    };

    /* Free slots absent from the authoritative map (ghost healing + snapshot reconciliation). */
    const freeStale = (): number => {
      let n = 0;
      for (const key of [...c.slotOf.keys()]) {
        if (cells.has(key)) continue;
        const [gi, gj] = ijOf(key, meta.n_theta);
        if (freeCell(gi, gj)) n++;
      }
      return n;
    };

    /* Heal cells in the map that have no renderer slot (inverse of freeStale). Bounded per call. */
    const healMissing = (caps: number): number => {
      let n = 0;
      for (const cell of cells.values()) {
        if (n >= caps) break;
        const key = keyOf(cell[0], cell[1], meta.n_theta);
        if (!c.slotOf.has(key)) {
          if (writeCell(cell)) n++;
        }
      }
      return n;
    };

    /* Compact slots: move live cells to lowest contiguous slots so mesh.count tracks
       the live scene, not the high-water mark. O(live); run on snapshots only. */
    const compactSlots = (): boolean => {
      // After a compaction every live slot moves, so the matrices for ALL slots
      // must re-upload; clear partial update ranges for a full flush.
      const fullFlush = (): void => {
        mesh.instanceMatrix.clearUpdateRanges();
        if (mesh.instanceColor) mesh.instanceColor.clearUpdateRanges();
      };
      if (c.free.length < 256 || c.free.length < c.slotOf.size) return false;
      fullFlush();
      const reindex = new Map<number, number>();
      const geo: (CellGeo | null)[] = [];
      const oldSlotOf = c.slotOf;
      for (const [key, oldSlot] of oldSlotOf) {
        const g = c.geo[oldSlot];
        if (!g) continue;
        const newSlot = geo.length;
        reindex.set(key, newSlot);
        geo.push(g);
      }
      // move geometry arrays into their new compacted slots
      const oldGeo = c.geo;
      for (let slot = 0; slot < geo.length; slot++) {
        oldGeo[slot] = geo[slot];
      }
      for (let slot = geo.length; slot < c.next; slot++) {
        oldGeo[slot] = null;
      }
      c.slotOf = reindex;
      c.geo = oldGeo;
      c.free = [];
      c.next = geo.length;
      // Rewrite every live slot at its new index (matrix + color) — the moved
      // slots carry their old buffers' contents.
      for (const [key, slot] of c.slotOf) {
        const g = c.geo[slot];
        if (!g) continue;
        const b = c.base.get(key);
        if (!b) continue;
        t.p.set(b.posX, g.pos[1], b.posZ);
        t.s.set(g.scale[0], g.scale[1], g.scale[2]);
        t.q.setFromAxisAngle(UP_AXIS, g.rotY);
        t.m.compose(t.p, t.q, t.s);
        mesh.setMatrixAt(slot, t.m);
        t.c.setHex(g.rgb); // exact stored final colour
        mesh.setColorAt(slot, t.c);
      }
      return true;
    };

    let dirty = false;
    let matDirty = false;
    let colDirty = false;
    let written = 0;
    let freed_count = 0;
    const t0 = performance.now();
    if (patch.kind === "reset") {
      clearAll();
      dirty = true;
      matDirty = true;
      colDirty = true;
      lastRenderedFrame.current = patch.frame;
    } else if (patch.kind === "delta") {
      for (const cell of patch.upserts) { if (writeCell(cell)) written++; dirty = true; }
      if (patch.frees.length > 0) {
        dirty = true;
        matDirty = true;
        for (const [i, j] of patch.frees) { if (freeCell(i, j)) freed_count++; }
      }
      /* Ghost healing on frame gaps: free stale slots, heal missing cells until next snapshot. */
      if (lastRenderedFrame.current >= 0 && patch.frame - lastRenderedFrame.current > 1) {
        const healed = freeStale();
        if (healed > 0) {
          freed_count += healed; dirty = true; matDirty = true; colDirty = true;
        }
        const added = healMissing(5000);
        if (added > 0) { written += added; dirty = true; }
      }
      lastRenderedFrame.current = patch.frame;
    } else {
      /* Snapshot: reconcile against authoritative full map, skipping unchanged slots (~free in steady state). */
      const healed = freeStale();
      if (healed > 0) {
        freed_count += healed; dirty = true; matDirty = true; colDirty = true;
      }
      for (const cell of patch.upserts) {
        const key = keyOf(cell[0], cell[1], meta.n_theta);
        if (!cells.has(key)) continue;
        if (writeCell(cell)) written++;
      }
      /* Heal any authoritative cell still missing a slot (e.g. dropped delta), then compact high-water slot count. */
      written += healMissing(5000);
      if (compactSlots()) { dirty = true; matDirty = true; colDirty = true; }
      lastRenderedFrame.current = patch.frame;
    }
    const layoutMs = performance.now() - t0;
    if (GRID_DEBUG && (written > 0 || freed_count > 0)) {
      console.log(
        `[grid:render] patch=${patch.kind} written=${written} freed=${freed_count} ` +
        `live=${c.slotOf.size} dirty=${dirty} layout=${layoutMs.toFixed(1)}ms`
      );
    }

    if (dirty) {
      mesh.count = c.next;
      // Partial-range uploads: only the slots [0, highWater) written since the
      // last flush move to the GPU, instead of the full MAX_INSTANCES buffers.
      const highWater = c.next;
      if (matDirty) {
        mesh.instanceMatrix.addUpdateRange(0, highWater * 16);
        mesh.instanceMatrix.needsUpdate = true;
      }
      if (colDirty && mesh.instanceColor) {
        mesh.instanceColor.addUpdateRange(0, highWater * 3);
        mesh.instanceColor.needsUpdate = true;
      }
    }
  }, [cells, meta, patch, opacity, travMode]);

  /* Ensure bounding sphere covers full scene so raycasting finds instances even when most slots are at origin. */
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    mesh.geometry.computeBoundingSphere();
    const sphere = mesh.geometry.boundingSphere;
    if (sphere) {
      sphere.radius = 200;
      sphere.center.set(0, 0, 0);
    }
  }, [meta]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      onPointerMove={(e) => {
        if (!onHover) return;
        // R3F event.instanceId is the slot index; fall back to raycast order.
        const id = (e as unknown as { instanceId?: number }).instanceId;
        if (id != null && cache.current?.geo[id]) {
          onHover(cache.current.geo[id]);
        } else {
          onHover(null);
        }
        e.stopPropagation();
      }}
      onPointerOut={() => onHover?.(null)}
    >
      <boxGeometry />
      <meshStandardMaterial
        roughness={0.9}
        metalness={0.1}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </instancedMesh>
  );
}

export const CellLayer = React.memo(function CellLayer({
  cells,
  meta,
  patch,
  onHover,
  opacity = 1,
  travMode = false,
}: {
  cells: CellMap;
  meta: GridMeta | null;
  patch: CellPatch | null;
  onHover?: (info: CellInfo | null) => void;
  opacity?: number;
  travMode?: boolean;
}) {
  return (
    <InstancedCells
      cells={cells}
      meta={meta}
      patch={patch}
      onHover={onHover}
      opacity={opacity}
      travMode={travMode}
    />
  );
});