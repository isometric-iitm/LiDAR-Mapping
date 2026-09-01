"use client";

import { useLayoutEffect, useRef } from "react";
import React from "react";
import * as THREE from "three";
import { CLASS_COLOR } from "@/lib/colors";
import { computeRingEdges, ijOf, keyOf } from "@/lib/gridGeometry";
import type { Cell, GridMeta } from "@/lib/types";
import type { CellMap, CellPatch } from "@/lib/useMapStream";

const MAX_INSTANCES = 500000;

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

// Slot-based incremental instancing. Geometry, matrices and colors are only
// recomputed for cells that appear in the delta patch, and per-ring/per-sector
// math is cached, so per-frame cost scales with the number of *changed* cells
// (~1-3k) instead of a full ~37k rebuild. Freed slots are zero-scaled inside
// the fixed-capacity InstancedMesh and recycled.
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
  // Last frame number actually rendered. When deltas arrive with a jump > 1 it
  // means intermediate frames were dropped on the wire; those may have carried
  // upserts/frees that never reached us, so we cheaply reconcile ghost slots
  // instead of waiting for the periodic full snapshot to heal them.
  const lastRenderedFrame = useRef(-1);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh || !meta || !patch) return;
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
      // Two-phase edges exactly mirror the server: uniform dr_0 through the
      // phase-1 rings, then dr_0*alpha^j growth, with the outer edge clamped
      // to r_max. (The old dr_0*alpha^k-for-all-rings formula drifted from the
      // server's real ring layout the farther out a cell sat.)
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
      // trav is wired to the per-instance colour in trav mode, so it must be
      // part of the dirty key (a trav change with unchanged geometry would
      // otherwise never recolor). Occ drives alpha/brightness.
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
      // Reuse the cached geometry: matrix only changes with (pos, rot, scale),
      // all of which are fixed by (i, j, zMax). If those are unchanged we skip
      // the compose + setMatrixAt entirely (~37k/frame saved in steady scenes).
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
      // Per-band donut shading: gentler falloff so the outer rings stay
      // legible (10m/25m/50m previously too dark). Pure range, no DK.
      // 0-5m:1.00  5-10m:0.90  10-25m:0.78  25-50m:0.62  50m+:0.48
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
      // Park freed instances far below ground with zero scale so they don't
      // cluster at the ego origin as degenerate flickering lines/dots (precise
      // mode frees ~30k slots per frame).
      t.p.set(0, -1000, 0);
      t.s.set(0, 0, 0);
      t.q.identity();
      t.m.compose(t.p, t.q, t.s);
      mesh.setMatrixAt(slot, t.m);
      t.c.setHex(0);
      mesh.setColorAt(slot, t.c);
      c.slotOf.delete(key);
      c.geo[slot] = null;
      c.free.push(slot);
      return true;
    };

    // Free any slot whose cell is absent from the authoritative map. Called for
    // ghost healing after dropped frames and for full snapshot reconciliation.
    const freeStale = (): number => {
      let n = 0;
      for (const key of [...c.slotOf.keys()]) {
        if (cells.has(key)) continue;
        const [gi, gj] = ijOf(key, meta.n_theta);
        if (freeCell(gi, gj)) n++;
      }
      return n;
    };

    // Add any cell that is in the authoritative map but has no renderer slot
    // (the inverse of freeStale); heals rows lost across a dropped frame that
    // the server or a missed chunk couldn't deliver. Bounded per call so a
    // poisoned map can never stall a frame indefinitely.
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

    // Compacts the slot layout when the recycled-free list has grown large
    // relative to the live count, so mesh.count (the GPU instance buffer width)
    // tracks the live scene instead of the historic high-water mark that would
    // otherwise climb toward MAX_INSTANCES and never come back down. Rebuilds
    // `slotOf` so live cells occupy the lowest contiguous slots. O(live);
    // only run periodically (on snapshots) to bound cost.
    const compactSlots = (): boolean => {
      if (c.free.length < 256 || c.free.length < c.slotOf.size) return false;
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
      for (const [i, j] of patch.frees) { if (freeCell(i, j)) { freed_count++; dirty = true; matDirty = true; colDirty = true; } }
      // Belt-and-suspenders ghost healing: if deltas jumped over dropped frames,
      // some cells may have been lost on the wire. Free renderer slots whose
      // cell is no longer authoritative, and (bounded) add authoritative cells
      // that have no slot, so no ghosts linger and no cells go missing until the
      // next snapshot.
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
      // snapshot: reconcile against the authoritative full map, skipping
      // slots/rows whose values are unchanged so steady-state snapshots are ~free.
      const healed = freeStale();
      if (healed > 0) {
        freed_count += healed; dirty = true; matDirty = true; colDirty = true;
      }
      for (const cell of patch.upserts) {
        const key = keyOf(cell[0], cell[1], meta.n_theta);
        if (!cells.has(key)) continue;
        if (writeCell(cell)) written++;
      }
      // heal any authoritative cell still missing a slot (e.g. a dropped delta),
      // then compact the freelist-driven high-water slot count back down.
      written += healMissing(5000);
      if (compactSlots()) { dirty = true; matDirty = true; colDirty = true; }
      lastRenderedFrame.current = patch.frame;
    }
    const layoutMs = performance.now() - t0;
    if (written > 0 || freed_count > 0) {
      console.log(
        `[grid:render] patch=${patch.kind} written=${written} freed=${freed_count} ` +
        `live=${c.slotOf.size} dirty=${dirty} layout=${layoutMs.toFixed(1)}ms`
      );
    }

    if (dirty) {
      mesh.count = c.next;
      if (matDirty) mesh.instanceMatrix.needsUpdate = true;
      if (colDirty && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }, [cells, meta, patch, opacity, travMode]);

  // Ensure the bounding sphere covers the full scene so raycasting finds
  // instances even when most slots are at the origin (freed / not yet written).
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