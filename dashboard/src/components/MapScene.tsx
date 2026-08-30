"use client";

import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { CLASS_COLOR } from "@/lib/colors";
import type { Cell, GridMeta } from "@/lib/types";
import type { CellMap, CellPatch } from "@/lib/useMapStream";

const MAX_INSTANCES = 130000;

export type CellInfo = {
  i: number;
  j: number;
  zMean: number;
  zMax: number;
  cls: number;
  occ: number;
  dyn: number;
  trav: number;
  pos: [number, number, number];
};

export type CellGeo = CellInfo & {
  rotY: number;
  scale: [number, number, number];
  color: string;
  alpha: number;
};

export const RING_RADII = [10, 25, 50, 120];

type SlotCache = {
  edges: number[];
  thetaStep: number;
  nRings: number;
  sec: { th: number; cosA: number; sinA: number; q: THREE.Quaternion }[];
  base: Map<string, { posX: number; posZ: number; scaleX: number; scaleZ: number }>;
  slotOf: Map<string, number>;
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
        t.p.set(0, 0, 0);
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
      c.edges = [meta.r_min];
      let cum = 0;
      for (let k = 0; k < meta.n_rings; k++) {
        cum += meta.dr_0 * meta.alpha ** k;
        c.edges.push(meta.r_min + cum);
      }
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
      const [i, j, zMean, zMax, cls, occ, dyn, trav] = cell;
      if (!Number.isFinite(zMax) || !Number.isFinite(zMean)) return false;
      if (i < 0 || i >= meta.n_rings || j < 0 || j >= meta.n_theta) return false;
      const key = `${i}:${j}`;
      const prevSlot = c.slotOf.get(key);
      const prev = prevSlot !== undefined ? c.geo[prevSlot] : null;
      if (prev && prev.zMax === zMax && prev.zMean === zMean && prev.occ === occ && prev.cls === cls) {
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
      t.p.set(b.posX, posY, b.posZ);
      t.q.copy(secA.q);
      t.s.set(b.scaleX, height, b.scaleZ);
      t.m.compose(t.p, t.q, t.s);
      mesh.setMatrixAt(slot, t.m);
      let cellColor: number;
      if (travMode) {
        if (trav >= 0.7) cellColor = 0x22c55e;       // green: drivable
        else if (trav >= 0.4) cellColor = 0xf59e0b;   // amber: caution
        else cellColor = 0xef4444;                      // red: blocked
      } else {
        cellColor = c.hex.get(cls) ?? 0xffffff;
      }
      const dynBoost = dyn > 0.5 ? 1.0 + 0.6 * Math.min(1, dyn) : 1.0;
      t.c.setHex(cellColor);
      mesh.setColorAt(slot, t.c.multiplyScalar((0.25 + 0.75 * Math.min(1, 0.4 + occ)) * dynBoost));
      c.geo[slot] = {
        i,
        j,
        zMean,
        zMax,
        cls,
        occ,
        dyn,
        trav,
        pos: [b.posX, posY, b.posZ],
        rotY: secA.th,
        scale: [b.scaleX, height, b.scaleZ],
        color: travMode
          ? (trav >= 0.7 ? "#22c55e" : trav >= 0.4 ? "#f59e0b" : "#ef4444")
          : (CLASS_COLOR.get(cls) ?? "#ffffff"),
        alpha: Math.min(1, 0.4 + occ),
      };
      return true;
    };

    const freeCell = (i: number, j: number): boolean => {
      const key = `${i}:${j}`;
      const slot = c.slotOf.get(key);
      if (slot === undefined) return false;
      t.p.set(0, 0, 0);
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

    let dirty = false;
    if (patch.kind === "reset") {
      clearAll();
      dirty = true;
    } else if (patch.kind === "delta") {
      for (const cell of patch.upserts) dirty = writeCell(cell) || dirty;
      for (const [i, j] of patch.frees) dirty = freeCell(i, j) || dirty;
    } else {
      // snapshot: reconcile against the authoritative full map, skipping
      // slots/rows whose values are unchanged so steady-state snapshots are ~free.
      for (const key of [...c.slotOf.keys()]) {
        if (!cells.has(key)) dirty = freeCell(Number(key.split(":")[0]), Number(key.split(":")[1])) || dirty;
      }
      for (const cell of patch.upserts) {
        if (!cells.has(`${cell[0]}:${cell[1]}`)) continue;
        dirty = writeCell(cell) || dirty;
      }
    }

    if (dirty) {
      mesh.count = c.next;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }, [cells, meta, patch, opacity]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      onPointerMove={(e) =>
        onHover?.(
          e.instanceId != null && cache.current && cache.current.geo[e.instanceId] ? cache.current.geo[e.instanceId] : null
        )
      }
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

export function CellLayer({
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
}

// ---- point cloud (raw sensor + segmented) ----
// Map convention: forward = +X, left = +Z, up = +Y.
// Sensor cloud rows are (x=forward, y=left, z=up) -> scene (x, z, y).
// Optional subset of point indices to pack (skips non-finite points).
function toMapPositions(src: Float32Array, n: number, idxs?: number[]): Float32Array {
  const out = new Float32Array((idxs ?? Array.from({ length: n }, (_, i) => i)).length * 3);
  const list = idxs ?? Array.from({ length: n }, (_, i) => i);
  for (let o = 0; o < list.length; o++) {
    const i = list[o];
    out[o * 3] = src[i * 3];
    out[o * 3 + 1] = src[i * 3 + 2];
    out[o * 3 + 2] = src[i * 3 + 1];
  }
  return out;
}

function heightColor(h: number, target: Float32Array, offset: number) {
  const c = new THREE.Color();
  if (h < 0) c.lerpColors(new THREE.Color("#64748b"), new THREE.Color("#94a3b8"), Math.min(1, (h + 3) / 3));
  else if (h < 2) c.lerpColors(new THREE.Color("#94a3b8"), new THREE.Color("#4ade80"), h / 2);
  else c.lerpColors(new THREE.Color("#4ade80"), new THREE.Color("#fbbf24"), Math.min(1, (h - 2) / 25));
  target[offset] = c.r;
  target[offset + 1] = c.g;
  target[offset + 2] = c.b;
}

function CloudLayer({
  cloud,
  colored,
  size,
}: {
  cloud: { xyz: Float32Array; cls: Uint8Array } | null;
  colored: "height" | "class";
  size: number;
}) {
  const ref = useRef<THREE.Points>(null);
  const colBuf = useRef<Float32Array | null>(null);

  useLayoutEffect(() => {
    const p = ref.current;
    if (!p || !cloud) return;
    const n = cloud.xyz.length / 3;
    const geom = p.geometry;
    // drop non-finite points (bad pose transforms etc.) so the position
    // attribute never feeds NaN into computeBoundingSphere
    const idxs: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = cloud.xyz[i * 3], b = cloud.xyz[i * 3 + 1], c = cloud.xyz[i * 3 + 2];
      if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c)) idxs.push(i);
    }
    const k = idxs.length;
    const scene = toMapPositions(cloud.xyz, n, idxs);
    const pos = geom.attributes.position as THREE.BufferAttribute | undefined;
    if (!pos || pos.array.length !== scene.length) {
      geom.setAttribute("position", new THREE.BufferAttribute(scene, 3));
    } else {
      pos.array.set(scene);
      pos.needsUpdate = true;
    }
    const col = colBuf.current ? shrink(colBuf.current, k) : new Float32Array(k * 3);
    const arr = col;
    const ccolor = new THREE.Color();
    for (let o = 0; o < k; o++) {
      const i = idxs[o];
      const z = cloud.xyz[i * 3 + 2];
      if (colored === "class") {
        ccolor.set(CLASS_COLOR.get(cloud.cls[i]) ?? "#ffffff");
        arr[o * 3] = ccolor.r;
        arr[o * 3 + 1] = ccolor.g;
        arr[o * 3 + 2] = ccolor.b;
      } else {
        heightColor(z, arr, o * 3);
      }
    }
    colBuf.current = arr;
    const ca = geom.attributes.color as THREE.BufferAttribute | undefined;
    if (!ca || ca.array.length !== arr.length) {
      geom.setAttribute("color", new THREE.BufferAttribute(arr, 3));
    } else {
      ca.array.set(arr);
      ca.needsUpdate = true;
    }
  }, [cloud, colored]);

  if (!cloud) return null;
  return (
    <points ref={ref} frustumCulled={false}>
      <bufferGeometry />
      <pointsMaterial size={size} sizeAttenuation={false} vertexColors transparent opacity={0.95} depthWrite={false} />
    </points>
  );
}

function shrink(a: Float32Array, n: number) {
  if (a.length === n * 3) return a;
  const b = new Float32Array(n * 3);
  b.set(a.subarray(0, n * 3));
  return b;
}

export function CloudComparisonView({
  cloud,
  size,
}: {
  cloud: { xyz: Float32Array; cls: Uint8Array } | null;
  size: number;
}) {
  return <CloudLayer cloud={cloud} colored="height" size={size} />;
}

export function CloudSegView({
  cloud,
  size,
}: {
  cloud: { xyz: Float32Array; cls: Uint8Array } | null;
  size: number;
}) {
  return <CloudLayer cloud={cloud} colored="class" size={size} />;
}

export function RangeRings({ maxR }: { maxR: number }) {
  const rings = useMemo(() => RING_RADII.filter((r) => r <= maxR), [maxR]);
  return (
    <group>
      {rings.map((r) => (
        <group key={r}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
            <ringGeometry args={[r - 0.06, r, 128]} />
            <meshBasicMaterial color="#52525b" side={THREE.DoubleSide} transparent opacity={0.7} />
          </mesh>
          <sprite position={[r + 1.5, 0.5, 0]} scale={[8, 2, 1]}>
            <spriteMaterial color="#a1a1aa" transparent opacity={0.8} depthTest={false} />
          </sprite>
        </group>
      ))}
    </group>
  );
}

export function EgoMarker() {
  return (
    <group position={[0, 0.1, 0]}>
      <mesh>
        <boxGeometry args={[2.4, 0.08, 1.2]} />
        <meshBasicMaterial color="#0ea5e9" />
      </mesh>
      <mesh position={[2.6, 0.08, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.35, 1.6, 4]} />
        <meshBasicMaterial color="#0ea5e9" />
      </mesh>
    </group>
  );
}

export function GroundPlane({ maxR }: { maxR: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
      <planeGeometry args={[2 * maxR, 2 * maxR]} />
      <meshBasicMaterial color="#18181b" />
    </mesh>
  );
}

const PERF_LOG_EVERY = 30;

type TimerQueryExt = {
  TIME_ELAPSED_EXT: number;
  QUERY_RESULT_AVAILABLE_EXT: number;
  QUERY_RESULT_EXT: number;
  createQueryEXT: () => unknown;
  beginQueryEXT: (target: number, q: unknown) => void;
  endQueryEXT: (q: unknown) => void;
  getQueryObjectEXT: (q: unknown, pname: number) => unknown;
  deleteQueryEXT: (q: unknown) => void;
};

// In-canvas render telemetry. Logs an aggregate line every n frames with the
// JS render cost (measured around gl.render, includes buffer uploads + draws),
// GPU time via EXT_disjoint_timer_query_webgl2 when available, draw-call /
// triangle counts, and how many stream patches were committed in that window.
// Joins the [map] cadence log from useMapStream to separate server / wire /
// JS / GPU stages of the pipeline.
export function FramePerf() {
  const gl = useThree((s) => s.gl);
  const last = useRef(0);
  const a = useRef({ n: 0, fps: 0, js: 0, gpu: 0, gpuN: 0, patches: 0 });
  const renderMs = useRef(0);
  const extRef = useRef<TimerQueryExt | null>(null);
  const queryRef = useRef<unknown | null>(null);

  useLayoutEffect(() => {
    if (!gl || extRef.current) return;
    // enable GPU timing only when every method the extension needs is present;
    // otherwise keep timing off and always render through the original method
    let ext: TimerQueryExt | null = null;
    try {
      const ctx = gl.getContext() as unknown as { getExtension: (name: string) => unknown };
      const raw = (ctx.getExtension("EXT_disjoint_timer_query_webgl2") ?? null) as Partial<TimerQueryExt> | null;
      if (
        raw &&
        typeof raw.createQueryEXT === "function" &&
        typeof raw.beginQueryEXT === "function" &&
        typeof raw.endQueryEXT === "function" &&
        typeof raw.getQueryObjectEXT === "function" &&
        typeof raw.deleteQueryEXT === "function"
      ) {
        ext = raw as TimerQueryExt;
      } else {
        console.warn("[render] GPU timer query extension unavailable; gpu= will be '-'");
      }
    } catch {
      ext = null;
      console.warn("[render] GPU timer query unavailable; gpu= will be '-'");
    }
    extRef.current = ext;
    const orig = gl.render.bind(gl);
    gl.render = (scene, camera) => {
      const t0 = performance.now();
      const ext_ = extRef.current;
      if (ext_ && !queryRef.current) {
        try {
          const q = ext_.createQueryEXT();
          queryRef.current = q;
          ext_.beginQueryEXT(ext_.TIME_ELAPSED_EXT, q);
          orig(scene, camera);
          ext_.endQueryEXT(q);
        } catch {
          try {
            if (queryRef.current) ext_.endQueryEXT(queryRef.current);
          } catch {
            /* context lost; skip GPU timing */
          }
          queryRef.current = null;
        }
      } else {
        orig(scene, camera);
      }
      renderMs.current = performance.now() - t0;
    };
  }, [gl]);

  useFrame(() => {
    const now = performance.now();
    if (!last.current) {
      last.current = now;
      return;
    }
    const dt = now - last.current;
    last.current = now;
    const ext = extRef.current;
    const q = queryRef.current;
    if (ext && q) {
      try {
        if (ext.getQueryObjectEXT(q, ext.QUERY_RESULT_AVAILABLE_EXT)) {
          const t = ext.getQueryObjectEXT(q, ext.QUERY_RESULT_EXT) as number;
          ext.deleteQueryEXT(q);
          queryRef.current = null;
          a.current.gpu += t / 1e6;
          a.current.gpuN++;
        }
      } catch {
        try {
          ext.deleteQueryEXT(q);
        } catch {
          /* ignore */
        }
        queryRef.current = null;
      }
    }
    a.current.n++;
    a.current.fps += 1000 / dt;
    a.current.js += renderMs.current;
    if (typeof window !== "undefined") {
      const w = window as unknown as { __pc2d?: { commits: number } };
      if (w.__pc2d) {
        a.current.patches += w.__pc2d.commits;
        w.__pc2d.commits = 0;
      }
    }
    if (a.current.n >= PERF_LOG_EVERY) {
      const info = gl.info.render;
      const gpuAvg = a.current.gpuN ? a.current.gpu / a.current.gpuN : NaN;
      console.log(
        `[render] fps=${(a.current.fps / a.current.n).toFixed(1)} ` +
          `js=${(a.current.js / a.current.n).toFixed(2)}ms ` +
          `gpu=${Number.isFinite(gpuAvg) ? gpuAvg.toFixed(2) : "-"}ms ` +
          `draws=${info.calls} tris=${(info.triangles / 1000).toFixed(1)}k ` +
          `pts=${info.points} patches=${a.current.patches}`
      );
      a.current.n = 0;
      a.current.fps = 0;
      a.current.js = 0;
      a.current.gpu = 0;
      a.current.gpuN = 0;
      a.current.patches = 0;
    }
  });

  return null;
}

export { THREE };