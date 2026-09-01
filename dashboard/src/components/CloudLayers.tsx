"use client";

import { useLayoutEffect, useRef } from "react";
import * as THREE from "three";
import { CLASS_COLOR } from "@/lib/colors";

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