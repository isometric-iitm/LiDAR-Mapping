"use client";
/* eslint-disable react-hooks/immutability */

import { useFrame, useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export const RING_RADII = [10, 25, 50, 120];

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
          <Html
            position={[r + 1.5, 0.3, 0]}
            center
            sprite
            style={{ pointerEvents: "none" }}
          >
            <span className="hud-val whitespace-nowrap rounded bg-black/70 px-1.5 py-0.5">
              {r}m
            </span>
          </Html>
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
      <meshBasicMaterial color="#000000" />
    </mesh>
  );
}

export function PerfOverlay({ onStats }: { onStats?: (s: { fps: number; js: number; draws: number; tris: number }) => void }) {
  const gl = useThree((s) => s.gl);
  const last = useRef<number | null>(null);
  const acc = useRef({ n: 0, sumFps: 0, sumJs: 0 });
  const t0 = useRef(0);
  useLayoutEffect(() => {
    const orig = gl.render.bind(gl);
    gl.render = ((scene: THREE.Scene, camera: THREE.Camera) => {
      t0.current = performance.now();
      orig(scene, camera);
      const dt = performance.now() - t0.current;
      acc.current.sumJs += dt;
    }) as unknown as typeof gl.render;
    return () => {
      gl.render = orig;
    };
  }, [gl]);
  useFrame(() => {
    if (last.current === null) {
      last.current = performance.now();
      return;
    }
    const now = performance.now();
    const dt = now - last.current;
    last.current = now;
    acc.current.n += 1;
    acc.current.sumFps += 1000 / Math.max(dt, 0.1);
    if (acc.current.n >= 30) {
      const info = gl.info.render;
      const avgFps = acc.current.sumFps / acc.current.n;
      const avgJs = acc.current.sumJs / acc.current.n;
      onStats?.({ fps: avgFps, js: avgJs, draws: info.calls, tris: info.triangles });
      acc.current = { n: 0, sumFps: 0, sumJs: 0 };
    }
  });
  return null;
}