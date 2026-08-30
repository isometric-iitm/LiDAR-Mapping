"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, OrthographicCamera, PerspectiveCamera } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import MetricsPanel, { type ViewMode } from "@/components/MetricsPanel";
import Timeline from "@/components/Timeline";
import { CLASSES } from "@/lib/colors";
import {
  CellLayer,
  CloudComparisonView,
  CloudSegView,
  RangeRings,
  EgoMarker,
  GroundPlane,
  type CellInfo,
} from "@/components/MapScene";
import { useMapStream } from "@/lib/useMapStream";

type CamMode = "persp" | "ortho" | "top";

function computeGridEdges(meta: { r_min: number; dr_0: number; alpha: number; n_rings: number }): number[] {
  const { r_min, dr_0, alpha, n_rings } = meta;
  const edges: number[] = [r_min];
  let cum = 0;
  for (let k = 0; k < n_rings; k++) {
    cum += dr_0 * alpha ** k;
    edges.push(r_min + cum);
  }
  return edges;
}

function DemandInvalidator({ patch, camMode }: { patch: unknown; camMode: string }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    invalidate();
  }, [patch, camMode]);
  return null;
}

function OrthoAutoFit({ maxR }: { maxR: number }) {
  const cam = useThree((s) => s.camera as THREE.OrthographicCamera | THREE.PerspectiveCamera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    if (!(cam as THREE.OrthographicCamera).isOrthographicCamera) return;
    const o = cam as THREE.OrthographicCamera;
    const worldDiameter = 2 * maxR;
    const zoom = Math.min(size.width, size.height) / (worldDiameter * 1.08) / 1.5;
    o.zoom = Math.max(0.8, Math.min(6, zoom));
    o.updateProjectionMatrix();
    invalidate();
  }, [size.width, size.height, maxR, cam, invalidate]);
  return null;
}

function hoverInfo(cell: CellInfo, gridEdges: number[], nTheta: number): {
  cls: string;
  zMax: number;
  zMean: number;
  occ: number;
  dyn: number;
  trav: number;
  r: number;
  deg: number;
  cellWidth: string;
} {
  const rIn = gridEdges[cell.i] ?? 0;
  const rOut = gridEdges[cell.i + 1] ?? gridEdges[gridEdges.length - 1];
  const r = (rIn + rOut) / 2;
  const deg = ((cell.j / nTheta) * 360 + 180) % 360;
  const clsCls = CLASSES.find((c) => c.id === cell.cls);
  const ringWidth = rOut - rIn;
  const cellWidth = ringWidth < 0.1 ? `${(ringWidth * 100).toFixed(1)} cm` : `${ringWidth.toFixed(2)} m`;
  return {
    cls: clsCls?.label ?? "other",
    zMax: cell.zMax,
    zMean: cell.zMean,
    occ: cell.occ,
    dyn: cell.dyn,
    trav: cell.trav,
    r,
    deg,
    cellWidth,
  };
}

export default function Home() {
  const {
    meta,
    cells,
    patch,
    lastFrame,
    stats,
    cellCount,
    cloud,
    cloudOn,
    setCloudOn,
    seeking,
    seekTo,
    send,
  } = useMapStream();
  const [paused, setPaused] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [pointSize, setPointSize] = useState(2);
  const [hover, setHover] = useState<ReturnType<typeof hoverInfo> | null>(null);
  const [camMode, setCamMode] = useState<CamMode>("persp");
  const controlsRef = useRef<OrbitControlsImpl>(null);

  const maxR = meta?.r_max ?? 100;
  const gridEdges = meta ? computeGridEdges(meta) : [];
  const nTheta = meta?.n_theta ?? 720;

  const resetNorth = () => {
    const c = controlsRef.current;
    if (!c) return;
    if (camMode === "top") {
      c.object.position.set(0, 180, 0.01);
      c.target.set(0, 0, 0);
    } else {
      c.object.position.set(0, 130, 95);
      c.target.set(0, 0, 0);
    }
    c.update();
  };

  const handlePause = (p: boolean) => {
    setPaused(p);
    send({ type: "control", action: p ? "pause" : "play" });
  };

  // Space bar toggles play/pause (ignore when focus is in an input/select).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      e.preventDefault();
      setPaused((prev) => {
        const next = !prev;
        send({ type: "control", action: next ? "pause" : "play" });
        return next;
      });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send]);

  const handleSpeed = (s: number) => {
    setSpeed(s);
    send({ type: "control", action: "speed", value: s });
  };

  const handleSeek = (idx: number) => seekTo(idx);

  const handleViewMode = (v: ViewMode) => {
    setViewMode(v);
    const needsCloud = v === "seg" || v === "raw" || v === "compare";
    if (needsCloud && !cloudOn) setCloudOn(true);
    else if (!needsCloud && cloudOn) setCloudOn(false);
  };

  const showCloud = viewMode === "seg" || viewMode === "raw" || viewMode === "compare";
  const gridOpacity = viewMode === "compare" ? 0.35 : 1;
  const showGrid = viewMode === "grid" || viewMode === "compare" || viewMode === "trav";

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <Canvas
            frameloop="demand"
            dpr={[1, 2]}
            onCreated={({ gl }) => {
              // demand mode still needs to wake on data changes — the
              // invalidate call is triggered by patch/cellCount effects below.
              gl.toneMappingExposure = 1;
            }}
          >
            {/* Camera rig: perspective (angled 3D), ortho (angled 3D), top (2D). */}
            {camMode === "ortho" ? (
              <OrthographicCamera key="ortho" makeDefault position={[0, 130, 95]} zoom={2.5} near={0.1} far={1000} />
            ) : camMode === "top" ? (
              <OrthographicCamera key="top" makeDefault position={[0, 180, 0.01]} zoom={2.5} near={0.1} far={1000} />
            ) : (
              <PerspectiveCamera key="persp" makeDefault position={[0, 130, 95]} fov={45} near={0.1} far={1000} />
            )}
            {(camMode === "ortho" || camMode === "top") && <OrthoAutoFit maxR={maxR} />}
            <color attach="background" args={["#0b0b0e"]} />
            <ambientLight intensity={0.7} />
            <directionalLight position={[60, 120, 40]} intensity={1.4} />
            <GroundPlane maxR={maxR} />
            <RangeRings maxR={maxR} />
            {/* heading-up: the grid/cloud are refreshed in the CURRENT ego
                frame every tick, so a fixed +90deg puts the ego's forward
                (+X scene axis) at the top of the screen permanently. */}
            <group rotation={[0, Math.PI / 2, 0]}>
              {showGrid && (
                <CellLayer
                  cells={cells}
                  meta={meta}
                  patch={patch}
                  onHover={(i) => setHover(i ? hoverInfo(i, gridEdges, nTheta) : null)}
                  opacity={gridOpacity}
                  travMode={viewMode === "trav"}
                />
              )}
              {showCloud &&
                (viewMode === "seg" ? (
                  <CloudSegView cloud={cloud} size={pointSize + 0.4} />
                ) : (
                  <CloudComparisonView cloud={cloud} size={pointSize} />
                ))}
              <EgoMarker />
            </group>
            <OrbitControls
              key={camMode}
              ref={controlsRef}
              makeDefault
              target={[0, 0, 0]}
              minDistance={10}
              maxDistance={320}
              enablePan
              maxPolarAngle={camMode === "top" ? Math.PI / 2.001 : Math.PI / 2.05}
              minPolarAngle={camMode === "top" ? 0 : undefined}
            />
            <DemandInvalidator patch={patch} camMode={camMode} />
          </Canvas>

          {/* Camera projection + north-reset */}
          <div className="pointer-events-auto absolute right-4 top-4 flex items-center gap-1 rounded-md bg-zinc-900/80 p-1 backdrop-blur">
            {(["persp", "ortho", "top"] as CamMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setCamMode(m)}
                className={`rounded px-2 py-1 text-[11px] font-medium ${camMode === m ? "bg-sky-600 text-white" : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"}`}
                title={m === "persp" ? "Perspective 3D" : m === "ortho" ? "Orthographic 3D" : "Top-down 2D"}
              >
                {m === "persp" ? "3D" : m === "ortho" ? "Ortho" : "Top"}
              </button>
            ))}
            <div className="mx-1 h-4 w-px bg-zinc-700" />
            <button
              onClick={resetNorth}
              className="rounded px-2 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              title="Reset to north-up"
            >
              N↑
            </button>
          </div>

          <div className="pointer-events-none absolute bottom-4 left-4 rounded-md bg-zinc-900/80 px-3 py-2 text-xs text-zinc-400 backdrop-blur">
            <span className="font-semibold text-zinc-200">
              {viewMode === "grid"
                ? "2.5D grid"
                : viewMode === "trav"
                  ? "traversability"
                  : viewMode === "seg"
                  ? "segmented cloud"
                  : viewMode === "raw"
                    ? "raw point cloud"
                    : "raw cloud over grid"}
            </span>
            {meta ? (
              <span className="ml-2 font-mono">
                {meta.n_rings} × {meta.n_theta}
              </span>
            ) : null}
            {showCloud && cloud ? (
              <span className="ml-2 font-mono text-zinc-500">{(cloud.xyz.length / 3).toLocaleString()} pts</span>
            ) : null}
            {viewMode === "compare" && stats ? (
              <div className="mt-1 text-[10px] text-sky-400/80">
                {stats.compression_ratio.toLocaleString()}× fewer cells than uniform 5 cm grid
              </div>
            ) : null}
          </div>

          <div className="pointer-events-none absolute left-4 top-4 rounded-md bg-zinc-900/80 px-3 py-2 text-xs backdrop-blur">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
              {viewMode === "trav" ? (
                <>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: "#22c55e" }} />
                    <span className="text-[11px] leading-none text-zinc-300">Drivable</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: "#f59e0b" }} />
                    <span className="text-[11px] leading-none text-zinc-300">Caution</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: "#ef4444" }} />
                    <span className="text-[11px] leading-none text-zinc-300">Blocked</span>
                  </div>
                </>
              ) : (
                CLASSES.map((c) => (
                  <div key={c.id} className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: c.color }} />
                    <span className="text-[11px] leading-none text-zinc-300">{c.label}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          <Timeline
            seqPos={stats?.seq_pos ?? 0}
            seqLen={stats?.seq_len ?? 0}
            paused={paused}
            speed={speed}
            seeking={seeking}
            onPause={handlePause}
            onSeek={handleSeek}
            onSpeed={handleSpeed}
          />
        </div>

        <MetricsPanel
          stats={stats}
          cellCount={cellCount}
          lastFrame={lastFrame}
          viewMode={viewMode}
          onViewMode={handleViewMode}
          pointSize={pointSize}
          onPointSize={setPointSize}
          hover={hover}
        />
      </div>
    </div>
  );
}