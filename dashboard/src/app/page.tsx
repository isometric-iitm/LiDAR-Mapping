"use client";

import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
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
  FramePerf,
  type CellInfo,
} from "@/components/MapScene";
import { useMapStream } from "@/lib/useMapStream";

const GRID_EDGES = (() => {
  const r0 = 0.5;
  const a = 1.05;
  const d0 = 0.05;
  const edges: number[] = [r0];
  let cum = 0;
  for (let k = 0; k < 95; k++) {
    cum += d0 * a ** k;
    edges.push(r0 + cum);
  }
  return edges;
})();

function hoverInfo(cell: CellInfo): {
  cls: string;
  zMax: number;
  zMean: number;
  occ: number;
  dyn: number;
  r: number;
  deg: number;
} {
  const rIn = GRID_EDGES[cell.i];
  const rOut = GRID_EDGES[cell.i + 1] ?? GRID_EDGES[GRID_EDGES.length - 1];
  const r = (rIn + rOut) / 2;
  const deg = ((cell.j / 720) * 360 + 180) % 360;
  const clsCls = CLASSES.find((c) => c.id === cell.cls);
  return {
    cls: clsCls?.label ?? "other",
    zMax: cell.zMax,
    zMean: cell.zMean,
    occ: cell.occ,
    dyn: cell.dyn,
    r,
    deg,
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

  const maxR = meta?.r_max ?? 100;

  const handlePause = (p: boolean) => {
    setPaused(p);
    send({ type: "control", action: p ? "pause" : "play" });
  };

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

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <Canvas
            camera={{ position: [0, 130, 95], fov: 45, near: 0.1, far: 1000 }}
            dpr={[1, 2]}
          >
            <color attach="background" args={["#0b0b0e"]} />
            <ambientLight intensity={0.7} />
            <directionalLight position={[60, 120, 40]} intensity={1.4} />
            <GroundPlane maxR={maxR} />
            <RangeRings maxR={maxR} />
            <FramePerf />
            {/* heading-up: the grid/cloud are refreshed in the CURRENT ego
                frame every tick, so a fixed +90deg puts the ego's forward
                (+X scene axis) at the top of the screen permanently. */}
            <group rotation={[0, Math.PI / 2, 0]}>
              {(viewMode === "grid" || viewMode === "compare") && (
                <CellLayer
                  cells={cells}
                  meta={meta}
                  patch={patch}
                  onHover={viewMode === "grid" ? (i) => setHover(i ? hoverInfo(i) : null) : undefined}
                  opacity={gridOpacity}
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
              makeDefault
              target={[0, 0, 0]}
              minDistance={30}
              maxDistance={260}
              enablePan
              maxPolarAngle={Math.PI / 2.05}
            />
          </Canvas>

          <div className="pointer-events-none absolute bottom-4 left-4 rounded-md bg-zinc-900/80 px-3 py-2 text-xs text-zinc-400 backdrop-blur">
            <span className="font-semibold text-zinc-200">
              {viewMode === "grid"
                ? "2.5D grid"
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
          </div>

          <div className="pointer-events-none absolute left-4 top-4 rounded-md bg-zinc-900/80 px-3 py-2 text-xs backdrop-blur">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
              {CLASSES.map((c) => (
                <div key={c.id} className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: c.color }} />
                  <span className="text-[11px] leading-none text-zinc-300">{c.label}</span>
                </div>
              ))}
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