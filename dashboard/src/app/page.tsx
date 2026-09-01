"use client";
/* eslint-disable react-hooks/immutability */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, OrthographicCamera, PerspectiveCamera } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Compass, Refresh } from "iconoir-react";
import MetricsPanel, { type ViewMode } from "@/components/MetricsPanel";
import Timeline from "@/components/Timeline";
import { CLASSES } from "@/lib/colors";
import {
  CellLayer,
  CloudComparisonView,
  CloudSegView,
  GroundPlane,
  PerfOverlay,
  RangeRings,
  EgoMarker,
  type CellInfo,
} from "@/components/MapScene";
import { useMapStream } from "@/lib/useMapStream";

type CamMode = "persp" | "ortho" | "top";

function computeGridEdges(meta: { r_min: number; dr_0: number; alpha: number; n_rings: number; phase1_rings?: number }): number[] {
  const { r_min, dr_0, alpha, n_rings, phase1_rings = 0 } = meta;
  const edges: number[] = [r_min];
  let cum = 0;
  for (let k = 0; k < n_rings; k++) {
    cum += k < phase1_rings ? dr_0 : dr_0 * alpha ** (k - phase1_rings);
    edges.push(r_min + cum);
  }
  return edges;
}

function DemandInvalidator({ patch, camMode }: { patch: unknown; camMode: string }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    invalidate();
  }, [patch, camMode, invalidate]);
  return null;
}

/** Smooth camera reset: lerps position + target each frame. */
function CameraAnimator({ target, running, onDone }: { target: { pos: THREE.Vector3; look: THREE.Vector3 }; running: boolean; onDone: () => void }) {
  const { camera } = useThree();
  const elapsed = useRef(0);
  const startPos = useRef(new THREE.Vector3());
  const startLook = useRef(new THREE.Vector3());
  const duration = 0.6; // seconds

  useEffect(() => {
    if (!running) return;
    elapsed.current = 0;
    startPos.current.copy(camera.position);
    startLook.current.set(0, 0, 0);
    // Try to get current target from controls
    const oc = camera.userData.__orbitControls as OrbitControlsImpl | undefined;
    if (oc) startLook.current.copy(oc.target);
  }, [running, camera]);

  useFrame((_, dt) => {
    if (!running) return;
    elapsed.current += dt;
    const t = Math.min(1, elapsed.current / duration);
    // Ease-out cubic
    const e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(startPos.current, target.pos, e);
    // Also smoothly move the orbit target
    const oc = camera.userData.__orbitControls as OrbitControlsImpl | undefined;
    if (oc) {
      oc.target.lerpVectors(startLook.current, target.look, e);
      oc.update();
    }
    if (t >= 1) onDone();
  });

  return null;
}

function OrthoAutoFit({ maxR }: { maxR: number }) {
  const cam = useThree((s) => s.camera as THREE.OrthographicCamera | THREE.PerspectiveCamera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  useLayoutEffect(() => {
    const o = cam as THREE.OrthographicCamera;
    if (!o.isOrthographicCamera) return;
    // Fit the full world diameter edge-to-edge to the viewport WIDTH.
    // Subtract bottom overlay height (~70px) so the map doesn't tuck
    // behind the timeline / mode label.
    const bottomMargin = 70;
    const worldDiameter = 2 * maxR;
    const zoomW = size.width / worldDiameter;
    const zoomH = (size.height - bottomMargin) / worldDiameter;
    // Use whichever zoom is *larger* (shows more, so the map fits in both axes).
    // Width-first for wide screens, height-first for tall.
    const zoom = Math.min(zoomW, zoomH);
    o.zoom = Math.max(0.5, zoom);
    o.updateProjectionMatrix();
    invalidate();
  }, [size.width, size.height, maxR, cam, invalidate]);
  return null;
}

function hoverInfo(cell: CellInfo, gridEdges: number[], nTheta: number): {
  cls: string;
  clsColor: string;
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
    clsColor: clsCls?.color ?? "#ffffff",
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

type SeqInfo = { id: string; frames: number; has_poses: boolean; has_labels: boolean };

const API = process.env.NEXT_PUBLIC_PC2D_API ?? "http://localhost:8000";

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
    seqId,
    switchSequence,
    status,
    statusMsg,
    buffering,
  } = useMapStream();
  const [paused, setPaused] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [pointSize, setPointSize] = useState(2);
  const [hover, setHover] = useState<ReturnType<typeof hoverInfo> | null>(null);
  const [camMode, setCamMode] = useState<CamMode>("persp");
  const [animTarget, setAnimTarget] = useState<{ pos: THREE.Vector3; look: THREE.Vector3 } | null>(null);
  const [availableSeqs, setAvailableSeqs] = useState<SeqInfo[]>([]);

  // client-side perf tracking (wired to PerfOverlay + sidebar FPS)
  const [clientFps, setClientFps] = useState(0);
  const [clientJs, setClientJs] = useState(0);
  const clientPerfN = useRef(0);
  const onClientPerf = useCallback((s: { fps: number; js: number; draws: number; tris: number }) => {
    setClientFps(s.fps);
    setClientJs(s.js);
    clientPerfN.current += 1;
    if (clientPerfN.current % 10 === 0) {
      console.log(
        `[render:perf] client_fps=${s.fps.toFixed(1)} js=${s.js.toFixed(1)}ms ` +
        `draws=${s.draws} tris=${(s.tris / 1000).toFixed(0)}K`
      );
    }
  }, []);

  const maxR = meta?.r_max ?? 100;
  const gridEdges = useMemo(() => meta ? computeGridEdges(meta) : [], [meta?.r_min, meta?.dr_0, meta?.alpha, meta?.n_rings, meta?.phase1_rings]);
  const nTheta = meta?.n_theta ?? 720;

  const resetNorth = () => {
    const pos = camMode === "top"
      ? new THREE.Vector3(0, 180, 0.01)
      : new THREE.Vector3(0, 130, 95);
    setAnimTarget({ pos, look: new THREE.Vector3(0, 0, 0) });
  };

  const onAnimDone = useCallback(() => setAnimTarget(null), []);

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

  useEffect(() => {
    fetch(`${API}/sequences`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => setAvailableSeqs(data.sequences ?? []))
      .catch(() => {});
  }, []);

  const showCloud = viewMode === "seg" || viewMode === "raw" || viewMode === "compare";
  const gridOpacity = viewMode === "compare" ? 0.35 : 1;
  const showGrid = viewMode === "grid" || viewMode === "compare" || viewMode === "trav";

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-black text-zinc-100">
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <Canvas
            frameloop="demand"
            dpr={[1, 2]}
            onCreated={({ gl }) => {
              gl.toneMappingExposure = 1;
            }}
          >
            {/* Camera rig: perspective (angled 3D), ortho (angled 3D), top (2D). */}
            {camMode === "ortho" ? (
              <OrthographicCamera key="cam-ortho" makeDefault position={[0, 130, 95]} zoom={2.5} near={0.1} far={1000} />
            ) : camMode === "top" ? (
              <OrthographicCamera key="cam-top" makeDefault position={[0, 180, 0.01]} zoom={2.5} near={0.1} far={1000} />
            ) : (
              <PerspectiveCamera key="cam-persp" makeDefault position={[0, 130, 95]} fov={45} near={0.1} far={1000} />
            )}
            {(camMode === "ortho" || camMode === "top") && <OrthoAutoFit maxR={maxR} />}
            <color attach="background" args={["#000000"]} />
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
              key={`oc-${camMode}`}
              ref={(r) => {
                if (r) (r.object as THREE.Camera).userData.__orbitControls = r;
              }}
              makeDefault
              target={[0, 0, 0]}
              minDistance={10}
              maxDistance={300}
              enablePan
              maxPolarAngle={camMode === "top" ? Math.PI / 2.001 : Math.PI / 2.05}
              minPolarAngle={camMode === "top" ? 0 : undefined}
              minZoom={0.3}
              maxZoom={12}
            />
            <PerfOverlay onStats={onClientPerf} />
            <DemandInvalidator patch={patch} camMode={camMode} />
            {animTarget && (
              <CameraAnimator
                target={animTarget}
                running={!!animTarget}
                onDone={onAnimDone}
              />
            )}
          </Canvas>

          {/* -- Loading / buffering overlay -------------------------------- */}
          {buffering && (
            <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60 backdrop-blur-[2px]">
              <div className="frost flex flex-col items-center gap-3 px-6 py-5">
                <Refresh className="h-6 w-6 animate-spin text-cyan-400" />
                <div className="text-sm font-medium text-zinc-100">{statusMsg ?? (status === "loading" ? "Loading model\u2026" : "Buffering first frame\u2026")}</div>
              </div>
            </div>
          )}
          {!buffering && cellCount === 0 && meta && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="frost px-4 py-2 text-xs text-zinc-400">Press Play to start</div>
            </div>
          )}

          {/* -- HUD top chrome: legend (left) / camera controls (right) --- */}
          <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-2 p-3">
            <div className="frost pointer-events-none px-3 py-2">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                {viewMode === "trav" ? (
                  <>
                    <LegendSwatch color="#22c55e" label="Drivable" />
                    <LegendSwatch color="#f59e0b" label="Caution" />
                    <LegendSwatch color="#ef4444" label="Blocked" />
                  </>
                ) : (
                  CLASSES.map((c) => <LegendSwatch key={c.id} color={c.color} label={c.label} />)
                )}
              </div>
            </div>

            <div className="frost pointer-events-auto flex items-center gap-1.5 p-1">
              {(["persp", "ortho", "top"] as CamMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setCamMode(m)}
                  className={`rounded-[2.5px] px-2.5 py-1 text-xs font-medium transition-colors ${
                    camMode === m ? "bg-cyan-500/20 text-white ring-1 ring-cyan-400/50" : "text-zinc-400 hover:bg-white/10 hover:text-zinc-200"
                  }`}
                  title={m === "persp" ? "Perspective 3D" : m === "ortho" ? "Orthographic 3D" : "Top-down 2D"}
                >
                  {m === "persp" ? "3D" : m === "ortho" ? "Ortho" : "Top"}
                </button>
              ))}
              <div className="mx-0.5 h-4 w-px bg-white/15" />
              <button
                onClick={resetNorth}
                className="flex items-center gap-1 rounded-[2.5px] px-2.5 py-1 text-xs font-medium text-zinc-400 transition-colors hover:bg-white/10 hover:text-zinc-200"
                title="Reset to north-up"
              >
                <Compass width={14} height={14} strokeWidth={2} />
                North
              </button>
            </div>
          </div>

          {/* -- bottom-left: current mode + grid size ------------------ */}
          <div className="frost pointer-events-none absolute bottom-4 left-4 px-4 py-2 text-xs text-zinc-400">
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
              <span className="hud-val ml-2">
                {meta.n_rings} × {meta.n_theta}
              </span>
            ) : null}
            {showCloud && cloud ? (
              <span className="hud-val ml-2 text-zinc-500">{(cloud.xyz.length / 3).toLocaleString()} pts</span>
            ) : null}
            {viewMode === "compare" && stats ? (
              <div className="hud-cap mt-1 text-cyan-400/90">
                {stats.compression_ratio.toLocaleString()}× fewer cells than uniform 5 cm grid
              </div>
            ) : null}
          </div>

          <Timeline
            seqPos={stats?.seq_pos ?? 0}
            seqLen={stats?.seq_len ?? 0}
            paused={paused}
            speed={speed}
            seeking={seeking}
            buffering={buffering}
            seqId={seqId}
            availableSeqs={availableSeqs}
            onPause={handlePause}
            onSeek={handleSeek}
            onSpeed={handleSpeed}
            onSwitchSeq={switchSequence}
          />
        </div>

        <MetricsPanel
          stats={stats}
          clientFps={clientFps}
          clientJs={clientJs}
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

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      <span className="text-xs leading-none text-zinc-300">{label}</span>
    </div>
  );
}