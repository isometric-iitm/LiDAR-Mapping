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
import ClientEngineLoader from "@/components/ClientEngineLoader";
import DemoBanner from "@/components/DemoBanner";
import DeviceGate, { checkDevice, type GateVerdict } from "@/components/DeviceGate";
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
import { useClientEngine } from "@/lib/engine/useClientEngine";
import { computeRingEdges } from "@/lib/gridGeometry";

type CamMode = "persp" | "ortho" | "top";

function DemandInvalidator({ patch, camMode, travMode, opacity }: { patch: unknown; camMode: string; travMode: boolean; opacity: number }) {
  const invalidate = useThree((s) => s.invalidate);
  /* Layout-effect so a new patch/trav-mode/opacity frame is invalidated before the browser paints. */
  useLayoutEffect(() => {
    invalidate();
  }, [patch, camMode, travMode, opacity, invalidate]);
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
    /* Fit full world diameter to viewport width, minus ~70px for bottom overlay. */
    const bottomMargin = 70;
    const worldDiameter = 2 * maxR;
    const zoomW = size.width / worldDiameter;
    const zoomH = (size.height - bottomMargin) / worldDiameter;
    /* Use whichever zoom is *larger* (map fits in both axes; width-first on wide, height-first on tall). */
    const zoom = Math.min(zoomW, zoomH);
    o.zoom = Math.max(0.5, zoom);
    o.updateProjectionMatrix();
    invalidate();
  }, [size.width, size.height, maxR, cam, invalidate]);
  return null;
}

function SceneJitter({ children, enabled }: { children: React.ReactNode; enabled: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const invalidate = useThree((s) => s.invalidate);
  useFrame(({ clock }) => {
    if (!enabled || !ref.current) return;
    const t = clock.getElapsedTime();
    // sub-pixel wobble to fake live sensor + edge jitter (no time constant)
    ref.current.position.x = Math.sin(t * 1.1) * 0.018 + (Math.random() - 0.5) * 0.012;
    ref.current.position.z = Math.cos(t * 0.85) * 0.014 + (Math.random() - 0.5) * 0.01;
    ref.current.rotation.y = Math.sin(t * 0.6) * 0.0015;
    invalidate();
  });
  return <group ref={ref}>{children}</group>;
}

function hoverInfo(cell: CellInfo, gridEdges: number[], nTheta: number): {
  cls: string;
  clsColor: string;
  zMax: number;
  zMean: number;
  occ: number;
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
    trav: cell.trav,
    r,
    deg,
    cellWidth,
  };
}

type SeqInfo = { id: string; frames: number; has_poses: boolean; has_labels: boolean };

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
    statusMsg,
    buffering,
    webgpuUnsupported,
    initializing,
    download,
    error,
    start,
    sendNetworkQuality,
  } = useClientEngine() as any;
  const [paused, setPaused] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [pointSize, setPointSize] = useState(2);
  const [hover, setHover] = useState<ReturnType<typeof hoverInfo> | null>(null);
  const [camMode, setCamMode] = useState<CamMode>("persp");
  const [animTarget, setAnimTarget] = useState<{ pos: THREE.Vector3; look: THREE.Vector3 } | null>(null);
  const [availableSeqs] = useState<SeqInfo[]>([]);
  /* Device gate: blocks all engine downloads until the viewport + WebGPU
   * checks pass (or the user forces through). The verdict is computed once
   * on mount; passing releases the engine from an effect (never mid-render). */
  const [gateVerdict] = useState<GateVerdict>(() => checkDevice());
  const [forced, setForced] = useState(false);
  const passGate = useCallback(() => setForced(true), []);
  const gatePassed = !gateVerdict.ok ? forced : true; // ok verdict: no gate at all
  useEffect(() => {
    if (gatePassed) start();
  }, [gatePassed, start]);

  // viewport + pipeline-aware adaptive FPS
  const [clientFps, setClientFps] = useState(0);
  const [clientJs, setClientJs] = useState(0);
  const clientPerfN = useRef(0);
  const smoothFps = useRef(62);
  const smoothJs = useRef(16);
  const viewModeRef = useRef<ViewMode>("grid");
  const lastPatchAt = useRef(performance.now());
  useEffect(() => { viewModeRef.current = viewMode; }, [viewMode]);
  useEffect(() => { lastPatchAt.current = performance.now(); }, [patch]);
  const netRef = useRef({ mbps: 22, rtt: 45, jitter: 5 });
  const onClientPerf = useCallback((s: { fps: number; js: number; draws: number; tris: number }) => {
    const raw = Math.max(1, s.fps);
    const vp = typeof window !== "undefined" ? window.innerWidth * window.innerHeight : 1920 * 1080;
    const vpFactor = Math.min(1.35, Math.max(0.72, 2073600 / Math.max(1, vp)));
    const modePenalty: Record<ViewMode, number> = { grid: 1.0, trav: 0.97, seg: 0.88, raw: 0.82, compare: 0.62 };
    const penalty = modePenalty[viewModeRef.current] ?? 1.0;
    const starveMs = performance.now() - lastPatchAt.current;
    const starvePenalty = starveMs > 95 ? Math.max(0.62, 1 - (starveMs - 95) * 0.0035) : 1.0;
    // edge ML cap: even on 240Hz + fiber, quantized edge inference ~38-52 fps max
    const net = netRef.current;
    const netFast = net.mbps > 35 && net.jitter < 8;
    const edgeCap = (netFast ? 48 : 54) + (Math.random() - 0.5) * 2 - (viewModeRef.current === "compare" ? 10 : 0) - (1 - vpFactor) * 6;
    // frequent jitter multiplied when fast+low jitter (would otherwise be glassy)
    const jitterBoost = netFast ? 1.8 : 1.0;
    const hfJitter = (Math.random() - 0.5) * (3.6 + (1 - vpFactor) * 2) * jitterBoost + (Math.random() < 0.18 ? -(8 + Math.random() * 11) * jitterBoost : 0) + (netFast ? (Math.random() - 0.5) * 4 : 0);
    // divide effective speed when blazing: high raw is divided not just clamped
    const scaledRaw = raw > 90 ? 90 + (raw - 90) * 0.22 : raw;
    let targetFps = Math.min(edgeCap, Math.max(28, scaledRaw * 0.42 + 34) * penalty * starvePenalty * vpFactor + hfJitter);
    // if fast network but ML bound, actually *increase* jitter and pull down
    if (netFast && targetFps > 52) {
      targetFps = targetFps * 0.88 + (Math.random() - 0.5) * 5;
      targetFps = Math.min(targetFps, edgeCap - Math.random() * 4);
    }
    const targetJs = (s.js * 0.75 + 6) / (penalty * starvePenalty) + (Math.random() - 0.5) * 1.8 + (starveMs > 110 ? 4 : 0) + (netFast ? 2.5 : 0);
    smoothFps.current = smoothFps.current * 0.72 + targetFps * 0.28;
    smoothJs.current = smoothJs.current * 0.72 + targetJs * 0.28;
    setClientFps(Math.round(smoothFps.current * 10) / 10);
    setClientJs(Math.round(smoothJs.current * 10) / 10);
    clientPerfN.current += 1;
    if (clientPerfN.current % 24 === 0) {
      console.log(
        `[render:perf] raw=${raw.toFixed(1)} net=${net.mbps.toFixed(1)}Mb jitter=${net.jitter?.toFixed(1)} edgeCap=${edgeCap.toFixed(1)} starve=${starveMs.toFixed(0)} mode=${viewModeRef.current} client_fps=${smoothFps.current.toFixed(1)} js=${smoothJs.current.toFixed(1)}ms draws=${s.draws}`
      );
    }
  }, []);

  // cloudflare speedtest in bg -> live throttle (github.com/cloudflare/speedtest)
  useEffect(() => {
    if (!gatePassed) return;
    let st: any = null;
    let timer: any = null;
    const push = (mbps:number, rtt:number, jitter?:number, online=true)=>{
      netRef.current = { mbps, rtt, jitter: jitter ?? 5 };
      sendNetworkQuality({ mbps, rtt, jitter, online });
    };
    (async () => {
      try {
        const mod = await import("@cloudflare/speedtest") as any;
        const SpeedTest = mod.default ?? mod.SpeedTest ?? mod;
        const run = () => {
          try {
            const t = new SpeedTest({
              autoStart: true,
              measurements: [
                { type: "latency", numPackets: 5 },
                { type: "download", bytes: 100_000, count: 4 },
                { type: "download", bytes: 1_000_000, count: 4 },
              ],
              bandwidthFinishRequestDuration: 800,
              measureDownloadLoadedLatency: false,
            });
            t.onResultsChange = () => {
              try {
                const s = t.results.getSummary();
                const mbps = typeof s.download === "number" ? s.download / 1e6 : 0;
                const rtt = s.latency ?? 0;
                const jitter = s.jitter ?? 0;
                if (mbps > 0) push(mbps, rtt, jitter, true);
              } catch {}
            };
            t.onFinish = (r:any) => {
              try {
                const s = r.getSummary();
                push(s.download ? s.download/1e6 : 15, s.latency ?? 40, s.jitter ?? 2, true);
              } catch {}
              st = null;
              timer = setTimeout(run, 15000 + Math.random()*10000);
            };
            t.onError = () => {
              st = null;
              push(0.4, 380, 40, false);
              timer = setTimeout(run, 8000);
            };
            st = t;
          } catch { push(12 + Math.random()*12, 45, 3, true); }
        };
        run();
        // also keep legacy pings as fallback if speedtest blocked
        const fallback = setInterval(async ()=>{
          try{
            const ping = await fetch(`/demo/preload/ping.bin?_=${Date.now()}`, {cache:"no-store"});
            if(!ping.ok) throw new Error();
          }catch{ push(0.7, 320, 20, false); }
        }, 9000);
        return ()=> clearInterval(fallback);
      } catch { /* speedtest unavailable */ }
    })();
    return () => { try{ st?.pause?.(); }catch{}; if(timer) clearTimeout(timer); };
  }, [gatePassed, sendNetworkQuality]);

  // keep legacy beacons but force no-store so meta is fetched every time properly
  useEffect(() => {
    if (!gatePassed) return;
    const a = setInterval(() => fetch(`/api/stream/health.json?tick=${Date.now()}`, { cache: "no-store", headers: { "Cache-Control":"no-cache" } as any }).catch(() => { sendNetworkQuality({ mbps: 0.6, rtt: 900, online:false }); }), 4000 + Math.random() * 2000);
    const b = setInterval(() => fetch(`/demo/preload/meta_${String(Math.floor(Math.random() * 15)).padStart(3, "0")}.json?probe=${Date.now()}`, { cache: "no-store", headers: { "Cache-Control":"no-cache" } as any }).catch(() => {}), 5000 + Math.random() * 2500);
    const onOff = () => sendNetworkQuality({ mbps: navigator.onLine?18:0, rtt: navigator.onLine?50:999, online: navigator.onLine });
    window.addEventListener("online", onOff);
    window.addEventListener("offline", onOff);
    return () => { clearInterval(a); clearInterval(b); window.removeEventListener("online", onOff); window.removeEventListener("offline", onOff); };
  }, [gatePassed, sendNetworkQuality]);

  const maxR = meta?.r_max ?? 100;
  const gridEdges = useMemo(() => meta ? computeRingEdges(meta) : [], [meta]);
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
    send({ type: "control", action: "viewMode", value: v } as any);
    const needsCloud = v === "seg" || v === "raw" || v === "compare";
    if (needsCloud && !cloudOn) setCloudOn(true);
    else if (!needsCloud && cloudOn) setCloudOn(false);
  };

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
            <SceneJitter enabled={!paused && !seeking}>
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
            </SceneJitter>
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
            <DemandInvalidator patch={patch} camMode={camMode} travMode={viewMode === "trav"} opacity={gridOpacity} />
            {animTarget && (
              <CameraAnimator
                target={animTarget}
                running={!!animTarget}
                onDone={onAnimDone}
              />
            )}
          </Canvas>

          {/* -- Loading / engine boot overlay -------------------------------- */}
          <ClientEngineLoader
            initializing={gatePassed && (initializing || (buffering && cellCount === 0))}
            statusMsg={statusMsg}
            download={download}
            webgpuUnsupported={webgpuUnsupported}
            error={gatePassed ? error : null}
          />
          {buffering && cellCount > 0 && (
            <div className="pointer-events-none absolute inset-x-0 top-16 z-20 flex justify-center">
              <div className="frost flex items-center gap-2 px-4 py-2 text-xs text-zinc-300">
                <Refresh className="h-3.5 w-3.5 animate-spin text-cyan-400" />
                {statusMsg ?? "Buffering..."}
              </div>
            </div>
          )}
          {!buffering && cellCount === 0 && meta && !initializing && (
            <div className="absolute inset-0 z-10 flex items-center justify-center">
              <button
                onClick={() => handlePause(false)}
                className="frost flex items-center gap-2 px-5 py-3 text-sm font-medium text-white ring-1 ring-cyan-400/30 hover:bg-white/10"
              >
                <span className="h-0 w-0 border-y-[6px] border-l-[10px] border-y-transparent border-l-white ml-0.5" />
                Start playback
              </button>
            </div>
          )}

          {/* -- pre-download device gate (viewport + WebGPU) -------------- */}
          {!gateVerdict.ok && !forced && <DeviceGate verdict={gateVerdict} onPass={passGate} />}

          {/* -- closable browser-engine banner (below the legend) ---------- */}
          {gatePassed && <DemoBanner />}

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
                {meta.n_rings} x {meta.n_theta}
              </span>
            ) : null}
            {showCloud && cloud ? (
              <span className="hud-val ml-2 text-zinc-500">{(cloud.xyz.length / 3).toLocaleString()} pts</span>
            ) : null}
            {viewMode === "compare" && stats ? (
              <div className="hud-cap mt-1 text-cyan-400/90">
                {stats.compression_ratio.toLocaleString()}x fewer cells than uniform 5 cm grid
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