"use client";
/**
 * Main-thread hook for the in-browser engine (demo mode).
 *
 * Returns the exact `UseMapStream` contract so page.tsx / InstancedCellLayer /
 * Timeline / MetricsPanel work unchanged. Spawns engineWorker, mirrors the
 * server message semantics (epoch guard, freeze frame, frees-before-upserts,
 * snapshot cadence) and reshapes worker messages into React state.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { Cell, GridMeta, Stats } from "../types";
import type { CellMap, CellPatch, ConnState, StatusState, UseMapStream } from "../useMapStream";
import { keyOf } from "../gridGeometry";

export type EngineDownload = {
  phase: "model" | "sequence";
  chunk?: number;
  fraction: number;
  loaded: number;
  total: number;
};

export type UseClientEngine = UseMapStream & {
  /** WebGPU missing / session creation failed -> show the gate panel. */
  webgpuUnsupported: boolean;
  /** True until the worker posted `ready` (drives the loader overlay). */
  initializing: boolean;
  /** Latest asset download progress (loader UI). */
  download: EngineDownload | null;
  error: string | null;
  /** Deferred start: the device gate calls this before any download begins. */
  start: () => void;
};

type WorkerPatch =
  | { type: "patch"; kind: "reset"; frame: number; epoch: number }
  | {
      type: "patch";
      kind: "snap";
      frame: number;
      epoch: number;
      /** (k,6) f32: i, j, zMean, zMax, occ, trav */
      rows: Float32Array;
      /** (k,) u8: cls */
      cls: Uint8Array;
      nUps: number;
      yaw?: number;
    }
  | {
      type: "patch";
      kind: "delta";
      frame: number;
      epoch: number;
      rows: Float32Array;
      cls: Uint8Array;
      nUps: number;
      /** (m,2) f32 flat [i,j,...] */
      freed: Float32Array;
      yaw?: number;
    };

/** Unpack a binary patch into the Cell tuple shape the renderer consumes.
 *  Single tight loop; frees are read from the flat f32 pairs (snap has none). */
function unpackPatch(msg: Extract<WorkerPatch, { kind: "snap" | "delta" }>): { upserts: Cell[]; frees: [number, number][] } {
  const k = msg.nUps;
  const upserts: Cell[] = new Array(k);
  const rows = msg.rows;
  const cls = msg.cls;
  for (let q = 0; q < k; q++) {
    const o6 = q * 6;
    upserts[q] = [rows[o6], rows[o6 + 1], rows[o6 + 2], rows[o6 + 3], cls[q], rows[o6 + 4], rows[o6 + 5]];
  }
  if (msg.kind !== "delta") return { upserts, frees: [] };
  const nf = msg.freed.length / 2;
  const frees: [number, number][] = new Array(nf);
  const f = msg.freed;
  for (let q = 0; q < nf; q++) frees[q] = [f[q * 2], f[q * 2 + 1]];
  return { upserts, frees };
}type WorkerMsg =
  | { type: "ready" }
  | { type: "grid_meta" } & GridMeta
  | { type: "status"; state: string; msg?: string }
  | { type: "stats" } & Stats
  | WorkerPatch
  | { type: "cloud"; frame: number; n: number; xyz: Float32Array; cls: Uint8Array; epoch: number; yaw?: number }
  | { type: "control_ack"; action: string; frame: number; idx?: number; epoch?: number }
  | { type: "download" } & EngineDownload
  | { type: "webgpu_unsupported" }
  | { type: "error"; message: string };

export function useClientEngine(): UseClientEngine {
  const [conn] = useState<ConnState>("open");
  const [meta, setMeta] = useState<GridMeta | null>(null);
  const [cells, setCells] = useState<CellMap>(() => new Map());
  const [lastFrame, setLastFrame] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [cloud, setCloud] = useState<{ frame: number; xyz: Float32Array; cls: Uint8Array } | null>(null);
  const [cloudOn, setCloudOn] = useState(false);
  const [seeking, setSeeking] = useState(false);
  const [heading, setHeading] = useState(0);
  const [patch, setPatch] = useState<CellPatch | null>(null);
  const [seqId, setSeqId] = useState("");
  const [status, setStatus] = useState<StatusState>("loading");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [buffering, setBuffering] = useState(true);
  const [webgpuUnsupported, setWebgpuUnsupported] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [download, setDownload] = useState<EngineDownload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveCount, setLiveCount] = useState(0);

  const workerRef = useRef<Worker | null>(null);
  const nThetaRef = useRef(720);
  const full = useRef<Map<number, Cell>>(new Map());
  const epochRef = useRef(-1);
  const lastAppliedFrame = useRef(-1);
  const freezeFrame = useRef(-1);
  const lastHeading = useRef(-1);
  const cloudOnRef = useRef(false);
  const sendRef = useRef<(msg: object) => void>(() => {});
  /* Device gate: start() records that the gate passed and spawns the worker
   * (which begins all downloads). If start() fires before this effect has
   * installed spawn(), the flag makes the effect spawn on mount instead. */
  const gateRef = useRef<{ started: boolean; spawn: (() => void) | null }>({ started: false, spawn: null });

  useEffect(() => {
    let worker: Worker | null = null;

    const handleMsg = (ev: MessageEvent) => {
      const msg = ev.data as WorkerMsg;
      switch (msg.type) {
        case "grid_meta": {
          const metaMsg = msg as GridMeta & { type: "grid_meta" };
          setMeta(metaMsg);
          nThetaRef.current = (metaMsg as GridMeta).n_theta;
          full.current.clear();
          epochRef.current = -1;
          lastAppliedFrame.current = -1;
          setCells(new Map());
          setLiveCount(0);
          setPatch({ kind: "reset", frame: 0 });
          setSeqId((metaMsg as GridMeta).seq_id ?? "");
          return;
        }
        case "status": {
          const s = msg.state as StatusState;
          setStatus(s);
          setStatusMsg(msg.msg ?? null);
          if (s === "loading" || s === "buffering") setBuffering(true);
          else if (s === "ready") setBuffering(false);
          return;
        }
        case "download": {
          setDownload({ phase: msg.phase, chunk: msg.chunk, fraction: msg.fraction, loaded: msg.loaded, total: msg.total });
          return;
        }
        case "webgpu_unsupported": {
          setWebgpuUnsupported(true);
          setInitializing(false);
          setStatus("error");
          setStatusMsg("WebGPU is not available in this browser");
          return;
        }
        case "error": {
          setError(msg.message);
          setStatus("error");
          setStatusMsg(msg.message);
          setInitializing(false);
          return;
        }
        case "ready": {
          setInitializing(false);
          setDownload(null);
          return;
        }
        case "control_ack": {
          if (msg.action === "seek") {
            epochRef.current = msg.epoch ?? epochRef.current;
            freezeFrame.current = -1;
            setSeeking(false);
          }
          return;
        }
        case "stats": {
          setStats(msg as unknown as Stats);
          return;
        }
        default:
          break;
      }

      // per-frame messages below carry the seek epoch
      if ("epoch" in msg) {
        const e = (msg as { epoch: number }).epoch;
        if (epochRef.current < 0) epochRef.current = e;
        else if (e !== epochRef.current) return;
      }
      if ("yaw" in msg && typeof (msg as { yaw?: number }).yaw === "number") {
        const yaw = (msg as { yaw: number }).yaw;
        if (Math.abs(yaw - lastHeading.current) > 1e-3) {
          lastHeading.current = yaw;
          setHeading(yaw);
        }
      }

      // freeze while paused (hold the newest frame)
      const frozen =
        freezeFrame.current >= 0 && "frame" in msg && (msg as { frame: number }).frame > freezeFrame.current;
      if (frozen) return;

      if (msg.type === "cloud") {
        if (!cloudOnRef.current) return;
        setCloud({ frame: msg.frame, xyz: msg.xyz, cls: msg.cls });
        return;
      }

      if (msg.type === "patch") {
        if (msg.kind === "reset") {
          full.current.clear();
          setCells(new Map());
          setLiveCount(0);
          setCloud(null);
          setPatch({ kind: "reset", frame: msg.frame });
          lastAppliedFrame.current = -1;
          return;
        }
        if (msg.frame <= lastAppliedFrame.current) return;
        const nTheta = nThetaRef.current;
        const { upserts, frees } = unpackPatch(msg);
        if (msg.kind === "snap") {
          const next = new Map<number, Cell>();
          for (const c of upserts) next.set(keyOf(c[0], c[1], nTheta), c);
          full.current = next;
          setCells(new Map(next));
          setLiveCount(next.size);
          setLastFrame(msg.frame);
          lastAppliedFrame.current = msg.frame;
          setPatch({ kind: "snap", frame: msg.frame, upserts });
        } else {
          // frees first, then upserts (a freed+re-added cell stays present)
          for (const [i, j] of frees) full.current.delete(keyOf(i, j, nTheta));
          for (const c of upserts) full.current.set(keyOf(c[0], c[1], nTheta), c);
          setCells(full.current);
          setLiveCount(full.current.size);
          setLastFrame(msg.frame);
          lastAppliedFrame.current = msg.frame;
          setPatch({ kind: "delta", frame: msg.frame, upserts, frees });
        }
        if (full.current.size > 0) setBuffering(false);
        return;
      }
    };

    const spawn = (): void => {
      if (gateRef.current.started) return; // already spawned
      gateRef.current.started = true;
      try {
        worker = new Worker(new URL("./engineWorker", import.meta.url), { type: "module" });
      } catch {
        gateRef.current.started = false;
        return;
      }
      workerRef.current = worker;
      worker.onmessage = handleMsg;
      worker.onerror = () => {
        setError("engine worker crashed");
        setInitializing(false);
      };
      sendRef.current = (msg: object) => worker!.postMessage(msg);
      worker.postMessage({ type: "init" });
    };

    // expose spawn for start(); honor an early start() that arrived before
    // this effect ran (its call was a no-op, the flag says go)
    gateRef.current.spawn = spawn;
    if (gateRef.current.started) spawn();

    return () => {
      worker?.terminate();
      workerRef.current = null;
    };
  }, []);

  const seekTo = useCallback((idx: number) => {
    setSeeking(true);
    sendRef.current({ type: "control", action: "seek", value: idx });
  }, []);

  /** Device-gate trigger: spawns the worker (downloads start only after this).
   *  Safe to call before mount: the flag is honored when the effect runs. */
  const start = useCallback(() => {
    const g = gateRef.current;
    if (g.spawn) g.spawn();
    else g.started = true; // effect not yet run; spawn on mount
  }, []);

  const setCloudOnLocal = useCallback((on: boolean) => {
    cloudOnRef.current = on;
    setCloudOn(on);
    if (!on) setCloud(null);
    sendRef.current({ type: "control", action: "cloud", value: on });
  }, []);

  /** Wrap send() so pause/play also toggle the freeze-frame bookkeeping
   *  exactly like the server's control_ack path did in useMapStream. */
  const sendWithFreeze = useCallback(
    (msg: object) => {
      const action = (msg as { action?: string }).action;
      if (action === "pause") {
        freezeFrame.current = lastAppliedFrame.current;
        sendRef.current(msg);
      } else if (action === "play") {
        freezeFrame.current = -1;
        sendRef.current(msg);
      } else {
        sendRef.current(msg);
      }
    },
    []
  );

  return {
    conn,
    meta,
    cells,
    lastFrame,
    stats,
    cellCount: liveCount,
    cloud,
    cloudOn,
    setCloudOn: setCloudOnLocal,
    patch,
    heading,
    seeking,
    seekTo,
    send: sendWithFreeze,
    seqId,
    switchSequence: () => {}, // single packed sequence in demo mode
    status,
    statusMsg,
    buffering,
    webgpuUnsupported,
    initializing,
    download,
    error,
    start,
  } as UseClientEngine;
}
