"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { inflateRaw } from "pako";
import type { AckMsg, Cell, DeltaMsg, GridMeta, SnapshotMsg, Stats } from "./types";
import { keyOf } from "./gridGeometry";

export type CellMap = Map<number, Cell>;
export type ConnState = "connecting" | "open" | "closed";
export type StatusState = "loading" | "buffering" | "ready" | "error";

/** Incremental render patch: only the cells that actually changed arrive
 *  from a delta, so the scene can update a few thousand instances instead of
 *  rebuilding all ~37k every frame. 'snap' is a full re-sync (carried by the
 *  periodic server snapshot), 'reset' clears state after a seek / new meta. */
export type CellPatch =
  | { kind: "reset"; frame: number }
  | { kind: "delta"; frame: number; upserts: Cell[]; frees: [number, number][] }
  | { kind: "snap"; frame: number; upserts: Cell[] };

export type UseMapStream = {
  conn: ConnState;
  meta: GridMeta | null;
  cells: CellMap;
  lastFrame: number;
  stats: Stats | null;
  cellCount: number;
  cloud: { frame: number; xyz: Float32Array; cls: Uint8Array } | null;
  cloudOn: boolean;
  setCloudOn: (on: boolean) => void;
  patch: CellPatch | null;
  heading: number;
  seeking: boolean;
  seekTo: (idx: number) => void;
  send: (msg: object) => void;
  seqId: string;
  switchSequence: (seqId: string) => void;
  status: StatusState;
  statusMsg: string | null;
  buffering: boolean;
};

const WS_URL = process.env.NEXT_PUBLIC_PC2D_WS ?? "ws://localhost:8000/ws/map";

const MAGIC = 0x50433244;
const K_SNAPSHOT = 1;
const K_DELTA = 2;
const K_CLOUD = 3;

/** Decompress a binary frame if it has the 'Z' prefix (server raw-DEFLATE +
 *  5-byte header: b'Z' + uint32 original length -- see ws_protocol.py).
 *  Synchronous pako inflate avoids an await + stream teardown per WebSocket
 *  message, which previously throttled high-fps binary chunk streams. */
function decompressFrame(buf: ArrayBuffer): ArrayBuffer {
  const u8 = new Uint8Array(buf);
  if (u8.length <= 5 || u8[0] !== 0x5a) return buf; // not compressed
  try {
    const out = inflateRaw(u8.subarray(5));
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
  } catch (e) {
    console.warn("[ws] decompress failed, using raw:", e);
    return buf;
  }
}

/** Decode a server binary frame (see src/server/ws_protocol.py, 44-byte
 *  header `struct "<IHHQQiiiii"`) into the legacy message shapes so the
 *  existing chunk/epoch/freeze logic is shared. */

/** ServerMsg plus the fields the binary channel carries that the legacy JSON
 *  shapes didn't (chunk bookkeeping on cloud frames, typed-array payloads,
 *  per-frame ego yaw in radians). */
type MapMsg =
  | GridMeta
  | Stats
  | AckMsg
  | { type: "status"; state: string; msg?: string; seq_id?: string }
  | (SnapshotMsg & { yaw?: number })
  | (DeltaMsg & { yaw?: number })
  | ({
      type: "cloud";
      frame: number;
      n: number;
      xyz: Float32Array | number[];
      cls: Uint8Array | number[];
      seq: number;
      total: number;
      epoch?: number;
      yaw?: number;
    });

function parseBinary(buf: ArrayBuffer): MapMsg | null {
  const dv = new DataView(buf);
  if (dv.byteLength < 44 || dv.getUint32(0, true) !== MAGIC) return null;
  const code = dv.getUint16(4, true);
  const frame = Number(dv.getBigInt64(8, true));
  const epoch = Number(dv.getBigInt64(16, true));
  const n = dv.getInt32(24, true);
  const nFreed = dv.getInt32(28, true);
  const seq = dv.getInt32(32, true);
  const total = dv.getInt32(36, true);
  const yaw = ((dv.getInt32(40, true) / 100) * Math.PI) / 180;
  const base = { frame, epoch, seq, total, yaw };

  if (code === K_CLOUD) {
    return {
      type: "cloud",
      ...base,
      n,
      xyz: new Float32Array(buf, 44, n * 3),
      cls: new Uint8Array(buf, 44 + n * 12, n),
    };
  }
  if (code !== K_SNAPSHOT && code !== K_DELTA) return null;

  // row record is 32 bytes: [i,j,z_mean,z_max,occ,dyn,trav] f32 (28) + cls u8 + 3 pad
  const f = new Float32Array(buf, 44, n * 7);
  const cls = new Uint8Array(buf, 44 + n * 28, n);
  const cells = new Array<Cell>(n);
  for (let k = 0; k < n; k++) {
    const o = k * 7;
    cells[k] = [f[o], f[o + 1], f[o + 2], f[o + 3], cls[k], f[o + 4], f[o + 5], f[o + 6]];
  }
  if (code === K_SNAPSHOT) {
    return { type: "snapshot", ...base, cells };
  }
  const freedNr = new Float32Array(buf, 44 + n * 32, nFreed * 2);
  const freed = new Array<[number, number]>(nFreed);
  for (let k = 0; k < nFreed; k++) freed[k] = [freedNr[2 * k], freedNr[2 * k + 1]];
  return { type: "delta", ...base, cells, freed };
}

export function useMapStream(): UseMapStream {
  const [conn, setConn] = useState<ConnState>("connecting");
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
  const [buffering, setBuffering] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const nThetaRef = useRef(720); // set on grid_meta; used for flat cell keys
  const full = useRef<Map<number, Cell>>(new Map());
  const snapBuf = useRef<Map<number, Cell>>(new Map());
  const snapFrame = useRef(-1);
  // Lightweight live-count state so the sidebar "Cells"/empty-hint update every
  // delta without an O(n) `cells` copy each frame (`cells` stays authoritative
  // on snapshot/reset only - same-ref mutation on deltas would otherwise bail out).
  const [liveCount, setLiveCount] = useState(0);
  const sendRef = useRef<(msg: object) => void>(() => {});
  const freezeFrame = useRef(-1); // -1 = live; else hold grid state at this frame
  const epochRef = useRef(-1);    // seek epoch; mismatched frames are stale
  const cloudOnRef = useRef(false);
  const lastHeading = useRef(-1);
  const cloudBuf = useRef<{ xyz: Float32Array; cls: Uint8Array; off: number } | null>(null);

  const deltaAcc = useRef<Cell[]>([]);

  // perf logging accumulators
  const perfRef = useRef({ n: 0, totalParseMs: 0, totalHandleMs: 0, totalBytes: 0 });

  useEffect(() => {
    let cancelled = false;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;

    const connect = () => {
      if (cancelled) return;
      setConn("connecting");
      let ws: WebSocket;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        const backoff = Math.min(500 * 2 ** attempts, 5000);
        attempts += 1;
        retry = setTimeout(connect, backoff);
        return;
      }
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      sendRef.current = (msg: object) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
      };

      ws.onopen = () => {
        attempts = 0;
        if (!cancelled) setConn("open");
      };

      const handleMsg = (msg: MapMsg) => {
        if ("yaw" in msg && typeof msg.yaw === "number" && Math.abs(msg.yaw - lastHeading.current) > 1e-3) {
          lastHeading.current = msg.yaw;
          setHeading(msg.yaw);
        }

        if (msg.type === "grid_meta") {
          setMeta(msg);
          nThetaRef.current = msg.n_theta;
          full.current.clear();
          snapBuf.current.clear();
          freezeFrame.current = -1;
          epochRef.current = -1;
          lastHeading.current = -1;
          setHeading(0);
          setSeeking(false);
          setCells(new Map());
          setLiveCount(0);
          deltaAcc.current = [];
          setPatch({ kind: "reset", frame: 0 });
          if ("seq_id" in msg && typeof msg.seq_id === "string") setSeqId(msg.seq_id);
          return;
        }

        if (msg.type === "status") {
          const s = (msg as { state: string }).state as StatusState;
          setStatus(s);
          setStatusMsg((msg as { msg?: string }).msg ?? null);
          if (s === "loading" || s === "buffering") setBuffering(true);
          else if (s === "ready") setBuffering(false);
          return;
        }

        if (msg.type === "control_ack") {
          if (msg.action === "pause") freezeFrame.current = msg.frame;
          else if (msg.action === "play") freezeFrame.current = -1;
          else if (msg.action === "seek") {
            epochRef.current = msg.epoch ?? epochRef.current;
            freezeFrame.current = -1;
            setSeeking(false);
            full.current.clear();
            snapBuf.current.clear();
            deltaAcc.current = [];
            setCells(new Map());
            setLiveCount(0);
            setCloud(null);
            setPatch({ kind: "reset", frame: msg.frame ?? 0 });
          } else if (msg.action === "switch_sequence") {
            epochRef.current = msg.epoch ?? epochRef.current;
            freezeFrame.current = -1;
            setSeeking(false);
            setSeqId(msg.seq_id ?? "");
            full.current.clear();
            snapBuf.current.clear();
            deltaAcc.current = [];
            setCells(new Map());
            setLiveCount(0);
            setCloud(null);
            setPatch({ kind: "reset", frame: msg.frame ?? 0 });
          }
          return;
        }

        // seek epoch guard: drop any frame produced before the latest seek
        if ("epoch" in msg) {
          const e = msg.epoch;
          if (e === undefined) return;
          if (epochRef.current < 0) epochRef.current = e;
          else if (e !== epochRef.current) return;
        }

        if (msg.type === "stats") {
          setStats(msg);
          return;
        }

        // freeze: hold the grid/cloud state at the ack frame while paused
        const frozen =
          msg.type === "delta" ||
          msg.type === "snapshot" ||
          msg.type === "cloud"
            ? freezeFrame.current >= 0 && msg.frame > freezeFrame.current
            : false;
        if (frozen) return;

        if (msg.type === "cloud") {
          if (!cloudOnRef.current) return;
          const n = msg.n;
          if (msg.seq === 0) cloudBuf.current = { xyz: new Float32Array(n * 3), cls: new Uint8Array(n), off: 0 };
          const cb = cloudBuf.current;
          if (cb) {
            // capacity counts points; each point is 3 floats in xyz
            if (cb.off + n > cb.xyz.length / 3) {
              const nx = new Float32Array(Math.max(cb.xyz.length * 2, (cb.off + n) * 3));
              nx.set(cb.xyz);
              cb.xyz = nx;
              const nc = new Uint8Array(nx.length / 3);
              nc.set(cb.cls);
              cb.cls = nc;
            }
            cb.xyz.set(msg.xyz, cb.off * 3);
            cb.cls.set(msg.cls, cb.off);
            cb.off += n;
          }
          if (msg.seq === msg.total - 1 && cb) {
            setCloud({ frame: msg.frame, xyz: cb.xyz.slice(0, cb.off * 3), cls: cb.cls.slice(0, cb.off) });
            cloudBuf.current = null;
          }
          return;
        }

        if (msg.type === "delta") {
          // Incremental frame. The server sends each frame as a chain of chunks
          // (seq 0..total-1); 'freed' rides on the final chunk. Apply upserts
          // eagerly (cheap per-chunk) and commit on the final chunk. A fresh
          // frame always starts at seq 0 — if we see one, any partial
          // accumulation from a prior frame is stale, so reset it to avoid
          // leaking upserts across dropped frames.
          const t0 = performance.now();
          if (msg.seq === 0) {
            deltaAcc.current = [];
          }
          for (const c of msg.cells) {
            full.current.set(keyOf(c[0], c[1], nThetaRef.current), c);
            deltaAcc.current.push(c);
          }
          for (const [i, j] of msg.freed) {
            full.current.delete(keyOf(i, j, nThetaRef.current));
          }
          if (msg.seq === msg.total - 1) {
            const mapCopy0 = performance.now();
            // Pass the live map reference (not a full copy): `cells` is only
            // read on snapshot frames and for the empty/hint check, so an
            // O(n) copy each delta frame is wasted work. Snapshots resync it.
            setCells(full.current);
            setLiveCount(full.current.size);
            setLastFrame(msg.frame);
            setPatch({ kind: "delta", frame: msg.frame, upserts: deltaAcc.current, frees: msg.freed });
            const mapCopyMs = performance.now() - mapCopy0;
            const totalMs = performance.now() - t0;
            console.log(
              `[ws:delta] f=${msg.frame} seq=${msg.seq}/${msg.total} ` +
              `cells=${msg.cells.length} freed=${msg.freed.length} ` +
              `map_size=${full.current.size} ` +
              `map_copy=${mapCopyMs.toFixed(1)}ms handle=${totalMs.toFixed(1)}ms`
            );
            deltaAcc.current = [];
            if (full.current.size > 0) setBuffering(false);
          }
          return;
        }

        if (msg.type === "snapshot") {
          const t0 = performance.now();
          if (msg.seq === 0) {
            snapBuf.current = new Map();
            snapFrame.current = msg.frame;
          }
          for (const c of msg.cells) {
            snapBuf.current.set(keyOf(c[0], c[1], nThetaRef.current), c);
          }
          if (msg.seq === msg.total - 1) {
            const t1 = performance.now();
            full.current = new Map(snapBuf.current);
            setCells(new Map(full.current));
            setLiveCount(full.current.size);
            setLastFrame(msg.frame);
            deltaAcc.current = [];
            setPatch({ kind: "snap", frame: msg.frame, upserts: [...snapBuf.current.values()] });
            const t2 = performance.now();
            console.debug(
              `[ws:snap] f=${msg.frame} seq=${msg.seq}/${msg.total} ` +
              `upserts=${snapBuf.current.size} ` +
              `map_copy=${(t2 - t1).toFixed(1)}ms total=${(t2 - t0).toFixed(1)}ms`
            );
            if (full.current.size > 0) setBuffering(false);
          }
        }
      };

ws.onmessage = (ev) => {
          if (cancelled) return;
          if (typeof ev.data === "string") {
            try {
              handleMsg(JSON.parse(ev.data) as MapMsg);
            } catch {
              /* ignore malformed */
            }
          } else {
            const t0 = performance.now();
            const buf = decompressFrame(ev.data as ArrayBuffer);
            const parseMs = performance.now() - t0; // decompress + still parse below
            const msg = parseBinary(buf);
          const t1 = performance.now();
          if (msg) handleMsg(msg);
          const handleMs = performance.now() - t1;
          // perf logging: aggregate every 30 frames
          const p = perfRef.current;
          p.n += 1;
          p.totalParseMs += parseMs;
          p.totalHandleMs += handleMs;
          p.totalBytes += (ev.data as ArrayBuffer).byteLength;
          if (p.n >= 30) {
            console.debug(
              `[ws:perf] frames=${p.n} ` +
              `avg_parse=${(p.totalParseMs / p.n).toFixed(2)}ms ` +
              `avg_handle=${(p.totalHandleMs / p.n).toFixed(2)}ms ` +
              `avg_bytes=${(p.totalBytes / p.n / 1024).toFixed(1)}KB ` +
              `total_bytes=${(p.totalBytes / 1024).toFixed(0)}KB`
            );
            perfRef.current = { n: 0, totalParseMs: 0, totalHandleMs: 0, totalBytes: 0 };
          }
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setConn("closed");
        const backoff = Math.min(500 * 2 ** attempts, 5000);
        attempts += 1;
        console.warn(`[ws] closed, reconnect in ${backoff}ms (attempt ${attempts})`);
        retry = setTimeout(connect, backoff);
      };

      ws.onerror = () => {
        // onerror is followed by onclose; just ensure close fires
        try { ws.close(); } catch {}
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (retry) clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((msg: object) => sendRef.current(msg), []);
  const setCloudOnLocal = useCallback(
    (on: boolean) => {
      cloudOnRef.current = on;
      setCloudOn(on);
      if (!on) setCloud(null);
      sendRef.current({ type: "control", action: "cloud", value: on });
    },
    []
  );

  const seekTo = useCallback((idx: number) => {
    setSeeking(true);
    sendRef.current({ type: "control", action: "seek", value: idx });
  }, []);

  const switchSequence = useCallback((newSeqId: string) => {
    setSeeking(true);
    sendRef.current({ type: "control", action: "switch_sequence", value: newSeqId });
  }, []);

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
    send,
    seqId,
    switchSequence,
    status,
    statusMsg,
    buffering,
  };
}