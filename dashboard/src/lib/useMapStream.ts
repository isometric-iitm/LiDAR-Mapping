"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AckMsg, Cell, DeltaMsg, GridMeta, SnapshotMsg, Stats } from "./types";
import { keyOf } from "./gridGeometry";
import { createFrameDecoder, type FrameDecoder } from "./frameDecoder";

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

/** Convert accumulated flat freed keys back to [i,j] pairs for the patch. */
function pairFrom(flatKeys: number[], nTheta: number): [number, number][] {
  const out: [number, number][] = new Array(flatKeys.length);
  for (let k = 0; k < flatKeys.length; k++) {
    const key = flatKeys[k];
    out[k] = [Math.floor(key / nTheta), key % nTheta];
  }
  return out;
}

/** ServerMsg plus the fields the binary channel carries that the legacy JSON
 *  shapes didn't (chunk bookkeeping on cloud frames, per-frame ego yaw). */
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
  const deltaFreed = useRef<number[]>([]);
  const deltaFrame = useRef(-1);
  // Binary frame decode (pako inflate + 28-byte row parse) runs in a worker so
  // it never contends with React/R3F for the main thread.
  const decoderRef = useRef<FrameDecoder | null>(null);
  if (decoderRef.current === null) decoderRef.current = createFrameDecoder();

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
          deltaFreed.current.length = 0;
          deltaFrame.current = -1;
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
            deltaFreed.current.length = 0;
            deltaFrame.current = -1;
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
            deltaFreed.current.length = 0;
            deltaFrame.current = -1;
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
          // (seq 0..total-1); 'freed' rides on the final chunk (but treat frees
          // as frame-scoped regardless of arrival chunk). We accumulate the
          // whole frame then apply it ONCE at the last chunk so that:
          //   1) a re-added cell is not wrongly deleted by its own frame's free
          //      (free-before-upsert): freed indices are collected, then all
          //      upserts win on the final apply.
          //   2) a fresh frame always discards any half-accumulated prior frame
          //      (reset on frame change, not just seq==0) so upserts never leak
          //      across a dropped frame boundary.
          const t0 = performance.now();
          if (msg.seq === 0 || msg.frame !== deltaFrame.current) {
            deltaAcc.current = [];
            deltaFreed.current.length = 0;
          }
          deltaFrame.current = msg.frame;
          for (const c of msg.cells) {
            deltaAcc.current.push(c);
          }
          for (const [i, j] of msg.freed) {
            deltaFreed.current.push(keyOf(i, j, nThetaRef.current));
          }
          if (msg.seq === msg.total - 1) {
            const mapCopy0 = performance.now();
            // Free first, then upsert so a cell both freed and re-added in the
            // same frame ends up present (the upsert wins).
            for (const k of deltaFreed.current) {
              full.current.delete(k);
            }
            for (const c of deltaAcc.current) {
              full.current.set(keyOf(c[0], c[1], nThetaRef.current), c);
            }
            // Pass the live map reference (not a full copy): `cells` is only
            // read on snapshot frames and for the empty/hint check, so an
            // O(n) copy each delta frame is wasted work. Snapshots resync it.
            setCells(full.current);
            setLiveCount(full.current.size);
            setLastFrame(msg.frame);
            setPatch({ kind: "delta", frame: msg.frame, upserts: deltaAcc.current, frees: pairFrom(deltaFreed.current, nThetaRef.current) });
            const mapCopyMs = performance.now() - mapCopy0;
            const totalMs = performance.now() - t0;
            console.log(
              `[ws:delta] f=${msg.frame} seq=${msg.seq}/${msg.total} ` +
              `cells=${msg.cells.length} freed=${msg.freed.length} ` +
              `map_size=${full.current.size} ` +
              `map_copy=${mapCopyMs.toFixed(1)}ms handle=${totalMs.toFixed(1)}ms`
            );
            deltaAcc.current = [];
            deltaFreed.current.length = 0;
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
            deltaFreed.current.length = 0;
            deltaFrame.current = msg.frame;
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
            return;
          }
          const dec = decoderRef.current;
          if (!dec) return;
          const t0 = performance.now();
          dec(ev.data as ArrayBuffer)
            .then((msg) => {
              if (cancelled || !msg) return;
              const t1 = performance.now();
              handleMsg(msg);
              const handleMs = performance.now() - t1;
              const parseMs = performance.now() - t0;
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
            })
            .catch(() => {});
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