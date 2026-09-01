/// <reference lib="webworker" />
/**
 * Web Worker: offloads the per-message binary decode + pako inflate that used
 * to run on the main thread (the hot path when the server streams hundreds of
 * KB/s of deflate-compressed binary frames at ~30-60fps).
 *
 * Protocol:
 *   main -> worker : { id: number, buf: ArrayBuffer }   (transferable)
 *   worker -> main : { id, ok: true, msg } | { id, ok: false }
 *
 * `msg` is the same structural shape parseBinary() produced before (grid rows
 * as plain JS arrays etc.) — the main-thread hook just hands these to the same
 * reducer logic. Working in a worker means pako's inflate + the row-array
 * building never contends with React/R3F for the main thread.
 */
import { inflateRaw } from "pako";
import type { AckMsg, Cell, DeltaMsg, GridMeta, SnapshotMsg, Stats } from "./types";

const MAGIC = 0x50433244;
const K_SNAPSHOT = 1;
const K_DELTA = 2;
const K_CLOUD = 3;

export type ParsedMsg =
  | GridMeta
  | Stats
  | AckMsg
  | { type: "status"; state: string; msg?: string; seq_id?: string }
  | (Omit<SnapshotMsg, "cells"> & { cells: Array<Cell & { __row?: never }>; yaw?: number })
  | (DeltaMsg & { yaw?: number })
  | ({
      type: "cloud";
      frame: number;
      n: number;
      xyz: number[];
      cls: number[];
      seq: number;
      total: number;
      epoch?: number;
      yaw?: number;
    });

function decompressFrame(u8: Uint8Array): Uint8Array {
  if (u8.length <= 5 || u8[0] !== 0x5a) return u8; // not compressed
  try {
    return inflateRaw(u8.subarray(5));
  } catch {
    return u8;
  }
}

function parseBinary(buf: ArrayBuffer): ParsedMsg | null {
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
    const xyz = new Float32Array(buf, 44, n * 3);
    const cls = new Uint8Array(buf, 44 + n * 12, n);
    return {
      type: "cloud",
      ...base,
      n,
      xyz: Array.from(xyz),
      cls: Array.from(cls),
    };
  }
  if (code !== K_SNAPSHOT && code !== K_DELTA) return null;

  // Row record is 28 bytes: [i,j,z_mean,z_max,occ,trav] f32 (24) + cls u8 + 3 pad
  const f = new Float32Array(buf, 44, n * 6);
  const cls = new Uint8Array(buf, 44 + n * 24, n);
  const cells = new Array<Cell>(n);
  for (let k = 0; k < n; k++) {
    const o = k * 6;
    cells[k] = [f[o], f[o + 1], f[o + 2], f[o + 3], cls[k], f[o + 4], f[o + 5]];
  }
  if (code === K_SNAPSHOT) {
    return { type: "snapshot", ...base, cells };
  }
  const freedNr = new Float32Array(buf, 44 + n * 28, nFreed * 2);
  const freed = new Array<[number, number]>(nFreed);
  for (let k = 0; k < nFreed; k++) freed[k] = [freedNr[2 * k], freedNr[2 * k + 1]];
  return { type: "delta", ...base, cells, freed };
}

self.onmessage = (ev: MessageEvent) => {
  const { id, buf } = ev.data as { id: number; buf: ArrayBuffer };
  try {
    // The server sends raw-deflate-compressed binary (b'Z' + u32 uncompressed
    // length + deflate payload) matching pako's inflateRaw. Decompress to a
    // fresh ArrayBuffer, then parse. Fall back to the raw buffer when it is
    // not actually compressed (plain path used by some tests/tools).
    const u8 = new Uint8Array(buf);
    const inflated = decompressFrame(u8);
    const msg = parseBinary(inflated.buffer as ArrayBuffer);
    (self as unknown as Worker).postMessage({ id, ok: true, msg });
  } catch {
    (self as unknown as Worker).postMessage({ id, ok: false });
  }
};
