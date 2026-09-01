"use client";

import type { ParsedMsg } from "./frameWorker";

/**
 * Thin promise wrapper around the dedicated frame-parse Web Worker. Each binary
 * message is assigned an id; the resolve happens when the worker echoes it back,
 * so ordering is preserved even though replies arrive asynchronously. Created
 * lazily on first use so builds/tests that never render a stream don't pay for a
 * worker. Returns null if Worker/URL is unavailable (defensive fallback).
 */
export type FrameDecoder = (buf: ArrayBuffer) => Promise<ParsedMsg | null>;

export function createFrameDecoder(): FrameDecoder | null {
  if (typeof Worker === "undefined") return null;
  let worker: Worker | null = null;
  let nextId = 0;
  const pending = new Map<number, { resolve: (m: ParsedMsg | null) => void; reject: (e: unknown) => void }>();

  const ensure = (): Worker | null => {
    if (worker) return worker;
    try {
      worker = new Worker(new URL("./frameWorker", import.meta.url), { type: "module" });
      worker.onmessage = (ev: MessageEvent) => {
        const { id, ok, msg } = ev.data as { id: number; ok: boolean; msg?: ParsedMsg };
        const p = pending.get(id);
        if (!p) return;
        pending.delete(id);
        if (ok) p.resolve(msg ?? null);
        else p.reject(new Error("frame worker parse failed"));
      };
      worker.onerror = () => {
        // Worker hard-failed: reject everything outstanding and mark it dead so
        // the next call spins up a fresh one.
        const all = [...pending.values()];
        pending.clear();
        all.forEach((p) => p.reject(new Error("frame worker crashed")));
        worker?.terminate();
        worker = null;
      };
    } catch {
      worker = null;
    }
    return worker;
  };

  return (buf: ArrayBuffer) => {
    const w = ensure();
    if (!w) return Promise.reject(new Error("no worker"));
    const id = nextId++;
    return new Promise<ParsedMsg | null>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      w.postMessage({ id, buf }, [buf]);
    });
  };
}
