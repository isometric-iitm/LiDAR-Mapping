/**
 * Asset fetch + IndexedDB cache with progress reporting.
 *
 * Assets (model .onnx, sequence chunks) are fetched from the same origin
 * (dashboard/public/demo/) with a ReadableStream pass so the loader UI can
 * show accurate byte progress, then cached in IndexedDB via idb-keyval keyed
 * "name:sha256" so repeat visits (and offline reloads) are instant.
 */
import { get, set } from "idb-keyval";

export interface FetchProgress {
  /** 0..1 */
  fraction: number;
  /** bytes downloaded so far */
  loaded: number;
  /** total bytes (from Content-Length; may be 0 when unknown) */
  total: number;
}

export type ProgressCb = (p: FetchProgress) => void;

/** Check whether a cached copy exists (no download). */
export async function hasCached(key: string): Promise<boolean> {
  try {
    const v = await get(key);
    return v instanceof ArrayBuffer || v instanceof Uint8Array;
  } catch {
    return false;
  }
}

/**
 * Fetch `url` with progress; returns the (possibly cached) bytes.
 * `key` is the cache key ("name:sha256"); `totalHint` is used when
 * Content-Length is missing (e.g. cached streams).
 */
export async function fetchAsset(
  url: string,
  key: string,
  totalHint: number,
  onProgress?: ProgressCb,
): Promise<ArrayBuffer> {
  // 1) cache hit
  try {
    const v = await get(key);
    if (v instanceof ArrayBuffer && v.byteLength > 0) {
      onProgress?.({ fraction: 1, loaded: v.byteLength, total: v.byteLength });
      return v;
    }
  } catch {
    /* cache unavailable -> straight to network */
  }

  // 2) network fetch with stream progress
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const total = Number(res.headers.get("Content-Length")) || totalHint || 0;
  let buf: ArrayBuffer;
  if (res.body && total > 0) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastPost = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      // throttle progress posts to ~10 Hz
      const now = Date.now();
      if (onProgress && now - lastPost > 100) {
        lastPost = now;
        onProgress({ fraction: Math.min(0.999, loaded / total), loaded, total });
      }
    }
    const out = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    onProgress?.({ fraction: 1, loaded, total: Math.max(total, loaded) });
    buf = out.buffer;
  } else {
    buf = await res.arrayBuffer();
    onProgress?.({ fraction: 1, loaded: buf.byteLength, total: buf.byteLength || total });
  }

  // 3) store (best-effort; a failed quota write must not break playback)
  try {
    await set(key, buf);
  } catch {
    /* e.g. storage full in private mode; the session still works */
  }
  return buf;
}
