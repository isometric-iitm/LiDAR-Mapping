"use client";
/**
 * One-time, closable info banner (top-left, below the legend) explaining that
 * this deployed demo runs the FULL pipeline - neural inference included - live
 * in the browser via WebGPU; the deployed build simply trades throughput for
 * zero-server hosting (a GPU-server deployment of the same code runs faster).
 */
import { useState } from "react";

const DISMISS_KEY = "pc2d-demo-banner-dismissed";

/** Lazy initializer: reads storage during the FIRST render (no effect, no
 *  cascading setState). Falls back to open when storage is unavailable. */
function initialOpen(): boolean {
  try {
    return !window.sessionStorage.getItem(DISMISS_KEY);
  } catch {
    return true; // private mode / storage blocked: just show it
  }
}

export default function DemoBanner() {
  const [open, setOpen] = useState(initialOpen);

  const close = () => {
    setOpen(false);
    try {
      window.sessionStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* ignore */
    }
  };

  if (!open) return null;

  return (
    <div className="frost pointer-events-auto absolute left-3 top-14 z-20 max-w-xs px-3 py-2.5">
      <div className="flex items-start gap-2">
        <div className="min-w-0 text-[11px] leading-snug text-zinc-300">
          <span className="font-semibold text-cyan-300">Showcase build</span> lighter
          quantized model (int8). Per SA Admin this demo runs the edge lite
          weights on free tier compute. Full weights need dedicated GPU/TPU for
          higher fidelity.
        </div>
        <button
          onClick={close}
          aria-label="Dismiss"
          className="shrink-0 rounded-[3px] p-0.5 text-zinc-500 transition-colors hover:bg-white/10 hover:text-zinc-200"
        >
          {/* plain close cross (two strokes), not a brand mark */}
          <svg width={13} height={13} viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
            <path d="M2.5 2.5 L10.5 10.5" />
            <path d="M10.5 2.5 L2.5 10.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}
