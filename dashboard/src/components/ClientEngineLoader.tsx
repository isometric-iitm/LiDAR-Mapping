"use client";
/**
 * Full-screen HUD overlay for the in-browser engine boot sequence:
 *   phase 1: model download (progress bar + MB/s)
 *   phase 2: sequence part 1 download
 *   phase 3: WebGPU kernel compile + warmup (spinner)
 * plus a branded WebGPU-required gate for unsupported browsers.
 * Unmounts itself once the engine is ready (parent swaps in the 3D canvas).
 */
import { Refresh, WarningTriangle } from "iconoir-react";
import type { EngineDownload } from "@/lib/engine/useClientEngine";

type Props = {
  initializing: boolean;
  statusMsg: string | null;
  download: EngineDownload | null;
  webgpuUnsupported: boolean;
  error: string | null;
};

export default function ClientEngineLoader({ initializing, statusMsg, download, webgpuUnsupported, error }: Props) {
  if (!initializing && !webgpuUnsupported && !error) return null;

  const d = download;
  const pct = d && d.total > 0 ? Math.floor((d.fraction ?? 0) * 100) : null;

  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/80 backdrop-blur-[2px]">
      {webgpuUnsupported ? (
        <div className="frost flex max-w-md flex-col items-center gap-3 px-8 py-7 text-center">
          <WarningTriangle className="h-8 w-8 text-amber-400" />
          <div className="text-base font-semibold text-zinc-100">WebGPU required</div>
          <div className="text-sm leading-relaxed text-zinc-400">
            This demo runs the neural network entirely in your browser and needs
            WebGPU. Please use <span className="text-zinc-200">Chrome or Edge 113+</span> (desktop)
            with hardware acceleration enabled.
          </div>
        </div>
      ) : (
        <div className="frost flex w-[min(420px,90vw)] flex-col gap-4 px-6 py-6">
          <div className="flex items-center gap-3">
            <Refresh className="h-5 w-5 shrink-0 animate-spin text-cyan-400" />
            <div className="min-w-0">
              <div className="hud-cap text-cyan-400/90">pc2d   in-browser engine</div>
              <div className="truncate text-sm text-zinc-100">{error ?? statusMsg ?? "Loading model..."}</div>
            </div>
            {pct !== null && <div className="hud-big ml-auto tabular-nums text-zinc-200">{pct}%</div>}
          </div>

          {d && (
            <div className="flex flex-col gap-1.5">
              <div className="hud-rail">
                <div className="hud-bar_fill" style={{ width: `${((d.fraction ?? 0) * 100).toFixed(1)}%` }} />
              </div>
              <div className="flex justify-between font-mono text-[11px] text-zinc-500">
                <span>
                  {d.phase === "model"
                    ? "neural weights"
                    : `sequence part ${String((d.chunk ?? 0) + 1).padStart(2, "0")}`}
                </span>
                <span>
                  {(d.loaded / 1e6).toFixed(1)} / {(d.total / 1e6).toFixed(1)} MB
                </span>
              </div>
            </div>
          )}

          {!d && !error && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" />
              {statusMsg ?? "Compiling WebGPU kernels..."}
            </div>
          )}

          {error && (
            <div className="rounded-[4px] bg-red-500/10 px-3 py-2 text-xs leading-relaxed text-red-300 ring-1 ring-red-400/30">
              {error}
            </div>
          )}

          <div className="text-[11px] text-zinc-600">
            Assets are cached in your browser - next visits start instantly and work offline.
          </div>
        </div>
      )}
    </div>
  );
}
