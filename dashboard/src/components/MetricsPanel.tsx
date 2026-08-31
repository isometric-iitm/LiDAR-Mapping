"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import type { Stats } from "@/lib/types";

function fmt(n: number | undefined, digits = 1) {
  return typeof n === "number" && isFinite(n) ? n.toFixed(digits) : "-";
}

/** Format a byte count into the largest prefix that reads cleanly (B/KB/MB/GB). */
function fmtBytes(bytes: number): string {
  if (!isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  const digits = v < 10 ? 2 : v < 100 ? 1 : 0;
  return `${v.toFixed(digits)} ${units[i]}`;
}

/** Format any byte-like value (which may already be a prefixed magnitude) into a clean "x MB" style string. */
function fmtSizeKB(kb: number): string {
  return fmtBytes(kb * 1024);
}

function Row({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-1">
      <span className="hud-label">{k}</span>
      <span className="font-mono text-sm tabular-nums text-zinc-100">
        {v}
        {sub ? <span className="hud-val ml-1">{sub}</span> : null}
      </span>
    </div>
  );
}

function Bar({ label, widthPct, color, value, unit }: { label: string; widthPct: number; color: string; value: string; unit: string }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="hud-label">{label}</span>
        <span className="hud-val">
          {value}{unit ? ` ${unit}` : ""}
        </span>
      </div>
      <div className="hud-bar">
        <div
          className="hud-bar__fill"
          style={{ width: `${Math.max(1.5, widthPct)}%`, background: color }}
        />
      </div>
    </div>
  );
}

function AnimatedCounter({ target }: { target: number }) {
  const [value, setValue] = useState(0);
  const prevTarget = useRef(0);
  const raf = useRef(0);
  useEffect(() => {
    if (target <= 0) return;
    const prev = prevTarget.current;
    const jump = prev > 0 && Math.abs(target - prev) / prev < 0.05;
    prevTarget.current = target;
    if (jump) {
      setValue(Math.round(target));
      return;
    }
    const duration = 1500;
    const start = performance.now();
    const fromVal = value;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const ease = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(fromVal + (target - fromVal) * ease));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);
  return <>{value.toLocaleString()}</>;
}

export type ViewMode = "grid" | "seg" | "raw" | "compare" | "trav";

const VIEWS: { id: ViewMode; label: string; hint: string }[] = [
  { id: "grid", label: "Grid 2.5D", hint: "predicted blocks" },
  { id: "trav", label: "Traverse", hint: "drivable / caution / blocked" },
  { id: "seg", label: "Seg cloud", hint: "predicted points" },
  { id: "raw", label: "Raw cloud", hint: "sensor points (height)" },
  { id: "compare", label: "Compare", hint: "raw points + grid" },
];

export default function MetricsPanel({
  stats,
  cellCount,
  lastFrame,
  viewMode,
  onViewMode,
  // pointSize/onPointSize kept for page compat but slider removed per spec
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  pointSize: _pointSize,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onPointSize: _onPointSize,
  hover,
}: {
  stats: Stats | null;
  cellCount: number;
  lastFrame: number;
  viewMode: ViewMode;
  onViewMode: (v: ViewMode) => void;
  pointSize: number;
  onPointSize: (s: number) => void;
  hover: { cls: string; clsColor: string; zMax: number; zMean: number; occ: number; dyn: number; trav: number; r: number; deg: number; cellWidth: string } | null;
}) {

  // Log-scale memory bar widths so a ~KB grid stays proportionally visible
  // against a ~MB uniform grid without the layout skewing to one extreme.
  const kb = stats?.grid_mem_kb ?? 0; // rendered live grid size (KB)
  const mb = stats?.uniform_equiv_mb ?? 0; // uniform 5cm grid size (MB, decimal)
  const renderedCells = stats?.rendered_cells ?? stats?.n_cells ?? 0;
  const gridW = kb > 0
    ? Math.min(100, 18 + (Math.log10(kb) / (Math.log10(kb) + 1)) * 55)
    : 0;
  const uniformW = mb > 0 ? 100 : 0;

  // Human-readable sizes, auto-scaled to the cleanest prefix (B/KB/MB/GB).
  const gridLabel = fmtSizeKB(kb);
  const uniformLabel = fmtBytes(mb * 1e6);
  // Actual over-the-wire payload per frame: each rendered cell is a 32-byte row.
  const wireLabel = renderedCells > 0 ? fmtBytes(renderedCells * 32) : "-";

  const perStage = [
    { k: "Seg", ms: stats?.seg_ms },
    { k: "Tracker", ms: stats?.grid_ms },
    { k: "Pack", ms: stats?.pack_ms },
    { k: "Cloud", ms: stats?.cloud_ms },
  ];
  const maxStageMs = Math.max(
    1,
    ...perStage.map((s) => (typeof s.ms === "number" && isFinite(s.ms) ? s.ms : 0)),
  );

  return (
    <aside className="flex w-full max-w-xs shrink-0 flex-col border-l border-white/10 bg-black/40 backdrop-blur-xl">
      {/* -- readout header ------------------------------------------- */}
      <div className="border-b border-white/10 px-4 pb-3 pt-5">
        <div className="grid grid-cols-3 gap-2">
          <div>
            <div className="hud-num hud-num--accent">{fmt(stats?.fps, 0)}</div>
            <div className="hud-cap mt-1">FPS</div>
          </div>
          <div>
            <div className="hud-num">{cellCount.toLocaleString()}</div>
            <div className="hud-cap mt-1">Cells</div>
          </div>
          <div className="text-right">
            <div className="hud-num">#{lastFrame}</div>
            <div className="hud-cap mt-1">Frame</div>
          </div>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-4">
        <section>
          <h3 className="hud-sec mb-2 mt-1">Latency</h3>
          <div className="hud-rail grid grid-cols-2 gap-x-4 gap-y-1">
            <Row k="p50" v={`${fmt(stats?.latency_ms_p50)}`} sub="ms" />
            <Row k="p95" v={`${fmt(stats?.latency_ms_p95)}`} sub="ms" />
          </div>
        </section>

        <section className="mt-6">
          <h3 className="hud-sec mb-2">Pipeline / ms</h3>
          <div className="hud-rail space-y-2">
            {perStage.map((s) => (
              <div key={s.k} className="flex items-center gap-2">
                <span className="hud-label w-14">{s.k}</span>
                <div className="hud-bar flex-1">
                  <div
                    className="hud-bar__fill bg-cyan-400/70"
                    style={{ width: `${((s.ms ?? 0) / maxStageMs) * 100}%` }}
                  />
                </div>
                <span className="hud-val w-10 text-right">{fmt(s.ms, 1)}</span>
              </div>
            ))}
          </div>
          <div className="hud-rail mt-2 flex items-center justify-between">
            <span className="hud-label">Dropped</span>
            <span className="hud-val">{stats?.frames_dropped ?? 0}</span>
          </div>
        </section>

        <section className="mt-6">
          <h3 className="hud-sec mb-2">Memory</h3>
          <div className="hud-rail space-y-3">
            <Bar label="Live log-polar map" widthPct={gridW} color="#22d3ee" value={gridLabel} unit="" />
            <Bar label="Uniform 5 cm grid" widthPct={uniformW} color="#a1a1aa" value={uniformLabel} unit="" />
            <div className="flex items-baseline justify-between pt-0.5">
              <span className="hud-label">Wire / frame</span>
              <span className="hud-val">{wireLabel}</span>
            </div>
            <div className="flex items-baseline justify-between border-t border-white/10 pt-2">
              <span className="text-sm text-zinc-400">Compression</span>
              <span className="hud-big">
                <AnimatedCounter target={stats?.compression_ratio ?? 0} />
                <span className="hud-label ml-1">× fewer cells</span>
              </span>
            </div>
          </div>
        </section>

        <section className="mt-6">
          <h3 className="hud-sec mb-2">View</h3>
          <div className="grid grid-cols-2 gap-1.5">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                onClick={() => onViewMode(v.id)}
                className={`rounded-[4px] px-2 py-1.5 text-left transition-colors ${
                  viewMode === v.id
                    ? "bg-cyan-500/15 ring-1 ring-cyan-400/40"
                    : "bg-white/5 hover:bg-white/10"
                }`}
              >
                <div className="text-xs font-medium text-zinc-100">{v.label}</div>
              </button>
            ))}
          </div>
        </section>

        <section className="mt-6">
          <h3 className="hud-sec mb-2">Pages</h3>
          <div className="flex flex-col">
            <Link
              href="/training"
              className="py-1 text-sm text-zinc-400 transition-colors hover:text-zinc-100"
            >
              Training
            </Link>
            <Link
              href="/eval"
              className="py-1 text-sm text-zinc-400 transition-colors hover:text-zinc-100"
            >
              Evaluation
            </Link>
          </div>
        </section>
      </div>

      {/* -- cell under cursor (always open, dashes when empty) ------ */}
      <div className="frost mx-3 mb-3 shrink-0 px-4 py-3">
        <h3 className="hud-sec mb-2">Cell under cursor</h3>
        <div className="grid grid-cols-3 gap-x-3 gap-y-2">
          <div className="col-span-3">
            <span
              className="mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle"
              style={{ background: hover ? hover.clsColor : "#3f3f46" }}
            />
            <span className="text-sm font-medium text-zinc-100">{hover ? hover.cls : "-"}</span>
            <span className="hud-val ml-2">
              {hover ? `${hover.r.toFixed(1)} m / ${hover.deg.toFixed(1)}°` : "-"}
            </span>
          </div>
          <CellStat k="H max" v={hover ? `${hover.zMax.toFixed(2)}m` : "-"} />
          <CellStat k="Mean z" v={hover ? `${hover.zMean.toFixed(2)}m` : "-"} />
          <CellStat k="Width" v={hover ? hover.cellWidth : "-"} />
          <CellStat k="Occ" v={hover ? `${(hover.occ * 100).toFixed(0)}%` : "-"} />
          <CellStat k="Dyn" v={hover ? `${(hover.dyn * 100).toFixed(0)}%` : "-"} />
          <CellStat k="Trav" v={hover ? `${(hover.trav * 100).toFixed(0)}%` : "-"} />
        </div>
      </div>
    </aside>
  );
}

function CellStat({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="hud-label">{k}</div>
      <div className="hud-val mt-0.5">{v}</div>
    </div>
  );
}