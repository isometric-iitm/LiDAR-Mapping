"use client";

import type { Stats } from "@/lib/types";

function fmt(n: number | undefined, digits = 1) {
  return typeof n === "number" && isFinite(n) ? n.toFixed(digits) : "–";
}

function Row({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-1">
      <span className="text-[11px] uppercase tracking-wider text-zinc-500">{k}</span>
      <span className="font-mono text-sm tabular-nums text-zinc-100">
        {v}
        {sub ? <span className="ml-1 text-[10px] text-zinc-500">{sub}</span> : null}
      </span>
    </div>
  );
}

function Bar({ label, widthPct, color, value, unit }: { label: string; widthPct: number; color: string; value: string; unit: string }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[10px] text-zinc-500">
        <span>{label}</span>
        <span className="font-mono tabular-nums text-zinc-300">
          {value} <span className="text-zinc-500">{unit}</span>
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded bg-zinc-800">
        <div
          className="h-full rounded transition-all duration-300"
          style={{ width: `${Math.max(1.5, widthPct)}%`, background: color }}
        />
      </div>
    </div>
  );
}

export type ViewMode = "grid" | "seg" | "raw" | "compare";

const VIEWS: { id: ViewMode; label: string; hint: string }[] = [
  { id: "grid", label: "Grid 2.5D", hint: "predicted blocks" },
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
  pointSize,
  onPointSize,
  hover,
}: {
  stats: Stats | null;
  cellCount: number;
  lastFrame: number;
  viewMode: ViewMode;
  onViewMode: (v: ViewMode) => void;
  pointSize: number;
  onPointSize: (s: number) => void;
  hover: { cls: string; zMax: number; zMean: number; occ: number; dyn: number; r: number; deg: number } | null;
}) {

  // log-scale widths so the 3 MB grid bar stays visible against the 720 MB uniform one
  const kb = stats?.grid_mem_kb ?? 0;
  const mb = stats?.uniform_equiv_mb ?? 0;
  const logGrid = kb > 0 ? Math.log10(kb) : 0;
  const logUniform = mb > 0 ? Math.log10(mb * 1024) : 0;
  const span = logUniform - logGrid || 1;
  const gridW = Math.min(100, 12 + (1 - logGrid / (logGrid + 1)) * 20);
  const uniformW = Math.min(100, 30 + (1 - span / 6) * 60);

  const showPointSize = viewMode === "seg" || viewMode === "raw" || viewMode === "compare";

  return (
    <aside className="flex w-[330px] shrink-0 flex-col gap-3 overflow-y-auto border-l border-zinc-800 bg-zinc-950/90 p-4">
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-lg font-bold tabular-nums text-zinc-100">{fmt(stats?.fps, 1)}</div>
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">FPS stream</div>
          </div>
          <div>
            <div className="text-lg font-bold tabular-nums text-zinc-100">
              {cellCount.toLocaleString()}
            </div>
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Cells drawn</div>
          </div>
        </div>
        <div className="mt-3 border-t border-zinc-800 pt-2">
          <Row k="Frame" v={`${lastFrame}`} />
          <Row k="Latency p50" v={`${fmt(stats?.latency_ms_p50)}`} sub="ms" />
          <Row k="Latency p95" v={`${fmt(stats?.latency_ms_p95)}`} sub="ms" />
          <div className="mt-2 border-t border-zinc-800 pt-2">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              per-stage ms
            </div>
            <Row k="Segment" v={`${fmt(stats?.seg_ms)}`} sub="ms" />
            <Row k="·Proj" v={`${fmt(stats?.project_ms)}`} sub="ms" />
            <Row k="·Fwd" v={`${fmt(stats?.forward_ms)}`} sub="ms" />
            <Row k="Grid" v={`${fmt(stats?.grid_ms)}`} sub="ms" />
            <Row k="Pack" v={`${fmt(stats?.pack_ms)}`} sub="ms" />
            <Row k="Cloud" v={`${fmt(stats?.cloud_ms)}`} sub="ms" />
          </div>
          <Row k="Compression" v={`${fmt(stats?.compression_ratio, 1)}`} sub="x" />
        </div>
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Memory · live
        </h3>
        <div className="space-y-3">
          <Bar
            label="Grid (log-polar)"
            widthPct={gridW}
            color="#38bdf8"
            value={fmt(kb, 1)}
            unit="KB"
          />
          <Bar
            label="Uniform 16M grid"
            widthPct={uniformW}
            color="#a3a3a3"
            value={fmt(mb, 1)}
            unit="MB"
          />
          <p className="text-[10px] leading-4 text-zinc-600">
            Same scene stored two ways. The bar lengths are log-scaled; the grid
            really is ~2,400× smaller per header above.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          View
        </h3>
        <div className="grid grid-cols-2 gap-1.5">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              onClick={() => onViewMode(v.id)}
              className={`rounded-md px-2 py-2 text-left transition-colors ${
                viewMode === v.id
                  ? "bg-sky-600/20 ring-1 ring-sky-500/50"
                  : "bg-zinc-800 hover:bg-zinc-700"
              }`}
            >
              <div className="text-xs font-medium text-zinc-100">{v.label}</div>
              <div className="text-[9px] text-zinc-500">{v.hint}</div>
            </button>
          ))}
        </div>
        {showPointSize && (
          <div className="mt-3 border-t border-zinc-800 pt-3">
            <div className="mb-1 flex justify-between text-[10px] text-zinc-500">
              <span>Point size</span>
              <span className="font-mono tabular-nums">{pointSize.toFixed(1)} px</span>
            </div>
            <input
              type="range"
              min={1}
              max={8}
              step={0.2}
              value={pointSize}
              onChange={(e) => onPointSize(parseFloat(e.target.value))}
              className="w-full accent-sky-500"
            />
          </div>
        )}
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Cell under cursor
        </h3>
        {hover ? (
          <div className="space-y-1">
            <Row k="Class" v={hover.cls} />
            <Row k="Height (z max)" v={`${hover.zMax.toFixed(2)}`} sub="m" />
            <Row k="Mean z" v={`${hover.zMean.toFixed(2)}`} sub="m" />
            <Row k="Occupancy" v={`${(hover.occ * 100).toFixed(0)}`} sub="%" />
            <Row k="Dynamic" v={`${(hover.dyn * 100).toFixed(0)}`} sub="%" />
            <Row k="Range" v={`${hover.r.toFixed(1)}`} sub="m" />
            <Row k="Bearing" v={`${hover.deg.toFixed(1)}`} sub="deg" />
          </div>
        ) : (
          <p className="text-[11px] text-zinc-600">Hover a block in the map to inspect its height.</p>
        )}
      </div>
    </aside>
  );
}