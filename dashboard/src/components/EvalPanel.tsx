"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Refresh, StatsReport } from "iconoir-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = process.env.NEXT_PUBLIC_PC2D_API ?? "http://localhost:8000";

type ClassIou = Record<string, number>;
type Band = Record<string, { miou: number; class_ious: ClassIou }>;
type Latency = { mean: number; std: number; min: number; max: number };

type EvalResult = {
  timestamp: string;
  device: string;
  precision: string;
  config: string;
  pixel: {
    n_samples: number;
    miou_5class: number;
    miou_4class: number;
    class_ious_5: ClassIou;
    class_ious_4: ClassIou;
    per_distance_band_4class: Band;
    latency_ms: Latency;
  };
  point: {
    n_scans: number;
    miou_5class: number;
    miou_4class: number;
    class_ious_5: ClassIou;
    class_ious_4: ClassIou;
    per_distance_band_4class: Band;
    latency_ms: Latency;
  };
  memory: { peak_rss_mb: number; gpu_peak_mb: number };
};

const CLASS_META: Record<string, { label: string; color: string }> = {
  drivable: { label: "Drivable", color: "#22c55e" },
  terrain_nondrivable: { label: "Terrain", color: "#a16207" },
  static_obstacle: { label: "Static", color: "#ef4444" },
  dynamic_vehicle: { label: "Dyn vehicle", color: "#f97316" },
  dynamic_pedestrian: { label: "Dyn pedestrian", color: "#3b82f6" },
  dynamic_object: { label: "Dynamic", color: "#3b82f6" },
};

function fmtPct(v: number | undefined): string {
  return typeof v === "number" && isFinite(v) ? `${(v * 100).toFixed(1)}%` : "-";
}

function classRows(dict: ClassIou): { name: string; value: number }[] {
  return Object.entries(dict)
    .map(([key, value]) => ({ name: CLASS_META[key]?.label ?? key, value: value * 100 }))
    .sort((a, b) => b.value - a.value);
}

function bandRows(band: Band): { name: string; miou: number }[] {
  return Object.entries(band).map(([name, v]) => ({ name, miou: v.miou * 100 }));
}

function Card({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="frost p-4">
      <div className="hud-cap">{label}</div>
      <div className="mt-1 font-mono text-base text-zinc-100">{value}</div>
      {sub ? <div className="mt-0.5 text-xs text-zinc-500">{sub}</div> : null}
    </div>
  );
}

function Loading({ label }: { label: string }) {
  return (
    <div className="mx-auto flex max-w-6xl items-center gap-2 px-6 py-8 text-sm text-zinc-400">
      <Refresh className="h-4 w-4 animate-spin" strokeWidth={2} />
      <span>{label}</span>
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number; color?: string; unit?: string }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[4px] border border-zinc-700 bg-zinc-900/95 px-3 py-2 text-xs">
      {label ? <div className="mb-1 font-mono text-zinc-400">{label}</div> : null}
      {payload.map((p, k) => (
        <div key={k} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-zinc-300">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="font-mono text-zinc-100">{p.value?.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function classColor(name: string): string {
  return Object.values(CLASS_META).find((m) => m.label === name)?.color ?? "#22d3ee";
}

const CHART_MARGIN = { top: 8, right: 24, bottom: 4, left: 0 };

function ClassIoUChart({ title, data }: { title: string; data: { name: string; value: number }[] }) {
  return (
    <div className="frost p-4">
      <h3 className="hud-sec mb-3">{title}</h3>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={CHART_MARGIN}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" stroke="#71717a" fontSize={10} tickLine={false} />
            <YAxis domain={[0, 100]} unit="%" stroke="#71717a" fontSize={10} width={46} tickLine={false} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            <Bar dataKey="value" name="IoU" radius={[2, 2, 0, 0]} isAnimationActive={false}>
              {data.map((d, i) => (
                <Cell key={i} fill={classColor(d.name)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function BandChart({ title, data, color }: { title: string; data: { name: string; miou: number }[]; color: string }) {
  return (
    <div className="frost p-4">
      <h3 className="hud-sec mb-3">{title}</h3>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={CHART_MARGIN}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" stroke="#71717a" fontSize={10} tickLine={false} />
            <YAxis domain={[0, 100]} unit="%" stroke="#71717a" fontSize={10} width={46} tickLine={false} />
            <Tooltip
              content={<ChartTooltip />}
              cursor={{ stroke: "#71717a", strokeDasharray: "3 3" }}
            />
            <Line
              type="monotone"
              dataKey="miou"
              name="mIoU"
              stroke={color}
              strokeWidth={2}
              dot={{ r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function EvalPanel() {
  const [data, setData] = useState<EvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/metrics/eval`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => {
        if (j && j.error) throw new Error(j.error);
        setData(j);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const pixelBars4 = useMemo(() => (data ? classRows(data.pixel.class_ious_4) : []), [data]);
  const pixelBars5 = useMemo(() => (data ? classRows(data.pixel.class_ious_5) : []), [data]);
  const pointBars4 = useMemo(() => (data ? classRows(data.point.class_ious_4) : []), [data]);
  const pointBars5 = useMemo(() => (data ? classRows(data.point.class_ious_5) : []), [data]);
  const pixelBands = useMemo(() => (data ? bandRows(data.pixel.per_distance_band_4class) : []), [data]);
  const pointBands = useMemo(() => (data ? bandRows(data.point.per_distance_band_4class) : []), [data]);

  if (loading) return <Loading label="Loading evaluation" />;
  if (error)
    return (
      <div className="mx-auto max-w-6xl px-6 py-8 text-sm text-rose-400">
        Failed to load eval results: {error}
      </div>
    );
  if (!data?.pixel)
    return (
      <div className="mx-auto max-w-6xl px-6 py-8 text-sm text-zinc-400">
        No eval results found (run eval/run_evaluation.py first).
      </div>
    );

  const ts = data.timestamp ? new Date(data.timestamp).toLocaleString() : "-";

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">Evaluation</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Run {ts} / {data.device} / {data.precision}
          </p>
        </div>
        <Link
          href="/training"
          className="flex items-center gap-1.5 rounded-[4px] bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-700"
        >
          <StatsReport width={14} height={14} strokeWidth={2} />
          Training curves
        </Link>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Card label="Pixel mIoU (4-class)" value={fmtPct(data.pixel.miou_4class)} sub={`${data.pixel.n_samples} samples`} />
        <Card label="Pixel mIoU (5-class)" value={fmtPct(data.pixel.miou_5class)} />
        <Card label="Point mIoU (4-class)" value={fmtPct(data.point.miou_4class)} sub={`${data.point.n_scans} scans`} />
        <Card label="Point mIoU (5-class)" value={fmtPct(data.point.miou_5class)} />
      </div>

      <div className="space-y-6">
        <h2 className="hud-sec">Pixel level</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <ClassIoUChart title="Per-class IoU (4-class)" data={pixelBars4} />
          <ClassIoUChart title="Per-class IoU (5-class)" data={pixelBars5} />
        </div>
        <BandChart title="Pixel mIoU by distance band (4-class)" data={pixelBands} color="#22d3ee" />
      </div>

      <div className="space-y-6">
        <h2 className="hud-sec">Point level</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <ClassIoUChart title="Per-class IoU (4-class)" data={pointBars4} />
          <ClassIoUChart title="Per-class IoU (5-class)" data={pointBars5} />
        </div>
        <BandChart title="Point mIoU by distance band (4-class)" data={pointBands} color="#a78bfa" />
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <Card
          label="Pixel latency"
          value={`${data.pixel.latency_ms.mean?.toFixed(1) ?? "-"} ms`}
          sub={`std ${data.pixel.latency_ms.std?.toFixed(1) ?? "-"} ms`}
        />
        <Card
          label="Point latency"
          value={`${data.point.latency_ms.mean?.toFixed(1) ?? "-"} ms`}
          sub={`std ${data.point.latency_ms.std?.toFixed(1) ?? "-"} ms`}
        />
        <Card
          label="Memory"
          value={`${data.memory.peak_rss_mb?.toFixed(0) ?? "-"} MB`}
          sub={data.memory.gpu_peak_mb ? `GPU ${data.memory.gpu_peak_mb.toFixed(0)} MB` : "-"}
        />
      </div>
    </div>
  );
}
