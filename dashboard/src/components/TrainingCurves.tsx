"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DashboardSpeed, Refresh } from "iconoir-react";

type HistoryEntry = {
  step: number;
  epoch: number;
  val_loss: number;
  miou_5class: number;
  miou_4class: number;
  class_ious_4: Record<string, number>;
  class_ious_5: Record<string, number>;
};

const API = process.env.NEXT_PUBLIC_PC2D_API ?? "http://localhost:8000";

const METRIC_SERIES = [
  { key: "miou_4class", label: "mIoU (4-class)", color: "#22c55e", axis: "iou" as const },
  { key: "miou_5class", label: "mIoU (5-class)", color: "#3b82f6", axis: "iou" as const },
  { key: "val_loss", label: "val loss", color: "#ef4444", axis: "loss" as const },
];

const PER_CLASS = [
  { key: "drivable", label: "Drivable", color: "#22c55e" },
  { key: "terrain_nondrivable", label: "Terrain", color: "#a16207" },
  { key: "static_obstacle", label: "Static", color: "#ef4444" },
  { key: "dynamic_object", label: "Dynamic", color: "#3b82f6" },
];

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number; color?: string }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[4px] border border-zinc-700 bg-zinc-900/95 px-3 py-2 text-xs">
      <div className="mb-1 font-mono text-zinc-400">step {label}</div>
      {payload.map((p, k) => (
        <div key={k} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-zinc-300">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="font-mono text-zinc-100">{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

function MetricChart({ title, rows }: { title: string; rows: Record<string, number | undefined>[] }) {
  return (
    <div className="frost p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">{title}</h3>
      <div className="h-[290px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 26, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="step"
              stroke="#71717a"
              fontSize={10}
              tickFormatter={(v: number) => v.toLocaleString()}
              tickLine={false}
            />
            <YAxis yAxisId="iou" domain={[0, 1]} stroke="#71717a" fontSize={10} width={46} tickLine={false} />
            <YAxis
              yAxisId="loss"
              orientation="right"
              domain={["auto", "auto"]}
              stroke="#71717a"
              fontSize={10}
              width={48}
              tickLine={false}
              tickFormatter={(v: number) => Number(v).toFixed(3)}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#71717a", strokeDasharray: "3 3" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
            {METRIC_SERIES.map((s) => (
              <Line
                key={s.label}
                yAxisId={s.axis}
                name={s.label}
                dataKey={s.label}
                stroke={s.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PerClassChart({ rows }: { rows: Record<string, number | undefined>[] }) {
  return (
    <div className="frost p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Per-class IoU (binned 4-class)
      </h3>
      <div className="h-[290px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 26, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="step"
              stroke="#71717a"
              fontSize={10}
              tickFormatter={(v: number) => v.toLocaleString()}
              tickLine={false}
            />
            <YAxis domain={[0, 100]} unit="%" stroke="#71717a" fontSize={10} width={48} tickLine={false} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#71717a", strokeDasharray: "3 3" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
            {PER_CLASS.map((s) => (
              <Line
                key={s.label}
                name={s.label}
                dataKey={s.label}
                stroke={s.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function TrainingCurves() {
  const [data, setData] = useState<HistoryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/metrics/history`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((rows) => setData(rows))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  // history.jsonl logs the same "step" once per epoch (step 500, 1000, ... in
  // epoch 0, 1, 2), so a raw "metrics vs step" line chart overplots 2-3 points
  // at every x. Dedupe to the most recent record per step and sort ascending so
  // the x-axis is monotonic and each step appears exactly once.
  const series = useMemo(() => {
    const byStep = new Map<number, HistoryEntry>();
    for (const e of data) byStep.set(e.step, e);
    return [...byStep.values()].sort((a, b) => a.step - b.step);
  }, [data]);

  const metricRows = useMemo(
    () =>
      series.map((e) => ({
        step: e.step,
        "mIoU (4-class)": e.miou_4class,
        "mIoU (5-class)": e.miou_5class,
        "val loss": e.val_loss,
      })),
    [series]
  );

  const perClassRows = useMemo(
    () =>
      series.map((e) => {
        const row: Record<string, number | undefined> = { step: e.step };
        for (const s of PER_CLASS) row[s.label] = (e.class_ious_4?.[s.key] ?? 0) * 100;
        return row;
      }),
    [series]
  );

  if (loading)
    return (
      <div className="mx-auto flex max-w-6xl items-center gap-2 px-6 py-8 text-sm text-zinc-400">
        <Refresh className="h-4 w-4 animate-spin" strokeWidth={2} />
        <span>Loading training history</span>
      </div>
    );
  if (error)
    return (
      <div className="mx-auto max-w-6xl px-6 py-8 text-sm text-rose-400">Failed: {error}</div>
    );
  if (!data.length)
    return (
      <div className="mx-auto max-w-6xl px-6 py-8 text-sm text-zinc-400">
        No history available (start the server / check ckpt_dir).
      </div>
    );

  const last = data[data.length - 1];
  const peak4 = Math.max(...data.map((d) => d.miou_4class));

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Training curves</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Range-image UNet / SemanticKITTI (4-class binned eval) / {data.length} validation checkpoints
          </p>
        </div>
        <Link
          href="/eval"
          className="flex items-center gap-1.5 rounded-[4px] bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-700"
        >
          <DashboardSpeed width={14} height={14} strokeWidth={2} />
          Evaluation
        </Link>
      </header>

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Best artifact" value="best_miou.pt" sub={`${last.step.toLocaleString()} steps`} />
        <MetricCard label="Peak mIoU (4-class)" value={peak4.toFixed(4)} sub="val" />
        <MetricCard label="Val loss (last)" value={last.val_loss.toFixed(4)} sub={`epoch ${last.epoch}`} />
      </div>

      <MetricChart title="Training metrics vs step" rows={metricRows} />
      <PerClassChart rows={perClassRows} />
      <p className="hud-cap text-center">
        Hover for exact values / click legend entries to toggle series
      </p>
    </div>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="frost p-4">
      <div className="hud-cap">{label}</div>
      <div className="mt-1 truncate font-mono text-sm text-zinc-100">{value}</div>
      <div className="text-xs text-zinc-500">{sub}</div>
    </div>
  );
}