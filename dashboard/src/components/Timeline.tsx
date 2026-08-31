"use client";

import { useRef, useState } from "react";
import { Pause, Play, Refresh } from "iconoir-react";

type SeqInfo = { id: string; frames: number; has_poses: boolean; has_labels: boolean };

type Props = {
  seqPos: number;
  seqLen: number;
  paused: boolean;
  speed: number;
  seeking: boolean;
  buffering: boolean;
  seqId: string;
  availableSeqs: SeqInfo[];
  onPause: (p: boolean) => void;
  onSeek: (idx: number) => void;
  onSpeed: (s: number) => void;
  onSwitchSeq: (seqId: string) => void;
};

const fmt = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

const SPEEDS = [0.5, 1, 2, 4];

export default function Timeline({ seqPos, seqLen, paused, speed, seeking, buffering, seqId, availableSeqs, onPause, onSeek, onSpeed, onSwitchSeq }: Props) {
  const [dragging, setDragging] = useState(false);
  const [scrubFrac, setScrubFrac] = useState<number | null>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const lastSend = useRef(0);
  const lastCommittedFrac = useRef(-1);

  const frac = seqLen > 0 ? Math.min(1, Math.max(0, seqPos / seqLen)) : 0;
  const shown = dragging && scrubFrac !== null ? scrubFrac : frac;
  const shownPos = Math.round(shown * seqLen);
  const totalSec = Math.round(seqLen / 10);

  const computeFrac = (clientX: number) => {
    const track = trackRef.current;
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  };

  const commitSeek = (f: number) => {
    if (lastCommittedFrac.current >= 0 && Math.abs(f - lastCommittedFrac.current) < 0.005) return;
    lastCommittedFrac.current = f;
    const idx = Math.max(0, Math.min(seqLen - 1, Math.round(f * (seqLen - 1))));
    onSeek(idx);
  };

  return (
    <div className="frost pointer-events-auto absolute inset-x-0 bottom-4 mx-auto flex w-[min(720px,94%)] items-center gap-3 px-4 py-2 text-xs text-zinc-200">
      <button
        onClick={() => onPause(!paused)}
        disabled={seeking || buffering}
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[4px] bg-zinc-800 text-zinc-200 transition-colors hover:bg-zinc-700 disabled:cursor-wait disabled:opacity-60"
        title={seeking || buffering ? "Loading\u2026" : paused ? "Resume playback" : "Freeze playback"}
      >
        {seeking || buffering ? (
          <Refresh className="h-4 w-4 animate-spin" strokeWidth={2} />
        ) : paused ? (
          <Play className="h-4 w-4" strokeWidth={2} />
        ) : (
          <Pause className="h-4 w-4" strokeWidth={2} />
        )}
      </button>

      <div
        ref={trackRef}
        className="relative h-9 min-w-0 flex-1 cursor-pointer touch-none select-none"
        onPointerDown={(e) => {
          const f = computeFrac(e.clientX);
          setDragging(true);
          setScrubFrac(f);
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!dragging) return;
          const f = computeFrac(e.clientX);
          setScrubFrac(f);
          const now = performance.now();
          if (now - lastSend.current > 120) {
            lastSend.current = now;
            commitSeek(f);
          }
        }}
        onPointerUp={(e) => {
          const f = computeFrac(e.clientX);
          setDragging(false);
          setScrubFrac(null);
          commitSeek(f);
        }}
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-zinc-700" />
        <div
          className="absolute top-1/2 left-0 h-1 -translate-y-1/2 rounded-full bg-emerald-500/80"
          style={{ width: `${shown * 100}%` }}
        />
        {seeking && (
          <div
            className="absolute top-1/2 left-0 h-1 -translate-y-1/2 animate-pulse rounded-full bg-sky-400/60"
            style={{ width: `${shown * 100}%` }}
          />
        )}
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-400"
          style={{ left: `${shown * 100}%` }}
        />
      </div>

      <div className="shrink-0 font-mono text-zinc-400">
        {seeking ? "..." : `${fmt(shownPos / 10)} / ${fmt(totalSec)}`}
      </div>
      <div className="hidden shrink-0 font-mono text-zinc-500 sm:block">
        {seeking ? "..." : `${shownPos.toLocaleString()} / ${seqLen.toLocaleString()}`}
      </div>
      <select
        value={seqId}
        onChange={(e) => onSwitchSeq(e.target.value)}
        disabled={seeking}
        className="shrink-0 rounded-[4px] bg-zinc-800 px-2 py-1 font-mono text-xs text-zinc-300 outline-none transition-colors hover:bg-zinc-700 focus:ring-1 focus:ring-zinc-500 disabled:cursor-wait disabled:opacity-60"
        title="Sequence"
      >
        {availableSeqs.map((s) => (
          <option key={s.id} value={s.id}>
            Seq {s.id}
          </option>
        ))}
      </select>
      <select
        value={speed}
        onChange={(e) => onSpeed(Number(e.target.value))}
        className="shrink-0 rounded-[4px] bg-zinc-800 px-2 py-1 font-mono text-xs text-zinc-300 outline-none transition-colors hover:bg-zinc-700 focus:ring-1 focus:ring-zinc-500"
        title="Playback speed"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s.toFixed(1)}×
          </option>
        ))}
      </select>
    </div>
  );
}