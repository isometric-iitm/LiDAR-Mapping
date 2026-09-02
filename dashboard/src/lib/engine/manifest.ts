/**
 * Demo manifest types + loader. The manifest is emitted by
 * scripts/pack_demo_sequence.py and lives at ${DEMO_BASE}/manifest.json.
 */

export interface DemoChunkMeta {
  file: string;
  sha256: string;
  bytes: number;
  frame_start: number;
  frames: number;
}

export interface DemoManifest {
  version: number;
  seq_id: string;
  seq_len: number;
  fps: number;
  start_idx: number;
  chunk_frames: number;
  n_chunks: number;
  point_dtype: "float16" | "float32";
  chunks: DemoChunkMeta[];
  /** [n][16] row-major world_T_velo 4x4 (pose_k @ Tr), or null */
  poses: number[][] | null;
  /** per-frame ego yaw, centi-degrees (matches server yaw_cd) */
  yaws_cd: number[];
  grid: {
    r_min: number;
    r_max: number;
    r_transition: number;
    dr_0: number;
    alpha: number;
    n_theta: number;
    z_min: number;
    z_max: number;
    n_classes: number;
    occupancy_gain: number;
    occ_threshold: number;
    phase1_rings: number;
    phase2_rings: number;
    n_rings: number;
    class_colors: Record<string, string>;
  };
  trav: {
    enabled: boolean;
    weights: [number, number, number, number];
    z_diff_thresh: number;
    slope_thresh: number;
    class_scores: number[];
  };
  cloud: { points_max: number; history_frames: number };
  /** Baked demo: no live inference. Optional legacy field kept for back-compat. */
  model?: {
    file: string;
    in_channels: number;
    num_classes: number;
    h: number;
    w: number;
    fov_top_deg: number;
    fov_bottom_deg: number;
    max_range: number;
    bin_5_to_4: number[];
  };
  /** Precomputed demo segments (pure replay, no live inference). Formerly `baked`. */
  baked?: {
    version: number;
    chunks: DemoChunkMeta[];
    total_bytes: number;
    snapshot_interval: number;
  };
  /** Obfuscated name for baked segments - used in production to avoid obvious `baked` fetches. */
  preload?: {
    version: number;
    chunks: DemoChunkMeta[];
    total_bytes: number;
    snapshot_interval: number;
  };
  total_bytes: number;
}

/** Resolve the demo asset base URL (env-configurable, same-origin default). */
export function demoBase(): string {
  const base = process.env.NEXT_PUBLIC_DEMO_BASE ?? "/demo/";
  return base.endsWith("/") ? base : `${base}/`;
}

export async function fetchManifest(): Promise<DemoManifest> {
  const res = await fetch(`${demoBase()}manifest.json`);
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  return (await res.json()) as DemoManifest;
}
