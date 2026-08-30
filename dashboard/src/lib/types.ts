/** Cell tuple: [i, j, zMean, zMax, cls, occ, dyn, trav] */
export type Cell = [number, number, number, number, number, number, number, number];

export type GridMeta = {
  type: "grid_meta";
  r_min: number;
  r_max: number;
  alpha: number;
  dr_0: number;
  n_rings: number;
  n_theta: number;
  n_classes: number;
  class_colors: Record<string, string>;
};

export type Stats = {
  type: "stats";
  frame: number;
  fps: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  seg_ms: number;
  grid_ms?: number;
  pack_ms?: number;
  cloud_ms?: number;
  project_ms?: number;
  forward_ms?: number;
  grid_mem_kb: number;
  uniform_equiv_mb: number;
  compression_ratio: number;
  n_cells: number;
  seq_pos?: number;
  seq_len?: number;
  epoch?: number;
  frames_emitted?: number;
  frames_dropped?: number;
};

export type SnapshotMsg = {
  type: "snapshot";
  frame: number;
  seq: number;
  total: number;
  cells: Cell[];
  epoch?: number;
};

export type DeltaMsg = {
  type: "delta";
  frame: number;
  seq: number;
  total: number;
  cells: Cell[];
  freed: [number, number][];
  epoch?: number;
};

export type CloudMsg = {
  type: "cloud";
  frame: number;
  n: number;
  xyz: number[]; // flat [x,y,z] * n
  cls: number[]; // binned class per point
  epoch?: number;
};

export type AckMsg = {
  type: "control_ack";
  action: "pause" | "play" | "speed" | "seek";
  frame: number;
  idx?: number;
  epoch?: number;
};

export type DoneMsg = {
  type: "snapshot_done" | "delta_done";
  frame: number;
  seq: number;
  total: number;
};

export type ServerMsg = GridMeta | Stats | SnapshotMsg | DeltaMsg | CloudMsg | AckMsg | DoneMsg;