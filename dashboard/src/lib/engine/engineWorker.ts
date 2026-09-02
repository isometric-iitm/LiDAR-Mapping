/// <reference lib="webworker" />
/**
 * Engine Web Worker: full client-side pipeline (replaces the Python server
 * in demo mode). Per frame:
 *   decompress (gzip member) -> project -> ONNX inference (WebGPU) ->
 *   softmax + 3x3 gather + argmax + bin 5->4 -> grid update -> snapshot/delta ->
 *   cloud accumulation (world-anchored via poses) -> post to main thread.
 *
 * Chunk streaming: chunk 0 is fetched up front; chunks 1..N prefetch in the
 * background while playback runs. If playback catches an unfetched chunk the
 * worker posts `status: buffering` until the chunk lands.
 *
 * Message protocol (mirrors the server's WS protocol semantics):
 *   main -> worker : { type: "init" }
 *                   | { type: "control", action: "play"|"pause"|"speed"|"seek"|"cloud", value? }
 *   worker -> main : { type: "ready" } | { type: "grid_meta", ... }
 *                   | { type: "status", state, msg? } | { type: "stats", ... }
 *                   | { type: "patch", kind, frame, rows?, freed?, epoch, yaw? }
 *                   | { type: "cloud", frame, n, xyz, cls, epoch, yaw? }
 *                   | { type: "download", phase, chunk?, fraction, loaded, total }
 *                   | { type: "control_ack", action: "seek", ... }
 *                   | { type: "webgpu_unsupported" } | { type: "error", message }
 */
import type { DemoManifest } from "./manifest";
import { fetchManifest, demoBase } from "./manifest";
import { fetchAsset, type ProgressCb } from "./assetCache";
import { RangeProjector } from "./projector";
import { LogPolarGrid } from "./logpolarGrid";
import { makeBinLut, gatherArgmax, binClasses } from "./postprocess";

/* eslint-disable @typescript-eslint/no-explicit-any */
type OrtModule = any;

const DEMO_BASE = demoBase();

interface ChunkState {
  buf: ArrayBuffer;
  index: { off: number; len: number }[];
  nFrames: number;
  pointDtype: 16 | 32;
}

// ---- module state ----
let ort: OrtModule | null = null;
let session: OrtModule | null = null;
let manifest: DemoManifest | null = null;
let binLut: Uint8Array | null = null;
let grid: LogPolarGrid | null = null;
let projector: RangeProjector | null = null;
let accScratch: Float32Array | null = null;
let cls5: Uint8Array | null = null;

const chunks: (ChunkState | undefined)[] = [];
const chunkFetch: (Promise<void> | undefined)[] = [];

// playback state
let running = false;
let speed = 1;
let cloudOn = false;
let epoch = 0;
let frameCursor = 0; // next demo frame index to process
let previewPending = false; // render one frame even while paused (seek preview)
let lastSnapshotFrame = -999;
let lastStatsFrame = -999;
let loopGen = 0; // invalidates in-flight loop ticks after seek
let loopTimer: ReturnType<typeof setTimeout> | null = null;
const SNAPSHOT_IV = 5;
const STATS_IV = 2;

/** Stage-1 result: GPU run launched, CPU work still pending. While the GPU
 *  executes frame k, the loop immediately drains frame k-1's CPU stages -
 *  the same SEG-ahead-of-GRID overlap the Python server's two threads give. */
interface Stage1 {
  fi: number;
  n: number;
  slot: import("./projector").ProjectorSlot;
  runP: Promise<any>;
  timings: { t0: number; projectMs: number; gpuStart: number };
}
let stage1: Stage1 | null = null;

const MAX_POINTS = 130000;
let cloudXyz = new Float32Array(0);
let cloudCls = new Uint8Array(0);

// half->float scratch (allocated on first fp16 decode)
let halfScratch32 = new Uint32Array(0);
let halfScratchF = new Float32Array(0);

// cloud accumulation (world-anchored, mirrors app.py _accum_stream)
interface CloudFrame {
  world: Float32Array;
  cls: Uint8Array;
}
let cloudHist: CloudFrame[] = [];

// metrics windows (mirrors metrics.py)
const MAX_WINDOW = 100;
const winProc: number[] = [];
const winSeg: number[] = [];
const winGrid: number[] = [];
const winPack: number[] = [];
const winCloud: number[] = [];
const winProject: number[] = [];
const winForward: number[] = [];
const winPost: number[] = [];
const winLat: number[] = [];

function post(msg: unknown, transfer?: Transferable[]): void {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

function pushWin(arr: number[], v: number): void {
  arr.push(v);
  if (arr.length > MAX_WINDOW) arr.shift();
}

function percentile(arr: number[], q: number): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const rank = (s.length - 1) * q;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (rank - lo);
}

function mean(arr: number[]): number {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

// ---- chunk decoding ----

async function inflateGzip(member: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream !== "undefined") {
    const ds = new DecompressionStream("gzip");
    const stream = new Blob([member as BlobPart]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const { inflate } = await import("pako");
  return inflate(member);
}

/** Fast IEEE-754 half -> float32 via bit manipulation (values in raw are all
 *  normals in practice; subnormal/inf paths included for completeness). */
function halfToFloat32(u16: Uint16Array, out: Float32Array, len: number): void {
  if (halfScratch32.length < len) {
    const n = Math.max(1024, len);
    const buf = new ArrayBuffer(n * 4);
    halfScratch32 = new Uint32Array(buf);
    halfScratchF = new Float32Array(buf);
  }
  const b32 = halfScratch32;
  const bf = halfScratchF;
  for (let k = 0; k < len; k++) {
    const v = u16[k];
    const e = (v >>> 10) & 0x1f;
    if (e === 0) {
      // subnormal (or zero): m * 2^-24
      out[k] = (v & 0x3ff) * 5.960464477539063e-8 * (v & 0x8000 ? -1 : 1);
    } else if (e === 31) {
      out[k] = v & 0x3ff ? NaN : v & 0x8000 ? -Infinity : Infinity;
    } else {
      b32[k] = ((v & 0x8000) << 16) | ((e + 112) << 23) | ((v & 0x3ff) << 13);
      out[k] = bf[k];
    }
  }
}

/** Read + decompress demo frame `fi` into the given slot's points buffer. */
async function readFrameInto(fi: number, slot: import("./projector").ProjectorSlot): Promise<number> {
  const m = manifest!;
  const ci = Math.floor(fi / m.chunk_frames);
  const ki = fi % m.chunk_frames;
  const st = chunks[ci];
  if (!st) throw new Error(`chunk ${ci} not loaded`);
  const entry = st.index[ki];
  const base = 12 + st.index.length * 12;
  const member = new Uint8Array(st.buf, base + entry.off, entry.len);
  const raw = await inflateGzip(member);
  const nFloats = (st.pointDtype === 16 ? raw.byteLength / 2 : raw.byteLength / 4) | 0;
  const n = (nFloats / 4) | 0;
  if (n > MAX_POINTS) throw new Error(`frame has ${n} points > MAX_POINTS`);
  if (st.pointDtype === 16) {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, nFloats);
    halfToFloat32(u16, slot.points, n * 4);
  } else {
    slot.points.set(new Float32Array(raw.buffer, raw.byteOffset, n * 4));
  }
  return n;
}

// ---- asset loading ----

async function ensureChunk(ci: number, onProgress?: ProgressCb): Promise<void> {
  if (chunks[ci]) return;
  if (!chunkFetch[ci]) {
    chunkFetch[ci] = (async () => {
      const m = manifest!;
      const cm = m.chunks[ci];
      const key = `seq:${m.seq_id}:${cm.sha256}`;
      const buf = await fetchAsset(`${DEMO_BASE}${cm.file}`, key, cm.bytes, onProgress);
      const dv = new DataView(buf);
      // magic is the ASCII bytes "PC2D" -> big-endian uint32 0x50433244
      if (dv.getUint32(0, false) !== 0x50433244) throw new Error(`chunk ${ci}: bad magic`);
      const version = dv.getUint16(4, true);
      if (version !== 1) throw new Error(`chunk ${ci}: unsupported version ${version}`);
      const nFrames = dv.getUint16(6, true);
      const pointWidth = dv.getUint16(8, true);
      if (pointWidth !== 4) throw new Error(`chunk ${ci}: bad point width ${pointWidth}`);
      const pointDtype = dv.getUint16(10, true) as 16 | 32;
      const index: { off: number; len: number }[] = new Array(nFrames);
      let off = 12;
      for (let k = 0; k < nFrames; k++) {
        index[k] = { off: Number(dv.getBigUint64(off, true)), len: dv.getUint32(off + 8, true) };
        off += 12;
      }
      chunks[ci] = { buf, index, nFrames, pointDtype };
    })().catch((e) => {
      chunkFetch[ci] = undefined; // allow retry
      throw e;
    });
  }
  await chunkFetch[ci];
}

async function prefetchAhead(): Promise<void> {
  const m = manifest!;
  const need = Math.ceil(frameCursor / m.chunk_frames); // chunk of next frame
  for (let ci = 0; ci < Math.min(need + 1, m.n_chunks); ci++) {
    if (!chunks[ci] && !chunkFetch[ci]) {
      try {
        await ensureChunk(ci, (p) =>
          post({ type: "download", phase: "sequence", chunk: ci, fraction: p.fraction, loaded: p.loaded, total: p.total }),
        );
      } catch (e) {
        post({ type: "error", message: `chunk ${ci} prefetch failed: ${e}` });
      }
    }
  }
}

// ---- pipeline ----

function gridMetaMsg() {
  const m = manifest!;
  const g = m.grid;
  return {
    type: "grid_meta" as const,
    r_min: g.r_min,
    r_max: g.r_max,
    r_transition: g.r_transition,
    alpha: g.alpha,
    dr_0: g.dr_0,
    n_rings: g.n_rings,
    phase1_rings: g.phase1_rings,
    n_theta: g.n_theta,
    n_classes: g.n_classes,
    class_colors: g.class_colors,
    seq_id: m.seq_id,
    seq_len: m.seq_len,
  };
}

/** Cloud in the CURRENT ego frame (mirrors app.py _accum_stream + ground rebase). */
function cloudEgoFrame(frameIdx: number, points: Float32Array, n: number, cls4: Uint8Array): { xyz: Float32Array; cls: Uint8Array } | null {
  const m = manifest!;
  const maxPts = m.cloud.points_max;
  if (!m.poses) {
    // sweep-only fallback
    const stride = Math.max(1, Math.ceil(n / maxPts));
    let cnt = 0;
    for (let k = 0; k < n; k += stride) {
      cloudXyz[cnt * 3] = points[k * 4];
      cloudXyz[cnt * 3 + 1] = points[k * 4 + 1];
      cloudXyz[cnt * 3 + 2] = points[k * 4 + 2];
      cloudCls[cnt] = cls4[k];
      cnt++;
    }
    return { xyz: cloudXyz.subarray(0, cnt * 3), cls: cloudCls.subarray(0, cnt) };
  }
  const pose = m.poses[frameIdx % m.poses.length];
  // row-major 4x4 -> R (3x3), t (3); world = p @ R^T + t (numpy row-vector form)
  const r00 = pose[0], r01 = pose[4], r02 = pose[8], tx = pose[3];
  const r10 = pose[1], r11 = pose[5], r12 = pose[9], ty = pose[7];
  const r20 = pose[2], r21 = pose[6], r22 = pose[10], tz = pose[11];
  const newWorld = new Float32Array(n * 3);
  const newCls = cls4.slice(0, n);
  for (let k = 0; k < n; k++) {
    const x = points[k * 4], y = points[k * 4 + 1], z = points[k * 4 + 2];
    newWorld[k * 3] = x * r00 + y * r10 + z * r20 + tx;
    newWorld[k * 3 + 1] = x * r01 + y * r11 + z * r21 + ty;
    newWorld[k * 3 + 2] = x * r02 + y * r12 + z * r22 + tz;
  }
  cloudHist.push({ world: newWorld, cls: newCls });
  if (cloudHist.length > m.cloud.history_frames) cloudHist.shift();

  let total = 0;
  for (const f of cloudHist) total += f.cls.length;
  const stride = Math.max(1, Math.ceil(total / maxPts));
  let cnt = 0;
  outer: for (const f of cloudHist) {
    for (let k = 0; k < f.cls.length; k += stride) {
      if (cnt >= maxPts) break outer;
      // ego = (world - t) @ R
      const wx = f.world[k * 3] - tx, wy = f.world[k * 3 + 1] - ty, wz = f.world[k * 3 + 2] - tz;
      cloudXyz[cnt * 3] = wx * r00 + wy * r01 + wz * r02;
      cloudXyz[cnt * 3 + 1] = wx * r10 + wy * r11 + wz * r12;
      cloudXyz[cnt * 3 + 2] = wx * r20 + wy * r21 + wz * r22;
      cloudCls[cnt] = f.cls[k];
      cnt++;
    }
  }
  // rebase onto the grid's ground reference (cxyz[:, 2] -= ground_z)
  const gz = grid!.groundZ;
  const xyz = cloudXyz.subarray(0, cnt * 3);
  for (let k = 0; k < cnt; k++) xyz[k * 3 + 2] -= gz;
  return { xyz, cls: cloudCls.subarray(0, cnt) };
}

// ---- two-stage pipelined frame processing ----

/**
 * Stage 1 (producer): decode -> project -> launch GPU run WITHOUT awaiting it.
 * The caller then drains the previous frame's stage 2 while the GPU executes,
 * mirroring the Python server's SEG-thread-ahead-of-GRID-thread overlap.
 */
async function runStage1(fi: number): Promise<void> {
  const m = manifest!;
  const t0 = performance.now();
  const slot = projector!.beginFrame();
  const n = await readFrameInto(fi, slot);
  const tp0 = performance.now();
  projector!.project(slot, n);
  const gpuStart = performance.now();
  const input = new ort!.Tensor("float32", slot.image, [1, m.model.in_channels, m.model.h, m.model.w]);
  const runP = session!.run({ range_image: input });
  stage1 = { fi, n, slot, runP, timings: { t0, projectMs: gpuStart - tp0, gpuStart } };
}

/** Drain stage 2 (consumer) for the pending stage-1 frame. */
async function runStage2(): Promise<void> {
  if (!stage1) return;
  const p = stage1;
  stage1 = null; // clear FIRST so an error path can't double-drain
  const m = manifest!;
  const tGpuEnd = performance.now();
  const probs = (await p.runP).logits_probs.data as Float32Array;

  const tpp0 = performance.now();
  const C = m.model.num_classes;
  gatherArgmax(probs, p.slot.proj, p.n, C, m.model.h, m.model.w, accScratch!, cls5!);
  binClasses(cls5!, binLut!);
  const tpp1 = performance.now();

  const tg0 = performance.now();
  grid!.update(p.slot.points, cls5!, p.n, p.slot.thetas, p.slot.planarRs);
  const tg1 = performance.now();

  const tk0 = performance.now();
  const isSnap = grid!.frame - lastSnapshotFrame >= SNAPSHOT_IV;
  const delta = isSnap ? grid!.computeSnapshot() : grid!.computeDelta();
  const tk1 = performance.now();
  const yawCd = m.yaws_cd[p.fi] ?? 0;
  const yaw = ((yawCd / 100) * Math.PI) / 180;
  post(
    {
      type: "patch",
      kind: isSnap ? "snap" : "delta",
      frame: grid!.frame,
      epoch,
      rows: delta.rows,
      cls: delta.cls,
      nUps: delta.upCount,
      freed: delta.freed,
      yaw,
    },
    [delta.rows.buffer, delta.cls.buffer, delta.freed.buffer],
  );
  grid!.commit(delta);
  if (isSnap) lastSnapshotFrame = grid!.frame;

  let cloudMs = 0;
  if (cloudOn) {
    const tc0 = performance.now();
    const c = cloudEgoFrame(p.fi, p.slot.points, p.n, cls5!);
    cloudMs = performance.now() - tc0;
    if (c) {
      const xyz = c.xyz.slice();
      const cls = c.cls.slice();
      post({ type: "cloud", frame: grid!.frame, n: cls.length, xyz, cls, epoch, yaw }, [xyz.buffer, cls.buffer]);
    }
  }

  const procMs = performance.now() - p.timings.t0;
  const gpuMs = tGpuEnd - p.timings.gpuStart;
  pushWin(winProc, procMs);
  pushWin(winSeg, p.timings.projectMs + gpuMs + (tpp1 - tpp0));
  pushWin(winGrid, tg1 - tg0);
  pushWin(winPack, tk1 - tk0);
  pushWin(winCloud, cloudMs);
  pushWin(winProject, p.timings.projectMs);
  pushWin(winForward, gpuMs);
  pushWin(winPost, tpp1 - tpp0);
  pushWin(winLat, procMs);
  if (grid!.frame - lastStatsFrame >= STATS_IV) {
    lastStatsFrame = grid!.frame;
    const mem = grid!.memoryReport();
    post({
      type: "stats",
      frame: grid!.frame,
      fps: 1000 / Math.max(mean(winProc), 0.001),
      latency_ms_p50: Math.round(percentile(winLat, 0.5) * 10) / 10,
      latency_ms_p95: Math.round(percentile(winLat, 0.95) * 10) / 10,
      seg_ms: Math.round(mean(winSeg) * 10) / 10,
      grid_ms: Math.round(mean(winGrid) * 10) / 10,
      pack_ms: Math.round(mean(winPack) * 10) / 10,
      cloud_ms: Math.round(mean(winCloud) * 10) / 10,
      project_ms: Math.round(mean(winProject) * 10) / 10,
      forward_ms: Math.round(mean(winForward) * 10) / 10,
      grid_mem_kb: mem.gridKb,
      uniform_equiv_mb: mem.uniformMb,
      compression_ratio: mem.compressionRatio,
      capacity_compression: mem.compressionRatio,
      rendered_cells: mem.renderedCells,
      n_cells: mem.nCells,
      seq_pos: p.fi,
      seq_len: m.seq_len,
      epoch,
      frames_emitted: grid!.frame,
      frames_dropped: 0,
    });
  }
}

/** Sequential fallback (seek preview / one-off renders): one full frame. */
async function processFrame(fi: number): Promise<void> {
  await runStage1(fi);
  await runStage2();
}

// ---- playback loop ----

// Seek previews are serialized on a promise chain; a newer seek supersedes
// (token-skips) any queued preview so rapid scrubbing can't pile up stale
// full-pipeline renders. An in-flight render still finishes, but its patch
// carries the old epoch and the main thread drops it.
let previewChain: Promise<void> = Promise.resolve();
let previewToken = 0;
let pendingPlay = false; // play control arrived before the session was ready

function requestPreview(): void {
  const token = ++previewToken;
  const wasRunning = running;
  previewChain = previewChain
    .then(async () => {
      if (token !== previewToken) return; // superseded
      const m = manifest!;
      const ci = Math.floor(frameCursor / m.chunk_frames);
      if (!chunks[ci]) {
        post({ type: "status", state: "buffering", msg: `Loading part ${ci + 1}/${m.n_chunks}...` });
        await ensureChunk(ci, (p) =>
          post({ type: "download", phase: "sequence", chunk: ci, fraction: p.fraction, loaded: p.loaded, total: p.total }),
        );
        post({ type: "status", state: "ready", msg: "Streaming" });
        if (token !== previewToken) return; // seeked while buffering
      }
      await processFrame(frameCursor);
      void prefetchAhead();
    })
    .then(() => {
      previewPending = false;
      if (wasRunning && running) scheduleNext(0);
    })
    .catch((e) => post({ type: "error", message: `preview failed: ${e}` }));
}

function scheduleNext(frameMs: number): void {
  const gen = loopGen;
  const period = 1000 / (manifest!.fps * speed) - frameMs;
  loopTimer = setTimeout(() => void loopTick(gen), Math.max(0, period));
}

async function loopTick(gen: number): Promise<void> {
  if (!running || gen !== loopGen) return; // stale tick (seeked/paused)
  const t0 = performance.now();
  const m = manifest!;
  let fi = frameCursor;
  if (fi >= m.seq_len) {
    fi = 0; // loop
    frameCursor = 0;
  }
  const ci = Math.floor(fi / m.chunk_frames);
  if (!chunks[ci]) {
    post({ type: "status", state: "buffering", msg: `Buffering part ${ci + 1}/${m.n_chunks}...` });
    try {
      await ensureChunk(ci, (p) =>
        post({ type: "download", phase: "sequence", chunk: ci, fraction: p.fraction, loaded: p.loaded, total: p.total }),
      );
      post({ type: "status", state: "ready", msg: "Streaming" });
    } catch (e) {
      post({ type: "error", message: `chunk ${ci} unavailable: ${e}` });
      running = false;
      return;
    }
    if (!running || gen !== loopGen) return; // seeked while buffering
  }
  try {
    // Stage 1: launch this frame's GPU run (no await of the GPU yet).
    await runStage1(fi);
    // Stage 2 of the PREVIOUS tick runs here while this frame's GPU executes:
    // decode+project (CPU) overlapped the GPU; now gather+grid+pack overlap it too.
    await runStage2();
  } catch (e) {
    post({ type: "error", message: `frame ${fi} failed: ${e}` });
    running = false;
    return;
  }
  frameCursor = fi + 1;
  void prefetchAhead();
  if (!running || gen !== loopGen) return; // paused/seeked during processing
  scheduleNext(performance.now() - t0);
}

function doSeek(fi: number): void {
  const m = manifest!;
  frameCursor = Math.max(0, Math.min(m.seq_len - 1, Math.floor(fi)));
  epoch += 1;
  loopGen += 1; // kill any in-flight tick
  if (loopTimer) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
  stage1 = null; // drop the stale in-flight frame; its epoch won't match
  grid!.reset();
  cloudHist = [];
  lastSnapshotFrame = -999;
  lastStatsFrame = -999;
  winProc.length = 0;
  winSeg.length = 0;
  winGrid.length = 0;
  winPack.length = 0;
  winCloud.length = 0;
  winProject.length = 0;
  winForward.length = 0;
  winPost.length = 0;
  winLat.length = 0;
  previewPending = true;
  post({ type: "patch", kind: "reset", frame: 0, epoch });
  post({ type: "control_ack", action: "seek", frame: 0, idx: frameCursor, epoch });
  requestPreview(); // renders preview; resumes the paced loop after if playing
}

// ---- init ----

async function init(): Promise<void> {
  try {
    // 1) WebGPU gate: distinguish "browser can't ever run this" (adapter API
    //    missing) from later failures (session/warmup), which report as errors.
    const nav = navigator as Navigator & { gpu?: unknown };
    if (typeof navigator === "undefined" || !nav.gpu) {
      post({ type: "webgpu_unsupported" });
      return;
    }

    post({ type: "status", state: "loading", msg: "Loading demo manifest..." });
    manifest = await fetchManifest();
    const m = manifest;

    post({ type: "status", state: "loading", msg: "Downloading neural weights..." });
    const modelKey = `model:${m.model.file}`;
    const modelBuf = await fetchAsset(`${DEMO_BASE}${m.model.file}`, modelKey, 0, (p) => {
      post({ type: "download", phase: "model", fraction: p.fraction, loaded: p.loaded, total: p.total });
    });

    post({ type: "status", state: "loading", msg: "Compiling WebGPU kernels..." });
    // ORT is copied to /ort/ at build time (scripts/copy-assets.mjs); loaded with
    // a runtime import so no bundler ever touches it (wasm-safe, static-export-safe).
    const ortUrl = new URL("/ort/ort.webgpu.bundle.min.mjs", self.location.origin).href;
    ort = (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ ortUrl)) as OrtModule;
    ort.env.wasm.wasmPaths = "/ort/";
    // Prefer the discrete GPU on dual-GPU machines (laptops). Windows currently
    // ignores powerPreference in requestAdapter (crbug.com/369219127) but this
    // still selects correctly on macOS/Linux and future Windows builds.
    if (ort.env.webgpu) {
      ort.env.webgpu.powerPreference = "high-performance";
    }
    let sessionReady = false;
    let sessionError = "";
    try {
      session = await ort.InferenceSession.create(modelBuf, {
        executionProviders: ["webgpu"],
        // graph optimization level: all (the default; kept explicit so a
        // future ORT default change can't silently regress speed)
        graphOptimizationLevel: "all",
      });
      // A full-size warmup run both compiles kernels and proves the EP works;
      // failing here (e.g. blocked adapter) means the browser can't run the demo.
      const warmup = new ort.Tensor(
        "float32",
        new Float32Array(m.model.in_channels * m.model.h * m.model.w),
        [1, m.model.in_channels, m.model.h, m.model.w],
      );
      await session.run({ range_image: warmup });
      sessionReady = true;
    } catch (e) {
      sessionError = String(e instanceof Error ? e.message : e);
    }
    if (!sessionReady) {
      post({
        type: "error",
        message: `ONNX session failed to start${sessionError ? `: ${sessionError.slice(0, 300)}` : ""}`,
      });
      return;
    }

    // 2) engine state
    projector = new RangeProjector(
      { h: m.model.h, w: m.model.w, fovTopDeg: m.model.fov_top_deg, fovBottomDeg: m.model.fov_bottom_deg, maxRange: m.model.max_range },
      MAX_POINTS,
    );
    grid = new LogPolarGrid(
      {
        rMin: m.grid.r_min,
        rMax: m.grid.r_max,
        dr0: m.grid.dr_0,
        rTransition: m.grid.r_transition,
        alpha: m.grid.alpha,
        nTheta: m.grid.n_theta,
        zMin: m.grid.z_min,
        zMax: m.grid.z_max,
        nClasses: m.grid.n_classes,
        occGain: m.grid.occupancy_gain,
        occThreshold: m.grid.occ_threshold,
      },
      {
        enabled: m.trav.enabled,
        weights: m.trav.weights,
        zDiffThresh: m.trav.z_diff_thresh,
        slopeThresh: m.trav.slope_thresh,
        classScores: m.trav.class_scores,
      },
      MAX_POINTS,
    );
    binLut = makeBinLut(m.model.bin_5_to_4);
    cloudXyz = new Float32Array(m.cloud.points_max * 3);
    cloudCls = new Uint8Array(m.cloud.points_max);
    accScratch = new Float32Array(m.model.num_classes);
    cls5 = new Uint8Array(MAX_POINTS);

    // 3) chunk 0 (blocking) then ready
    post({ type: "status", state: "loading", msg: "Downloading first sequence part..." });
    await ensureChunk(0, (p) =>
      post({ type: "download", phase: "sequence", chunk: 0, fraction: p.fraction, loaded: p.loaded, total: p.total }),
    );

    post(gridMetaMsg());
    post({ type: "ready" });
    post({ type: "status", state: "ready", msg: "Ready" });
    void prefetchAhead();
    if (pendingPlay) {
      // play control arrived during boot; honor it now
      pendingPlay = false;
      running = true;
      scheduleNext(0);
    }
  } catch (e) {
    post({ type: "error", message: String(e) });
  }
}

// ---- control plane ----

self.onmessage = (ev: MessageEvent) => {
  const data = ev.data as { type: string; action?: string; value?: number | boolean };
  if (data.type === "init") {
    void init();
    return;
  }
  if (data.type !== "control") return;
  const action = data.action;
  if (action === "play") {
    if (!session) {
      pendingPlay = true; // session still booting; init() resumes playback
      return;
    }
    if (!running) {
      running = true;
      loopGen += 1; // invalidate any stale in-flight tick from before a pause
      if (previewPending) {
        requestPreview();
      } else {
        scheduleNext(0);
      }
    }
  } else if (action === "pause") {
    if (!session) {
      pendingPlay = false;
      return;
    }
    running = false;
    loopGen += 1;
    if (loopTimer) {
      clearTimeout(loopTimer);
      loopTimer = null;
    }
    // Drain the in-flight frame so the final patch isn't lost on pause.
    if (stage1) void runStage2().catch(() => {});
  } else if (action === "speed") {
    speed = Math.max(0.05, Number(data.value) || 1);
  } else if (action === "cloud") {
    cloudOn = Boolean(data.value);
    if (!cloudOn) cloudHist = [];
  } else if (action === "seek") {
    if (!session) return;
    doSeek(Number(data.value) || 0);
  }
};
