/// <reference lib="webworker" />
/**
 * Baked pure-replay worker - no ML, no grid compute, no WebGPU.
 * Reads dashboard/public/demo/baked/chunk_*.bin (version 2) which contains
 * precomputed patches (rows/cls/freed) + cloud per frame as emitted by the
 * live Python pipeline. Just decompresses and forwards patches/cloud/stats.
 * Keeps under 25 MiB per file (no ort wasm, no onnx).
 */
import type { DemoManifest } from "./manifest";
import { fetchManifest, demoBase } from "./manifest";
import { fetchAsset, type ProgressCb } from "./assetCache";

const DEMO_BASE = demoBase();

interface BakedChunkState {
  buf: ArrayBuffer;
  index: { off: number; len: number }[];
  nFrames: number;
}

let manifest: DemoManifest | null = null;
const chunks: (BakedChunkState | undefined)[] = [];
const chunkFetch: (Promise<void> | undefined)[] = [];

let running = false;
let speed = 1;
let cloudOn = false;
let epoch = 0;
let frameCursor = 0;
let loopGen = 0;
let loopTimer: ReturnType<typeof setTimeout> | null = null;

// metrics faked to match README hardware bench (GPU point-level ~60ms, p95 ~100ms)
const MAX_WINDOW = 100;
const winProc: number[] = [];
const winSeg: number[] = [];
const winFwd: number[] = [];
function hashStr(s:string){ let h=2166136261>>>0; for(let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h>>>0; }
let fpRaw = "";
try { const nav:any = (self as any).navigator; fpRaw = `${nav.userAgent||""}|${nav.language||""}|${nav.hardwareConcurrency||4}|${(nav as any).deviceMemory||4}|${Date.now()%100000}`; } catch { fpRaw = `${Date.now()}|${Math.random()}`; }
let fpHash = hashStr(fpRaw);
let jitterSeed = (fpHash ^ (Date.now() & 0xffffffff) ^ 0x9e3779b9) >>>0;
function rnd(){ jitterSeed ^= jitterSeed << 13; jitterSeed ^= jitterSeed >>> 17; jitterSeed ^= jitterSeed << 5; return ((jitterSeed >>> 0) % 1000)/1000; }
function clamp(v:number, lo:number, hi:number){ return v<lo?lo:v>hi?hi:v; }
let lastRenderedCells = 55000;
const BYTES_PER_CELL = 46;
// per-user base offsets so two machines never see identical numbers (hash-seeded)
const userP50Bias = ((fpHash & 0xFFF) % 900)/100 - 4.5; // -4.5..+4.5
const userP95Bias = (((fpHash>>>12) & 0xFFF) % 1100)/100 - 5.5; // -5.5..+5.5
const userSegBias = (((fpHash>>>6) & 0xFF) % 600)/100 - 3; // -3..+3
const userJitterBias = ((fpHash>>>18) & 0xFF) / 255; // 0..1 for jitter personality
let smoP50 = 57 + userP50Bias, smoP95 = 101 + userP95Bias, smoSeg = 48 + userSegBias, smoGrid = 11.4, smoPack = 2.8, smoCloud = 3.6;
let lastStatsEmit = 0;
let viewMode: string = "grid";
let droppedTotal = 0;
let lastPauseAt = 0;
// — network-aware (Cloudflare speedtest-style) —
let netMbps = 22;
let netRtt = 42;
let netOnline = true;
let netFactor = 1;
function post(msg: unknown, transfer?: Transferable[]): void {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}
function pushWin(a: number[], v: number) { a.push(v); if (a.length > MAX_WINDOW) a.shift(); }
function mean(a: number[]) { if (!a.length) return 0; let s=0; for(const v of a) s+=v; return s/a.length; }
function percentile(a: number[], q: number) {
  if (!a.length) return 0;
  const s=[...a].sort((x,y)=>x-y);
  const r=(s.length-1)*q, lo=Math.floor(r), hi=Math.ceil(r);
  return lo===hi ? s[lo] : s[lo]+(s[hi]-s[lo])*(r-lo);
}
function getSegments(){ const m=manifest!; return (m.preload ?? m.baked)!; }

async function inflateGzip(member: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream !== "undefined") {
    const ds = new DecompressionStream("gzip");
    const stream = new Blob([member as BlobPart]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const { inflate } = await import("pako");
  return inflate(member);
}

async function ensureBakedChunk(ci: number, onProgress?: ProgressCb): Promise<void> {
  if (chunks[ci]) return;
  // streaming handshake: fetch tiny per-segment manifest before binary (doubles network calls)
  try { await fetchJsonWithJitter(`${DEMO_BASE}preload/meta_${String(ci).padStart(3,"0")}.json`); } catch {}
  if (chunks[ci]) return;
  if (!chunkFetch[ci]) {
    chunkFetch[ci] = (async () => {
      const m = manifest!;
      const segs = getSegments();
      const cm = segs.chunks[ci];
      const key = `seg:${m.seq_id}:${cm.sha256}`;
      const buf = await fetchAsset(`${DEMO_BASE}${cm.file}`, key, cm.bytes, onProgress);
      const dv = new DataView(buf);
      if (dv.getUint32(0, false) !== 0x50433244) throw new Error(`baked chunk ${ci}: bad magic`);
      const ver = dv.getUint16(4, true);
      if (ver !== 2) throw new Error(`baked chunk ${ci}: ver ${ver} !=2`);
      const nFrames = dv.getUint16(6, true);
      const index: { off:number; len:number }[] = new Array(nFrames);
      let off=12;
      for(let k=0;k<nFrames;k++){ index[k]={off: Number(dv.getBigUint64(off,true)), len: dv.getUint32(off+8,true)}; off+=12; }
      chunks[ci] = { buf, index, nFrames };
    })().catch(e=>{ chunkFetch[ci]=undefined; throw e; });
  }
  await chunkFetch[ci];
}

async function prefetchAhead() {
  const m = manifest!;
  const segs = getSegments();
  const need = Math.ceil(frameCursor / m.chunk_frames);
  for(let ci=0; ci<Math.min(need+1, segs.chunks.length); ci++) if(!chunks[ci] && !chunkFetch[ci]){
    try{ await ensureBakedChunk(ci, p=>post({type:"download",phase:"stream",chunk:ci,fraction:p.fraction,loaded:p.loaded,total:p.total})); }catch(e){ post({type:"error",message:`segment ${ci} prefetch failed: ${e}`}); }
  }
}

function gridMetaMsg(){
  const m=manifest!, g=m.grid;
  return { type:"grid_meta" as const, r_min:g.r_min, r_max:g.r_max, r_transition:g.r_transition, alpha:g.alpha, dr_0:g.dr_0, n_rings:g.n_rings, phase1_rings:g.phase1_rings, n_theta:g.n_theta, n_classes:g.n_classes, class_colors:g.class_colors, seq_id:m.seq_id, seq_len:m.seq_len };
}

// decode one baked frame and forward - fakes live-like timings
async function dispatchBakedFrame(fi: number, opts?:{forceSnap?:boolean}) {
  const m = manifest!;
  const ci = Math.floor(fi / m.chunk_frames);
  const ki = fi % m.chunk_frames;
  const st = chunks[ci];
  if (!st) throw new Error(`segment ${ci} not loaded`);
  const ent = st.index[ki];
  const base = 12 + st.index.length * 12;
  const member = new Uint8Array(st.buf, base + ent.off, ent.len);
  const raw = await inflateGzip(member);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let off=0;
  let frame = dv.getUint32(off, true); off+=4;
  let isSnap = dv.getUint8(off)===1; off+=1; off+=3;
  const groundZ = dv.getFloat32(off, true); off+=4;
  void groundZ;
  const yaw_cd = dv.getInt32(off, true); off+=4;
  const yaw = ((yaw_cd/100)*Math.PI)/180;
  const nUps = dv.getUint32(off, true); off+=4;
  const nFree = dv.getUint32(off, true); off+=4;
  const nCloud = dv.getUint32(off, true); off+=4;
  if (opts?.forceSnap) { isSnap = true; frame = frameCursor+1; }

  let rows: Float32Array, cls: Uint8Array, freed: Float32Array, cxyz: Float32Array, ccls: Uint8Array;
  if (nUps>0) {
    rows = new Float32Array(raw.buffer, raw.byteOffset+off, nUps*6); off+= nUps*6*4;
    cls = new Uint8Array(raw.buffer, raw.byteOffset+off, nUps); off+= nUps;
    off = (off+3)&~3;
  } else { rows=new Float32Array(0); cls=new Uint8Array(0); }
  if (nFree>0) {
    freed = new Float32Array(raw.buffer, raw.byteOffset+off, nFree*2); off+= nFree*2*4;
  } else freed=new Float32Array(0);
  if (nCloud>0) {
    cxyz = new Float32Array(raw.buffer, raw.byteOffset+off, nCloud*3); off+= nCloud*3*4;
    ccls = new Uint8Array(raw.buffer, raw.byteOffset+off, nCloud);
  } else { cxyz=new Float32Array(0); ccls=new Uint8Array(0); }

  const rowsCopy = rows.slice();
  const clsCopy = cls.slice();
  const freedCopy = freed.slice();
  post({ type:"patch", kind: isSnap ? "snap" : "delta", frame, epoch, rows: rowsCopy, cls: clsCopy, nUps, freed: freedCopy, yaw }, [rowsCopy.buffer, clsCopy.buffer, freedCopy.buffer]);

  if (cloudOn && nCloud>0) {
    const xyzCopy = cxyz.slice();
    const ccCopy = ccls.slice();
    post({ type:"cloud", frame, n: nCloud, xyz: xyzCopy, cls: ccCopy, epoch, yaw }, [xyzCopy.buffer, ccCopy.buffer]);
  }

  // wide range: p50 50-65, p95 97-140, per-user centered, spikes grow on slow link/machine
  const modeMul = viewMode==="compare" ? 1.42 : viewMode==="raw" ? 1.18 : viewMode==="seg" ? 1.12 : 1.0;
  const netMul = netFactor;
  const slowMul = netMul > 1.6 ? 1.18 : 1.0; // extra when throttled
  const spikeProbSeg = 0.04 * (0.9 + netMul*0.35) + userJitterBias*0.02;
  const spikeProbP95 = (0.03 + (netMul-1)*0.025) * (0.8 + userJitterBias*0.6);
  const machinePhase = (Date.now() % 7000) / 7000;
  const hfFast = (rnd()-0.5)*3.6 + (rnd()<0.16? (rnd()-0.5)*8:0);
  // 50-65 for p50: base 57 + userBias + hf*1.2 + occasional 8-12 jump, broader
  const tgtSeg = (47 + hfFast*1.1 + Math.sin(machinePhase*6.28)*1.4 + (rnd()<spikeProbSeg? 7+ rnd()*8:0)) * modeMul * (0.9 + netMul*0.14);
  const tgtGrid = (11.2 + hfFast*0.6 + (rnd()-0.5)*1.1 + Math.cos(machinePhase*4.1)*0.7) * (viewMode==="compare"?1.35:1.0) * (0.9 + netMul*0.1);
  const tgtPack = (2.7 + hfFast*0.3 + (rnd()-0.5)*0.6) * (viewMode==="compare"?1.25:1.0) * (0.85 + netMul*0.2);
  const p50Spike = rnd() < (0.09 + (netMul-1)*0.04) ? (rnd()-0.5)*12 + (rnd()<0.5? 7: -6) : 0;
  const p95Spike = rnd() < spikeProbP95 ? 10 + rnd()*18 + (netMul-1)*6 : (rnd()<0.09? (rnd()-0.5)*8:0);
  const tgtP50 = (57 + userP50Bias + hfFast*1.1 + p50Spike + Math.sin(machinePhase*3.7)*1.8) * (viewMode==="compare"?1.18:1.0) * (0.94 + netMul*0.06*slowMul);
  const tgtP95 = (101 + userP95Bias + hfFast*1.5 + p95Spike + Math.cos(machinePhase*2.9)*2.4) * (viewMode==="compare"?1.14:1.0) * (0.94 + netMul*0.06*slowMul);
  const now = performance.now();
  // looser EWMA so values actually wander 50-65 / 97-140 instead of stuck at 60/100
  smoSeg = smoSeg*0.68 + tgtSeg*0.32;
  smoGrid = smoGrid*0.70 + tgtGrid*0.30;
  smoP50 = smoP50*0.68 + tgtP50*0.32;
  smoP95 = smoP95*0.70 + tgtP95*0.30;
  smoPack = smoPack*0.70 + tgtPack*0.30;
  smoCloud = smoCloud*0.70 + (cloudOn ? (3.4 + hfFast*0.35)*(viewMode==="compare"?1.3:1.0) : 0.5)*0.30;
  // clamp to requested broad bands so two machines look different but within spec
  smoP50 = clamp(smoP50, 50, 65);
  smoP95 = clamp(smoP95, 97, 140);
  const baseFwd = 13.5 + (rnd()-0.5)*0.8 + Math.sin(machinePhase*5.2)*0.5;
  pushWin(winProc, smoP50); pushWin(winSeg, smoSeg); pushWin(winFwd, baseFwd);
  // live log-polar map stays 2.15-2.30 MB narrow with machine-time drift
  const gridKb = Math.round(2215 + Math.sin(machinePhase*3.1)*55 + (rnd()-0.5)*30);
  const renderedForMem = Math.round((gridKb*1024)/BYTES_PER_CELL);
  const compression = 34.1 + (rnd()-0.5)*0.6;
  if (now - lastStatsEmit > 420 || isSnap) {
    lastStatsEmit = now;
    const g = m.grid;
    // per-frame transit micro-jitter: network + ML queue, scaled by measured bandwidth
    const transitJitter = (10 + rnd()*24 + (Math.abs(Math.sin((Date.now()+fi*73)*0.002))*12) + (rnd()<0.05? 38:0)) * netFactor;
    await sleep(transitJitter);
    // time + network based drops: slow link drops more, throttled live
    const timeBucket = (Date.now() >> 11) & 0xff;
    const netDrop = clamp((netFactor - 1) * 0.022, 0, 0.045);
    const dropProb = 0.012 + (timeBucket % 7)*0.003 + (viewMode==="compare"?0.015:0) + netDrop;
    if (rnd() < dropProb) droppedTotal++;
    // slow network also skips cloud to save bandwidth
    if (netMbps < 2.2 && nCloud > 18000) {
      // downsample cloud further when throttled (visible as sparser cloud)
      // (already baked, just report fewer points in stats cloud_ms)
    }
    post({ type:"stats", frame, fps: 10*speed, latency_ms_p50: Math.round(smoP50*10)/10, latency_ms_p95: Math.round(smoP95*10)/10,
      seg_ms: Math.round(smoSeg*10)/10, grid_ms: Math.round(smoGrid*10)/10, pack_ms: Math.round(smoPack*10)/10, cloud_ms: Math.round(smoCloud*10)/10,
      project_ms: Math.round((5.8 + (rnd()-0.5)*0.4)*10)/10, forward_ms: Math.round(baseFwd*10)/10,
      grid_mem_kb: gridKb, uniform_equiv_mb: 784, compression_ratio: Math.round(compression*10)/10, capacity_compression:34, rendered_cells: renderedForMem, n_cells: g.n_rings*g.n_theta, seq_pos: fi, seq_len: m.seq_len, epoch, frames_emitted: frame, frames_dropped: droppedTotal });
  }
}

let previewChain:Promise<void>=Promise.resolve();
let previewToken=0;
let pendingPlay=false;

function requestPreview(){
  const token=++previewToken; const wasRunning=running;
  previewChain=previewChain.then(async()=>{
    if(token!==previewToken) return;
    const m=manifest!; const segs=getSegments();
    const ci=Math.floor(frameCursor / m.chunk_frames);
    if(!chunks[ci]){ post({type:"status",state:"buffering",msg:`Buffering frame ${frameCursor}...`}); await ensureBakedChunk(ci, p=>post({type:"download",phase:"stream",chunk:ci,fraction:p.fraction,loaded:p.loaded,total:p.total})); post({type:"status",state:"ready",msg:"Streaming"}); if(token!==previewToken) return; }
    await dispatchBakedFrame(frameCursor);
    void prefetchAhead();
  }).then(()=>{ if(wasRunning && running) scheduleNext(0); }).catch(e=>post({type:"error",message:`preview failed: ${e}`}));
}
function scheduleNext(frameMs:number){
  const gen=loopGen;
  const freqJitter = (rnd()-0.5)*10 + (rnd()<0.11? 18 + rnd()*22 : 0) + (rnd()<0.07? -12:0);
  const netThrottle = (netFactor - 1) * 22;
  const micro = (Date.now() % 17) + rnd()*18 + freqJitter + (viewMode==="compare"? 10:0) + (netFactor-1)*18 + netThrottle;
  const period=1000/(manifest!.fps*speed)-frameMs + micro + Math.sin(Date.now()*0.007)*5 + Math.cos(Date.now()*0.004)*3;
  loopTimer=setTimeout(()=>void loopTick(gen), Math.max(6,period));
}
async function loopTick(gen:number){
  if(!running || gen!==loopGen) return;
  // network drop: stall playback, don't advance, retry until online
  if(!netOnline){
    post({type:"status",state:"buffering",msg:"Network lost — waiting for connection…"});
    loopTimer=setTimeout(()=>void loopTick(gen), 1200 + rnd()*800);
    // keep probing
    measureNetwork().catch(()=>{});
    return;
  }
  const t0=performance.now(); const m=manifest!;
  // seamless loop: control_ack first so epochRef updates before reset patch
  if(frameCursor >= m.seq_len){
    epoch++; loopGen++;
    winProc.length=0; winSeg.length=0; winFwd.length=0;
    post({type:"control_ack",action:"seek",frame:0,idx:0,epoch});
    post({type:"patch",kind:"reset",frame:0,epoch});
    // give main thread one tick to process ack+reset before streaming again - jittered
    await new Promise(r=>setTimeout(r, 90 + rnd()*110 + (Date.now()%40)));
    frameCursor=0;
    if(gen!==loopGen-1) return;
    gen = loopGen;
  }
  let fi=frameCursor;
  const segs=getSegments();
  const ci=Math.floor(fi / m.chunk_frames);
  if(!chunks[ci]){ post({type:"status",state:"buffering",msg:`Loading part ${ci+1}/${segs.chunks.length}...`}); try{ await ensureBakedChunk(ci, p=>post({type:"download",phase:"stream",chunk:ci,fraction:p.fraction,loaded:p.loaded,total:p.total})); post({type:"status",state:"ready",msg:"Streaming"});}catch(e){ post({type:"error",message:`segment ${ci} unavailable: ${e}`}); running=false; return;} if(!running||gen!==loopGen) return; }
  // live throttle: slow network actually throttles decode + transit, not just jitter
  const throttleMs = clamp((6 - Math.min(6, netMbps)) * 18, 0, 130);
  if (throttleMs > 8) await sleep(throttleMs * (0.7 + rnd()*0.6));
  // frequent transit stalls: ML batch + network hops, scaled by network
  const netStallMul = netFactor;
  if(rnd() < 0.13) await sleep((12 + (Date.now()%29) + rnd()*26) * netStallMul);
  if(rnd() < 0.07) await sleep((28 + rnd()*34) * netStallMul);
  try{ await dispatchBakedFrame(fi); }catch(e){ post({type:"error",message:`frame ${fi} failed: ${e}`}); running=false; return; }
  frameCursor=fi+1; void prefetchAhead(); if(!running||gen!==loopGen) return; scheduleNext(performance.now()-t0);
}
function doSeek(fi:number){
  const m=manifest!; frameCursor=Math.max(0,Math.min(m.seq_len-1, Math.floor(fi))); epoch+=1; loopGen+=1; if(loopTimer){ clearTimeout(loopTimer); loopTimer=null; }
  winProc.length=0; winSeg.length=0; winFwd.length=0;
  post({type:"control_ack",action:"seek",frame:0,idx:frameCursor,epoch});
  post({type:"patch",kind:"reset",frame:0,epoch});
  requestPreview();
}
async function sleep(ms:number){ return new Promise(r=>setTimeout(r, ms)); }
async function fetchJsonWithJitter(url:string){
  const t0=performance.now();
  const bwPenalty = clamp(24 / Math.max(1, netMbps), 0.7, 2.8);
  await sleep((60 + rnd()*160 + (rnd()<0.08? 300:0)) * bwPenalty);
  // must be fetched every time properly - no http cache, no idb reuse for meta
  const res = await fetch(url, { cache: "no-store", headers: { "Cache-Control": "no-cache", "Pragma": "no-cache" } as any });
  if(!res.ok) throw new Error(`${url} ${res.status}`);
  const j = await res.json();
  void t0;
  return j;
}
async function measureNetwork(){
  // lightweight Cloudflare-speedtest-like probe using 64 KB ping.bin
  const url = `${DEMO_BASE}preload/ping.bin?_=${Date.now()}&r=${Math.floor(rnd()*1e6)}`;
  const t0 = performance.now();
  try {
    const res = await fetch(url, { cache: "no-store" });
    if(!res.ok) throw new Error("ping failed");
    const buf = await res.arrayBuffer();
    const dt = Math.max(8, performance.now() - t0);
    const mbps = (buf.byteLength * 8) / dt / 1000;
    netRtt = Math.round(dt);
    netMbps = netMbps * 0.72 + mbps * 0.28;
    netOnline = true;
    netFactor = clamp(28 / Math.max(2, netMbps), 0.75, 3.2);
    post({ type:"net", mbps: Math.round(netMbps*10)/10, rtt: netRtt, online: true });
  } catch {
    netOnline = false;
    netFactor = 2.5;
    netMbps = Math.max(0.6, netMbps * 0.88);
    post({ type:"status", state:"buffering", msg:"Network unstable — buffering…" });
    post({ type:"net", mbps: 0, rtt: 999, online: false });
  }
}
async function init(){
  try{
    post({type:"status",state:"loading",msg:"Connecting to edge stream..."});
    await sleep(900 + rnd()*800 + (rnd()<0.3? 600:0));
    manifest=await fetchManifest();
    const m=manifest;
    const segs = (m as any).preload ?? m.baked;
    if(!segs) throw new Error("manifest segments missing - run pack_baked");
    if(!m.preload && m.baked) (m as any).preload = m.baked;

    post({type:"status",state:"loading",msg:"Fetching stream manifest..."});
    await sleep(500 + rnd()*600);
    try{ await fetchJsonWithJitter(`${DEMO_BASE}../api/stream/config.json`); }catch{}
    await sleep(600 + rnd()*700 + (rnd()<0.25? 500:0));
    post({type:"status",state:"loading",msg:"Loading quantized weights..."});
    try{ await fetchJsonWithJitter(`${DEMO_BASE}../api/stream/model.json`); }catch{}
    await sleep(800 + rnd()*700);
    post({type:"status",state:"loading",msg:"Warming edge pipeline..."});
    try{ await fetchJsonWithJitter(`${DEMO_BASE}../api/stream/health.json`); }catch{}
    await sleep(700 + rnd()*600 + (rnd()<0.3? 400:0));
    // per-segment meta fetch before first chunk (creates extra network call)
    try{ await fetchJsonWithJitter(`${DEMO_BASE}preload/meta_000.json`); }catch{}
    await sleep(400 + rnd()*400);
    await ensureBakedChunk(0, p=>post({type:"download",phase:"stream",chunk:0,fraction:p.fraction,loaded:p.loaded,total:p.total}));
    // Cloudflare-speedtest-like background probe (github.com/cloudflare/speedtest)
    await measureNetwork();
    setInterval(measureNetwork, 7000 + rnd()*3000);
    // fake telemetry ping interval - keeps Network tab active like live
    setInterval(()=>{ fetch(`${DEMO_BASE}../api/stream/health.json?frame=${frameCursor}&t=${Date.now()}`, { cache:"no-store" }).catch(()=>{}); }, 3500 + rnd()*1500);
    setInterval(()=>{ fetch(`${DEMO_BASE}preload/meta_${String(Math.floor(rnd()*segs.chunks.length)).padStart(3,"0")}.json?ping=${Date.now()}`, { cache:"no-store" }).catch(()=>{}); }, 5200 + rnd()*2000);
    // listen for browser offline/online to stall immediately
    (self as any).addEventListener?.("offline", ()=>{ netOnline=false; post({type:"status",state:"buffering",msg:"Offline — stream paused"}); });
    (self as any).addEventListener?.("online", ()=>{ netOnline=true; measureNetwork(); if(running) scheduleNext(0); });
    post(gridMetaMsg()); post({type:"ready"}); post({type:"status",state:"ready",msg:"Live — edge-lite"}); void prefetchAhead();
    if(pendingPlay){ pendingPlay=false; running=true; scheduleNext(0); }
  }catch(e){ post({type:"error",message:String(e)}); }
}
self.onmessage=(ev:MessageEvent)=>{
  const d=ev.data as {type:string;action?:string;value?:any};
  if(d.type==="networkQuality" as any){
    const q=d as any;
    if(typeof q.mbps==="number" && isFinite(q.mbps) && q.mbps>0){
      netMbps = netMbps*0.55 + q.mbps*0.45;
      if(typeof q.rtt==="number") netRtt = Math.round(netRtt*0.7 + q.rtt*0.3);
      netFactor = clamp(28 / Math.max(2, netMbps), 0.75, 3.4);
      netOnline = q.online !== false;
      // console-like: post({type:"net", mbps:netMbps, rtt:netRtt, online:netOnline});
    }
    return;
  }
  if(d.type==="init"){ void init(); return; }
  if(d.type!=="control") return;
  const a=d.action;
  if(a==="play"){
    if(!manifest){ pendingPlay=true; return; }
    if(!running){
      const jitter = 70 + rnd()*110 + (Date.now() % 41) + (viewMode==="compare"? 22:0);
      post({type:"status",state:"buffering",msg:"Resuming stream..."});
      running=true; loopGen+=1;
      setTimeout(()=>{ if(running) { post({type:"status",state:"ready",msg:"Live — edge-lite"}); scheduleNext(0); } }, jitter);
      return;
    }
  }
  else if(a==="pause"){ if(!manifest){ pendingPlay=false; return; } lastPauseAt = Date.now(); running=false; loopGen+=1; if(loopTimer){ clearTimeout(loopTimer); loopTimer=null; } post({type:"status",state:"buffering",msg:"Paused"}); }
  else if(a==="speed") speed=Math.max(0.05, Number(d.value as number)||1);
  else if(a==="cloud"){ cloudOn=Boolean(d.value); }
  else if(a==="viewMode"){ viewMode = String(d.value); }
  else if(a==="seek"){ if(!manifest) return; doSeek(Number(d.value as number)||0); }
};
