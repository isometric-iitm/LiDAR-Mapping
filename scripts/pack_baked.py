#!/usr/bin/env python3
"""Bake true segmentation + grid patches into static chunks for pure replay.

Reads dashboard/public/demo/sequence/chunk_*.bin (raw float16 points) as source,
runs Segmenter (CPU) + LogPolarGrid exactly like the live server, and writes
dashboard/public/demo/baked/chunk_*.bin containing per-frame patches + cloud.

Each baked member (one gzip per frame) layout (little-endian):
  u32 frame
  u8  isSnap
  u8  _pad[3]
  f32 groundZ
  i32 yaw_cd
  u32 nUps
  u32 nFree
  u32 nCloud
  // then:
  nUps*6 f32 rows  (i,j,zMean,zMax,occ,trav)
  nUps   u8  cls   + pad to 4 bytes
  nFree*2 f32 freed (i,j)
  nCloud*3 f32 xyz (already ground-rebased)
  nCloud   u8 clsCloud + pad
Baked chunk header (same as raw): PC2D u32, u16 version=2, u16 n_frames, u16 point_width=0, u16 point_dtype=0, then index.

Also writes dashboard/public/demo/baked_manifest.json style update to manifest.json (adds baked flag).
"""
import gzip, hashlib, io, json, math, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "dashboard" / "public" / "demo"
SEQ_DIR = DEMO_DIR / "sequence"
BAKED_DIR = DEMO_DIR / "preload"
MANIFEST_PATH = DEMO_DIR / "manifest.json"

SNAP_IV = 5
MAX_POINTS = 130000

def half_to_float32(u16):
    # use numpy half -> float32 for speed
    return u16.astype(np.float16).astype(np.float32)

def decode_chunk(chunk_path):
    data = chunk_path.read_bytes()
    dv = memoryview(data)
    magic = int.from_bytes(data[0:4], 'little') if data[0:4]==b'PC2D' else int.from_bytes(data[0:4], 'big')
    # actually pack used b'PC2D' raw bytes, DataView getUint32(0,false) checks big endian 0x50433244
    assert data[0:4]==b'PC2D', f"bad magic {data[0:4]}"
    import struct
    ver, n_frames, pw, pd = struct.unpack_from('<HHHH', data, 4)
    assert ver==1, f"ver {ver}"
    index = []
    off=12
    for i in range(n_frames):
        o, l = struct.unpack_from('<QI', data, off)  # u64 offset, u32 len
        index.append((o,l))
        off+=12
    base=12+n_frames*12
    frames=[]
    for o,l in index:
        member = data[base+o:base+o+l]
        raw = gzip.decompress(member)
        n_floats = len(raw)//2 if pd==16 else len(raw)//4
        n = n_floats//4
        if pd==16:
            pts = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(-1,4)
        else:
            pts = np.frombuffer(raw, dtype=np.float32).reshape(-1,4)
        frames.append(pts)
    return frames, ver

def load_manifest():
    m=json.loads(MANIFEST_PATH.read_text())
    return m

def main():
    import torch
    from src.grid_engine.logpolar_grid import LogPolarGrid
    from src.models.predict import Segmenter
    from src.data.label_mapping import bin_5_to_4

    manifest = load_manifest()
    seq_len = manifest["seq_len"]
    chunk_frames = manifest["chunk_frames"]
    poses = np.array(manifest["poses"], dtype=np.float32).reshape(-1,4,4) if manifest["poses"] else None
    yaws = manifest["yaws_cd"]

    # decode all frames from sequence chunks into list
    all_pts = []
    n_chunks = manifest["n_chunks"]
    for ci in range(n_chunks):
        p = SEQ_DIR / f"chunk_{ci:03d}.bin"
        frames,_ = decode_chunk(p)
        all_pts.extend(frames)
    assert len(all_pts) >= seq_len, f"decoded {len(all_pts)} < {seq_len}"
    all_pts = all_pts[:seq_len]
    print(f"[bake] decoded {len(all_pts)} frames")

    # init segmenter - auto GPU if available
    ckpt = ROOT / "checkpoints" / "best_miou.pt"
    # allow env override
    import os, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None, help="auto|cuda|cpu")
    ap.add_argument("--precision", type=str, default=None, help="fp32|fp16")
    args, _ = ap.parse_known_args()
    ckpt = Path(os.getenv("PC2D_CHECKPOINT", str(ckpt)))
    # auto-detect GPU
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision:
        precision = args.precision
    else:
        precision = "fp16" if device == "cuda" else "fp32"
    print(f"[bake] loading segmenter {ckpt} device={device} precision={precision} (cuda_available={torch.cuda.is_available()})")
    seg = Segmenter(str(ckpt), device=device, precision=precision)
    grid = LogPolarGrid()
    print(f"[bake] grid {grid.n_rings}x{grid.n_theta} n_cells={grid.n_cells}")

    # cloud accumulation (mirror app.py)
    from collections import deque
    cloud_hist = deque(maxlen=manifest["cloud"]["history_frames"])
    cloud_max = manifest["cloud"]["points_max"]
    # need pose mats same as server: already have

    baked_frames = []  # list of bytes (members)
    last_snap = -999

    for fi, pts in enumerate(all_pts):
        if pts.shape[0]==0:
            cls4 = np.zeros(0,dtype=np.uint8)
        else:
            cls5, _ = seg.segment(pts)
            cls4 = bin_5_to_4(cls5)
        grid.update(pts, cls4)
        is_snap = (grid.frame - last_snap) >= SNAP_IV
        if is_snap:
            snap = grid.compute_snapshot()
            rows, cls = snap["rows"], snap["cls"]
            freed = np.zeros((0,2),dtype=np.float32)
            last_snap = grid.frame
            # commit after
            grid.commit_snapshot()
        else:
            delta = grid.compute_delta()
            rows, cls = delta["rows"], delta["cls"]
            freed = delta["freed"]
            grid.commit_delta(delta)
        # cloud
        # _accum_stream
        if poses is not None:
            pose = poses[fi % len(poses)]
            R = pose[:3,:3]; t = pose[:3,3]
            new_world = (pts[:,:3].astype(np.float64) @ R.T.astype(np.float64) + t).astype(np.float32)
            cloud_hist.append((new_world.copy(), cls4.astype(np.uint8)))
            total = sum(len(w) for w,_ in cloud_hist)
            stride = max(1, int(np.ceil(total / cloud_max)))
            parts = [(w[::stride], c[::stride]) for w,c in cloud_hist]
            world = np.concatenate([w for w,_ in parts]) if parts else np.zeros((0,3),dtype=np.float32)
            clsc = np.concatenate([c for _,c in parts]) if parts else np.zeros(0,dtype=np.uint8)
            ego = ((world.astype(np.float64) - t) @ R.astype(np.float64)).astype(np.float32)
            if ego.shape[0] > cloud_max:
                ego = ego[:cloud_max]; clsc = clsc[:cloud_max]
            cxyz = np.round(ego,2)
            cxyz[:,2] -= float(grid.ground_z)
        else:
            n=pts.shape[0]
            stride=max(1,int(np.ceil(n/cloud_max)))
            cxyz = pts[::stride,:3].astype(np.float32)
            clsc = cls4[::stride].astype(np.uint8)
            cxyz[:,2] -= float(grid.ground_z)
        # pack member
        nUps = rows.shape[0]
        nFree = freed.shape[0]
        nCloud = cxyz.shape[0]
        yaw_cd = int(yaws[fi]) if fi < len(yaws) else 0
        groundZ = float(grid.ground_z)
        frame = int(grid.frame)
        # build payload
        buf = io.BytesIO()
        buf.write(struct.pack('<I', frame))
        buf.write(struct.pack('B', 1 if is_snap else 0))
        buf.write(b'\x00\x00\x00')
        buf.write(struct.pack('<f', groundZ))
        buf.write(struct.pack('<i', yaw_cd))
        buf.write(struct.pack('<III', nUps, nFree, nCloud))
        if nUps>0:
            buf.write(rows.astype(np.float32).tobytes())
            # cls with pad to 4
            cls_u8 = cls.astype(np.uint8).tobytes()
            buf.write(cls_u8)
            pad = (-len(cls_u8)) % 4
            if pad: buf.write(b'\x00'*pad)
        if nFree>0:
            buf.write(freed.astype(np.float32).tobytes())
        if nCloud>0:
            buf.write(cxyz.astype(np.float32).tobytes())
            clsC = clsc.astype(np.uint8).tobytes()
            buf.write(clsC)
            pad = (-len(clsC)) % 4
            if pad: buf.write(b'\x00'*pad)
        member = gzip.compress(buf.getvalue(), compresslevel=6)
        baked_frames.append(member)
        if fi % 50==0:
            print(f"[bake] frame {fi}/{seq_len} nUps={nUps} nFree={nFree} nCloud={nCloud} isSnap={is_snap}")

    # pack into baked chunks
    BAKED_DIR.mkdir(parents=True, exist_ok=True)
    baked_chunks_meta=[]
    total_bytes=0
    for c in range(n_chunks):
        s = c*chunk_frames
        e = min(s+chunk_frames, seq_len)
        members = baked_frames[s:e]
        n = len(members)
        index = np.zeros(n, dtype=[("offset","<u8"),("length","<u4")])
        off=0
        for k,m in enumerate(members):
            index[k]=(off, len(m))
            off+=len(m)
        buf=io.BytesIO()
        buf.write(b"PC2D")
        buf.write(struct.pack('<HHHH', 2, n, 0, 0))  # version 2
        buf.write(index.tobytes())
        for m in members:
            buf.write(m)
        data=buf.getvalue()
        p = BAKED_DIR / f"seg_{c:03d}.bin"
        p.write_bytes(data)
        sha=hashlib.sha256(data).hexdigest()
        baked_chunks_meta.append({"file": f"preload/seg_{c:03d}.bin","sha256":sha,"bytes":len(data),"frame_start":s,"frames":n})
        total_bytes+=len(data)
        print(f"[bake] chunk {c:02d} {n} frames {len(data)/1e6:.2f} MB")

    # update manifest - use obfuscated `preload` name, keep `baked` alias for compat
    payload = {"version":2, "chunks": baked_chunks_meta, "total_bytes": total_bytes, "snapshot_interval": SNAP_IV}
    manifest["preload"] = payload
    manifest["baked"] = payload
    MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[bake] done total {total_bytes/1e6:.2f} MB -> {BAKED_DIR}")

if __name__=="__main__":
    main()
