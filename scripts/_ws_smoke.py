"""Smoke test the live PC2D server on a spare port (never collides with the demo server).

Spawns uvicorn as a subprocess (same pattern as run_live_demo.py), then drives a
websocket client: streams cloud + mesh, checks pause freeze, and verifies seek
rebuild + epoch semantics.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8010


async def ws_smoke():
    await asyncio.sleep(2.5)
    import websockets
    uri = f"ws://127.0.0.1:{PORT}/ws/map"
    counts = {}
    cloud_ok = mesh_ok = ack_ok = frozen_ok = epoch_ok = seek_ok = None
    cloud_frames = mesh_frames = stats_seen = 0
    cloud_max_n = 0
    mesh_max_n = 0
    epoch = None
    all_ok = False
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "control", "action": "cloud", "value": True}))
        await ws.send(json.dumps({"type": "control", "action": "mesh", "value": True}))
        deadline = time.time() + 40
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=12)
            data = json.loads(raw)
            t = data["type"]
            counts[t] = counts.get(t, 0) + 1
            if epoch is None and data.get("epoch") is not None:
                epoch = data["epoch"]
            if t == "stats":
                stats_seen += 1
                if stats_seen == 2:
                    print("stats seq_len=%s seq_pos=%s epoch=%s" % (data.get("seq_len"), data.get("seq_pos"), data.get("epoch")))
                    assert data.get("seq_len") and data.get("seq_pos") is not None
            elif t == "cloud":
                cloud_frames += 1
                n = data["n"]
                cloud_max_n = max(cloud_max_n, n)
                if len(data["xyz"]) == n * 3 and len(data["cls"]) == n and len(raw) < 1024 * 1024:
                    cloud_ok = True
                else:
                    cloud_ok = False
            elif t == "mesh":
                mesh_frames += 1
                n = data["n"]
                mesh_max_n = max(mesh_max_n, n)
                if len(data["xyz"]) == n * 3 and len(data["cls"]) == n and len(raw) < 1024 * 1024:
                    mesh_ok = True
                else:
                    mesh_ok = False
            if cloud_frames >= 6 and mesh_frames >= 6 and stats_seen >= 2:
                break

        # pause -> expect ack; deltas after ack must not exceed its frame
        await ws.send(json.dumps({"type": "control", "action": "pause"}))
        got_ack = None
        max_frame_after_pause = None
        pause_end = time.time() + 3
        while time.time() < pause_end:
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(raw)
            counts[data["type"]] = counts.get(data["type"], 0) + 1
            if data["type"] == "control_ack":
                got_ack = data
            elif data["type"] in ("delta", "snapshot", "cloud", "mesh"):
                f = data.get("frame")
                max_frame_after_pause = f if max_frame_after_pause is None else max(max_frame_after_pause, f)
        ack_ok = got_ack is not None and got_ack["action"] == "pause"
        frozen_ok = ack_ok and max_frame_after_pause is not None and max_frame_after_pause <= got_ack["frame"]
        if got_ack:
            print("pause ack: %s frame=%s | max frame after ack=%s" % (got_ack["action"], got_ack["frame"], max_frame_after_pause))

        # seek -> ack with bumped epoch; following frames must all carry the new epoch
        await ws.send(json.dumps({"type": "control", "action": "seek", "value": 2000}))
        seek_ack = None
        seen_after = []
        seek_end = time.time() + 3
        while time.time() < seek_end:
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(raw)
            counts[data["type"]] = counts.get(data["type"], 0) + 1
            if data["type"] == "control_ack" and data.get("action") == "seek":
                seek_ack = data
            if data["type"] in ("delta", "snapshot", "cloud", "mesh", "stats"):
                seen_after.append(data)
        if seek_ack:
            print("seek ack: idx=%s epoch=%s (was %s) | frames after: %s" % (
                seek_ack.get("idx"), seek_ack.get("epoch"), epoch,
                [d.get("frame") for d in seen_after[-5:]]))
            new_epoch = seek_ack.get("epoch")
            epoch_ok = new_epoch is not None and new_epoch != epoch
            seek_ok = all(d.get("epoch") == new_epoch for d in seen_after)

        await ws.send(json.dumps({"type": "control", "action": "play"}))
        await asyncio.sleep(0.6)

    print("message counts:", counts)
    print("cloud max n:", cloud_max_n, "| mesh max n:", mesh_max_n)
    print("cloud ok:", cloud_ok, "| mesh ok:", mesh_ok,
          "| pause ok:", ack_ok and frozen_ok, "| epoch bump:", epoch_ok, "| seek consistent:", seek_ok)
    all_ok = (cloud_ok and mesh_ok and ack_ok and frozen_ok
              and epoch_ok and seek_ok and cloud_max_n >= 6000)
    return all_ok


def main():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn",
         "src.server.app:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    try:
        passed = asyncio.run(asyncio.wait_for(ws_smoke(), timeout=140))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("SMOKE", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()