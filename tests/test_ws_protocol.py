import struct

import numpy as np
import pytest

from src.grid_engine.logpolar_grid import LogPolarGrid, load_grid_config
from src.server import ws_protocol


class TestBinarySnapshotHeader:
    def test_header_magic_and_code(self):
        grid = LogPolarGrid(load_grid_config())
        rng = np.random.default_rng(0)
        n = 50
        pts = np.zeros((n, 4), dtype=np.float32)
        r = rng.uniform(1.0, 20.0, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        pts[:, 0] = r * np.cos(theta)
        pts[:, 1] = r * np.sin(theta)
        pts[:, 2] = rng.uniform(-2, 3, n)
        pts[:, 3] = 1.0
        labels = rng.integers(0, 4, n, dtype=np.uint8)
        grid.update(pts, labels)
        snap = grid.snapshot()
        frames = list(ws_protocol.iter_snapshot_frames(
            snap["frame"], epoch=0, rows6=snap["rows"], cls=snap["cls"],
            chunk=ws_protocol.CHUNK_CELLS, yaw_cd=0,
        ))
        assert len(frames) >= 1
        buf = frames[0]
        assert len(buf) >= 44
        magic, code, version = struct.unpack_from("<IHH", buf, 0)
        assert magic == 0x50433244
        assert code == 1
        assert version == 2

    def test_header_frame_and_counts(self):
        grid = LogPolarGrid(load_grid_config())
        rng = np.random.default_rng(1)
        n = 30
        pts = np.zeros((n, 4), dtype=np.float32)
        r = rng.uniform(1.0, 15.0, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        pts[:, 0] = r * np.cos(theta)
        pts[:, 1] = r * np.sin(theta)
        pts[:, 2] = rng.uniform(-1, 2, n)
        pts[:, 3] = 1.0
        labels = rng.integers(0, 4, n, dtype=np.uint8)
        grid.update(pts, labels)
        snap = grid.snapshot()
        k = snap["rows"].shape[0]
        frames = list(ws_protocol.iter_snapshot_frames(
            snap["frame"], epoch=7, rows6=snap["rows"], cls=snap["cls"],
            chunk=ws_protocol.CHUNK_CELLS, yaw_cd=123,
        ))
        buf = frames[0]
        frame_val = struct.unpack_from("<Q", buf, 8)[0]
        epoch_val = struct.unpack_from("<Q", buf, 16)[0]
        n_a = struct.unpack_from("<i", buf, 24)[0]
        n_b = struct.unpack_from("<i", buf, 28)[0]
        yaw = struct.unpack_from("<i", buf, 40)[0]
        assert frame_val == snap["frame"]
        assert epoch_val == 7
        assert n_a == k
        assert n_b == 0
        assert yaw == 123


class TestBinaryDeltaHeader:
    def test_delta_code_and_freed(self):
        grid = LogPolarGrid(load_grid_config())
        rng = np.random.default_rng(2)
        n = 40
        pts = np.zeros((n, 4), dtype=np.float32)
        r = rng.uniform(1.0, 15.0, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        pts[:, 0] = r * np.cos(theta)
        pts[:, 1] = r * np.sin(theta)
        pts[:, 2] = rng.uniform(-1, 2, n)
        pts[:, 3] = 1.0
        labels = rng.integers(0, 4, n, dtype=np.uint8)
        grid.update(pts, labels)
        grid.snapshot()

        other_pts = pts.copy()
        other_pts[:, 0] += 50.0
        for _ in range(15):
            grid.update(other_pts, labels)

        delta = grid.delta()
        frames = list(ws_protocol.iter_delta_frames(
            delta["frame"], epoch=3, rows6=delta["rows"], cls=delta["cls"],
            freed=delta["freed"], chunk=ws_protocol.CHUNK_CELLS, yaw_cd=0,
        ))
        assert len(frames) >= 1
        buf = frames[-1]
        magic, code, version = struct.unpack_from("<IHH", buf, 0)
        assert magic == 0x50433244
        assert code == 2
        n_freed_in_header = struct.unpack_from("<i", buf, 28)[0]
        assert n_freed_in_header == delta["freed"].shape[0]


class TestBinaryRowLayout:
    def test_row_record_32_bytes(self):
        grid = LogPolarGrid(load_grid_config())
        rng = np.random.default_rng(3)
        n = 20
        pts = np.zeros((n, 4), dtype=np.float32)
        r = rng.uniform(1.0, 10.0, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        pts[:, 0] = r * np.cos(theta)
        pts[:, 1] = r * np.sin(theta)
        pts[:, 2] = rng.uniform(-1, 2, n)
        pts[:, 3] = 1.0
        labels = rng.integers(0, 4, n, dtype=np.uint8)
        grid.update(pts, labels)
        snap = grid.snapshot()
        k = snap["rows"].shape[0]
        if k == 0:
            pytest.skip("no rendered cells")
        frames = list(ws_protocol.iter_snapshot_frames(
            snap["frame"], epoch=0, rows6=snap["rows"], cls=snap["cls"],
            chunk=ws_protocol.CHUNK_CELLS, yaw_cd=0,
        ))
        buf = frames[0]
        body = buf[44:]
        assert len(body) == k * 32
        floats = np.frombuffer(body, dtype="<f4")
        first_row = floats[:7]
        i_val, j_val = first_row[0], first_row[1]
        trav_val = first_row[6]
        assert i_val >= 0
        assert j_val >= 0
        assert 0.0 <= trav_val <= 1.0


class TestWireCompression:
    def test_roundtrip_preserves_payload(self):
        rng = np.random.default_rng(7)
        # build a payload larger than the 512-byte threshold
        payload = bytes(rng.integers(0, 256, 4096, dtype=np.uint8))
        comp = ws_protocol._maybe_compress(payload, enabled=True)
        # large incompressible data should fall back to raw (no benefit)
        assert comp == payload or comp[0:1] == b"Z"
        dec = ws_protocol.decompress_if_needed(comp)
        assert dec == payload

    def test_compressible_data_actually_compresses(self):
        payload = b"ABCDEFGHIJKLMNOP" * 512  # highly compressible, 8192 bytes
        comp = ws_protocol._maybe_compress(payload, enabled=True)
        assert comp[0:1] == b"Z"
        assert len(comp) < len(payload)
        # verify the exact original size is encoded in the header
        orig_size = struct.unpack("<I", comp[1:5])[0]
        assert orig_size == len(payload)
        assert ws_protocol.decompress_if_needed(comp) == payload

    def test_disabled_and_small_payloads_passthrough(self):
        small = b"hello"
        assert ws_protocol._maybe_compress(small, enabled=True) == small
        big = b"A" * 2000
        assert ws_protocol._maybe_compress(big, enabled=False) == big

    def test_compressed_output_is_raw_deflate(self):
        import zlib
        payload = b"data-raw-deflate-check" * 64
        comp = ws_protocol._maybe_compress(payload, enabled=True)
        assert comp[0:1] == b"Z"
        # raw DEFLATE must decompress with wbits=-15 (no zlib header)
        inflated = zlib.decompress(comp[5:], wbits=-15)
        assert inflated == payload

