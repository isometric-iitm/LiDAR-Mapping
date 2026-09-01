import numpy as np
import pytest

from src.data.projection import (
    build_range_image,
    build_range_image_gpu,
    compute_projection,
    knn_project_back,
    project_points_gpu,
)


class TestCpuGpuProjectionParity:
    def test_row_col_match_synthetic(self):
        rng = np.random.default_rng(0)
        n = 500
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, 0] = rng.uniform(0.5, 40.0, n)
        pts[:, 1] = rng.uniform(-20, 20, n)
        pts[:, 2] = rng.uniform(-2, 3, n)
        pts[:, 3] = 1.0
        proj_cpu, ranges_cpu = compute_projection(pts, h=64, w=2048)
        import torch
        pts_t = torch.from_numpy(pts[:, :4].copy())
        if torch.cuda.is_available():
            pts_t = pts_t.to("cuda")
        row, col, r = project_points_gpu(pts_t, h=64, w=2048)
        row_np = row.cpu().numpy() if row.is_cuda else row.numpy()
        col_np = col.cpu().numpy() if col.is_cuda else col.numpy()
        np.testing.assert_array_equal(proj_cpu[:, 0], row_np)
        np.testing.assert_array_equal(proj_cpu[:, 1], col_np)

    def test_range_values_match(self):
        rng = np.random.default_rng(1)
        n = 200
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, 0] = rng.uniform(1.0, 30.0, n)
        pts[:, 1] = rng.uniform(-10, 10, n)
        pts[:, 2] = rng.uniform(-1, 2, n)
        pts[:, 3] = 1.0
        _, ranges_cpu = compute_projection(pts, h=64, w=2048)
        import torch
        pts_t = torch.from_numpy(pts[:, :4].copy())
        if torch.cuda.is_available():
            pts_t = pts_t.to("cuda")
        _, _, r_gpu = project_points_gpu(pts_t, h=64, w=2048)
        r_np = r_gpu.cpu().numpy() if r_gpu.is_cuda else r_gpu.numpy()
        np.testing.assert_allclose(ranges_cpu, r_np, rtol=1e-5)


class TestBuildRangeImageNearestWins:
    def test_cpu_nearest_point_per_pixel(self):
        h, w, max_range = 64, 2048, 80.0
        pts = np.array([
            [5.0, 0.0, 1.0, 0.9],
            [3.0, 0.0, 0.5, 0.5],
        ], dtype=np.float32)
        proj, ranges = compute_projection(pts, h=h, w=w)
        ri = build_range_image(pts, proj, pts[:, 3], ranges, h=h, w=w, max_range=max_range)
        row, col = proj[0]
        assert ri[0, row, col] == pytest.approx(3.0 / max_range, rel=0.05)

    def test_gpu_nearest_point_per_pixel(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        h, w, max_range = 64, 2048, 80.0
        pts_np = np.array([
            [5.0, 0.0, 1.0, 0.9],
            [3.0, 0.0, 0.5, 0.5],
        ], dtype=np.float32)
        pts_t = torch.from_numpy(pts_np).to("cuda")
        row, col, r = project_points_gpu(pts_t, h=h, w=w)
        ri = build_range_image_gpu(pts_t, row, col, r, h=h, w=w, max_range=max_range)
        r0 = row[0].item()
        c0 = col[0].item()
        assert ri[0, r0, c0].item() == pytest.approx(3.0 / max_range, rel=0.05)


class TestKnnProjectBack:
    def test_output_shape_and_probs_sum(self):
        import torch
        b, c, h, w, n = 1, 5, 64, 2048, 100
        rng = np.random.default_rng(42)
        logits = torch.randn(b, c, h, w)
        proj_np = np.stack([
            rng.integers(0, h, n),
            rng.integers(0, w, n),
        ], axis=-1).astype(np.int64)
        proj_t = torch.from_numpy(proj_np).unsqueeze(0)
        out = knn_project_back(logits, proj_t, k=3)
        assert out.shape == (b, n, c)
        sums = out.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_segmenter_knn_probs_output_shape_and_valid(self):
        import torch
        from src.models.predict import Segmenter
        b, c, h, w, n = 1, 5, 64, 2048, 50
        rng = np.random.default_rng(7)
        pixel_probs = torch.softmax(torch.randn(b, c, h, w), dim=1)
        proj_np = np.stack([
            rng.integers(0, h, n),
            rng.integers(0, w, n),
        ], axis=-1).astype(np.int64)
        proj_t = torch.from_numpy(proj_np).unsqueeze(0)
        result = Segmenter._knn_probs(pixel_probs, proj_t, k=3)
        assert result.shape == (n, c)
        assert (result >= 0).all()
        row_sums = result.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(n), atol=1e-4, rtol=1e-4)

    def test_knn_project_back_output_shape_and_valid(self):
        import torch
        b, c, h, w, n = 1, 5, 64, 2048, 50
        rng = np.random.default_rng(7)
        pixel_probs = torch.softmax(torch.randn(b, c, h, w), dim=1)
        proj_np = np.stack([
            rng.integers(0, h, n),
            rng.integers(0, w, n),
        ], axis=-1).astype(np.int64)
        proj_t = torch.from_numpy(proj_np).unsqueeze(0)
        result = knn_project_back(pixel_probs, proj_t, k=3)
        assert result.shape == (b, n, c)
        assert (result >= 0).all()
        row_sums = result.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(b, n), atol=1e-4, rtol=1e-4)


class TestProjectionEdgeCases:
    def test_single_point_no_crash(self):
        pts = np.array([[5.0, 0.0, 1.0, 0.5]], dtype=np.float32)
        proj, ranges = compute_projection(pts, h=64, w=2048)
        ri = build_range_image(pts, proj, pts[:, 3], ranges, h=64, w=2048)
        assert ri.shape == (5, 64, 2048)
        assert ri.any()

    def test_jit_projection_matches_numpy(self):
        from src.data import projection
        rng = np.random.default_rng(3)
        n = 3000
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, 0] = rng.uniform(-60, 60, n)
        pts[:, 1] = rng.uniform(-60, 60, n)
        pts[:, 2] = rng.uniform(-3, 5, n)
        pts[:, 3] = 1.0
        p_jit, r_jit = compute_projection(pts, h=64, w=2048)
        p_np, r_np = projection._compute_projection_numpy(pts, 64, 2048, 2.0, -24.8)
        np.testing.assert_array_equal(p_jit, p_np)
        np.testing.assert_allclose(r_jit, r_np, atol=1e-5)

    def test_range_image_values(self):
        h, w, max_range = 64, 2048, 80.0
        pts = np.array([[5.0, 0.0, 1.0, 0.7]], dtype=np.float32)
        proj, ranges = compute_projection(pts, h=h, w=w)
        ri = build_range_image(pts, proj, pts[:, 3], ranges, h=h, w=w, max_range=max_range)
        row, col = proj[0]
        expected_range_norm = ranges[0] / max_range
        assert ri[0, row, col] == pytest.approx(expected_range_norm, rel=1e-4)
        assert ri[4, row, col] == pytest.approx(0.7, rel=1e-4)
