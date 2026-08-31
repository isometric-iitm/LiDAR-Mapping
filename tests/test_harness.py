import numpy as np
import pytest


class TestGridFixture:
    def test_grid_created(self, grid):
        assert grid is not None
        assert grid.n_rings > 0
        assert grid.n_theta > 0
        assert grid.n_cells == grid.n_rings * grid.n_theta

    def test_grid_cfg_defaults(self, grid):
        assert grid.r_min == pytest.approx(0.5)
        assert grid.r_max == pytest.approx(100.0)
        assert grid.dr_0 == pytest.approx(0.05)
        assert grid.alpha == pytest.approx(1.05)
        assert grid.n_theta == 720
        assert grid.n_classes == 4

    def test_grid_memory_report(self, grid):
        mem = grid.memory_report()
        assert mem["grid_kb"] > 0
        assert mem["grid_kb"] < 5000
        assert mem["compression_ratio"] > 100
        assert mem["n_cells"] == grid.n_cells


class TestSamplePointsFixture:
    def test_shape(self, sample_points):
        assert sample_points.shape == (1000, 4)
        assert sample_points.dtype == np.float32

    def test_labels_shape(self, sample_labels):
        assert sample_labels.shape == (1000,)
        assert sample_labels.dtype == np.uint8
        assert sample_labels.max() < 4


class TestLabelMappingFixture:
    def test_class_mapping_loaded(self, class_mapping):
        assert isinstance(class_mapping, dict)
        assert 40 in class_mapping
        assert class_mapping[40] == 0

    def test_bin_mapping_loaded(self, bin_mapping):
        assert isinstance(bin_mapping, dict)
        assert bin_mapping[0] == 0
        assert bin_mapping[3] == 3
        assert bin_mapping[4] == 3


class TestProjectionSmoke:
    def test_cpu_projection_runs(self, sample_points):
        from src.data.projection import compute_projection, build_range_image
        proj, ranges = compute_projection(sample_points, h=64, w=2048)
        assert proj.shape == (sample_points.shape[0], 2)
        assert proj.dtype == np.int32
        ri = build_range_image(sample_points, proj, sample_points[:, 3], ranges)
        assert ri.shape == (5, 64, 2048)

    def test_gpu_projection_parity(self, sample_points):
        pytest.importorskip("torch")
        import torch
        from src.data.projection import compute_projection, project_points_gpu
        proj_cpu, ranges_cpu = compute_projection(sample_points, h=64, w=2048)
        pts_t = torch.from_numpy(sample_points[:, :4].copy())
        if torch.cuda.is_available():
            pts_t = pts_t.to("cuda")
            row, col, r = project_points_gpu(pts_t, h=64, w=2048)
            row_np = row.cpu().numpy()
            col_np = col.cpu().numpy()
            np.testing.assert_array_equal(proj_cpu[:, 0], row_np)
            np.testing.assert_array_equal(proj_cpu[:, 1], col_np)
        else:
            row, col, r = project_points_gpu(pts_t, h=64, w=2048)
            np.testing.assert_array_equal(proj_cpu[:, 0], row.numpy())
            np.testing.assert_array_equal(proj_cpu[:, 1], col.numpy())


class TestGridGeometrySmoke:
    def test_ring_index(self, grid):
        r = np.array([0.51, 1.0, 10.0, 50.0, 99.0])
        idx = grid.ring_index(r)
        assert idx.shape == r.shape
        assert (idx >= 0).all()
        assert (idx < grid.n_rings).all()
        assert idx[0] == 0

    def test_sector_index(self, grid):
        theta = np.array([-np.pi, 0.0, np.pi])
        idx = grid.sector_index(theta)
        assert idx.shape == theta.shape
        assert (idx >= 0).all()
        assert (idx < grid.n_theta).all()

    def test_cell_ids_from_xy(self, grid, sample_points):
        ids, valid = grid.cell_ids_from_xy(sample_points[:, 0], sample_points[:, 1])
        assert ids.shape == (sample_points.shape[0],)
        assert valid.shape == (sample_points.shape[0],)
        assert (ids[valid] >= 0).all()
        assert (ids[valid] < grid.n_cells).all()
