import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.label_mapping import (
    bin_5_to_4,
    compute_class_weights,
    load_bin_mapping,
    load_class_mapping,
    remap_labels,
)


class TestRemapLabels:
    def test_known_raw_ids(self, class_mapping):
        raw = np.array([40, 50, 11], dtype=np.uint8)
        mapped = remap_labels(raw, class_mapping)
        assert mapped[0] == 0
        assert mapped[1] == 2
        assert mapped[2] == 4

    def test_ignore_index_for_unlabeled(self, class_mapping):
        raw = np.array([0, 1], dtype=np.uint8)
        mapped = remap_labels(raw, class_mapping)
        assert (mapped == 255).all()

    def test_raw_255_maps_to_class_4(self, class_mapping):
        raw = np.array([255], dtype=np.uint8)
        mapped = remap_labels(raw, class_mapping)
        assert mapped[0] == 4

    def test_large_array(self, class_mapping):
        raw = np.full(10000, 40, dtype=np.uint8)
        mapped = remap_labels(raw, class_mapping)
        assert (mapped == 0).all()


class TestBin5To4:
    def test_dynamic_merge(self, bin_mapping):
        labels = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        binned = bin_5_to_4(labels, bin_mapping)
        np.testing.assert_array_equal(binned, [0, 1, 2, 3, 3])

    def test_identity_for_non_dynamic(self, bin_mapping):
        labels = np.array([0, 1, 2], dtype=np.uint8)
        binned = bin_5_to_4(labels, bin_mapping)
        np.testing.assert_array_equal(binned, [0, 1, 2])

    def test_ignore_index_preserved(self, bin_mapping):
        labels = np.array([255], dtype=np.uint8)
        binned = bin_5_to_4(labels, bin_mapping)
        assert binned[0] == 255

    def test_lut_consistency(self, bin_mapping):
        from src.data.label_mapping import bin_lut
        _BIN_LUT_BACKUP = None
        import src.data.label_mapping as _mod
        if hasattr(_mod, "_BIN_LUT") and _mod._BIN_LUT is not None:
            _BIN_LUT_BACKUP = _mod._BIN_LUT
            _mod._BIN_LUT = None
        try:
            lut = bin_lut(bin_mapping)
            assert lut.shape == (256,)
            assert lut[0] == 0
            assert lut[3] == 3
            assert lut[4] == 3
            assert lut[255] == 255
        finally:
            if _BIN_LUT_BACKUP is not None:
                _mod._BIN_LUT = _BIN_LUT_BACKUP


class TestComputeClassWeights:
    def test_synth_labels_no_nan_in_range(self, class_mapping):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n_points = 2000
            rng = np.random.default_rng(42)
            raw_ids = [40, 48, 50, 11, 255]
            raw_labels = np.array(rng.choice(raw_ids, n_points), dtype=np.uint32)
            raw_labels = (raw_labels & 0xFFFF).astype(np.uint32)
            raw_labels.tofile(str(td / "000000.label"))
            weights = compute_class_weights(td, class_mapping, num_classes=5)
            assert weights.shape == (5,)
            assert not np.any(np.isnan(weights))
            assert (weights >= 0.1).all()
            assert (weights <= 5.0).all()
