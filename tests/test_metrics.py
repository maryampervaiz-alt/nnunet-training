"""Unit tests for segmentation metrics.

Run with:
    pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_metrics, MetricResult


SPACING = (1.0, 1.0, 1.0)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_sphere(shape: tuple, radius: int, center: tuple | None = None) -> np.ndarray:
    """Create a binary sphere mask."""
    if center is None:
        center = tuple(s // 2 for s in shape)
    grid = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
    return (dist <= radius).astype(np.uint8)


# ─── Perfect prediction ───────────────────────────────────────────────────────

class TestPerfectPrediction:
    def setup_method(self):
        self.gt = _make_sphere((64, 64, 64), radius=10)
        self.pred = self.gt.copy()

    def test_dice_is_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.dice == pytest.approx(1.0, abs=1e-5)

    def test_hd95_is_zero(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.hd95 == pytest.approx(0.0, abs=1.0)

    def test_nsd_is_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING, nsd_tolerance_mm=2.0)
        assert result.nsd == pytest.approx(1.0, abs=1e-3)

    def test_sensitivity_is_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.sensitivity == pytest.approx(1.0, abs=1e-5)

    def test_precision_is_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.precision == pytest.approx(1.0, abs=1e-5)


# ─── No prediction ───────────────────────────────────────────────────────────

class TestEmptyPrediction:
    def setup_method(self):
        self.gt = _make_sphere((64, 64, 64), radius=10)
        self.pred = np.zeros_like(self.gt)

    def test_dice_is_zero(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.dice == pytest.approx(0.0, abs=1e-5)

    def test_nsd_is_zero(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.nsd == pytest.approx(0.0, abs=1e-5)

    def test_sensitivity_is_zero(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.sensitivity == pytest.approx(0.0, abs=1e-5)


# ─── Both empty ───────────────────────────────────────────────────────────────

class TestBothEmpty:
    def setup_method(self):
        self.gt = np.zeros((64, 64, 64), dtype=np.uint8)
        self.pred = np.zeros_like(self.gt)

    def test_dice_is_one_for_empty_both(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.dice == pytest.approx(1.0, abs=1e-5)

    def test_nsd_is_one_for_empty_both(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.nsd == pytest.approx(1.0, abs=1e-5)


# ─── Partial overlap ─────────────────────────────────────────────────────────

class TestPartialOverlap:
    def setup_method(self):
        shape = (64, 64, 64)
        center = (32, 32, 32)
        self.gt = _make_sphere(shape, radius=10, center=center)
        # Shift prediction by 5 voxels
        self.pred = _make_sphere(shape, radius=10, center=(37, 32, 32))

    def test_dice_between_zero_and_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert 0.0 < result.dice < 1.0

    def test_hd95_positive(self):
        result = compute_metrics(self.pred, self.gt, SPACING)
        assert result.hd95 > 0.0

    def test_nsd_between_zero_and_one(self):
        result = compute_metrics(self.pred, self.gt, SPACING, nsd_tolerance_mm=2.0)
        assert 0.0 <= result.nsd <= 1.0


# ─── MetricResult ─────────────────────────────────────────────────────────────

def test_metric_result_to_dict():
    gt = _make_sphere((32, 32, 32), radius=5)
    pred = gt.copy()
    result = compute_metrics(pred, gt, SPACING, case_id="test_case")
    d = result.to_dict()
    assert d["case_id"] == "test_case"
    assert "dice" in d
    assert "hd95" in d
    assert "nsd" in d


# ─── Anisotropic spacing ─────────────────────────────────────────────────────

def test_anisotropic_spacing():
    gt = _make_sphere((64, 64, 64), radius=10)
    pred = gt.copy()
    result = compute_metrics(pred, gt, spacing_mm=(1.0, 1.0, 1.5))
    assert result.dice == pytest.approx(1.0, abs=1e-5)
