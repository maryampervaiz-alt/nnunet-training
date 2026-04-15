"""Segmentation metrics for binary GTV segmentation.

Metrics
-------
- Dice Similarity Coefficient (DSC / F1)
- Hausdorff Distance — 95th percentile (HD95) and full (HD)
- Normalised Surface Distance (NSD) — BraTS 2024 official
- Precision (PPV)
- Recall (Sensitivity / TPR)
- Specificity (TNR)
- Volume Similarity
- Absolute Volume Error (ml)

All metric functions operate on boolean/integer numpy arrays (0/1) and
require an explicit voxel spacing so distances are in physical units (mm).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """All segmentation metrics for a single case."""

    case_id: str

    # Overlap
    dice: float = float("nan")          # Dice / F1
    precision: float = float("nan")     # PPV
    recall: float = float("nan")        # Sensitivity / TPR
    specificity: float = float("nan")   # TNR
    volume_similarity: float = float("nan")

    # Surface
    hd95: float = float("nan")          # 95th-percentile Hausdorff Distance (mm)
    hd: float = float("nan")            # Full (max) Hausdorff Distance (mm)
    nsd: float = float("nan")           # Normalised Surface Distance (BraTS official)

    # Volume
    pred_volume_ml: float = float("nan")
    gt_volume_ml: float = float("nan")
    abs_volume_error_ml: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)

    # Aliases kept for compatibility
    @property
    def sensitivity(self) -> float:
        return self.recall

    @property
    def f1(self) -> float:
        return self.dice


# ─── Surface geometry helpers ─────────────────────────────────────────────────

def _surface_mask(binary: np.ndarray) -> np.ndarray:
    """Boolean mask of surface voxels using full (26-connectivity) erosion."""
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    struct = generate_binary_structure(binary.ndim, binary.ndim)
    eroded = binary_erosion(binary.astype(bool), structure=struct, border_value=1)
    return binary.astype(bool) & ~eroded


class _SurfaceDistances(NamedTuple):
    """Directed and symmetric surface distances in mm."""
    pred_to_gt: np.ndarray   # distances from each pred-surface voxel to nearest gt-surface
    gt_to_pred: np.ndarray   # distances from each gt-surface voxel to nearest pred-surface
    symmetric: np.ndarray    # concatenation of both


def _compute_surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...],
) -> _SurfaceDistances | None:
    """Return surface distances using the ``surface_distance`` library if available,
    falling back to a pure-scipy implementation.

    Returns ``None`` when either surface is empty.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    # ── surface_distance library (preferred, O(N log N)) ─────────────────────
    try:
        import surface_distance as sd

        raw = sd.compute_surface_distances(gt_b, pred_b, spacing_mm=spacing_mm)
        p2g = raw["distances_pred_to_gt"]
        g2p = raw["distances_gt_to_pred"]
        return _SurfaceDistances(
            pred_to_gt=p2g,
            gt_to_pred=g2p,
            symmetric=np.concatenate([p2g, g2p]),
        )
    except (ImportError, Exception):
        pass

    # ── scipy fallback ────────────────────────────────────────────────────────
    pred_surf = _surface_mask(pred_b)
    gt_surf = _surface_mask(gt_b)

    if not pred_surf.any() or not gt_surf.any():
        return None

    gt_dt = distance_transform_edt(~gt_surf, sampling=spacing_mm)
    pred_dt = distance_transform_edt(~pred_surf, sampling=spacing_mm)

    p2g = gt_dt[pred_surf]
    g2p = pred_dt[gt_surf]
    return _SurfaceDistances(p2g, g2p, np.concatenate([p2g, g2p]))


# ─── Individual metric functions ──────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = float(np.sum(pred_b & gt_b))
    fp = float(np.sum(pred_b & ~gt_b))
    fn = float(np.sum(~pred_b & gt_b))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0 if not gt_b.any() else 0.0
    return 2 * tp / denom


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...],
    percentile: float = 100.0,
) -> float:
    """Hausdorff distance at *percentile* (100 = full HD, 95 = HD95) in mm."""
    dists = _compute_surface_distances(pred, gt, spacing_mm)
    if dists is None:
        pred_b, gt_b = pred.astype(bool), gt.astype(bool)
        return float("inf") if (pred_b.any() or gt_b.any()) else float("nan")
    return float(np.percentile(dists.symmetric, percentile))


def normalised_surface_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...],
    tolerance_mm: float = 2.0,
) -> float:
    """Normalised Surface Distance (BraTS 2024 official metric)."""
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)

    # surface_distance library has its own NSD implementation
    try:
        import surface_distance as sd

        raw = sd.compute_surface_distances(gt_b, pred_b, spacing_mm=spacing_mm)
        return float(sd.compute_surface_dice_at_tolerance(raw, tolerance_mm=tolerance_mm))
    except (ImportError, Exception):
        pass

    # scipy fallback
    pred_surf = _surface_mask(pred_b)
    gt_surf = _surface_mask(gt_b)

    if not pred_surf.any() and not gt_surf.any():
        return 1.0
    if not pred_surf.any() or not gt_surf.any():
        return 0.0

    gt_dt = distance_transform_edt(~gt_surf, sampling=spacing_mm)
    pred_dt = distance_transform_edt(~pred_surf, sampling=spacing_mm)

    n_pred_within = int(np.sum(gt_dt[pred_surf] <= tolerance_mm))
    n_gt_within = int(np.sum(pred_dt[gt_surf] <= tolerance_mm))
    denom = pred_surf.sum() + gt_surf.sum()
    return float((n_pred_within + n_gt_within) / denom) if denom > 0 else 0.0


# ─── Main compute function ────────────────────────────────────────────────────

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...] = (1.0, 1.0, 1.0),
    case_id: str = "unknown",
    nsd_tolerance_mm: float = 2.0,
) -> MetricResult:
    """Compute all segmentation metrics for one case.

    Parameters
    ----------
    pred:
        Predicted binary segmentation (H × W × D), dtype int or bool.
    gt:
        Ground-truth binary segmentation, same shape.
    spacing_mm:
        Voxel spacing in mm ordered consistently with array axes.
    case_id:
        Identifier embedded in the result record.
    nsd_tolerance_mm:
        Surface tolerance for NSD (BraTS 2024 default: 2 mm).

    Returns
    -------
    MetricResult
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    # ── Overlap counts ────────────────────────────────────────────────────────
    tp = float(np.sum(pred_b & gt_b))
    fp = float(np.sum(pred_b & ~gt_b))
    fn = float(np.sum(~pred_b & gt_b))
    tn = float(np.sum(~pred_b & ~gt_b))

    # ── Overlap metrics ───────────────────────────────────────────────────────
    denom_dice = 2 * tp + fp + fn
    if denom_dice == 0:
        dice = 1.0 if not gt_b.any() else 0.0
    else:
        dice = 2 * tp / denom_dice

    recall = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    v_pred = float(pred_b.sum())
    v_gt = float(gt_b.sum())
    vs_denom = v_pred + v_gt
    volume_similarity = (
        1.0 - abs(v_pred - v_gt) / vs_denom if vs_denom > 0 else float("nan")
    )

    # ── Surface metrics ───────────────────────────────────────────────────────
    if pred_b.any() and gt_b.any():
        dists = _compute_surface_distances(pred_b, gt_b, spacing_mm)
        if dists is not None:
            hd95_val = float(np.percentile(dists.symmetric, 95))
            hd_val = float(np.max(dists.symmetric))
        else:
            hd95_val = float("inf")
            hd_val = float("inf")
        nsd_val = normalised_surface_distance(pred_b, gt_b, spacing_mm, nsd_tolerance_mm)
    elif pred_b.any() or gt_b.any():
        hd95_val = float("inf")
        hd_val = float("inf")
        nsd_val = 0.0
    else:
        hd95_val = float("nan")
        hd_val = float("nan")
        nsd_val = 1.0

    # ── Volumes ───────────────────────────────────────────────────────────────
    voxel_vol_ml = float(np.prod(spacing_mm)) / 1000.0
    pred_vol_ml = v_pred * voxel_vol_ml
    gt_vol_ml = v_gt * voxel_vol_ml
    abs_vol_err_ml = abs(pred_vol_ml - gt_vol_ml)

    def _r(x: float, n: int = 6) -> float:
        return round(x, n) if math.isfinite(x) else x

    return MetricResult(
        case_id=case_id,
        dice=_r(dice),
        precision=_r(precision),
        recall=_r(recall),
        specificity=_r(specificity),
        volume_similarity=_r(volume_similarity),
        hd95=_r(hd95_val, 4),
        hd=_r(hd_val, 4),
        nsd=_r(nsd_val),
        pred_volume_ml=_r(pred_vol_ml, 4),
        gt_volume_ml=_r(gt_vol_ml, 4),
        abs_volume_error_ml=_r(abs_vol_err_ml, 4),
    )
