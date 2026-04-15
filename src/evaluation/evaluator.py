"""Batch segmentation evaluator.

Handles:
  - Automatic voxel spacing from NIfTI headers
  - Per-case metric computation (parallel-ready, single-threaded by default)
  - Outlier tagging (cases with Dice below a threshold)
  - Per-tag and per-fold CSV export
  - Publication-ready aggregate table (mean ± std, median [IQR])
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .metrics import MetricResult, compute_metrics

# Columns in display order for the summary table
_SUMMARY_COLS = ["dice", "hd95", "hd", "nsd", "precision", "recall",
                 "specificity", "volume_similarity", "abs_volume_error_ml"]


class SegmentationEvaluator:
    """Evaluate predicted segmentations against ground-truth labels.

    Parameters
    ----------
    pred_dir:
        Directory with predicted ``{case_id}.nii.gz`` files.
    gt_dir:
        Directory with ground-truth ``{case_id}.nii.gz`` files.
    results_dir:
        Output directory for CSV files.
    nsd_tolerance_mm:
        NSD surface tolerance (BraTS 2024 default: 2 mm).
    low_dice_threshold:
        Cases with Dice below this value are flagged as outliers.
    """

    def __init__(
        self,
        pred_dir: str | Path,
        gt_dir: str | Path,
        results_dir: str | Path = "results",
        nsd_tolerance_mm: float = 2.0,
        low_dice_threshold: float = 0.5,
    ) -> None:
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.nsd_tolerance_mm = nsd_tolerance_mm
        self.low_dice_threshold = low_dice_threshold
        self._records: list[MetricResult] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        case_ids: list[str] | None = None,
        tag: str = "eval",
    ) -> pd.DataFrame:
        """Evaluate cases and write CSVs.

        Parameters
        ----------
        case_ids:
            Explicit list of case IDs.  Auto-discovered from ``pred_dir``
            when ``None``.
        tag:
            Prefix for output file names.

        Returns
        -------
        pd.DataFrame
            Per-case metrics table (one row per case).
        """
        if case_ids is None:
            case_ids = self._discover_cases()

        if not case_ids:
            logger.warning(f"[{tag}] No cases to evaluate.")
            return pd.DataFrame()

        logger.info(f"[{tag}] Evaluating {len(case_ids)} cases …")
        self._records = []

        for case_id in tqdm(case_ids, desc=f"Eval [{tag}]"):
            pred_path = self.pred_dir / f"{case_id}.nii.gz"
            gt_path = self.gt_dir / f"{case_id}.nii.gz"

            if not pred_path.exists():
                logger.warning(f"[{case_id}] Prediction not found: {pred_path.name}")
                continue
            if not gt_path.exists():
                logger.warning(f"[{case_id}] Ground-truth not found: {gt_path.name}")
                continue

            try:
                result = self._evaluate_case(case_id, pred_path, gt_path)
                self._records.append(result)
            except Exception as exc:
                logger.error(f"[{case_id}] Metric computation failed: {exc}")

        df = pd.DataFrame([r.to_dict() for r in self._records])
        if df.empty:
            return df

        # Tag outliers
        df["outlier"] = df["dice"] < self.low_dice_threshold

        # Write outputs
        self._write_per_case_csv(df, tag)
        self._write_aggregate_csv(df, tag)
        self._log_summary(df, tag)

        return df

    # ── Internals ─────────────────────────────────────────────────────────────

    def _discover_cases(self) -> list[str]:
        return sorted(
            p.name.replace(".nii.gz", "")
            for p in self.pred_dir.glob("*.nii.gz")
            if not p.name.startswith(".")
        )

    def _evaluate_case(
        self, case_id: str, pred_path: Path, gt_path: Path
    ) -> MetricResult:
        pred_nib = nib.load(pred_path)
        gt_nib = nib.load(gt_path)

        pred_arr = np.asarray(pred_nib.dataobj).astype(np.uint8)
        gt_arr = np.asarray(gt_nib.dataobj).astype(np.uint8)

        spacing = tuple(float(s) for s in pred_nib.header.get_zooms()[:3])

        result = compute_metrics(
            pred=pred_arr,
            gt=gt_arr,
            spacing_mm=spacing,
            case_id=case_id,
            nsd_tolerance_mm=self.nsd_tolerance_mm,
        )
        logger.debug(
            f"  {case_id}: DSC={result.dice:.4f} | "
            f"HD95={result.hd95:.2f}mm | NSD={result.nsd:.4f} | "
            f"Rec={result.recall:.4f} | Prec={result.precision:.4f}"
        )
        return result

    def _write_per_case_csv(self, df: pd.DataFrame, tag: str) -> Path:
        path = self.results_dir / f"{tag}_per_case.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        logger.info(f"Per-case CSV → {path}")
        return path

    def _write_aggregate_csv(self, df: pd.DataFrame, tag: str) -> Path:
        cols = [c for c in _SUMMARY_COLS if c in df.columns]
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")

        rows: list[dict] = []
        for stat_name, func in [
            ("mean", lambda x: x.mean()),
            ("std", lambda x: x.std()),
            ("median", lambda x: x.median()),
            ("q25", lambda x: x.quantile(0.25)),
            ("q75", lambda x: x.quantile(0.75)),
            ("min", lambda x: x.min()),
            ("max", lambda x: x.max()),
        ]:
            row = {"statistic": stat_name}
            row.update(func(numeric).to_dict())
            rows.append(row)

        agg_df = pd.DataFrame(rows).set_index("statistic")
        path = self.results_dir / f"{tag}_aggregate.csv"
        agg_df.to_csv(path, float_format="%.6f")
        logger.info(f"Aggregate CSV → {path}")
        return path

    def _log_summary(self, df: pd.DataFrame, tag: str) -> None:
        cols = [c for c in ["dice", "hd95", "nsd", "precision", "recall"] if c in df.columns]
        lines = [f"[{tag}] Results ({len(df)} cases):"]
        for col in cols:
            vals = df[col].dropna()
            # Replace inf with NaN for stats
            vals = vals.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if vals.empty:
                lines.append(f"  {col:<25} all NaN/inf")
            else:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                lines.append(
                    f"  {col:<25} {vals.mean():.4f} ± {vals.std():.4f}  "
                    f"[{vals.median():.4f}]  IQR=[{q1:.4f}, {q3:.4f}]"
                )
        n_out = int(df["outlier"].sum()) if "outlier" in df.columns else 0
        if n_out:
            outliers = df[df["outlier"]]["case_id"].tolist()
            lines.append(f"  Outliers (Dice<{self.low_dice_threshold}): {outliers}")
        logger.info("\n".join(lines))
