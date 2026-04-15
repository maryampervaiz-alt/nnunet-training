"""Cross-fold results aggregation utilities."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

_PUB_METRICS = [
    "dice",
    "hd95",
    "hd",
    "precision",
    "recall",
    "specificity",
    "nsd",
    "volume_similarity",
    "abs_volume_error_ml",
]

_DISPLAY_NAMES = {
    "dice": "DSC",
    "hd95": "HD95 (mm)",
    "hd": "HD (mm)",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "nsd": "NSD",
    "volume_similarity": "Volume similarity",
    "abs_volume_error_ml": "Abs. volume error (ml)",
}

_HIGHER_IS_BETTER = {
    "dice",
    "precision",
    "recall",
    "specificity",
    "nsd",
    "volume_similarity",
}


def _clean_values(series: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
        .to_numpy(dtype=float)
    )


def _bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    samples = np.array(
        [
            statistic(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_resamples)
        ],
        dtype=float,
    )
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(samples, alpha * 100.0)),
        float(np.percentile(samples, (1.0 - alpha) * 100.0)),
    )


class ResultsAggregator:
    """Aggregate per-case metrics across folds and export summaries."""

    def __init__(
        self,
        results_dir: str | Path = "results",
        metrics: list[str] | None = None,
        bootstrap_n: int = 2000,
        ci_level: float = 0.95,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or list(_PUB_METRICS)
        self.bootstrap_n = bootstrap_n
        self.ci_level = ci_level

        self._fold_dfs: dict[str, pd.DataFrame] = {}
        self._combined: pd.DataFrame | None = None

    def load_fold_csvs(self, pattern: str = "fold_*_per_case.csv") -> None:
        csvs = sorted(self.results_dir.glob(pattern))
        if not csvs:
            logger.warning(f"No CSVs matching '{pattern}' in {self.results_dir}")
            return
        for csv_path in csvs:
            label = csv_path.stem.replace("_per_case", "")
            self._fold_dfs[label] = pd.read_csv(csv_path)
            logger.debug(f"Loaded {label}: {len(self._fold_dfs[label])} rows")

    def load_combined_csv(self, path: str | Path | None = None) -> None:
        csv_path = Path(path) if path is not None else self.results_dir / "cv_combined.csv"
        if not csv_path.exists():
            logger.warning(f"Combined CSV not found: {csv_path}")
            return
        self._combined = pd.read_csv(csv_path)
        logger.debug(f"Loaded combined CSV: {len(self._combined)} rows")

    def set_dataframe(self, df: pd.DataFrame, label: str = "all") -> None:
        self._fold_dfs[label] = df.copy()

    def fold_summary(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fold_label, df in self._fold_dfs.items():
            row: dict[str, object] = {"fold": fold_label, "n_cases": len(df)}
            for metric in self.metrics:
                if metric not in df.columns:
                    continue
                values = _clean_values(df[metric])
                row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
                row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                )
                row[f"{metric}_median"] = (
                    float(np.median(values)) if len(values) else float("nan")
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()

    def overall_summary(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        source = self._resolve_dataframe(df)
        if source is None:
            logger.error("No data loaded. Provide a DataFrame or load result CSVs first.")
            return pd.DataFrame()

        available = [metric for metric in self.metrics if metric in source.columns]
        stats: dict[str, dict[str, float]] = {
            "mean": {},
            "std": {},
            "median": {},
            "q25": {},
            "q75": {},
            "min": {},
            "max": {},
        }
        if self.bootstrap_n > 0:
            stats["ci95_lo"] = {}
            stats["ci95_hi"] = {}

        for metric in available:
            values = _clean_values(source[metric])
            stats["mean"][metric] = float(np.mean(values)) if len(values) else float("nan")
            stats["std"][metric] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
            stats["median"][metric] = float(np.median(values)) if len(values) else float("nan")
            stats["q25"][metric] = (
                float(np.percentile(values, 25.0)) if len(values) else float("nan")
            )
            stats["q75"][metric] = (
                float(np.percentile(values, 75.0)) if len(values) else float("nan")
            )
            stats["min"][metric] = float(np.min(values)) if len(values) else float("nan")
            stats["max"][metric] = float(np.max(values)) if len(values) else float("nan")
            if self.bootstrap_n > 0:
                lo, hi = _bootstrap_ci(
                    values,
                    n_resamples=self.bootstrap_n,
                    ci=self.ci_level,
                )
                stats["ci95_lo"][metric] = lo
                stats["ci95_hi"][metric] = hi

        return pd.DataFrame(stats).T[available]

    def rank_cases(
        self,
        df: pd.DataFrame | None = None,
        by: str = "dice",
        n: int = 5,
    ) -> dict[str, pd.DataFrame]:
        source = self._resolve_dataframe(df)
        if source is None or by not in source.columns:
            return {}

        ranked = source.dropna(subset=[by]).sort_values(
            by,
            ascending=by not in _HIGHER_IS_BETTER,
        )
        if ranked.empty:
            return {}

        mid = len(ranked) // 2
        half_window = max(1, n) // 2
        return {
            "best": ranked.head(n),
            "worst": ranked.tail(n).iloc[::-1],
            "median": ranked.iloc[max(0, mid - half_window): mid + half_window + 1],
        }

    def export_overall_csv(
        self,
        df: pd.DataFrame | None = None,
        tag: str = "overall",
    ) -> Path:
        summary = self.overall_summary(df)
        path = self.results_dir / f"{tag}_summary.csv"
        summary.to_csv(path, float_format="%.6f")
        logger.info(f"Overall summary CSV -> {path}")
        return path

    def export_fold_csv(self, tag: str = "fold") -> Path:
        summary = self.fold_summary()
        path = self.results_dir / f"{tag}_summary.csv"
        summary.to_csv(path, float_format="%.6f")
        logger.info(f"Fold summary CSV -> {path}")
        return path

    def to_latex(
        self,
        df: pd.DataFrame | None = None,
        caption: str = "Segmentation results.",
        label: str = "tab:results",
        include_ci: bool = True,
    ) -> str:
        summary = self.overall_summary(df)
        if summary.empty:
            return ""

        available = [metric for metric in self.metrics if metric in summary.columns]
        header_names = [_DISPLAY_NAMES.get(metric, metric) for metric in available]

        def _fmt(value: float, decimals: int = 3) -> str:
            return f"{value:.{decimals}f}" if math.isfinite(value) else "--"

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\small",
            r"\begin{tabular}{l" + "c" * len(available) + "}",
            r"\toprule",
            "Statistic & " + " & ".join(header_names) + r" \\",
            r"\midrule",
        ]

        mean_cells = []
        for metric in available:
            mean_cells.append(
                f"{_fmt(summary.at['mean', metric])} $\\pm$ {_fmt(summary.at['std', metric])}"
            )
        lines.append(r"Mean $\pm$ Std & " + " & ".join(mean_cells) + r" \\")

        median_cells = []
        for metric in available:
            median_cells.append(
                f"{_fmt(summary.at['median', metric])} "
                f"[{_fmt(summary.at['q25', metric])}, {_fmt(summary.at['q75', metric])}]"
            )
        lines.append(r"Median [IQR] & " + " & ".join(median_cells) + r" \\")

        if include_ci and "ci95_lo" in summary.index:
            ci_cells = []
            for metric in available:
                ci_cells.append(
                    f"[{_fmt(summary.at['ci95_lo', metric])}, "
                    f"{_fmt(summary.at['ci95_hi', metric])}]"
                )
            lines.append(r"95\% CI & " + " & ".join(ci_cells) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def export_latex(
        self,
        df: pd.DataFrame | None = None,
        tag: str = "results",
        **kwargs,
    ) -> Path:
        latex = self.to_latex(df=df, **kwargs)
        path = self.results_dir / f"{tag}_table.tex"
        path.write_text(latex, encoding="utf-8")
        logger.info(f"LaTeX table -> {path}")
        return path

    def export_rankings_json(
        self,
        df: pd.DataFrame | None = None,
        by: str = "dice",
        n: int = 10,
        tag: str = "rankings",
    ) -> Path:
        rankings = self.rank_cases(df=df, by=by, n=n)
        payload: dict[str, list[str]] = {}
        for group, group_df in rankings.items():
            payload[group] = (
                group_df["case_id"].astype(str).tolist() if "case_id" in group_df.columns else []
            )
        path = self.results_dir / f"{tag}_by_{by}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(f"Rankings JSON -> {path}")
        return path

    def print_summary(self, df: pd.DataFrame | None = None, tag: str = "Results") -> None:
        summary = self.overall_summary(df)
        if summary.empty:
            return

        available = [metric for metric in self.metrics if metric in summary.columns]
        lines = [f"[{tag}]"]
        for metric in available:
            name = _DISPLAY_NAMES.get(metric, metric)
            mean = summary.at["mean", metric]
            std = summary.at["std", metric]
            median = summary.at["median", metric]
            q25 = summary.at["q25", metric]
            q75 = summary.at["q75", metric]
            lines.append(
                f"  {name:<24} mean={mean:.4f} std={std:.4f} "
                f"median={median:.4f} iqr=[{q25:.4f}, {q75:.4f}]"
            )
        logger.info("\n".join(lines))

    def _resolve_dataframe(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is not None:
            return df
        if self._combined is not None:
            return self._combined
        if self._fold_dfs:
            return pd.concat(self._fold_dfs.values(), ignore_index=True)
        return None
