#!/usr/bin/env python3
"""Step 6 — Generate publication-ready visualizations.

Produces:
  overlays/          Prediction vs GT contour overlays with TP/FP/FN difference maps.
                     Cases are sampled at best / worst / median Dice when a
                     metrics CSV is available, otherwise randomly.
  metrics_boxplot.png     Boxplot of DSC, HD95, NSD, precision, recall.
  metrics_violin.png      Violin + strip plot of the same metrics.
  volume_scatter.png      Predicted vs GT volume scatter coloured by DSC.
  fold_comparison_*.png   Per-fold bar charts (DSC, HD95, etc.) from CV results.
  training_curve_*.png    Per-fold loss + Dice training curves.
  all_folds_training.png  All-folds Dice overlay on a single axes.

Usage
-----
::

    # All visualizations (CV mode)
    python scripts/06_visualize.py --all --cv-mode

    # Overlay plots only (best/worst/median by Dice)
    python scripts/06_visualize.py --overlays --cv-mode

    # Metric plots from a specific CSV
    python scripts/06_visualize.py --metric-plots --results-csv results/cv_combined.csv

    # Training curves from metrics CSV written by the training pipeline
    python scripts/06_visualize.py --training-curves --metrics-dir metrics

    # Fold comparison bar charts
    python scripts/06_visualize.py --fold-comparison
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitter import load_case_ids
from src.evaluation.results_aggregator import ResultsAggregator
from src.utils.env_utils import dataset_folder_name, get_path_env, load_env
from src.utils.logging_utils import get_logger
from src.visualization.plotter import (
    SegmentationPlotter,
    plot_all_folds_training,
    plot_fold_comparison,
    plot_metrics_boxplot,
    plot_metrics_violin,
    plot_training_curve,
    plot_volume_scatter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Visualization selection ───────────────────────────────────────────────
    p.add_argument("--all", action="store_true", help="Run all visualizations.")
    p.add_argument("--overlays", action="store_true", help="Generate overlay plots.")
    p.add_argument("--metric-plots", action="store_true", help="Boxplot + violin + volume scatter.")
    p.add_argument("--fold-comparison", action="store_true", help="Per-fold bar charts.")
    p.add_argument("--training-curves", action="store_true", help="Training loss/Dice curves.")

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--cv-mode",
        action="store_true",
        help="Use CV predictions and combined results by default.",
    )

    # ── Overlay options ───────────────────────────────────────────────────────
    p.add_argument(
        "--n-cases",
        type=int,
        default=15,
        help="Total overlay cases (best+worst+median split equally, default: 15).",
    )
    p.add_argument(
        "--random-cases",
        action="store_true",
        help="Sample overlay cases randomly instead of best/worst/median.",
    )
    p.add_argument(
        "--pred-dir",
        default=None,
        metavar="DIR",
        help="Prediction directory (default: inferred from --cv-mode).",
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument(
        "--results-csv",
        default=None,
        metavar="CSV",
        help="Per-case metrics CSV (default: inferred from --cv-mode).",
    )
    p.add_argument("--results-dir", default="results", help="Results directory.")
    p.add_argument("--metrics-dir", default="metrics", help="Training metrics CSV directory.")
    p.add_argument("--output-dir", default="visualizations", help="Visualization output root.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--num-slices", type=int, default=5, help="Axial slices per overlay case.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


# ── Overlay plots ─────────────────────────────────────────────────────────────

def _select_overlay_cases(
    metrics_df: pd.DataFrame | None,
    n_total: int,
    random_sample: bool,
    seed: int,
) -> list[str]:
    """Return case IDs sorted by Dice (best/worst/median) or randomly."""
    if metrics_df is None or "case_id" not in metrics_df.columns:
        return []

    rng = random.Random(seed)
    all_cases = metrics_df["case_id"].dropna().astype(str).tolist()

    if random_sample or "dice" not in metrics_df.columns:
        return rng.sample(all_cases, min(n_total, len(all_cases)))

    aggregator = ResultsAggregator()
    aggregator.set_dataframe(metrics_df)
    n_per_group = max(1, n_total // 3)
    rankings = aggregator.rank_cases(metrics_df, by="dice", n=n_per_group)

    selected: list[str] = []
    for group in ("best", "median", "worst"):
        gdf = rankings.get(group, pd.DataFrame())
        if not gdf.empty:
            selected += gdf["case_id"].tolist()

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:n_total]


def _run_overlays(
    args: argparse.Namespace,
    pred_dir: Path,
    metrics_df: pd.DataFrame | None,
    log,
) -> None:
    raw_dir = get_path_env("nnUNet_raw", required=True)
    dataset_dir = raw_dir / dataset_folder_name()
    image_dir = dataset_dir / "imagesTr"
    gt_dir = dataset_dir / "labelsTr"

    if not image_dir.exists():
        log.warning(f"imagesTr not found: {image_dir} — skipping overlays.")
        return

    case_ids = _select_overlay_cases(
        metrics_df=metrics_df,
        n_total=args.n_cases,
        random_sample=args.random_cases,
        seed=args.seed,
    )

    if not case_ids:
        # Fallback: random sample from training split
        all_tr = load_case_ids(split="train")
        random.seed(args.seed)
        case_ids = random.sample(all_tr, min(args.n_cases, len(all_tr))) if all_tr else []

    if not case_ids:
        log.warning("No cases available for overlay plots.")
        return

    mode = "random" if args.random_cases else "best/median/worst by Dice"
    log.info(f"Generating {len(case_ids)} overlay plots ({mode}) …")

    plotter = SegmentationPlotter(
        output_dir=Path(args.output_dir) / "overlays",
        num_slices=args.num_slices,
    )
    paths = plotter.plot_batch(
        case_ids=case_ids,
        image_dir=image_dir,
        pred_dir=pred_dir if pred_dir.exists() else None,
        gt_dir=gt_dir if gt_dir.exists() else None,
        metrics_df=metrics_df,
    )
    log.success(f"Saved {len(paths)} overlay plots → {plotter.output_dir}")


# ── Metric plots ──────────────────────────────────────────────────────────────

def _run_metric_plots(
    args: argparse.Namespace,
    metrics_df: pd.DataFrame | None,
    log,
) -> None:
    if metrics_df is None or metrics_df.empty:
        log.warning("No metrics DataFrame — skipping metric plots. Run Step 5 first.")
        return

    out_dir = Path(args.output_dir)
    stem = Path(args.results_csv).stem if args.results_csv else "metrics"

    plot_metrics_boxplot(
        df=metrics_df,
        output_path=out_dir / "metrics_boxplot.png",
        title=f"Segmentation Metrics — {stem}",
    )
    plot_metrics_violin(
        df=metrics_df,
        output_path=out_dir / "metrics_violin.png",
        title=f"Segmentation Metric Distributions — {stem}",
    )
    plot_volume_scatter(
        df=metrics_df,
        output_path=out_dir / "volume_scatter.png",
        title=f"Predicted vs Ground-Truth Volume — {stem}",
    )
    log.success(f"Metric plots saved → {out_dir}")


# ── Fold comparison ───────────────────────────────────────────────────────────

def _run_fold_comparison(args: argparse.Namespace, log) -> None:
    results_dir = Path(args.results_dir)
    fold_csvs = sorted(results_dir.glob("fold_*_per_case.csv"))

    if not fold_csvs:
        log.warning("No per-fold CSV files found. Run Step 5 with --cv-mode first.")
        return

    fold_dfs = {
        csv_f.stem.replace("_per_case", ""): pd.read_csv(csv_f)
        for csv_f in fold_csvs
    }
    out_dir = Path(args.output_dir)
    for metric in ["dice", "hd95", "nsd", "precision", "recall"]:
        plot_fold_comparison(
            fold_dfs=fold_dfs,
            metric=metric,
            output_path=out_dir / f"fold_comparison_{metric}.png",
        )
    log.success(f"Fold comparison plots → {out_dir}")


# ── Training curves ───────────────────────────────────────────────────────────

def _run_training_curves(args: argparse.Namespace, log) -> None:
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)

    # All-folds combined CSV written by CrossValidationOrchestrator
    all_folds_csv = metrics_dir / "all_folds_training.csv"
    if all_folds_csv.exists():
        plot_all_folds_training(
            all_folds_csv=all_folds_csv,
            output_path=out_dir / "all_folds_training.png",
        )
        log.success(f"All-folds training curve → {out_dir}/all_folds_training.png")

    # Per-fold CSVs: fold_0_training.csv, fold_1_training.csv, …
    fold_csvs = sorted(metrics_dir.glob("fold_*_training.csv"))
    for csv_path in fold_csvs:
        fold_label = csv_path.stem.replace("_training", "")
        try:
            fold_int = int(fold_label.split("_")[-1])
        except ValueError:
            fold_int = None

        out_path = out_dir / f"training_curve_{fold_label}.png"
        plot_training_curve(
            progress_csv=csv_path,
            output_path=out_path,
            fold=fold_int,
        )
        log.info(f"  Training curve {fold_label} → {out_path}")

    if not all_folds_csv.exists() and not fold_csvs:
        log.warning(f"No training CSVs found in {metrics_dir}. Run Step 3 first.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="06_visualize", log_dir=args.log_dir)

    do_overlays = args.all or args.overlays
    do_metrics = args.all or args.metric_plots
    do_fold_cmp = args.all or args.fold_comparison
    do_training = args.all or args.training_curves

    if not any([do_overlays, do_metrics, do_fold_cmp, do_training]):
        log.warning("No visualization mode selected. Use --all or a specific flag.")
        log.warning("  --overlays       GT/pred contour overlays (best/worst/median cases)")
        log.warning("  --metric-plots   Boxplot, violin, volume scatter")
        log.warning("  --fold-comparison  Per-fold bar charts")
        log.warning("  --training-curves  Loss + Dice training curves")
        sys.exit(0)

    # Resolve default paths based on mode
    pred_dir = Path(
        args.pred_dir
        or ("inference_outputs/cv/combined" if args.cv_mode else "inference_outputs/ensemble")
    )
    results_csv = args.results_csv or (
        str(Path(args.results_dir) / "cv_combined.csv")
        if args.cv_mode
        else str(Path(args.results_dir) / f"eval_per_case.csv")
    )

    # Auto-discover latest per-case CSV if default not found
    if not Path(results_csv).exists():
        candidates = sorted(Path(args.results_dir).glob("*_per_case.csv"))
        if candidates:
            results_csv = str(candidates[-1])
            log.info(f"Auto-selected results CSV: {results_csv}")

    metrics_df = pd.read_csv(results_csv) if Path(results_csv).exists() else None

    log.info("=" * 64)
    log.info("  Step 6: Visualization")
    log.info("=" * 64)
    log.info(f"  Mode            : {'CV' if args.cv_mode else 'Standard'}")
    log.info(f"  Predictions dir : {pred_dir}")
    log.info(f"  Results CSV     : {results_csv}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info(f"  Cases (overlays): {args.n_cases}")
    log.info("=" * 64)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if do_overlays:
        _run_overlays(args=args, pred_dir=pred_dir, metrics_df=metrics_df, log=log)

    if do_metrics:
        _run_metric_plots(args=args, metrics_df=metrics_df, log=log)

    if do_fold_cmp:
        _run_fold_comparison(args=args, log=log)

    if do_training:
        _run_training_curves(args=args, log=log)

    log.success(f"Visualization complete. All figures in: {args.output_dir}/")


if __name__ == "__main__":
    main()
