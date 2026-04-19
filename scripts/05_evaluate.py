#!/usr/bin/env python3
"""Step 5 — Evaluate predicted segmentations against ground-truth labels.

Computes per-case metrics (DSC, HD95, HD, NSD, precision, recall, specificity,
volume similarity, absolute volume error), writes per-case and aggregate CSVs,
and optionally exports a publication-ready LaTeX table plus best/worst/median
case rankings.

Modes
-----
standard (default)
    Evaluate predictions in --pred against --gt labels.
    Output → results/<tag>_*.csv  and  results/<tag>_*.tex

cv (--cv-mode)
    Evaluate each fold's held-out validation split using per-fold prediction
    directories (produced by ``04_inference.py --cv-mode``).
    Output → results/fold_N_*.csv  +  results/cv_*.csv

Usage
-----
::

    # Standard evaluation
    python scripts/05_evaluate.py \\
        --pred inference_outputs/ensemble \\
        --gt   nnunet_raw/Dataset001_BraTSMENRT/labelsTr \\
        --tag  ensemble

    # Cross-validation evaluation
    python scripts/05_evaluate.py --cv-mode

    # CV + LaTeX table
    python scripts/05_evaluate.py --cv-mode --latex

    # Show best / worst 10 cases by Dice
    python scripts/05_evaluate.py --cv-mode --show-rankings --top-n 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitter import load_splits
from src.evaluation.evaluator import SegmentationEvaluator
from src.evaluation.results_aggregator import ResultsAggregator
from src.utils.env_utils import dataset_folder_name, get_path_env, load_env
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Input / output ────────────────────────────────────────────────────────
    p.add_argument(
        "--pred",
        default=None,
        metavar="DIR",
        help="Prediction directory "
             "(default: inference_outputs/ensemble or inference_outputs/cv)",
    )
    p.add_argument(
        "--gt",
        default=None,
        metavar="DIR",
        help="Ground-truth label directory (default: nnunet_raw/<dataset>/labelsTr).",
    )
    p.add_argument("--results-dir", default="results", help="Output results directory.")
    p.add_argument("--tag", default="eval", help="Tag prefix for output file names.")

    # ── Metric options ────────────────────────────────────────────────────────
    p.add_argument("--nsd-tolerance", type=float, default=2.0, help="NSD tolerance in mm.")
    p.add_argument(
        "--low-dice",
        type=float,
        default=0.5,
        help="Dice threshold below which a case is flagged as outlier (default: 0.5).",
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--cv-mode", action="store_true", help="Evaluate each fold's validation split.")

    # ── Publication outputs ───────────────────────────────────────────────────
    p.add_argument(
        "--latex",
        action="store_true",
        help="Write a publication-ready LaTeX table to results_dir.",
    )
    p.add_argument(
        "--show-rankings",
        action="store_true",
        help="Log best / worst / median cases ranked by Dice.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of cases shown per ranking group (default: 5).",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=2000,
        metavar="N",
        help="Bootstrap resamples for 95 %% CI (0 = skip, default: 2000).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_gt_dir(args: argparse.Namespace) -> Path:
    if args.gt is not None:
        return Path(args.gt)
    raw_dir = get_path_env("nnUNet_raw", required=True)
    return raw_dir / dataset_folder_name() / "labelsTr"


def _validate_cv_consistency(splits: list[dict], log) -> None:
    """Warn if dataset has been modified since training started.

    This prevents incomparable CV results if user trains on 50 cases,
    then accidentally preprocesses the full dataset and trains folds
    on different splits.
    """
    total_cases = sum(len(f['train']) + len(f['val']) for f in splits)

    if total_cases not in (20, 50, 100, 200, 500):
        log.warning(
            f"Dataset has {total_cases} total cases across all folds. "
            f"For 5-fold CV, this should be stable across all fold training. "
            f"If folds were trained on different dataset sizes, results may be incomparable."
        )
    else:
        log.info(f"Dataset: {total_cases} cases ({total_cases // 5} per fold average)")


def _export_publication_outputs(
    df: pd.DataFrame,
    results_dir: Path,
    tag: str,
    caption: str,
    bootstrap_n: int,
    do_latex: bool,
    show_rankings: bool,
    top_n: int,
    log,
) -> None:
    """Write aggregate CSV, optional LaTeX table, optional rankings, and print summary."""
    aggregator = ResultsAggregator(
        results_dir=results_dir,
        bootstrap_n=bootstrap_n,
    )

    # Overall summary CSV
    csv_path = aggregator.export_overall_csv(df=df, tag=f"{tag}_overall")
    log.info(f"Overall summary CSV → {csv_path}")

    # Human-readable summary
    aggregator.print_summary(df=df, tag=tag.upper())

    # LaTeX table
    if do_latex:
        tex_path = aggregator.export_latex(
            df=df,
            tag=f"{tag}_results",
            caption=caption,
            label=f"tab:{tag.replace('-', '_')}_results",
        )
        log.info(f"LaTeX table → {tex_path}")

    # Rankings
    if show_rankings:
        rankings = aggregator.rank_cases(df=df, by="dice", n=top_n)
        for group, gdf in rankings.items():
            id_col = "case_id" if "case_id" in gdf.columns else gdf.columns[0]
            log.info(
                f"  [{tag.upper()}] {group.capitalize()} {top_n} cases (by Dice):\n"
                + "\n".join(
                    f"    {row[id_col]}  DSC={row['dice']:.4f}"
                    + (f"  HD95={row['hd95']:.2f}" if "hd95" in row else "")
                    for _, row in gdf.iterrows()
                )
            )
        json_path = aggregator.export_rankings_json(df=df, tag=f"{tag}_rankings", n=top_n)
        log.info(f"Rankings JSON → {json_path}")

    log.info(f"All publication outputs written to: {results_dir}")


# ── Evaluation runners ────────────────────────────────────────────────────────

def _run_standard(args: argparse.Namespace, gt_dir: Path, log) -> pd.DataFrame:
    pred_dir = Path(args.pred or "inference_outputs/ensemble")
    if not pred_dir.exists():
        log.error(f"Prediction directory not found: {pred_dir}")
        sys.exit(1)

    evaluator = SegmentationEvaluator(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        results_dir=args.results_dir,
        nsd_tolerance_mm=args.nsd_tolerance,
        low_dice_threshold=args.low_dice,
    )
    return evaluator.run(tag=args.tag)


def _run_cv(args: argparse.Namespace, gt_dir: Path, log) -> pd.DataFrame:
    """Evaluate each fold's held-out validation split."""
    pred_root = Path(args.pred or "inference_outputs/cv")

    try:
        splits = load_splits()
    except Exception as exc:
        log.error(f"Could not load splits_final.json: {exc}")
        sys.exit(1)

    # ── Warn if dataset size looks unusual ────────────────────────────────────
    _validate_cv_consistency(splits, log)

    all_dfs: list[pd.DataFrame] = []

    for fold_idx, fold in enumerate(splits):
        val_cases = sorted(fold.get("val", []))
        fold_pred_dir = pred_root / f"fold_{fold_idx}"

        if not fold_pred_dir.exists():
            log.warning(f"Fold {fold_idx} prediction dir missing: {fold_pred_dir} — skipping.")
            continue

        log.info(f"[fold_{fold_idx}] Evaluating {len(val_cases)} validation cases …")
        evaluator = SegmentationEvaluator(
            pred_dir=fold_pred_dir,
            gt_dir=gt_dir,
            results_dir=args.results_dir,
            nsd_tolerance_mm=args.nsd_tolerance,
            low_dice_threshold=args.low_dice,
        )
        df = evaluator.run(case_ids=val_cases, tag=f"fold_{fold_idx}")
        if df.empty:
            log.warning(f"Fold {fold_idx}: no results produced.")
            continue
        df["fold"] = fold_idx
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = Path(args.results_dir) / "cv_combined.csv"
    combined.to_csv(combined_path, index=False, float_format="%.6f")
    log.info(f"Combined CV results → {combined_path}")

    # Per-fold aggregate
    agg = ResultsAggregator(results_dir=Path(args.results_dir))
    agg.load_fold_csvs(pattern="fold_*_per_case.csv")
    fold_summary_path = agg.export_fold_csv(tag="cv_fold")
    log.info(f"Per-fold summary → {fold_summary_path}")

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="05_evaluate", log_dir=args.log_dir)

    gt_dir = _resolve_gt_dir(args)
    if not gt_dir.exists():
        log.error(f"Ground-truth directory not found: {gt_dir}")
        sys.exit(1)

    log.info("=" * 64)
    log.info("  Step 5: Evaluation")
    log.info("=" * 64)
    log.info(f"  Mode           : {'CV per-fold' if args.cv_mode else 'Standard'}")
    log.info(f"  Ground-truth   : {gt_dir}")
    log.info(f"  NSD tolerance  : {args.nsd_tolerance} mm")
    log.info(f"  Outlier thresh : Dice < {args.low_dice}")
    log.info(f"  LaTeX output   : {args.latex}")
    log.info(f"  Bootstrap n    : {args.bootstrap_n}")
    log.info("=" * 64)

    # ── Load and validate CV splits if in CV mode ──────────────────────────────
    if args.cv_mode:
        try:
            splits = load_splits()
            _validate_cv_consistency(splits, log)
        except Exception as exc:
            log.warning(f"Could not validate CV consistency: {exc}")

    df = _run_cv(args, gt_dir, log) if args.cv_mode else _run_standard(args, gt_dir, log)

    if df.empty:
        log.warning("Evaluation produced no results.")
        sys.exit(1)

    tag = "cv" if args.cv_mode else args.tag
    caption = (
        "Cross-validation segmentation performance on BraTS MEN-RT."
        if args.cv_mode
        else f"Segmentation performance ({args.tag})."
    )

    _export_publication_outputs(
        df=df,
        results_dir=Path(args.results_dir),
        tag=tag,
        caption=caption,
        bootstrap_n=args.bootstrap_n,
        do_latex=args.latex,
        show_rankings=args.show_rankings,
        top_n=args.top_n,
        log=log,
    )

    log.success(
        f"Evaluation complete. {len(df)} cases | "
        f"DSC={df['dice'].mean():.4f} ± {df['dice'].std():.4f} | "
        f"HD95={df['hd95'].replace([float('inf'), float('-inf')], float('nan')).mean():.2f} | "
        f"Precision={df['precision'].mean():.4f} | "
        f"Recall={df['recall'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
