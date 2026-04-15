#!/usr/bin/env python3
"""Standalone integrity check for an nnU-Net v2 raw dataset.

Runs all checks defined in ``IntegrityChecker`` and prints a detailed report.
Exits with code 0 when all checks pass, 1 otherwise.

Checks
------
1. ``dataset.json`` — present, schema-valid, numTraining matches actual files
2. ``imagesTr/`` — all cases have correct number of channel files
3. ``labelsTr/`` — all training cases have a label; values in expected set
4. Image ↔ label spatial shape consistency per case
5. NIfTI readability of every file
6. ``imagesTs/`` — all cases have correct channel files

Usage
-----
::

    # Check the dataset configured in .env
    python scripts/check_integrity.py

    # Check a specific dataset directory
    python scripts/check_integrity.py --dataset-dir /path/to/Dataset001_BraTSMENRT

    # Quick mode: check at most 20 cases per split
    python scripts/check_integrity.py --max-cases 20

    # Custom expected label values
    python scripts/check_integrity.py --label-values 0 1 2

    # Export full per-case report to CSV
    python scripts/check_integrity.py --csv results/integrity_report.csv

    # Exit 0 even when failures found (useful in CI to collect report only)
    python scripts/check_integrity.py --no-fail
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.env_utils import load_env, get_path_env, dataset_folder_name
from src.utils.logging_utils import get_logger
from src.data.integrity_checker import IntegrityChecker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset-dir",
        default=None,
        metavar="PATH",
        help="Path to Dataset{ID}_{NAME} directory. "
             "Auto-derived from env if omitted.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of cases to check per split (default: all).",
    )
    p.add_argument(
        "--label-values",
        nargs="+",
        type=int,
        default=[0, 1],
        metavar="INT",
        help="Expected integer label values (default: 0 1).",
    )
    p.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Write per-case report to this CSV path.",
    )
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even if integrity checks fail.",
    )
    p.add_argument("--log-dir", default="logs", help="Log file directory")
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="check_integrity", log_dir=args.log_dir)

    # ── Resolve dataset directory ─────────────────────────────────────────────
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        raw = get_path_env("nnUNet_raw", required=True)
        dataset_dir = raw / dataset_folder_name()

    if not dataset_dir.exists():
        log.error(f"Dataset directory not found: {dataset_dir}")
        log.error("Run scripts/01_prepare_dataset.py first.")
        sys.exit(1)

    log.info("=" * 62)
    log.info("  Dataset Integrity Check")
    log.info("=" * 62)
    log.info(f"  Directory     : {dataset_dir}")
    log.info(f"  Label values  : {args.label_values}")
    log.info(f"  Max cases/split: {args.max_cases or 'all'}")
    log.info("=" * 62)

    # ── Run checker ───────────────────────────────────────────────────────────
    checker = IntegrityChecker(
        dataset_dir=dataset_dir,
        expected_label_values=set(args.label_values),
    )
    report = checker.run(max_cases=args.max_cases)

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + report.summary())

    # ── Per-case failures ─────────────────────────────────────────────────────
    failed = [r for r in report.case_reports if not r.ok]
    if failed:
        log.warning(f"\nFailed cases ({len(failed)}):")
        for r in failed:
            log.warning(f"  [{r.split}] {r.case_id}")
            for err in r.errors:
                log.warning(f"      → {err}")

    # ── CSV export ────────────────────────────────────────────────────────────
    if args.csv:
        checker.export_csv(report, args.csv)
    elif failed:
        default_csv = Path("results") / "integrity_report.csv"
        checker.export_csv(report, default_csv)
        log.info(f"Failure report → {default_csv}")

    # ── Exit code ─────────────────────────────────────────────────────────────
    if report.passed:
        log.success("All integrity checks passed.")
        sys.exit(0)
    else:
        msg = (
            f"Integrity check FAILED: {report.n_failed} case(s) with errors. "
            f"dataset.json {'OK' if report.json_valid else 'INVALID'}."
        )
        if args.no_fail:
            log.warning(msg + " (--no-fail: exiting 0)")
            sys.exit(0)
        else:
            log.error(msg)
            sys.exit(1)


if __name__ == "__main__":
    main()
