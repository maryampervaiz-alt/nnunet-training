#!/usr/bin/env python3
"""Step 1 — Prepare the BraTS MEN-RT dataset for nnU-Net v2.

What this script does
---------------------
1. Accepts training and validation sources as either **zip files** or
   **extracted directories** (auto-detected).
2. Auto-discovers all modality suffixes from the source files (no hardcoding).
3. Writes files into nnU-Net raw format:
      imagesTr/  {case_id}_{channel:04d}.nii.gz
      labelsTr/  {case_id}.nii.gz
      imagesTs/  {case_id}_{channel:04d}.nii.gz
4. Writes a sidecar ``.channel_map.json`` so the dataset.json builder can
   recover human-readable channel names.
5. Auto-generates a valid ``dataset.json`` (nnU-Net v2 schema).
6. Runs a quick integrity check and exits non-zero on failures.

Environment variables (set in .env)
------------------------------------
Required:
    nnUNet_raw          Root for nnU-Net raw datasets
    DATASET_ID          Integer ID, e.g. 001
    DATASET_NAME        Dataset name, e.g. BraTSMENRT

Optional (can be passed as CLI args instead):
    TRAIN_SOURCE        Path to training zip or directory
    VAL_SOURCE          Path to validation zip or directory
    LABEL_SUFFIX        File suffix identifying the label (default: gtv)

Usage
-----
::

    # Use env vars for paths
    python scripts/01_prepare_dataset.py

    # Override paths on CLI
    python scripts/01_prepare_dataset.py \\
        --train  "BraTs MEN-RT/BraTS2024-MEN-RT-TrainingData.zip" \\
        --val    "BraTs MEN-RT/BraTS2024-MEN-RT-ValidationData.zip" \\
        --label-suffix gtv

    # Overwrite existing files
    python scripts/01_prepare_dataset.py --overwrite

    # Skip validation set
    python scripts/01_prepare_dataset.py --skip-val

    # Skip integrity check after conversion
    python scripts/01_prepare_dataset.py --skip-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.env_utils import load_env, get_env, get_path_env, dataset_folder_name
from src.utils.logging_utils import get_logger
from src.data.converter import BraTSMENRTConverter
from src.data.dataset_json import build_dataset_json, write_channel_map_sidecar
from src.data.integrity_checker import IntegrityChecker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--train",
        default=None,
        metavar="PATH",
        help="Training source: zip file or extracted directory. "
             "Falls back to $TRAIN_SOURCE env var.",
    )
    p.add_argument(
        "--val",
        default=None,
        metavar="PATH",
        help="Validation source: zip file or extracted directory. "
             "Falls back to $VAL_SOURCE env var.",
    )
    p.add_argument(
        "--label-suffix",
        default=None,
        metavar="SUFFIX",
        help="Suffix identifying the label file (e.g. 'gtv'). "
             "Falls back to $LABEL_SUFFIX env var, then 'gtv'.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--skip-val", action="store_true", help="Do not convert validation set")
    p.add_argument("--skip-check", action="store_true", help="Skip post-conversion integrity check")
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="NAME=INT",
        help="Label class definitions, e.g. --labels background=0 GTV=1",
    )
    p.add_argument("--log-dir", default="logs", help="Directory for log files")
    return p.parse_args()


def _parse_labels(raw: list[str]) -> dict[str, int]:
    """Parse ``["background=0", "GTV=1"]`` into ``{"background": 0, "GTV": 1}``."""
    result: dict[str, int] = {}
    for item in raw:
        name, _, val = item.partition("=")
        result[name.strip()] = int(val.strip())
    return result


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="01_prepare", log_dir=args.log_dir)

    # ── Resolve sources ───────────────────────────────────────────────────────
    train_source = args.train or get_env("TRAIN_SOURCE", required=False)
    val_source = args.val or get_env("VAL_SOURCE", required=False)
    label_suffix = args.label_suffix or get_env("LABEL_SUFFIX", default="gtv")

    if train_source is None:
        log.error(
            "Training source not specified.\n"
            "Use --train <path> or set TRAIN_SOURCE in .env"
        )
        sys.exit(1)

    train_source = Path(train_source)
    if not train_source.exists():
        log.error(f"Training source not found: {train_source}")
        sys.exit(1)

    if val_source is not None:
        val_source = Path(val_source)
        if not val_source.exists():
            log.warning(f"Validation source not found: {val_source}  — skipping val.")
            val_source = None

    if args.skip_val:
        val_source = None

    # ── Resolve labels dict ───────────────────────────────────────────────────
    labels: dict[str, int] | None = None
    if args.labels:
        labels = _parse_labels(args.labels)
        log.info(f"Custom labels: {labels}")

    # ── Summary ───────────────────────────────────────────────────────────────
    raw_dir = get_path_env("nnUNet_raw", required=True)
    dataset_folder = dataset_folder_name()

    log.info("=" * 62)
    log.info("  Step 1: Dataset Preparation")
    log.info("=" * 62)
    log.info(f"  Train source : {train_source}")
    log.info(f"  Val source   : {val_source or '(skipped)'}")
    log.info(f"  Label suffix : _{label_suffix}")
    log.info(f"  Output dir   : {raw_dir / dataset_folder}")
    log.info(f"  Overwrite    : {args.overwrite}")
    log.info("=" * 62)

    # ── Convert ───────────────────────────────────────────────────────────────
    converter = BraTSMENRTConverter(
        train_source=train_source,
        val_source=val_source,
        label_suffix=label_suffix,
        overwrite=args.overwrite,
    )

    try:
        train_cases = converter.convert_training()
        val_cases = converter.convert_validation()
    finally:
        converter.cleanup()  # remove temp extraction dirs

    if not train_cases:
        log.error("No training cases were converted — aborting.")
        sys.exit(1)

    # ── Write channel map sidecar ─────────────────────────────────────────────
    dataset_dir = raw_dir / dataset_folder
    write_channel_map_sidecar(dataset_dir, converter.channel_map)

    # ── Build dataset.json ────────────────────────────────────────────────────
    build_dataset_json(
        dataset_dir=dataset_dir,
        channel_names=converter.channel_names,
        labels=labels,
        description="BraTS 2024 MEN-RT: Meningioma GTV segmentation from post-Gd T1c MRI",
        reference="BraTS 2024 Challenge",
        licence="CC-BY-NC-4.0",
        release="2024",
        overwrite=True,
    )

    # ── Report flagged cases ──────────────────────────────────────────────────
    if converter.flagged_cases:
        log.warning(
            f"{len(converter.flagged_cases)} cases flagged for low foreground ratio:"
        )
        for c in converter.flagged_cases:
            log.warning(f"    {c}")

    # ── Integrity check ───────────────────────────────────────────────────────
    if not args.skip_check:
        log.info("Running post-conversion integrity check …")
        checker = IntegrityChecker(dataset_dir=dataset_dir)
        report = checker.run()

        log.info("\n" + report.summary())

        if report.n_failed > 0:
            log.warning(f"{report.n_failed} cases failed integrity checks.")
            checker.export_csv(report, "results/integrity_report.csv")
        else:
            log.success("Integrity check passed.")
    else:
        log.info("Integrity check skipped (--skip-check).")

    log.success(
        f"Dataset preparation complete.\n"
        f"  Train : {len(train_cases)} cases\n"
        f"  Val   : {len(val_cases)} cases\n"
        f"  Output: {dataset_dir}"
    )


if __name__ == "__main__":
    main()
