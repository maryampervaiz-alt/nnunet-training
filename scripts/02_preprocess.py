#!/usr/bin/env python3
"""Step 2 — Run nnU-Net v2 planning and preprocessing.

What this script does
---------------------
1. Validates that all three required nnU-Net environment variables are set
   and that the dataset directory + ``dataset.json`` exist.
2. Calls ``nnUNetv2_plan_and_preprocess`` which automatically determines:
      - Target spacing
      - Patch size
      - Batch size
      - Network architecture (topology)
      - Normalization scheme (CT vs. MRI vs. non-image)
   None of these are overridden here.
3. Optionally runs ``nnUNetv2_determine_postprocessing`` afterwards.

Environment variables (set in .env)
------------------------------------
    nnUNet_raw              nnU-Net raw data root
    nnUNet_preprocessed     nnU-Net preprocessed output root
    nnUNet_results          nnU-Net results root
    DATASET_ID              Integer dataset ID

Usage
-----
::

    # Default: plan + preprocess all configurations
    python scripts/02_preprocess.py

    # Only 3d_fullres
    python scripts/02_preprocess.py --configurations 3d_fullres

    # Multiple configurations
    python scripts/02_preprocess.py --configurations 3d_fullres 3d_lowres

    # Verify dataset.json without running preprocessing
    python scripts/02_preprocess.py --verify-only

    # Control worker count (default: all CPUs)
    python scripts/02_preprocess.py --np 8

    # Skip fingerprint extraction (already done)
    python scripts/02_preprocess.py --no-pp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.env_utils import load_env, get_env, nnunet_env, dataset_folder_name
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--configurations",
        nargs="+",
        default=None,
        metavar="CONFIG",
        help="Configurations to preprocess. If omitted, nnU-Net processes all "
             "applicable ones. Example: 3d_fullres 3d_lowres",
    )
    p.add_argument(
        "--np",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel preprocessing workers. Default: nnU-Net auto.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dataset.json schema; do not run preprocessing.",
    )
    p.add_argument(
        "--no-pp",
        action="store_true",
        help="Skip preprocessing (only run planning/fingerprint).",
    )
    p.add_argument(
        "--overwrite-plans",
        action="store_true",
        help="Force re-planning even if plans already exist.",
    )
    p.add_argument("--log-dir", default="logs", help="Directory for log files")
    return p.parse_args()


def _validate_environment(env: dict[str, str], dataset_id: int, log) -> bool:
    """Return True if all preconditions are met."""
    ok = True

    for var, path in env.items():
        p = Path(path)
        if not p.exists():
            log.warning(f"  {var} directory does not exist, will be created: {path}")
            p.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(env["nnUNet_raw"]) / dataset_folder_name()
    if not dataset_dir.exists():
        log.error(f"Dataset directory not found: {dataset_dir}")
        log.error("Run scripts/01_prepare_dataset.py first.")
        ok = False
        return ok

    ds_json = dataset_dir / "dataset.json"
    if not ds_json.exists():
        log.error(f"dataset.json missing: {ds_json}")
        ok = False
    else:
        with ds_json.open() as fh:
            data = json.load(fh)
        missing = [k for k in ("channel_names", "labels", "numTraining", "file_ending") if k not in data]
        if missing:
            log.error(f"dataset.json missing required keys: {missing}")
            ok = False
        else:
            n = data["numTraining"]
            log.info(f"dataset.json OK — numTraining={n}, channels={data['channel_names']}")

    images_tr = dataset_dir / "imagesTr"
    if not images_tr.exists() or not any(images_tr.glob("*_0000.nii.gz")):
        log.error(f"imagesTr is empty: {images_tr}")
        ok = False

    labels_tr = dataset_dir / "labelsTr"
    if not labels_tr.exists() or not any(labels_tr.glob("*.nii.gz")):
        log.error(f"labelsTr is empty: {labels_tr}")
        ok = False

    return ok


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="02_preprocess", log_dir=args.log_dir)

    dataset_id = int(get_env("DATASET_ID", default="001"))
    env = {**os.environ, **nnunet_env()}

    log.info("=" * 62)
    log.info("  Step 2: nnU-Net Planning & Preprocessing")
    log.info("=" * 62)
    log.info(f"  Dataset ID      : {dataset_id}")
    log.info(f"  nnUNet_raw      : {env['nnUNet_raw']}")
    log.info(f"  nnUNet_preproc  : {env['nnUNet_preprocessed']}")
    log.info(f"  nnUNet_results  : {env['nnUNet_results']}")
    log.info(f"  Configurations  : {args.configurations or 'auto'}")
    log.info("=" * 62)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not _validate_environment(nnunet_env(), dataset_id, log):
        sys.exit(1)

    if args.verify_only:
        log.info("--verify-only: skipping preprocessing.")
        return

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
    ]

    if args.configurations:
        cmd += ["-c"] + args.configurations

    if args.np is not None:
        # nnU-Net ≥ 2.2: --np applies per configuration
        np_values = [str(args.np)] * (len(args.configurations) if args.configurations else 1)
        cmd += ["--np"] + np_values

    if args.no_pp:
        cmd.append("--no_pp")

    if args.overwrite_plans:
        cmd.append("--clean")

    log.info(f"Command: {' '.join(cmd)}")

    # ── Run ───────────────────────────────────────────────────────────────────
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        log.error(f"nnUNetv2_plan_and_preprocess failed (rc={result.returncode})")
        sys.exit(result.returncode)

    # ── Show generated plans ──────────────────────────────────────────────────
    preproc_dir = Path(env["nnUNet_preprocessed"]) / dataset_folder_name()
    if preproc_dir.exists():
        plans_files = list(preproc_dir.glob("nnUNetPlans*.json"))
        for pf in plans_files:
            with pf.open() as fh:
                plans = json.load(fh)
            log.info(f"\nPlans from {pf.name}:")
            for cfg_name, cfg in plans.get("configurations", {}).items():
                log.info(
                    f"  [{cfg_name}]  "
                    f"patch_size={cfg.get('patch_size')}  "
                    f"spacing={cfg.get('spacing')}  "
                    f"batch_size={cfg.get('batch_size')}"
                )

    log.success("Preprocessing complete.")


if __name__ == "__main__":
    main()
