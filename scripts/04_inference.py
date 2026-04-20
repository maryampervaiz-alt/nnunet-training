#!/usr/bin/env python3
"""Step 4 — Run inference on test/validation images.

Supports:
  - Single-fold or ensemble (all folds) prediction
  - Arbitrary input directory

Usage
-----
::

    # Predict on challenge validation set (imagesTs)
    python scripts/04_inference.py

    # Predict on a custom directory
    python scripts/04_inference.py --input /path/to/images --output inference_outputs/custom

    # Single fold
    python scripts/04_inference.py --folds 0

    # Ensemble (default)
    python scripts/04_inference.py --folds all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitter import load_splits
from src.utils.env_utils import load_env, get_env, get_path_env, dataset_folder_name
from src.utils.logging_utils import get_logger
from src.inference.predictor import NNUNetPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=None,
        help="Input directory with *_0000.nii.gz files (default: nnunet_raw/imagesTs)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory for predictions "
             "(default: inference_outputs/ensemble or inference_outputs/cv)",
    )
    p.add_argument(
        "--folds",
        nargs="+",
        default=["all"],
        help="Fold indices or 'all' for ensemble (default: all)",
    )
    p.add_argument(
        "--cv-mode",
        action="store_true",
        help="Run fold-wise validation inference using nnU-Net splits_final.json",
    )
    p.add_argument("--configuration", default=None, help="nnU-Net configuration override")
    p.add_argument("--trainer", default=None, help="Trainer class; auto-detected when omitted")
    p.add_argument("--plans", default=None, help="Plans identifier; auto-detected when omitted")
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("--no-gaussian", action="store_true", help="Disable Gaussian windowing")
    p.add_argument("--disable-tta", action="store_true", help="Disable test-time augmentation")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def _parse_folds(raw_folds: list[str], num_splits: int | None = None) -> list[int] | str:
    if raw_folds == ["all"] or "all" in raw_folds:
        return "all"
    folds = sorted({int(f) for f in raw_folds})
    if num_splits is not None:
        invalid = [f for f in folds if f < 0 or f >= num_splits]
        if invalid:
            raise ValueError(f"Invalid folds for {num_splits}-fold split: {invalid}")
    return folds


def _link_or_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_cases(case_ids: list[str], image_dir: Path, staging_dir: Path) -> int:
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for case_id in case_ids:
        matches = sorted(image_dir.glob(f"{case_id}_*.nii.gz"))
        for src in matches:
            _link_or_copy(src, staging_dir / src.name, overwrite=True)
        staged += len(matches)
    return staged


def _run_standard_inference(args: argparse.Namespace, config: str, log) -> int:
    output_dir = args.output or "inference_outputs/ensemble"

    if args.input is None:
        raw_dir = get_path_env("nnUNet_raw", required=True)
        input_dir = raw_dir / dataset_folder_name() / "imagesTs"
    else:
        input_dir = Path(args.input)

    if not input_dir.exists():
        log.error(f"Input directory does not exist: {input_dir}")
        return 1

    try:
        folds = _parse_folds(args.folds)
    except ValueError as exc:
        log.error(str(exc))
        return 1

    log.info("=" * 60)
    log.info("Step 4: Batch Inference")
    log.info(f"  Input dir     : {input_dir}")
    log.info(f"  Output dir    : {output_dir}")
    log.info(f"  Configuration : {config}")
    log.info(f"  Folds         : {folds}")
    log.info("=" * 60)

    predictor = NNUNetPredictor(
        input_dir=input_dir,
        output_dir=output_dir,
        configuration=config,
        folds=folds,
        trainer_class=args.trainer,
        plans_identifier=args.plans,
        step_size=args.step_size,
        use_gaussian=not args.no_gaussian,
        disable_tta=args.disable_tta,
        device=args.device,
        save_probabilities=args.save_probabilities,
        overwrite_existing=args.overwrite,
    )
    return predictor.predict()


def _run_cv_inference(args: argparse.Namespace, config: str, log) -> int:
    raw_dir = get_path_env("nnUNet_raw", required=True)
    dataset_dir = raw_dir / dataset_folder_name()
    image_dir = dataset_dir / "imagesTr"
    if not image_dir.exists():
        log.error(f"Training image directory not found: {image_dir}")
        return 1

    splits = load_splits()
    try:
        folds = _parse_folds(args.folds, num_splits=len(splits))
    except ValueError as exc:
        log.error(str(exc))
        return 1

    fold_indices = list(range(len(splits))) if folds == "all" else folds
    output_root = Path(args.output or "inference_outputs/cv")
    combined_dir = output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Step 4: Batch Inference (CV mode)")
    log.info(f"  Input dir     : {image_dir}")
    log.info(f"  Output root   : {output_root}")
    log.info(f"  Combined dir  : {combined_dir}")
    log.info(f"  Configuration : {config}")
    log.info(f"  Folds         : {fold_indices}")
    log.info("=" * 60)

    manifest: list[dict[str, object]] = []
    for fold_idx in fold_indices:
        val_cases = sorted(splits[fold_idx].get("val", []))
        if not val_cases:
            log.warning(f"Fold {fold_idx} has no validation cases; skipping.")
            continue

        fold_output = output_root / f"fold_{fold_idx}"
        log.info(
            f"[fold_{fold_idx}] Running inference for {len(val_cases)} validation cases "
            f"-> {fold_output}"
        )

        with tempfile.TemporaryDirectory(prefix=f"nnunet_fold_{fold_idx}_") as tmp:
            staged_dir = Path(tmp) / "input"
            staged_files = _stage_cases(val_cases, image_dir=image_dir, staging_dir=staged_dir)
            log.info(f"[fold_{fold_idx}] Staged {staged_files} image files.")

            predictor = NNUNetPredictor(
                input_dir=staged_dir,
                output_dir=fold_output,
                configuration=config,
                folds=[fold_idx],
                trainer_class=args.trainer,
                plans_identifier=args.plans,
                step_size=args.step_size,
                use_gaussian=not args.no_gaussian,
                disable_tta=args.disable_tta,
                device=args.device,
                save_probabilities=args.save_probabilities,
                overwrite_existing=args.overwrite,
            )
            rc = predictor.predict()
            if rc != 0:
                return rc

        linked = 0
        for case_id in val_cases:
            pred_path = fold_output / f"{case_id}.nii.gz"
            if pred_path.exists():
                _link_or_copy(
                    pred_path,
                    combined_dir / pred_path.name,
                    overwrite=args.overwrite,
                )
                linked += 1

        manifest.append(
            {
                "fold": fold_idx,
                "num_cases": len(val_cases),
                "num_predictions": linked,
                "output_dir": str(fold_output),
            }
        )

    manifest_path = output_root / "cv_prediction_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "configuration": config,
                "trainer_class": args.trainer,
                "plans_identifier": args.plans,
                "folds": fold_indices,
                "combined_dir": str(combined_dir),
                "fold_runs": manifest,
            },
            fh,
            indent=2,
        )
    log.success(f"CV inference complete. Manifest: {manifest_path}")
    return 0


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="04_inference", log_dir=args.log_dir)

    config = args.configuration or get_env("NNUNET_CONFIGURATION", default="3d_fullres")
    rc = _run_cv_inference(args, config, log) if args.cv_mode else _run_standard_inference(
        args, config, log
    )
    if rc != 0:
        log.error("Inference failed.")
        sys.exit(rc)

    final_output = args.output or ("inference_outputs/cv" if args.cv_mode else "inference_outputs/ensemble")
    log.success(f"Inference complete. Predictions in: {final_output}")


if __name__ == "__main__":
    main()
