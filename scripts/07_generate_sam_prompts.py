#!/usr/bin/env python3
"""Step 7 — Generate SAM-Med3D prompts from nnU-Net coarse segmentations.

This script converts predicted masks (e.g., nnU-Net outputs) into prompt JSON
payloads that can be consumed by a SAM-Med3D refinement stage.

Outputs
-------
1. One JSON file per case in <output-dir>/cases/
2. A manifest JSON: <output-dir>/sam_prompt_manifest.json
3. A summary CSV: <output-dir>/sam_prompt_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.prompt_builder import build_case_prompt_payload
from src.utils.env_utils import dataset_folder_name, get_path_env, load_env
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mask-dir",
        default=None,
        help="Directory with coarse masks (.nii.gz). "
             "Default: inference_outputs/ensemble",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Directory with input images for geometry + case matching. "
             "Default: nnUNet_raw/<dataset>/imagesTs if exists else imagesTr.",
    )
    p.add_argument(
        "--output-dir",
        default="prompts/sammed3d",
        help="Directory for generated prompt artifacts.",
    )
    p.add_argument(
        "--max-positive-points",
        type=int,
        default=4,
        help="Max positive points per component.",
    )
    p.add_argument(
        "--max-negative-points",
        type=int,
        default=4,
        help="Max negative points per component.",
    )
    p.add_argument(
        "--min-component-voxels",
        type=int,
        default=10,
        help="Ignore connected components smaller than this voxel count.",
    )
    p.add_argument(
        "--negative-shell-iters",
        type=int,
        default=4,
        help="Dilation iterations used to sample near-boundary negative points.",
    )
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def _default_image_dir() -> Path:
    raw_dir = get_path_env("nnUNet_raw", required=True)
    ds = raw_dir / dataset_folder_name()
    images_ts = ds / "imagesTs"
    if images_ts.exists() and any(images_ts.glob("*_0000.nii.gz")):
        return images_ts
    return ds / "imagesTr"


def _resolve_image_path(image_dir: Path, case_id: str) -> Path | None:
    cand = image_dir / f"{case_id}_0000.nii.gz"
    if cand.exists():
        return cand
    # Fallback for rare non-gz inputs.
    cand2 = image_dir / f"{case_id}_0000.nii"
    if cand2.exists():
        return cand2
    return None


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="sam_prompt_gen", log_dir=args.log_dir)

    mask_dir = Path(args.mask_dir or "inference_outputs/ensemble")
    image_dir = Path(args.image_dir) if args.image_dir else _default_image_dir()

    out_root = Path(args.output_dir)
    case_dir = out_root / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    if not mask_dir.exists():
        log.error(f"Mask directory not found: {mask_dir}")
        raise SystemExit(1)
    if not image_dir.exists():
        log.warning(f"Image directory not found: {image_dir}. Proceeding with mask geometry only.")

    masks = sorted(mask_dir.glob("*.nii.gz"))
    if not masks:
        log.error(f"No .nii.gz masks found in: {mask_dir}")
        raise SystemExit(1)

    log.info("=" * 68)
    log.info("Step 7: Generate SAM-Med3D Prompt Artifacts")
    log.info(f"  Mask dir   : {mask_dir}")
    log.info(f"  Image dir  : {image_dir}")
    log.info(f"  Output dir : {out_root}")
    log.info(f"  Cases      : {len(masks)}")
    log.info("=" * 68)

    summary_rows: list[dict[str, object]] = []
    manifest_cases: list[dict[str, object]] = []

    for mask_path in masks:
        case_id = mask_path.name[:-7] if mask_path.name.endswith(".nii.gz") else mask_path.stem
        img_path = _resolve_image_path(image_dir, case_id) if image_dir.exists() else None

        payload = build_case_prompt_payload(
            image_path=img_path,
            mask_path=mask_path,
            max_positive_points=args.max_positive_points,
            max_negative_points=args.max_negative_points,
            min_component_voxels=args.min_component_voxels,
            negative_shell_iters=args.negative_shell_iters,
        )

        out_json = case_dir / f"{case_id}.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        n_comp = int(payload["num_components"])
        comps = payload["components"]
        largest_vox = int(comps[0]["voxel_count"]) if n_comp > 0 else 0

        summary_rows.append(
            {
                "case_id": case_id,
                "mask_path": str(mask_path),
                "image_path": str(img_path) if img_path else None,
                "prompt_json": str(out_json),
                "num_components": n_comp,
                "largest_component_voxels": largest_vox,
            }
        )
        manifest_cases.append(
            {
                "case_id": case_id,
                "prompt_json": str(out_json),
                "num_components": n_comp,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("case_id")
    summary_csv = out_root / "sam_prompt_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest = {
        "mask_dir": str(mask_dir),
        "image_dir": str(image_dir),
        "output_dir": str(out_root),
        "num_cases": int(len(summary_rows)),
        "max_positive_points": int(args.max_positive_points),
        "max_negative_points": int(args.max_negative_points),
        "min_component_voxels": int(args.min_component_voxels),
        "negative_shell_iters": int(args.negative_shell_iters),
        "cases": manifest_cases,
    }
    manifest_path = out_root / "sam_prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.success("Prompt generation complete.")
    log.info(f"  Case JSONs : {case_dir}")
    log.info(f"  Manifest   : {manifest_path}")
    log.info(f"  Summary    : {summary_csv}")


if __name__ == "__main__":
    main()
