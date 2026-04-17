"""Utilities for loading case IDs and inspecting nnU-Net cross-validation splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from ..utils.env_utils import dataset_folder_name, get_path_env


def load_case_ids(split: str = "train") -> list[str]:
    """Return sorted case IDs present in ``imagesTr`` or ``imagesTs``.

    Parameters
    ----------
    split:
        ``"train"`` → ``imagesTr``, ``"test"`` → ``imagesTs``.
    """
    raw = get_path_env("nnUNet_raw", required=True)
    dataset_dir = raw / dataset_folder_name()
    subfolder = "imagesTr" if split == "train" else "imagesTs"
    img_dir = dataset_dir / subfolder
    if not img_dir.exists():
        logger.warning(f"{img_dir} does not exist yet.")
        return []
    case_ids = sorted({p.name.replace("_0000.nii.gz", "") for p in img_dir.glob("*_0000.nii.gz")})
    logger.info(f"Found {len(case_ids)} cases in {subfolder}.")
    return case_ids


def load_splits(preprocessed_dir: str | Path | None = None) -> list[dict]:
    """Load the nnU-Net auto-generated cross-validation splits JSON.

    The splits file is created by ``nnUNetv2_plan_and_preprocess`` and lives at::

        <nnUNet_preprocessed>/Dataset{ID}_{NAME}/splits_final.json

    Returns
    -------
    list[dict]
        List of dicts with keys ``"train"`` and ``"val"`` per fold.
    """
    if preprocessed_dir is None:
        preprocessed_dir = get_path_env("nnUNet_preprocessed", required=True)
    splits_file = Path(preprocessed_dir) / dataset_folder_name() / "splits_final.json"
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            "Run nnUNetv2_plan_and_preprocess first."
        )
    with splits_file.open() as fh:
        splits = json.load(fh)
    logger.info(f"Loaded {len(splits)}-fold splits from {splits_file}")
    return splits


def summarise_splits(splits: list[dict]) -> None:
    """Log a human-readable summary of fold sizes."""
    for i, fold in enumerate(splits):
        n_tr = len(fold.get("train", []))
        n_val = len(fold.get("val", []))
        logger.info(f"  Fold {i}: {n_tr} train | {n_val} val")
