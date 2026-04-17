"""Convert BraTS MEN-RT source data (zip or extracted) into nnU-Net v2 raw format.

Auto-discovery
--------------
The converter scans the source tree to find every unique file suffix
(e.g. ``_t1c``, ``_gtv``) and builds a modality-to-channel mapping
without hardcoding any modality name.  The caller tells it which suffix
is the *label*; everything else becomes an image channel ordered
alphabetically.

Supported source layouts
------------------------
1. **Zip file** – A single ``.zip`` whose root is a folder that contains
   one subfolder per case::

       BraTS-MEN-RT-Train-v2.zip
       └── BraTS-MEN-RT-Train-v2/
           └── BraTS-MEN-RT-0100-1/
               ├── BraTS-MEN-RT-0100-1_t1c.nii.gz
               └── BraTS-MEN-RT-0100-1_gtv.nii.gz

2. **Extracted directory** – The top-level directory (with or without an
   additional common root folder) that contains one subfolder per case.

nnU-Net v2 raw output layout
-----------------------------
::

    <nnUNet_raw>/Dataset{ID}_{NAME}/
        imagesTr/   {case_id}_{channel:04d}.nii.gz
        labelsTr/   {case_id}.nii.gz
        imagesTs/   {case_id}_{channel:04d}.nii.gz
        dataset.json
"""
from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Generator

import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..utils.env_utils import dataset_folder_name, get_env, get_path_env


# ─── Source scanner ───────────────────────────────────────────────────────────

class SourceLayout:
    """Inspects a source directory or zip and exposes a uniform case iterator.

    Parameters
    ----------
    source:
        Path to either a zip file or an extracted directory.
    label_suffix:
        The file suffix that identifies the segmentation mask
        (e.g. ``"gtv"`` matches ``*_gtv.nii.gz``).
    """

    _NIFTI_RE = re.compile(r"^(.+?)_([^_]+)\.nii\.gz$", re.IGNORECASE)

    def __init__(self, source: Path, label_suffix: str) -> None:
        self.source = source
        self.label_suffix = label_suffix.lstrip("_").lower()
        self._is_zip = source.suffix.lower() == ".zip"
        self._tmpdir: tempfile.TemporaryDirectory | None = None

        if self._is_zip:
            logger.info(f"Source is a zip file: {source.name}")
        else:
            logger.info(f"Source is a directory: {source}")

    # ── Public API ────────────────────────────────────────────────────────────

    def discover_modalities(self) -> tuple[dict[str, int], str]:
        """Scan a sample of cases and return (channel_map, label_suffix).

        Returns
        -------
        channel_map:
            Dict mapping suffix string → nnU-Net channel index (0-based),
            sorted alphabetically (label excluded).
        label_suffix:
            Confirmed label suffix string.
        """
        suffixes: set[str] = set()
        for case_id, file_map in self._iter_cases(max_cases=10):
            suffixes.update(file_map.keys())

        image_suffixes = sorted(s for s in suffixes if s != self.label_suffix)
        if not image_suffixes:
            raise ValueError(
                f"No image modalities found (only label '{self.label_suffix}' detected). "
                "Check label_suffix or source directory."
            )

        channel_map = {suf: idx for idx, suf in enumerate(image_suffixes)}
        logger.info(f"Discovered modalities → {channel_map}  |  label='{self.label_suffix}'")
        return channel_map, self.label_suffix

    def iter_cases(self) -> Generator[tuple[str, dict[str, Path]], None, None]:
        """Yield ``(case_id, {suffix: path})`` for every case in the source."""
        yield from self._iter_cases()

    def cleanup(self) -> None:
        """Remove any temporary extraction directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _iter_cases(
        self, max_cases: int | None = None
    ) -> Generator[tuple[str, dict[str, Path]], None, None]:
        root = self._get_root()
        count = 0
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue
            file_map = self._parse_case_dir(case_dir)
            if file_map:
                yield case_dir.name, file_map
                count += 1
                if max_cases is not None and count >= max_cases:
                    return

    def _get_root(self) -> Path:
        """Return the directory that directly contains per-case subfolders."""
        if not self._is_zip:
            return self._resolve_extracted_root(self.source)

        # Extract zip to temp dir (lazy)
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="nnunet_src_")
            logger.info(f"Extracting {self.source.name} → {self._tmpdir.name} …")
            with zipfile.ZipFile(self.source) as zf:
                zf.extractall(self._tmpdir.name)
        return self._resolve_extracted_root(Path(self._tmpdir.name))

    @staticmethod
    def _resolve_extracted_root(path: Path) -> Path:
        """Iteratively unwrap wrapper directories until case folders are found.

        Keeps descending as long as the current directory has exactly one
        child that is itself a directory (i.e. a single-item wrapper folder
        like BraTS2024-MEN-RT-TrainingData/ or BraTS-MEN-RT-Train-v2/).
        Stops when multiple children or non-directory children are found.
        """
        current = path
        for _ in range(6):  # guard: max 6 levels deep
            try:
                contents = [p for p in current.iterdir()
                            if not p.name.startswith(".") and not p.name.startswith("__")]
            except PermissionError:
                break
            dirs = [p for p in contents if p.is_dir()]
            # If exactly one child and it is a directory, descend into it
            if len(contents) == 1 and len(dirs) == 1:
                current = dirs[0]
            else:
                break
        logger.debug(f"Resolved dataset root: {current}")
        return current

    def _parse_case_dir(self, case_dir: Path) -> dict[str, Path]:
        """Return {suffix: filepath} for all NIfTI files in *case_dir*."""
        file_map: dict[str, Path] = {}
        for f in case_dir.glob("*.nii.gz"):
            m = self._NIFTI_RE.match(f.name)
            if m:
                suffix = m.group(2).lower()
                file_map[suffix] = f
        return file_map


# ─── Main converter ───────────────────────────────────────────────────────────

class BraTSMENRTConverter:
    """Convert BraTS MEN-RT source data to nnU-Net v2 raw dataset format.

    Parameters
    ----------
    train_source:
        Path to training data zip or extracted directory.
    val_source:
        Path to validation data zip or extracted directory (no labels).
    nnunet_raw_dir:
        ``nnUNet_raw`` root.  Auto-read from ``$nnUNet_raw`` when ``None``.
    label_suffix:
        Suffix that identifies the segmentation file within each case folder.
        E.g. ``"gtv"`` matches ``*_gtv.nii.gz``.
    min_fg_ratio:
        Minimum foreground fraction; cases below this threshold are flagged.
    overwrite:
        Overwrite existing files in the nnU-Net raw directory.
    """

    def __init__(
        self,
        train_source: str | Path,
        val_source: str | Path | None = None,
        nnunet_raw_dir: str | Path | None = None,
        label_suffix: str = "gtv",
        min_fg_ratio: float = 1e-5,
        overwrite: bool = False,
    ) -> None:
        self.train_src = SourceLayout(Path(train_source), label_suffix=label_suffix)
        self.val_src = SourceLayout(Path(val_source), label_suffix=label_suffix) if val_source else None
        self.label_suffix = label_suffix.lstrip("_").lower()
        self.min_fg_ratio = min_fg_ratio
        self.overwrite = overwrite

        if nnunet_raw_dir is None:
            nnunet_raw_dir = get_path_env("nnUNet_raw", required=True)
        self.nnunet_raw_dir = Path(nnunet_raw_dir)

        folder = dataset_folder_name()
        self.dataset_dir = self.nnunet_raw_dir / folder
        self.images_tr = self.dataset_dir / "imagesTr"
        self.labels_tr = self.dataset_dir / "labelsTr"
        self.images_ts = self.dataset_dir / "imagesTs"

        for d in (self.images_tr, self.labels_tr, self.images_ts):
            d.mkdir(parents=True, exist_ok=True)

        # Populated during conversion
        self.channel_map: dict[str, int] = {}
        self._flagged: list[str] = []
        self._train_cases: list[str] = []
        self._val_cases: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def convert_training(self, max_cases: int | None = None) -> list[str]:
        """Convert training cases (images + labels) to nnU-Net format.

        Parameters
        ----------
        max_cases:
            If set, only the first *max_cases* cases (sorted alphabetically)
            are converted.  Useful for quick subset experiments.

        Returns sorted list of successfully converted case IDs.
        """
        self.channel_map, _ = self.train_src.discover_modalities()
        cases = list(self.train_src.iter_cases())
        if max_cases is not None and max_cases < len(cases):
            logger.info(f"Subsetting to first {max_cases} cases (out of {len(cases)} total).")
            cases = cases[:max_cases]
        logger.info(f"Converting {len(cases)} training cases …")

        converted: list[str] = []
        for case_id, file_map in tqdm(cases, desc="Training"):
            try:
                self._write_images(case_id, file_map, self.images_tr)
                label_path = file_map.get(self.label_suffix)
                if label_path is None:
                    logger.warning(f"[{case_id}] Label file '{self.label_suffix}' missing — skipped.")
                    continue
                self._write_label(case_id, label_path, self.labels_tr)
                converted.append(case_id)
            except Exception as exc:
                logger.error(f"[{case_id}] Conversion failed: {exc}")

        self._train_cases = converted
        logger.info(f"Training done: {len(converted)}/{len(cases)} OK")
        if self._flagged:
            logger.warning(f"Flagged (low foreground): {self._flagged}")
        return converted

    def convert_validation(self) -> list[str]:
        """Convert validation cases (images only, no labels)."""
        if self.val_src is None:
            logger.info("No validation source provided — skipping.")
            return []

        if not self.channel_map:
            self.channel_map, _ = self.val_src.discover_modalities()

        cases = list(self.val_src.iter_cases())
        logger.info(f"Converting {len(cases)} validation cases …")

        converted: list[str] = []
        for case_id, file_map in tqdm(cases, desc="Validation"):
            try:
                self._write_images(case_id, file_map, self.images_ts)
                converted.append(case_id)
            except Exception as exc:
                logger.error(f"[{case_id}] Conversion failed: {exc}")

        self._val_cases = converted
        logger.info(f"Validation done: {len(converted)}/{len(cases)} OK")
        return converted

    def cleanup(self) -> None:
        """Remove temporary extraction directories."""
        self.train_src.cleanup()
        if self.val_src:
            self.val_src.cleanup()

    @property
    def channel_names(self) -> dict[str, str]:
        """Return nnU-Net channel_names dict: ``{"0": "T1c", …}``."""
        return {str(idx): suf.upper() for suf, idx in self.channel_map.items()}

    @property
    def flagged_cases(self) -> list[str]:
        return list(self._flagged)

    @property
    def train_case_ids(self) -> list[str]:
        return list(self._train_cases)

    @property
    def val_case_ids(self) -> list[str]:
        return list(self._val_cases)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write_images(self, case_id: str, file_map: dict[str, Path], dst_dir: Path) -> None:
        for suffix, channel_idx in self.channel_map.items():
            src = file_map.get(suffix)
            if src is None:
                logger.warning(f"[{case_id}] Modality '{suffix}' missing.")
                continue
            dst = dst_dir / f"{case_id}_{channel_idx:04d}.nii.gz"
            self._copy(src, dst)

    def _write_label(self, case_id: str, src: Path, dst_dir: Path) -> None:
        dst = dst_dir / f"{case_id}.nii.gz"
        self._copy(src, dst)
        self._validate_label(case_id, dst)

    def _copy(self, src: Path, dst: Path) -> None:
        if dst.exists() and not self.overwrite:
            return
        shutil.copy2(src, dst)

    def _validate_label(self, case_id: str, label_path: Path) -> None:
        try:
            img = nib.load(label_path)
            data = np.asarray(img.dataobj, dtype=np.uint8)
            unique = np.unique(data).tolist()
            fg_ratio = float(np.sum(data > 0)) / float(data.size)

            if fg_ratio < self.min_fg_ratio:
                logger.warning(
                    f"[{case_id}] Very small foreground: {fg_ratio:.2e}  labels={unique}"
                )
                self._flagged.append(case_id)
            if any(v > 1 for v in unique):
                logger.warning(f"[{case_id}] Multi-class label detected: {unique}")
        except Exception as exc:
            logger.error(f"[{case_id}] Label validation error: {exc}")
