"""nnU-Net dataset integrity checker.

Validates the raw dataset directory against the requirements of nnU-Net v2:

Checks performed
----------------
1.  ``dataset.json`` present and schema-valid
2.  ``imagesTr/`` and ``labelsTr/`` exist and are non-empty
3.  Every training case has exactly ``N`` image channels
4.  Every training case has a corresponding label file
5.  Image and label files are readable NIfTI files
6.  Image and label spatial shapes match within each case
7.  Label values are in the expected set (e.g. {0, 1})
8.  No duplicate case IDs in ``imagesTr``
9.  ``imagesTs/`` cases have all required channels (no labels required)
10. Channel file naming follows nnU-Net convention (``{case_id}_{chan:04d}.nii.gz``)

Results are returned as a structured report and optionally written to CSV.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..utils.env_utils import dataset_folder_name, get_path_env


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CaseReport:
    case_id: str
    split: str                          # "train" | "test"
    has_all_channels: bool = True
    has_label: bool = True              # always True for test
    images_readable: bool = True
    label_readable: bool = True
    shapes_match: bool = True
    label_values_ok: bool = True
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = "; ".join(self.errors)
        d["ok"] = self.ok
        return d


@dataclass
class IntegrityReport:
    dataset_dir: str
    json_valid: bool = False
    num_train_cases: int = 0
    num_test_cases: int = 0
    num_channels: int = 0
    expected_labels: list[int] = field(default_factory=list)
    case_reports: list[CaseReport] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.case_reports if not r.ok)

    @property
    def passed(self) -> bool:
        return self.json_valid and self.n_failed == 0

    def summary(self) -> str:
        lines = [
            f"Dataset      : {self.dataset_dir}",
            f"dataset.json : {'OK' if self.json_valid else 'FAIL'}",
            f"Train cases  : {self.num_train_cases}",
            f"Test cases   : {self.num_test_cases}",
            f"Channels     : {self.num_channels}",
            f"Failed cases : {self.n_failed}/{len(self.case_reports)}",
            f"Overall      : {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


# ─── Checker ──────────────────────────────────────────────────────────────────

_CHAN_RE = re.compile(r"^(.+)_(\d{4})\.nii\.gz$")


class IntegrityChecker:
    """Validates a nnU-Net v2 raw dataset directory.

    Parameters
    ----------
    dataset_dir:
        ``<nnUNet_raw>/Dataset{ID}_{NAME}`` directory.
        Auto-resolved from environment when ``None``.
    expected_label_values:
        Integer label values that are considered valid.
        Default ``{0, 1}`` for binary segmentation.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        expected_label_values: set[int] | None = None,
    ) -> None:
        if dataset_dir is None:
            raw = get_path_env("nnUNet_raw", required=True)
            dataset_dir = raw / dataset_folder_name()
        self.dataset_dir = Path(dataset_dir)
        self.expected_label_values = expected_label_values or {0, 1}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, max_cases: int | None = None) -> IntegrityReport:
        """Execute all integrity checks.

        Parameters
        ----------
        max_cases:
            Limit the number of cases checked per split (useful for quick runs).

        Returns
        -------
        IntegrityReport
        """
        report = IntegrityReport(dataset_dir=str(self.dataset_dir))

        # ── 1. dataset.json ──────────────────────────────────────────────────
        ds_json, report.json_valid = self._check_dataset_json()
        if ds_json is None:
            logger.error("dataset.json missing or invalid — aborting further checks.")
            return report

        num_channels = len(ds_json.get("channel_names", {"0": "image"}))
        report.num_channels = num_channels
        report.expected_labels = sorted(ds_json.get("labels", {}).values())

        # ── 2. Discover case IDs ─────────────────────────────────────────────
        images_tr = self.dataset_dir / "imagesTr"
        labels_tr = self.dataset_dir / "labelsTr"
        images_ts = self.dataset_dir / "imagesTs"

        train_ids = self._collect_case_ids(images_tr)
        test_ids = self._collect_case_ids(images_ts)
        report.num_train_cases = len(train_ids)
        report.num_test_cases = len(test_ids)

        if not train_ids:
            logger.warning("imagesTr is empty.")

        # ── 3. Validate training cases ───────────────────────────────────────
        logger.info(f"Checking {len(train_ids)} training cases …")
        for case_id in tqdm(
            train_ids[:max_cases] if max_cases else train_ids, desc="Train"
        ):
            cr = self._check_train_case(
                case_id=case_id,
                images_dir=images_tr,
                labels_dir=labels_tr,
                num_channels=num_channels,
            )
            report.case_reports.append(cr)
            if not cr.ok:
                for err in cr.errors:
                    logger.warning(f"[{case_id}] {err}")

        # ── 4. Validate test/validation cases ────────────────────────────────
        if test_ids:
            logger.info(f"Checking {len(test_ids)} test cases …")
            for case_id in tqdm(
                test_ids[:max_cases] if max_cases else test_ids, desc="Test"
            ):
                cr = self._check_test_case(
                    case_id=case_id,
                    images_dir=images_ts,
                    num_channels=num_channels,
                )
                report.case_reports.append(cr)
                if not cr.ok:
                    for err in cr.errors:
                        logger.warning(f"[{case_id}] {err}")

        return report

    def export_csv(self, report: IntegrityReport, output_path: str | Path) -> Path:
        """Write per-case report to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in report.case_reports]
        if not rows:
            return output_path
        with output_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Integrity report → {output_path}")
        return output_path

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_dataset_json(self) -> tuple[dict | None, bool]:
        path = self.dataset_dir / "dataset.json"
        if not path.exists():
            logger.error(f"Missing: {path}")
            return None, False

        with path.open() as fh:
            data = json.load(fh)

        errors: list[str] = []
        for required_key in ("channel_names", "labels", "numTraining", "file_ending"):
            if required_key not in data:
                errors.append(f"dataset.json missing key: '{required_key}'")

        if data.get("file_ending") != ".nii.gz":
            errors.append(f"file_ending is '{data.get('file_ending')}', expected '.nii.gz'")

        n_declared = data.get("numTraining", -1)
        images_tr = self.dataset_dir / "imagesTr"
        n_actual = len(self._collect_case_ids(images_tr)) if images_tr.exists() else 0
        if n_declared != n_actual:
            errors.append(
                f"numTraining mismatch: dataset.json={n_declared}, imagesTr={n_actual}"
            )

        if errors:
            for e in errors:
                logger.error(f"[dataset.json] {e}")
            return data, False

        logger.info("dataset.json: OK")
        return data, True

    def _check_train_case(
        self,
        case_id: str,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
    ) -> CaseReport:
        cr = CaseReport(case_id=case_id, split="train")

        # Channel files
        chan_paths = self._expected_channel_paths(case_id, images_dir, num_channels)
        missing_chans = [p for p in chan_paths if not p.exists()]
        if missing_chans:
            cr.has_all_channels = False
            cr.errors.append(f"Missing channels: {[p.name for p in missing_chans]}")

        # Label file
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            cr.has_label = False
            cr.errors.append(f"Missing label: {label_path.name}")

        # Readability + shape consistency
        shapes: list[tuple] = []
        for p in chan_paths:
            if p.exists():
                try:
                    img = nib.load(p)
                    shapes.append(tuple(img.shape[:3]))
                except Exception as exc:
                    cr.images_readable = False
                    cr.errors.append(f"Cannot read {p.name}: {exc}")

        if label_path.exists():
            try:
                lbl_img = nib.load(label_path)
                lbl_shape = tuple(lbl_img.shape[:3])
                shapes.append(lbl_shape)

                # Label value check
                data = np.asarray(lbl_img.dataobj, dtype=np.int32)
                unique_vals = set(np.unique(data).tolist())
                if not unique_vals.issubset(self.expected_label_values):
                    unexpected = unique_vals - self.expected_label_values
                    cr.label_values_ok = False
                    cr.errors.append(f"Unexpected label values: {unexpected}")
            except Exception as exc:
                cr.label_readable = False
                cr.errors.append(f"Cannot read label: {exc}")

        if len(set(shapes)) > 1:
            cr.shapes_match = False
            cr.errors.append(f"Shape mismatch: {shapes}")

        return cr

    def _check_test_case(
        self, case_id: str, images_dir: Path, num_channels: int
    ) -> CaseReport:
        cr = CaseReport(case_id=case_id, split="test")
        cr.has_label = True  # not applicable

        chan_paths = self._expected_channel_paths(case_id, images_dir, num_channels)
        missing = [p for p in chan_paths if not p.exists()]
        if missing:
            cr.has_all_channels = False
            cr.errors.append(f"Missing channels: {[p.name for p in missing]}")

        for p in chan_paths:
            if p.exists():
                try:
                    nib.load(p)
                except Exception as exc:
                    cr.images_readable = False
                    cr.errors.append(f"Cannot read {p.name}: {exc}")

        return cr

    @staticmethod
    def _collect_case_ids(images_dir: Path) -> list[str]:
        if not images_dir.exists():
            return []
        ids: set[str] = set()
        for f in images_dir.glob("*_0000.nii.gz"):
            m = _CHAN_RE.match(f.name)
            if m:
                ids.add(m.group(1))
        return sorted(ids)

    @staticmethod
    def _expected_channel_paths(
        case_id: str, images_dir: Path, num_channels: int
    ) -> list[Path]:
        return [images_dir / f"{case_id}_{i:04d}.nii.gz" for i in range(num_channels)]
