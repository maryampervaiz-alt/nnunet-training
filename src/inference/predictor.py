"""Inference wrapper around ``nnUNetv2_predict``.

Supports:
  - Single-fold prediction
  - Ensemble (all folds) prediction
  - Subprocess-based invocation with real-time logging
  - Per-case timing via prediction manifest JSON
  - CPU / GPU device selection (correct multi-token flag handling)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from ..utils.env_utils import dataset_folder_name, get_env, nnunet_env


class NNUNetPredictor:
    """Runs ``nnUNetv2_predict`` for a given input directory.

    Parameters
    ----------
    input_dir:
        Directory containing ``*_0000.nii.gz`` images.
    output_dir:
        Directory where predictions are written.
    configuration:
        nnU-Net configuration.
    folds:
        List of fold indices to use, or ``"all"`` for ensemble.
    trainer_class:
        nnU-Net trainer class name.
    plans_identifier:
        Plans identifier.
    step_size:
        Sliding-window overlap (0 < step_size ≤ 1).
    use_gaussian:
        Apply Gaussian weighting when aggregating patches (nnU-Net default-on).
    disable_tta:
        Disable test-time augmentation (mirroring).  Set True to speed up
        inference at a small accuracy cost.
    device:
        ``"cuda"`` or ``"cpu"``.  Passed as two separate tokens to the CLI so
        the subprocess receives ``--device cuda`` (not ``"--device cuda"`` as a
        single string, which nnU-Net rejects).
    save_probabilities:
        Save softmax probability maps alongside predictions.
    overwrite_existing:
        Overwrite already predicted files.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        configuration: str | None = None,
        folds: list[int] | str = "all",
        trainer_class: str | None = None,
        plans_identifier: str | None = None,
        step_size: float = 0.5,
        use_gaussian: bool = True,
        disable_tta: bool = False,
        device: str = "cuda",
        save_probabilities: bool = False,
        overwrite_existing: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configuration = configuration or get_env("NNUNET_CONFIGURATION", default="3d_fullres")
        self.folds = folds
        self.trainer_class = trainer_class
        self.plans_identifier = plans_identifier
        self.step_size = step_size
        self.use_gaussian = use_gaussian
        self.disable_tta = disable_tta
        self.device = device.lower()
        self.save_probabilities = save_probabilities
        self.overwrite_existing = overwrite_existing
        self._resolved_model_dir: Path | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self) -> int:
        """Run ``nnUNetv2_predict`` and return the subprocess exit code.

        Writes a ``prediction_manifest.json`` to ``output_dir`` on success.
        """
        dataset_id = int(get_env("DATASET_ID", default="001"))
        env = {**os.environ, **nnunet_env()}
        trainer_class, plans_identifier = self._resolve_model_identifiers()
        self.trainer_class = trainer_class
        self.plans_identifier = plans_identifier

        cmd = self._build_cmd(dataset_id)
        logger.info(f"Inference command: {' '.join(cmd)}")
        if self._resolved_model_dir is not None:
            logger.info(
                f"Resolved model directory: {self._resolved_model_dir} "
                f"(trainer={self.trainer_class}, plans={self.plans_identifier})"
            )

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        proc.wait()
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            logger.error(
                f"nnUNetv2_predict exited with code {proc.returncode} "
                f"after {elapsed:.1f}s"
            )
        else:
            preds = list(self.output_dir.glob("*.nii.gz"))
            logger.success(
                f"Prediction complete: {len(preds)} files in {elapsed:.1f}s "
                f"→ {self.output_dir}"
            )
            self._write_manifest(dataset_id, elapsed, len(preds))

        return proc.returncode

    def predict_single(self, image_path: str | Path, case_id: str | None = None) -> Path:
        """Predict a single image by copying it to a temp folder and running inference.

        Parameters
        ----------
        image_path:
            Path to a ``*_0000.nii.gz`` image file.
        case_id:
            Optional override for the output file stem.

        Returns
        -------
        Path
            Path to the predicted segmentation NIfTI.
        """
        import shutil
        import tempfile

        image_path = Path(image_path)
        if case_id is None:
            case_id = image_path.name.replace("_0000.nii.gz", "")

        with tempfile.TemporaryDirectory(prefix="nnunet_predict_") as tmp:
            tmp_in = Path(tmp) / "input"
            tmp_in.mkdir()
            shutil.copy2(image_path, tmp_in / image_path.name)

            old_input = self.input_dir
            self.input_dir = tmp_in
            rc = self.predict()
            self.input_dir = old_input

        if rc != 0:
            raise RuntimeError(f"Prediction failed for {image_path}")

        pred_path = self.output_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Expected prediction not found: {pred_path}")
        return pred_path

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_cmd(self, dataset_id: int) -> list[str]:
        assert self.trainer_class is not None
        assert self.plans_identifier is not None
        folds_tokens: list[str] = (
            ["all"] if self.folds == "all" else [str(f) for f in self.folds]
        )

        cmd: list[str] = [
            "nnUNetv2_predict",
            "-i", str(self.input_dir),
            "-o", str(self.output_dir),
            "-d", str(dataset_id),
            "-c", self.configuration,
            "-tr", self.trainer_class,
            "-p", self.plans_identifier,
            "-f", *folds_tokens,
            "-step_size", str(self.step_size),
            "-device", self.device,
        ]

        if self.disable_tta:
            cmd.append("--disable_tta")

        if self.save_probabilities:
            cmd.append("--save_probabilities")

        return cmd

    def _resolve_model_identifiers(self) -> tuple[str, str]:
        """Resolve trainer / plans identifiers from env or trained-model folders."""
        trainer_class = self.trainer_class or get_env(
            "NNUNET_TRAINER_CLASS", default=None, required=False
        )
        plans_identifier = self.plans_identifier or get_env(
            "NNUNET_PLANS_IDENTIFIER", default=None, required=False
        )

        if trainer_class and plans_identifier:
            return trainer_class, plans_identifier

        results_root = Path(nnunet_env()["nnUNet_results"]) / dataset_folder_name()
        candidates: list[tuple[int, float, str, str, Path]] = []
        if results_root.exists():
            for model_dir in results_root.iterdir():
                if not model_dir.is_dir():
                    continue
                parts = model_dir.name.split("__")
                if len(parts) != 3:
                    continue
                cand_trainer, cand_plans, cand_config = parts
                if cand_config != self.configuration:
                    continue
                if trainer_class and cand_trainer != trainer_class:
                    continue
                if plans_identifier and cand_plans != plans_identifier:
                    continue

                # Prefer directories that already contain fold outputs.
                fold_dirs = list(model_dir.glob("fold_*"))
                score = len(fold_dirs)
                newest_mtime = max(
                    (p.stat().st_mtime for p in fold_dirs),
                    default=model_dir.stat().st_mtime,
                )
                candidates.append(
                    (score, newest_mtime, cand_trainer, cand_plans, model_dir)
                )

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best = candidates[0]
            if len(candidates) > 1:
                logger.warning(
                    "Multiple nnU-Net model directories matched "
                    f"configuration '{self.configuration}'. "
                    f"Using '{best[4].name}'."
                )
            self._resolved_model_dir = best[4]
            return trainer_class or best[2], plans_identifier or best[3]

        return trainer_class or "nnUNetTrainer", plans_identifier or "nnUNetPlans"

    def _write_manifest(
        self, dataset_id: int, elapsed_s: float, n_preds: int
    ) -> Path:
        """Write a JSON manifest summarising this prediction run."""
        preds = sorted(p.name for p in self.output_dir.glob("*.nii.gz"))
        manifest = {
            "dataset_id": dataset_id,
            "configuration": self.configuration,
            "trainer_class": self.trainer_class,
            "plans_identifier": self.plans_identifier,
            "folds": self.folds if isinstance(self.folds, str) else list(self.folds),
            "step_size": self.step_size,
            "device": self.device,
            "disable_tta": self.disable_tta,
            "save_probabilities": self.save_probabilities,
            "n_predictions": n_preds,
            "elapsed_seconds": round(elapsed_s, 2),
            "predictions": preds,
        }
        path = self.output_dir / "prediction_manifest.json"
        with path.open("w") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info(f"Prediction manifest → {path}")
        return path
