"""K-fold cross-validation orchestrator.

Responsibilities
----------------
- Iterate over all (or selected) folds in sequence
- For each fold: create :class:`FoldTrainer`, run, collect metrics
- Write a per-fold CSV summary via :class:`FoldLogger`
- Write an aggregate CV summary CSV
- Archive checkpoints via :class:`CheckpointManager`
- Track the global best fold (by best validation Dice)
- Log hyperparameters + metrics to MLflow / CSV via :class:`ExperimentTracker`
- Load the nnU-Net auto-generated splits to show fold sizes before training
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from ..utils.env_utils import get_env
from ..utils.experiment_tracker import ExperimentTracker
from .checkpoint_manager import CheckpointManager
from .fold_logger import FoldLogger
from .trainer import FoldTrainer


class CrossValidationOrchestrator:
    """Orchestrate K-fold cross-validation using nnU-Net.

    Parameters
    ----------
    num_folds:
        Number of folds (must match nnU-Net's splits file).
    configuration:
        nnU-Net configuration (e.g. ``"3d_fullres"``).
    trainer_class:
        nnU-Net trainer class name.
    plans_identifier:
        Plans identifier.
    continue_training:
        Resume each fold from its existing checkpoint.
    seed:
        Global random seed (applied per-fold via env var propagation).
    es_patience:
        Early stopping patience.  ``0`` disables early stopping.
    es_min_delta:
        Minimum Dice improvement to reset patience.
    es_warmup:
        Warmup epochs before early stopping is active.
    experiment_name:
        MLflow experiment name.
    metrics_dir:
        Directory for CSV logs.
    checkpoints_dir:
        Root of the managed checkpoint directory.
    """

    def __init__(
        self,
        num_folds: int | None = None,
        configuration: str | None = None,
        trainer_class: str = "nnUNetTrainerEarlyStopping",
        plans_identifier: str = "nnUNetPlans",
        continue_training: bool = False,
        seed: int = 42,
        num_epochs: int | None = None,
        es_patience: int = 50,
        es_min_delta: float = 1e-4,
        es_warmup: int = 50,
        experiment_name: str | None = None,
        metrics_dir: str | Path = "metrics",
        checkpoints_dir: str | Path = "checkpoints",
    ) -> None:
        self.num_folds = num_folds or int(get_env("NUM_FOLDS", default="5"))
        self.configuration = configuration or get_env("NNUNET_CONFIGURATION", default="3d_fullres")
        self.trainer_class = trainer_class
        self.plans_identifier = plans_identifier
        self.continue_training = continue_training
        self.seed = seed
        self.num_epochs = num_epochs
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.es_warmup = es_warmup

        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        exp_name = experiment_name or get_env("EXPERIMENT_NAME", default="nnunet_men_rt")
        mlflow_uri = get_env("MLFLOW_TRACKING_URI", default="experiments/mlruns")
        self.tracker = ExperimentTracker(experiment_name=exp_name, tracking_uri=mlflow_uri)
        self.ckpt_manager = CheckpointManager(root=checkpoints_dir)

        # Populated during run()
        self._fold_results: dict[int, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, folds: list[int] | None = None) -> dict[int, int]:
        """Run all (or selected) folds sequentially.

        Parameters
        ----------
        folds:
            Explicit fold indices to train.  Defaults to
            ``range(self.num_folds)``.

        Returns
        -------
        dict[int, int]
            Fold index → subprocess return code (0 = success).
        """
        fold_indices = folds if folds is not None else list(range(self.num_folds))
        self._log_splits_summary(fold_indices)

        logger.info(
            f"Starting {len(fold_indices)}-fold CV | "
            f"config={self.configuration} | "
            f"trainer={self.trainer_class} | "
            f"seed={self.seed} | "
            f"ES patience={self.es_patience}"
        )

        return_codes: dict[int, int] = {}

        for fold in fold_indices:
            rc = self._run_fold(fold)
            return_codes[fold] = rc

        # ── Aggregate summary ─────────────────────────────────────────────────
        self._write_cv_summary()
        self.ckpt_manager.write_global_best_manifest(self._fold_results)

        n_ok = sum(v == 0 for v in return_codes.values())
        logger.info(
            f"Cross-validation complete: {n_ok}/{len(fold_indices)} folds OK"
        )

        if self._fold_results:
            best_fold = max(
                self._fold_results,
                key=lambda f: self._fold_results[f].get("best_val_dice") or -1,
            )
            logger.info(
                f"Best fold: {best_fold} | "
                f"Dice = {self._fold_results[best_fold].get('best_val_dice'):.4f}"
            )

        return return_codes

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_fold(self, fold: int) -> int:
        """Train a single fold and record results."""
        run_name = f"fold_{fold}"
        t0 = time.monotonic()

        with self.tracker.start_run(run_name=run_name, tags={"fold": str(fold), "seed": str(self.seed)}):
            self.tracker.log_params({
                "fold": fold,
                "configuration": self.configuration,
                "trainer_class": self.trainer_class,
                "plans_identifier": self.plans_identifier,
                "num_folds": self.num_folds,
                "seed": self.seed,
                "es_patience": self.es_patience,
                "es_min_delta": self.es_min_delta,
                "es_warmup": self.es_warmup,
            })

            trainer = FoldTrainer(
                configuration=self.configuration,
                fold=fold,
                trainer_class=self.trainer_class,
                plans_identifier=self.plans_identifier,
                continue_training=self.continue_training,
                seed=self.seed,
                num_epochs=self.num_epochs,
                es_patience=self.es_patience,
                es_min_delta=self.es_min_delta,
                es_warmup=self.es_warmup,
                metrics_dir=self.metrics_dir,
            )

            rc = trainer.run()
            elapsed = time.monotonic() - t0

            fold_metrics: dict = {"fold": fold, "rc": rc, "wall_time_s": elapsed}

            if trainer.fold_logger is not None:
                fold_metrics.update(trainer.fold_logger.final_metrics())
                # Upload per-fold CSV as artifact
                self.tracker.log_artifact(trainer.fold_logger.csv_path)

            self._fold_results[fold] = fold_metrics

            if rc == 0:
                # Archive nnU-Net checkpoints
                ckpt_paths = self.ckpt_manager.archive_fold(
                    fold=fold,
                    configuration=self.configuration,
                    trainer_class=self.trainer_class,
                    plans_identifier=self.plans_identifier,
                )
                self.tracker.log_metrics(
                    {
                        "training_success": 1.0,
                        "best_val_dice": fold_metrics.get("best_val_dice") or 0.0,
                        "epochs_trained": fold_metrics.get("epochs_trained") or 0,
                        "wall_time_s": elapsed,
                    },
                    step=fold,
                )
            else:
                logger.error(f"[fold_{fold}] FAILED (rc={rc})")
                self.tracker.log_metrics({"training_success": 0.0}, step=fold)

        return rc

    def _write_cv_summary(self) -> None:
        """Write aggregate CSV from all fold results."""
        if not self._fold_results:
            return

        rows = list(self._fold_results.values())
        summary_csv = self.metrics_dir / "cv_summary.csv"
        fieldnames = sorted({k for r in rows for k in r})
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CV summary → {summary_csv}")

        # Also write a concatenated per-epoch CSV across all folds
        fold_csvs = sorted(self.metrics_dir.glob("fold_*_training.csv"))
        if fold_csvs:
            dfs = [pd.read_csv(f) for f in fold_csvs]
            combined = pd.concat(dfs, ignore_index=True)
            all_epochs_csv = self.metrics_dir / "all_folds_training.csv"
            combined.to_csv(all_epochs_csv, index=False)
            logger.info(f"All-folds epoch log → {all_epochs_csv}")
            self.tracker.log_artifact(all_epochs_csv)

        self.tracker.export_csv(self.metrics_dir / "mlflow_metrics.csv")

    def _log_splits_summary(self, fold_indices: list[int]) -> None:
        """Attempt to load splits_final.json and log fold sizes."""
        try:
            from ..data.splitter import load_splits, summarise_splits
            splits = load_splits()
            logger.info("CV splits loaded from nnU-Net preprocessed directory:")
            summarise_splits(splits)
        except Exception as exc:
            logger.debug(f"Could not load splits: {exc}")
