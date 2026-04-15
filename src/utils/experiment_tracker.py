"""Dual-backend experiment tracker: MLflow + CSV.

Usage
-----
::

    tracker = ExperimentTracker(experiment_name="baseline", tracking_uri="experiments/mlruns")
    with tracker.start_run(run_name="fold_0"):
        tracker.log_params({"fold": 0, "configuration": "3d_fullres"})
        tracker.log_metrics({"dice": 0.91, "hd95": 3.4}, step=1000)
    tracker.export_csv("metrics/summary.csv")
"""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger


class ExperimentTracker:
    """Thin wrapper around MLflow that also maintains an in-process CSV log."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | Path = "experiments/mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = str(tracking_uri)
        self._rows: list[dict[str, Any]] = []
        self._current_run_name: str | None = None
        self._active_run = None

        try:
            import mlflow

            Path(tracking_uri).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            logger.info(f"MLflow tracking at: {self.tracking_uri}")
        except ImportError:
            logger.warning("mlflow not installed — CSV-only tracking mode.")
            self._mlflow = None

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        """Context manager that wraps a single training run."""
        self._current_run_name = run_name
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name, tags=tags or {}):
                yield self
        else:
            yield self
        self._current_run_name = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyper-parameters."""
        logger.debug(f"[{self._current_run_name}] params: {params}")
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics and append to CSV buffer."""
        row = {
            "run_name": self._current_run_name,
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **metrics,
        }
        self._rows.append(row)
        logger.info(f"[{self._current_run_name}] step={step} | {metrics}")
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file artifact to MLflow."""
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(local_path))

    def export_csv(self, csv_path: str | Path) -> Path:
        """Write all accumulated metric rows to *csv_path*."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            logger.warning("No metrics to export.")
            return csv_path
        fieldnames = list(self._rows[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)
        logger.info(f"Metrics exported → {csv_path}")
        return csv_path
