"""Single-fold nnU-Net trainer with real-time log parsing and early stopping.

Responsibilities
----------------
- Build the ``nnUNetv2_train`` command (no internal config overrides)
- Inject env vars: nnU-Net paths, seed, early stopping params, GPU selection
- Launch subprocess and consume stdout line-by-line
- Feed each line to :class:`~src.training.fold_logger.FoldLogger` for
  real-time CSV writing and early stopping decisions
- On early-stopping signal: terminate subprocess gracefully (SIGTERM → SIGKILL)
- After subprocess exits: parse any remaining log lines and write final CSV
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

from loguru import logger

from ..utils.env_utils import dataset_folder_name, get_env, nnunet_env
from .early_stopping import EarlyStoppingState
from .fold_logger import FoldLogger
from .reproducibility import cuda_info, seed_env_for_subprocess, set_global_seed


class FoldTrainer:
    """Train a single nnU-Net fold.

    Parameters
    ----------
    configuration:
        nnU-Net configuration (e.g. ``"3d_fullres"``).  Resolved from
        ``$NNUNET_CONFIGURATION`` when ``None``.
    fold:
        Fold index (0-based).
    trainer_class:
        nnU-Net trainer class name.  Use ``"nnUNetTrainerEarlyStopping"``
        to enable early stopping inside the nnU-Net process.
    plans_identifier:
        nnU-Net plans identifier.
    continue_training:
        Pass ``--c`` to nnU-Net to resume from an existing checkpoint.
    seed:
        Random seed propagated to the subprocess via ``$NNUNET_SEED``.
    es_patience:
        Early stopping patience (epochs).  ``0`` disables subprocess-level ES.
    es_min_delta:
        Minimum validation Dice improvement to reset patience.
    es_warmup:
        Epochs before early stopping is active.
    metrics_dir:
        Directory for per-fold CSV logs.
    extra_args:
        Additional CLI arguments forwarded verbatim.
    """

    # Grace period (seconds) between SIGTERM and SIGKILL
    _TERM_TIMEOUT: int = 60

    def __init__(
        self,
        configuration: str | None = None,
        fold: int = 0,
        trainer_class: str = "nnUNetTrainerEarlyStopping",
        plans_identifier: str = "nnUNetPlans",
        continue_training: bool = False,
        seed: int = 42,
        num_epochs: int | None = None,
        es_patience: int = 50,
        es_min_delta: float = 1e-4,
        es_warmup: int = 50,
        metrics_dir: str | Path = "metrics",
        extra_args: list[str] | None = None,
    ) -> None:
        self.configuration = configuration or get_env("NNUNET_CONFIGURATION", default="3d_fullres")
        self.fold = fold
        self.trainer_class = trainer_class
        self.plans_identifier = plans_identifier
        self.continue_training = continue_training
        self.seed = seed
        self.num_epochs = num_epochs
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.es_warmup = es_warmup
        self.metrics_dir = Path(metrics_dir)
        self.extra_args = extra_args or []

        self._fold_logger: FoldLogger | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> int:
        """Execute training for this fold.

        Returns
        -------
        int
            Subprocess exit code (0 = success).
        """
        set_global_seed(self.seed)
        gpu_info = cuda_info()
        logger.info(f"[fold_{self.fold}] GPU info: {gpu_info}")

        env = self._build_env()
        cmd = self._build_cmd()
        logger.info(f"[fold_{self.fold}] Command: {' '.join(cmd)}")

        # Early stopping monitor (subprocess-level, independent of in-process ES)
        es_state: EarlyStoppingState | None = None
        if self.es_patience > 0:
            es_state = EarlyStoppingState(
                patience=self.es_patience,
                min_delta=self.es_min_delta,
                mode="max",
                warmup_epochs=self.es_warmup,
            )

        self._fold_logger = FoldLogger(
            fold=self.fold,
            metrics_dir=self.metrics_dir,
            es_state=es_state,
        )

        rc = self._run_subprocess(cmd, env)
        return rc

    @property
    def fold_logger(self) -> FoldLogger | None:
        return self._fold_logger

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_env(self) -> dict[str, str]:
        env = {
            **os.environ,
            **nnunet_env(),
            **seed_env_for_subprocess(self.seed),
            # Early stopping params read by nnUNetTrainerEarlyStopping
            "ES_PATIENCE": str(self.es_patience),
            "ES_MIN_DELTA": str(self.es_min_delta),
            "ES_WARMUP": str(self.es_warmup),
            **({"NNUNET_NUM_EPOCHS": str(self.num_epochs)} if self.num_epochs is not None else {}),
        }
        return env

    def _build_cmd(self) -> list[str]:
        dataset_id = int(get_env("DATASET_ID", default="001"))
        cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            self.configuration,
            str(self.fold),
            "-tr", self.trainer_class,
            "-p", self.plans_identifier,
        ]
        # num_epochs passed via NNUNET_NUM_EPOCHS env var (nnUNetv2_train CLI has no --num_epochs)
        if self.continue_training:
            cmd.append("--c")
        cmd += self.extra_args
        return cmd

    def _run_subprocess(self, cmd: list[str], env: dict[str, str]) -> int:
        """Run the subprocess, feed lines to FoldLogger, handle ES termination."""
        assert self._fold_logger is not None

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                # Echo to our stdout
                sys.stdout.write(raw_line)
                sys.stdout.flush()

                # Feed to fold logger (returns True when ES fires)
                stop = self._fold_logger.feed_line(raw_line)

                if stop and proc.poll() is None:
                    logger.warning(
                        f"[fold_{self.fold}] Early stopping triggered — "
                        "sending SIGTERM to training process …"
                    )
                    self._terminate_gracefully(proc)
                    break

        finally:
            self._fold_logger.close()

        proc.wait()
        rc = proc.returncode

        if rc not in (0, -15, -9):  # 0=OK, -15=SIGTERM, -9=SIGKILL (ES)
            logger.error(f"[fold_{self.fold}] nnUNetv2_train exited with rc={rc}")
        elif self._fold_logger.stop_requested:
            logger.info(
                f"[fold_{self.fold}] Training stopped early at epoch "
                f"{self._fold_logger.epoch_count}. "
                f"Best Dice = {self._fold_logger.best_dice()}"
            )
        else:
            logger.success(
                f"[fold_{self.fold}] Training complete. "
                f"Best Dice = {self._fold_logger.best_dice()}"
            )

        # Treat early-stopped processes as successful (rc was SIGTERM)
        return 0 if self._fold_logger.stop_requested else rc

    def _terminate_gracefully(self, proc: subprocess.Popen) -> None:
        """Send SIGTERM; escalate to SIGKILL after _TERM_TIMEOUT seconds."""
        try:
            proc.terminate()
        except OSError:
            return

        deadline = time.monotonic() + self._TERM_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(1.0)

        if proc.poll() is None:
            logger.warning(f"[fold_{self.fold}] Escalating to SIGKILL.")
            try:
                proc.kill()
            except OSError:
                pass
