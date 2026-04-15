"""Early stopping state machine.

Decoupled from any framework: tracks a monitored metric over epochs and
signals when training should stop.  Used both by the subprocess monitor
(:class:`~src.training.fold_logger.FoldLogger`) and by the custom
nnU-Net trainer class (:mod:`src.training.nnunet_trainer_es`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger


@dataclass
class EarlyStoppingState:
    """Mutable state for one early-stopping monitor.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    min_delta:
        Minimum change in monitored metric to qualify as improvement.
    mode:
        ``"max"`` (higher is better, e.g. Dice) or ``"min"`` (lower is
        better, e.g. loss).
    warmup_epochs:
        Number of initial epochs during which early stopping is disabled.
    """

    patience: int = 50
    min_delta: float = 1e-4
    mode: Literal["max", "min"] = "max"
    warmup_epochs: int = 50

    # Internal state (not constructor args)
    best_value: float = field(init=False)
    wait: int = field(default=0, init=False)
    stopped_epoch: int = field(default=-1, init=False)
    history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.best_value = float("-inf") if self.mode == "max" else float("inf")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, value: float, epoch: int) -> bool:
        """Record a new metric value and decide whether to stop.

        Parameters
        ----------
        value:
            The monitored metric for this epoch.
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        bool
            ``True`` if training should stop now.
        """
        self.history.append(value)

        if epoch < self.warmup_epochs:
            return False

        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.wait = 0
            logger.debug(
                f"[EarlyStopping] Epoch {epoch}: improvement → best={self.best_value:.6f}"
            )
        else:
            self.wait += 1
            logger.debug(
                f"[EarlyStopping] Epoch {epoch}: no improvement ({self.wait}/{self.patience})"
            )

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(
                f"[EarlyStopping] Triggered at epoch {epoch}. "
                f"Best {self.mode}={self.best_value:.6f} "
                f"(patience={self.patience})"
            )
            return True

        return False

    @property
    def triggered(self) -> bool:
        """Return True if early stopping has fired."""
        return self.stopped_epoch >= 0

    def status_line(self, epoch: int) -> str:
        return (
            f"ES wait={self.wait}/{self.patience} | "
            f"best={self.best_value:.4f} | "
            f"epoch={epoch}"
        )
