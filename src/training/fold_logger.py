"""Per-fold real-time logger: parses nnU-Net stdout and writes epoch metrics to CSV.

Architecture
------------
``FoldLogger`` is used inside ``FoldTrainer`` to consume stdout lines from
the nnU-Net subprocess as they arrive.  For each complete epoch it:

1. Appends a row to an in-memory list.
2. Immediately flushes the row to ``metrics/fold_{N}_training.csv``.
3. Queries :class:`~src.training.early_stopping.EarlyStoppingState`
   and signals the caller when stopping is required.

CSV columns
-----------
``fold, epoch, train_loss, val_loss, val_dice, epoch_time_s,
learning_rate, es_wait, es_best_dice``
"""
from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger

from .early_stopping import EarlyStoppingState
from .log_parser import EpochMetrics, NNUNetLogParser


class FoldLogger:
    """Real-time per-fold metric logger with integrated early stopping.

    Parameters
    ----------
    fold:
        Fold index.
    metrics_dir:
        Directory where the CSV is written.
    es_state:
        Early stopping monitor.  When ``None`` early stopping is disabled.
    """

    _CSV_FIELDS = [
        "fold",
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "epoch_time_s",
        "learning_rate",
        "es_wait",
        "es_best_dice",
    ]

    def __init__(
        self,
        fold: int,
        metrics_dir: str | Path = "metrics",
        es_state: EarlyStoppingState | None = None,
    ) -> None:
        self.fold = fold
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.es = es_state

        self._parser = NNUNetLogParser(fold=fold)
        self._rows: list[dict] = []
        self._stop_requested = False

        self._csv_path = self.metrics_dir / f"fold_{fold}_training.csv"
        self._csv_fh = self._csv_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._csv_fh, fieldnames=self._CSV_FIELDS, extrasaction="ignore"
        )
        self._writer.writeheader()
        self._csv_fh.flush()
        logger.debug(f"[fold_{fold}] CSV log → {self._csv_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def feed_line(self, line: str) -> bool:
        """Process one stdout line.

        Returns
        -------
        bool
            ``True`` when early stopping has been triggered and the caller
            should terminate the subprocess.
        """
        finished = self._parser.feed_line(line)
        if finished is not None:
            self._record_epoch(finished)
        return self._stop_requested

    def flush(self) -> None:
        """Finalise the last in-progress epoch record."""
        finished = self._parser.flush()
        if finished is not None:
            self._record_epoch(finished)
        self._csv_fh.flush()

    def close(self) -> None:
        """Close the CSV file handle."""
        self.flush()
        self._csv_fh.close()
        logger.info(
            f"[fold_{self.fold}] Training log closed. "
            f"{len(self._rows)} epochs written → {self._csv_path}"
        )

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def epoch_count(self) -> int:
        return len(self._rows)

    def best_dice(self) -> float | None:
        """Return the best validation Dice recorded so far."""
        dices = [r["val_dice"] for r in self._rows if r["val_dice"] is not None]
        return max(dices) if dices else None

    def final_metrics(self) -> dict:
        """Summary dict for the completed fold (used by orchestrator)."""
        if not self._rows:
            return {"fold": self.fold, "epochs": 0}
        last = self._rows[-1]
        best = self.best_dice()
        return {
            "fold": self.fold,
            "epochs_trained": len(self._rows),
            "best_val_dice": best,
            "final_val_dice": last.get("val_dice"),
            "final_train_loss": last.get("train_loss"),
            "final_val_loss": last.get("val_loss"),
            "early_stopped": self._stop_requested,
            "es_stopped_epoch": self.es.stopped_epoch if self.es else -1,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _record_epoch(self, em: EpochMetrics) -> None:
        es_wait = self.es.wait if self.es else 0
        es_best = self.es.best_value if self.es else None

        row = {
            "fold": self.fold,
            "epoch": em.epoch,
            "train_loss": em.train_loss,
            "val_loss": em.val_loss,
            "val_dice": em.val_dice,
            "epoch_time_s": em.epoch_time_s,
            "learning_rate": em.learning_rate,
            "es_wait": es_wait,
            "es_best_dice": es_best,
        }
        self._rows.append(row)
        self._writer.writerow(row)
        self._csv_fh.flush()

        logger.info(
            f"[fold_{self.fold}] epoch={em.epoch:04d} | "
            f"loss={em.train_loss:.4f} | "
            f"val_loss={em.val_loss:.4f} | "
            f"dice={em.val_dice:.4f}"
            + (f" | {self.es.status_line(em.epoch)}" if self.es else "")
        )

        # Early stopping check
        if self.es is not None and em.val_dice is not None:
            if self.es.update(em.val_dice, em.epoch):
                self._stop_requested = True
