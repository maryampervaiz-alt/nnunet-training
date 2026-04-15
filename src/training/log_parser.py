"""Real-time parser for nnU-Net v2 stdout / training log files.

nnU-Net v2 (``nnUNetTrainer``) writes its training progress to:
  - stdout (captured by our subprocess pipe)
  - ``<fold_dir>/training_log_<timestamp>.txt``

The log entries of interest follow this format (timestamp is optional):

    [timestamp] Epoch: <N>
    [timestamp] train_loss <float>
    [timestamp] val_loss <float>
    [timestamp] Pseudo dice [<float>, ...]
    [timestamp] Epoch time: <float> s

This module provides:
  - :class:`NNUNetLogParser` — stateful line-by-line parser
  - :func:`parse_training_log_file` — batch parse a completed log file
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from loguru import logger


# ─── Epoch metric record ─────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Metrics extracted for a single training epoch."""
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    val_dice: float | None = None       # mean foreground Dice
    epoch_time_s: float | None = None
    learning_rate: float | None = None

    @property
    def complete(self) -> bool:
        """True when the three primary metrics are all present."""
        return (
            self.train_loss is not None
            and self.val_loss is not None
            and self.val_dice is not None
        )

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_dice": self.val_dice,
            "epoch_time_s": self.epoch_time_s,
            "learning_rate": self.learning_rate,
        }


# ─── Regex patterns ───────────────────────────────────────────────────────────

# Optional leading timestamp: "2024-01-01 12:00:00.000000 "
_TS = r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ )?"

_PATTERNS = {
    # "Epoch: 42" or "Epoch 42"
    "epoch":     re.compile(_TS + r"[Ee]poch[:\s]+(\d+)", re.IGNORECASE),
    # "train_loss -0.8234" or "train loss : -0.8234"
    "train_loss": re.compile(_TS + r"train[_ ]loss\s*:?\s*([-\d.eE+]+)", re.IGNORECASE),
    # "val_loss -0.9123" or "val loss : -0.9123"
    "val_loss":  re.compile(_TS + r"val[_ ]loss\s*:?\s*([-\d.eE+]+)", re.IGNORECASE),
    # "Pseudo dice [0.8745]" or "Pseudo dice 0.8745" or "Mean fg Dice [0.87]"
    "val_dice":  re.compile(
        _TS + r"(?:Pseudo\s+dice|Mean\s+fg\s+[Dd]ice)\s*:?\s*\[?([-\d.,\s eE+]+)\]?",
        re.IGNORECASE,
    ),
    # "Epoch time: 234.5 s" or "This epoch took 234.5 s"
    "epoch_time": re.compile(
        _TS + r"(?:[Ee]poch\s+time|[Tt]his\s+epoch\s+took)\s*:?\s*([\d.]+)\s*s",
        re.IGNORECASE,
    ),
    # "Current learning rate: 0.01"
    "lr":        re.compile(_TS + r"[Cc]urrent\s+learning\s+rate\s*:?\s*([\d.eE+-]+)", re.IGNORECASE),
}


def _parse_dice_value(raw: str) -> float | None:
    """Extract a single float from a possibly comma-separated list."""
    parts = [p.strip() for p in raw.replace("[", "").replace("]", "").split(",")]
    floats = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError:
            pass
    if not floats:
        return None
    return sum(floats) / len(floats)  # mean over classes if multi-class


# ─── Stateful line-by-line parser ────────────────────────────────────────────

class NNUNetLogParser:
    """Stateful parser that consumes lines one at a time.

    Each time a complete :class:`EpochMetrics` becomes available (i.e. when
    a new ``Epoch:`` line is seen after a previous one), the finished record
    is appended to ``self.completed``.

    Parameters
    ----------
    fold:
        Fold index — only used for log messages.
    """

    def __init__(self, fold: int = 0) -> None:
        self.fold = fold
        self._pending: EpochMetrics | None = None
        self.completed: list[EpochMetrics] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def feed_line(self, line: str) -> EpochMetrics | None:
        """Process a single log line.

        Returns
        -------
        EpochMetrics | None
            The *just-completed* epoch record if this line triggered
            finalization; ``None`` otherwise.
        """
        line = line.rstrip()
        if not line:
            return None

        # ── epoch marker ──────────────────────────────────────────────────────
        m = _PATTERNS["epoch"].search(line)
        if m:
            epoch_idx = int(m.group(1))
            # Finalise previous pending record
            finished = self._finalise()
            self._pending = EpochMetrics(epoch=epoch_idx)
            return finished

        if self._pending is None:
            return None

        # ── per-epoch fields ──────────────────────────────────────────────────
        for key, pat in _PATTERNS.items():
            if key == "epoch":
                continue
            m = pat.search(line)
            if m:
                raw = m.group(1)
                try:
                    if key == "val_dice":
                        self._pending.val_dice = _parse_dice_value(raw)
                    elif key == "train_loss":
                        self._pending.train_loss = float(raw)
                    elif key == "val_loss":
                        self._pending.val_loss = float(raw)
                    elif key == "epoch_time":
                        self._pending.epoch_time_s = float(raw)
                    elif key == "lr":
                        self._pending.learning_rate = float(raw)
                except ValueError:
                    logger.debug(f"[fold_{self.fold}] Could not parse '{key}' from: {line!r}")
                break

        return None

    def flush(self) -> EpochMetrics | None:
        """Finalise any in-progress epoch record (call after subprocess ends)."""
        return self._finalise()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _finalise(self) -> EpochMetrics | None:
        if self._pending is not None and self._pending.complete:
            self.completed.append(self._pending)
            finished = self._pending
            self._pending = None
            return finished
        self._pending = None
        return None


# ─── Batch log file parser ────────────────────────────────────────────────────

def parse_training_log_file(log_path: str | Path, fold: int = 0) -> list[EpochMetrics]:
    """Parse a completed nnU-Net training log file.

    Parameters
    ----------
    log_path:
        Path to ``training_log_<timestamp>.txt``.
    fold:
        Fold index for log messages.

    Returns
    -------
    list[EpochMetrics]
        One record per completed epoch, in training order.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        logger.warning(f"Training log not found: {log_path}")
        return []

    parser = NNUNetLogParser(fold=fold)
    with log_path.open(errors="replace") as fh:
        for line in fh:
            parser.feed_line(line)
    parser.flush()

    logger.info(f"Parsed {len(parser.completed)} epochs from {log_path.name}")
    return parser.completed


def find_training_log(fold_dir: str | Path) -> Path | None:
    """Find the most recent ``training_log_*.txt`` in *fold_dir*."""
    fold_dir = Path(fold_dir)
    candidates = sorted(fold_dir.glob("training_log_*.txt"))
    if not candidates:
        # Also check for plain training_log.txt
        plain = fold_dir / "training_log.txt"
        return plain if plain.exists() else None
    return candidates[-1]  # most recent by name (timestamp-sorted)
