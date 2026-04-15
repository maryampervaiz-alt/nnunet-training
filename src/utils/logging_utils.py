"""Centralised logging configuration using loguru + rich."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from rich.logging import RichHandler


_CONFIGURED: bool = False


def get_logger(
    name: str = "nnunet-men-rt",
    log_dir: str | Path | None = None,
    level: str = "INFO",
) -> "logger":  # noqa: F821
    """Configure and return a loguru logger.

    First call initialises the global logger (removes loguru's default sink and
    adds a rich console sink + an optional rotating file sink).  Subsequent calls
    with the same *name* are no-ops and return the same global logger.

    Parameters
    ----------
    name:
        Logical name embedded in the log file name.
    log_dir:
        Directory for the rotating file sink.  If ``None``, file logging is skipped.
    level:
        Minimum log level (DEBUG / INFO / WARNING / ERROR).
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return logger

    logger.remove()  # remove default stderr sink

    # Rich console sink
    logger.add(
        RichHandler(rich_tracebacks=True, markup=True),
        format="{message}",
        level=level,
        colorize=False,
    )

    # File sink (rotating, one file per process start)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / f"{name}_{{time:YYYY-MM-DD_HH-mm-ss}}.log",
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

    _CONFIGURED = True
    return logger
