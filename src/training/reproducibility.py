"""Reproducibility utilities: deterministic seeding across all RNG sources.

nnU-Net does not expose a seed argument.  We seed everything that lives
*outside* the nnU-Net subprocess (Python stdlib, NumPy, PyTorch) before
launching training, and pass the seed into the subprocess via an env var
so the custom trainer class can read and apply it.
"""
from __future__ import annotations

import os
import random

from loguru import logger


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.  Must be in [0, 2**32 - 1].
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms — may slow down training slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seeded: {seed} | cudnn.deterministic=True")
    except ImportError:
        pass

    logger.info(f"Global seed set to {seed}")


def seed_env_for_subprocess(seed: int) -> dict[str, str]:
    """Return env-var additions that propagate the seed into child processes.

    The custom nnU-Net trainer reads ``NNUNET_SEED`` and calls
    :func:`set_global_seed` at init time.
    """
    return {
        "NNUNET_SEED": str(seed),
        "PYTHONHASHSEED": str(seed),
    }


def cuda_info() -> dict[str, str | int | bool]:
    """Return a dict of GPU availability info for logging."""
    info: dict[str, str | int | bool] = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "unknown"
    except ImportError:
        info["cuda_available"] = False
    return info
