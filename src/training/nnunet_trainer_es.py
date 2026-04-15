"""Custom nnU-Net v2 trainer with early stopping and seed support.

This module defines ``nnUNetTrainerEarlyStopping``, a minimal subclass of
``nnUNetTrainer`` that adds:

  - **Early stopping** — hooks into ``on_epoch_end`` to monitor validation
    Dice and short-circuits the training loop when improvement stalls.
  - **Reproducibility** — reads ``NNUNET_SEED`` from the environment and
    applies it at init time via :func:`~src.training.reproducibility.set_global_seed`.

No internal nnU-Net configs are overridden (patch size, spacing, batch size,
augmentations, optimiser, LR schedule are all left to auto-configuration).

Usage
-----
Pass the class name to ``nnUNetv2_train``::

    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerEarlyStopping

Or via the :class:`~src.training.trainer.FoldTrainer` ``trainer_class`` parameter.

Environment variables
---------------------
``NNUNET_SEED``
    Integer seed applied at trainer init (default: 42).
``ES_PATIENCE``
    Epochs without improvement before stopping (default: 50).
``ES_MIN_DELTA``
    Minimum Dice improvement to reset patience counter (default: 1e-4).
``ES_WARMUP``
    Initial epochs during which early stopping is disabled (default: 50).
"""
from __future__ import annotations

import os

# Guard: only importable when nnunetv2 is installed (not needed at parse time).
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    _NNUNET_AVAILABLE = True
except ImportError:
    _NNUNET_AVAILABLE = False
    # Provide a stub so the file can be imported for syntax checking
    class nnUNetTrainer:  # type: ignore[no-redef]
        pass


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """nnUNetTrainer subclass with early stopping and seeded reproducibility.

    All constructor arguments are forwarded to ``nnUNetTrainer.__init__``;
    early stopping parameters are read from environment variables so that they
    can be controlled without touching source code.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ── Reproducibility ───────────────────────────────────────────────────
        seed = int(os.environ.get("NNUNET_SEED", "42"))
        self._apply_seed(seed)

        # ── Early stopping parameters ────────────────────────────────────────
        self._es_patience: int = int(os.environ.get("ES_PATIENCE", "50"))
        self._es_min_delta: float = float(os.environ.get("ES_MIN_DELTA", "1e-4"))
        self._es_warmup: int = int(os.environ.get("ES_WARMUP", "50"))

        # Internal state
        self._es_best_dice: float = float("-inf")
        self._es_wait: int = 0
        self._es_triggered: bool = False

        self.print_to_log_file(
            f"nnUNetTrainerEarlyStopping | "
            f"seed={seed} | "
            f"patience={self._es_patience} | "
            f"min_delta={self._es_min_delta} | "
            f"warmup={self._es_warmup}"
        )

    # ── Hook: called once per epoch by nnUNetTrainer.run_training() ───────────

    def on_epoch_end(self) -> None:
        """Delegate to super, then evaluate early stopping condition."""
        super().on_epoch_end()

        if self._es_triggered:
            return

        current_epoch = self.current_epoch  # already incremented by super()
        if current_epoch < self._es_warmup:
            return

        # Extract latest validation Dice from nnU-Net's internal logger
        dice_log = self.logger.my_fantastic_logging.get("mean_fg_dice", [])
        if not dice_log:
            return

        current_dice = float(dice_log[-1])

        if current_dice > self._es_best_dice + self._es_min_delta:
            self._es_best_dice = current_dice
            self._es_wait = 0
        else:
            self._es_wait += 1
            self.print_to_log_file(
                f"EarlyStopping: no improvement {self._es_wait}/{self._es_patience} "
                f"(best={self._es_best_dice:.4f}, current={current_dice:.4f})"
            )

            if self._es_wait >= self._es_patience:
                self.print_to_log_file(
                    f"EarlyStopping: triggered at epoch {current_epoch}. "
                    f"Best val Dice = {self._es_best_dice:.4f}"
                )
                self._es_triggered = True
                # Short-circuit the training loop: nnU-Net checks
                # ``self.current_epoch < self.num_epochs`` before each epoch.
                self.num_epochs = current_epoch

    # ── Seed helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_seed(seed: int) -> None:
        import random
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
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
