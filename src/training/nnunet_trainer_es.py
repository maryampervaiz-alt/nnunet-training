"""Custom nnU-Net v2 trainer with early stopping and seed support."""
from __future__ import annotations

import os

# Guard: only importable when nnunetv2 is installed (not needed at parse time).
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
except ImportError:
    class nnUNetTrainer:  # type: ignore[no-redef]
        pass


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """nnUNetTrainer subclass with early stopping and seeded reproducibility."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device=None,
    ) -> None:
        # IMPORTANT: keep signature aligned with installed nnunetv2
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        seed = int(os.environ.get("NNUNET_SEED", "42"))
        self._apply_seed(seed)

        self._es_patience: int = int(os.environ.get("ES_PATIENCE", "50"))
        self._es_min_delta: float = float(os.environ.get("ES_MIN_DELTA", "1e-4"))
        self._es_warmup: int = int(os.environ.get("ES_WARMUP", "50"))

        _num_epochs_env = os.environ.get("NNUNET_NUM_EPOCHS")
        if _num_epochs_env is not None:
            self.num_epochs = int(_num_epochs_env)

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

    def on_epoch_end(self) -> None:
        super().on_epoch_end()

        if self._es_triggered:
            return

        current_epoch = self.current_epoch
        if current_epoch < self._es_warmup:
            return

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
                self.num_epochs = current_epoch

    def print_to_log_file(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], str) and "Yayy!" in args[0]:
            args = (args[0].replace("Yayy! New best EMA pseudo Dice:", "New best EMA pseudo Dice:"),) + args[1:]
        super().print_to_log_file(*args, **kwargs)

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