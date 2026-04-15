"""Checkpoint management.

Mirrors nnU-Net's internal checkpoints (stored under ``nnUNet_results``) into
a structured, human-navigable ``checkpoints/`` directory.

Directory layout produced
--------------------------
::

    checkpoints/
        fold_0/
            best_model.pth          ← checkpoint_best.pth from nnU-Net
            last_model.pth          ← checkpoint_final.pth from nnU-Net
            metadata.json           ← fold metrics snapshot
        fold_1/
            ...
        fold_4/
            ...
        global_best/
            best_model.pth          ← copy of the fold with highest val Dice
            metadata.json           ← which fold, what Dice
        manifest.json               ← per-fold Dice + global best summary

nnU-Net checkpoint names (v2)
------------------------------
- ``checkpoint_best.pth``  — saved whenever validation Dice improves
- ``checkpoint_final.pth`` — saved at the very end of training
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from ..utils.env_utils import dataset_folder_name, get_path_env


class CheckpointManager:
    """Manages the project-level ``checkpoints/`` directory.

    Parameters
    ----------
    root:
        Root of the managed checkpoint tree.
    """

    _NNUNET_BEST = "checkpoint_best.pth"
    _NNUNET_FINAL = "checkpoint_final.pth"

    def __init__(self, root: str | Path = "checkpoints") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def archive_fold(
        self,
        fold: int | str,
        configuration: str,
        trainer_class: str = "nnUNetTrainerEarlyStopping",
        plans_identifier: str = "nnUNetPlans",
        fold_metrics: dict | None = None,
    ) -> dict[str, Path | None]:
        """Copy best + last checkpoints for *fold* into managed tree.

        Parameters
        ----------
        fold:
            Fold index.
        configuration:
            nnU-Net configuration string.
        trainer_class:
            nnU-Net trainer class name.
        plans_identifier:
            Plans identifier.
        fold_metrics:
            Optional metrics dict written to ``metadata.json``.

        Returns
        -------
        dict[str, Path | None]
            ``{"best": path | None, "last": path | None}``
        """
        nnunet_dir = self._nnunet_fold_dir(fold, configuration, trainer_class, plans_identifier)
        dst_dir = self.root / f"fold_{fold}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        best = self._copy(nnunet_dir / self._NNUNET_BEST, dst_dir / "best_model.pth")
        last = self._copy(nnunet_dir / self._NNUNET_FINAL, dst_dir / "last_model.pth")

        if fold_metrics is not None:
            meta_path = dst_dir / "metadata.json"
            with meta_path.open("w") as fh:
                json.dump(fold_metrics, fh, indent=2, default=str)
            logger.debug(f"Metadata written: {meta_path}")

        logger.info(
            f"Archived fold {fold}: best={'OK' if best else 'MISSING'}, "
            f"last={'OK' if last else 'MISSING'}"
        )
        return {"best": best, "last": last}

    def write_global_best_manifest(self, fold_results: dict[int, dict]) -> None:
        """Identify the globally best fold by val Dice and write manifest + copy.

        Parameters
        ----------
        fold_results:
            Dict mapping fold index → metrics dict (must include
            ``"best_val_dice"`` key).
        """
        if not fold_results:
            return

        # Pick fold with highest best_val_dice
        valid = {
            f: r for f, r in fold_results.items()
            if r.get("best_val_dice") is not None
        }
        if not valid:
            logger.warning("No valid Dice scores found — skipping global best selection.")
            return

        best_fold = max(valid, key=lambda f: valid[f]["best_val_dice"])
        best_dice = valid[best_fold]["best_val_dice"]

        # Copy best_model.pth of best fold → checkpoints/global_best/
        src = self.root / f"fold_{best_fold}" / "best_model.pth"
        if src.exists():
            global_dir = self.root / "global_best"
            global_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, global_dir / "best_model.pth")
            meta = {
                "best_fold": best_fold,
                "best_val_dice": best_dice,
                "fold_results": {str(f): r for f, r in valid.items()},
            }
            with (global_dir / "metadata.json").open("w") as fh:
                json.dump(meta, fh, indent=2, default=str)
            logger.success(
                f"Global best: fold_{best_fold} | Dice={best_dice:.4f} → {global_dir}"
            )

        # Write root manifest
        manifest = {
            "best_fold": best_fold,
            "best_val_dice": best_dice,
            "folds": {
                str(f): {
                    "best_val_dice": r.get("best_val_dice"),
                    "epochs_trained": r.get("epochs_trained"),
                    "early_stopped": r.get("early_stopped"),
                }
                for f, r in fold_results.items()
            },
        }
        manifest_path = self.root / "manifest.json"
        with manifest_path.open("w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info(f"Manifest written → {manifest_path}")

    def get_checkpoint(self, fold: int | str, which: str = "best") -> Path:
        """Return the path to a managed checkpoint file.

        Parameters
        ----------
        fold:
            Fold index, or ``"global_best"``.
        which:
            ``"best"`` or ``"last"`` (``"last"`` ignored for ``global_best``).
        """
        if str(fold) == "global_best":
            p = self.root / "global_best" / "best_model.pth"
        else:
            fname = "best_model.pth" if which == "best" else "last_model.pth"
            p = self.root / f"fold_{fold}" / fname
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    def list_available(self) -> dict[str, list[str]]:
        """Return ``{folder_name: [files]}`` for all checkpoint directories."""
        result: dict[str, list[str]] = {}
        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                result[d.name] = [f.name for f in d.glob("*")]
        return result

    def load_manifest(self) -> dict:
        """Load and return ``manifest.json`` if present."""
        p = self.root / "manifest.json"
        if not p.exists():
            return {}
        with p.open() as fh:
            return json.load(fh)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _nnunet_fold_dir(
        fold: int | str,
        configuration: str,
        trainer_class: str,
        plans_identifier: str,
    ) -> Path:
        results_root = get_path_env("nnUNet_results", required=True)
        dataset = dataset_folder_name()
        return (
            results_root
            / dataset
            / f"{trainer_class}__{plans_identifier}__{configuration}"
            / f"fold_{fold}"
        )

    @staticmethod
    def _copy(src: Path, dst: Path) -> Path | None:
        if not src.exists():
            logger.warning(f"Source checkpoint not found: {src}")
            return None
        shutil.copy2(src, dst)
        logger.debug(f"Copied: {src.name} → {dst}")
        return dst
