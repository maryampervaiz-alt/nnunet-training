"""Publication-ready visualization helpers for segmentation experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

_PUB_RC = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

_COLORS = {
    "gt": "#E74C3C",
    "pred": "#3498DB",
    "fp": "#F39C12",
    "fn": "#9B59B6",
    "tp": "#2ECC71",
    "neutral": "#4C72B0",
}


class SegmentationPlotter:
    """Plot image slices with ground-truth and prediction overlays."""

    def __init__(
        self,
        output_dir: str | Path = "visualizations",
        num_slices: int = 5,
        overlay_alpha: float = 0.35,
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_slices = num_slices
        self.overlay_alpha = overlay_alpha
        self.dpi = dpi

    def plot_case(
        self,
        case_id: str,
        image_path: str | Path,
        pred_path: str | Path | None = None,
        gt_path: str | Path | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        with plt.rc_context(_PUB_RC):
            image = self._load_volume(image_path)
            pred = self._load_volume(pred_path).astype(bool) if pred_path else None
            gt = self._load_volume(gt_path).astype(bool) if gt_path else None

            focus_mask = gt if gt is not None else pred
            slices = self._select_slices(image=image, mask=focus_mask)

            has_gt = gt is not None
            has_pred = pred is not None
            has_diff = has_gt and has_pred
            n_cols = 1 + int(has_gt) + int(has_pred) + int(has_diff)

            fig, axes = plt.subplots(
                len(slices),
                n_cols,
                figsize=(n_cols * 3.5, len(slices) * 3.5),
                dpi=self.dpi,
            )
            if len(slices) == 1:
                axes = axes[np.newaxis, :]
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            for row_idx, sl in enumerate(slices):
                img_slice = image[:, :, sl].T
                img_norm = self._normalize(img_slice)
                col_idx = 0

                ax = axes[row_idx, col_idx]
                ax.imshow(img_norm, cmap="gray", origin="upper")
                ax.set_title(f"T1c z={sl}", fontsize=8)
                ax.axis("off")

                if has_gt:
                    col_idx += 1
                    ax = axes[row_idx, col_idx]
                    ax.imshow(img_norm, cmap="gray", origin="upper")
                    gt_slice = gt[:, :, sl].T.astype(float)
                    self._overlay_mask(ax, gt_slice > 0, _COLORS["gt"])
                    self._draw_contour(ax, gt_slice, _COLORS["gt"])
                    ax.set_title(f"GT z={sl}", fontsize=8)
                    ax.axis("off")

                if has_pred:
                    col_idx += 1
                    ax = axes[row_idx, col_idx]
                    ax.imshow(img_norm, cmap="gray", origin="upper")
                    pred_slice = pred[:, :, sl].T.astype(float)
                    self._overlay_mask(ax, pred_slice > 0, _COLORS["pred"])
                    self._draw_contour(ax, pred_slice, _COLORS["pred"])
                    ax.set_title(f"Pred z={sl}", fontsize=8)
                    ax.axis("off")

                if has_diff:
                    col_idx += 1
                    ax = axes[row_idx, col_idx]
                    ax.imshow(img_norm, cmap="gray", origin="upper")
                    gt_slice = gt[:, :, sl].T.astype(bool)
                    pred_slice = pred[:, :, sl].T.astype(bool)
                    overlays = [
                        (pred_slice & gt_slice, _COLORS["tp"], "TP"),
                        (pred_slice & ~gt_slice, _COLORS["fp"], "FP"),
                        (~pred_slice & gt_slice, _COLORS["fn"], "FN"),
                    ]
                    legend_patches: list[mpatches.Patch] = []
                    for mask, color, label in overlays:
                        if mask.any():
                            self._overlay_mask(ax, mask, color)
                        legend_patches.append(mpatches.Patch(color=color, label=label))
                    ax.legend(handles=legend_patches, loc="lower right", fontsize=6, framealpha=0.7)
                    ax.set_title(f"TP/FP/FN z={sl}", fontsize=8)
                    ax.axis("off")

            title = case_id
            if metrics:
                summary = "  ".join(f"{key}={value:.3f}" for key, value in metrics.items())
                if summary:
                    title = f"{title} | {summary}"
            plt.suptitle(title, fontsize=10, fontweight="bold", y=1.01)
            plt.tight_layout()

            out_path = self.output_dir / f"{case_id}_overlay.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)
            logger.info(f"Saved overlay -> {out_path}")
            return out_path

    def plot_batch(
        self,
        case_ids: list[str],
        image_dir: str | Path,
        pred_dir: str | Path | None = None,
        gt_dir: str | Path | None = None,
        metrics_df: pd.DataFrame | None = None,
        image_suffix: str = "_0000.nii.gz",
        seg_suffix: str = ".nii.gz",
    ) -> list[Path]:
        image_dir = Path(image_dir)
        pred_dir = Path(pred_dir) if pred_dir is not None else None
        gt_dir = Path(gt_dir) if gt_dir is not None else None

        outputs: list[Path] = []
        for case_id in case_ids:
            image_path = image_dir / f"{case_id}{image_suffix}"
            pred_path = pred_dir / f"{case_id}{seg_suffix}" if pred_dir is not None else None
            gt_path = gt_dir / f"{case_id}{seg_suffix}" if gt_dir is not None else None

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            metrics = None
            if metrics_df is not None and "case_id" in metrics_df.columns:
                row = metrics_df[metrics_df["case_id"] == case_id]
                if not row.empty:
                    metrics = {
                        metric: float(row.iloc[0][metric])
                        for metric in ["dice", "hd95", "precision", "recall"]
                        if metric in row.columns and pd.notna(row.iloc[0][metric])
                    }

            try:
                outputs.append(
                    self.plot_case(
                        case_id=case_id,
                        image_path=image_path,
                        pred_path=pred_path if pred_path is not None and pred_path.exists() else None,
                        gt_path=gt_path if gt_path is not None and gt_path.exists() else None,
                        metrics=metrics,
                    )
                )
            except Exception as exc:
                logger.warning(f"[{case_id}] Overlay failed: {exc}")
        return outputs

    def _overlay_mask(self, ax, mask: np.ndarray, color: str) -> None:
        ax.imshow(
            np.where(mask, 1.0, np.nan),
            cmap=mcolors.ListedColormap([color]),
            alpha=self.overlay_alpha,
            origin="upper",
            vmin=0,
            vmax=1,
        )

    @staticmethod
    def _load_volume(path: str | Path) -> np.ndarray:
        return np.asarray(nib.load(path).dataobj)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = float(arr.min()), float(arr.max())
        if max_val == min_val:
            return np.zeros_like(arr, dtype=float)
        return (arr.astype(float) - min_val) / (max_val - min_val)

    @staticmethod
    def _draw_contour(ax, mask: np.ndarray, color: str) -> None:
        if np.any(mask):
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=[0.8])

    def _select_slices(self, image: np.ndarray, mask: np.ndarray | None) -> list[int]:
        depth = image.shape[2]
        if mask is not None and mask.any():
            fg_slices = np.where(mask.any(axis=(0, 1)))[0]
            if len(fg_slices) >= self.num_slices:
                return np.linspace(
                    fg_slices[0],
                    fg_slices[-1],
                    self.num_slices,
                    dtype=int,
                ).tolist()
        return np.linspace(0, depth - 1, self.num_slices, dtype=int).tolist()


def plot_metrics_violin(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    output_path: str | Path = "visualizations/metrics_violin.png",
    title: str = "Segmentation metric distributions",
    dpi: int = 150,
) -> Path:
    metrics = metrics or [m for m in ["dice", "hd95", "precision", "recall"] if m in df.columns]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(1, len(metrics)) * 3.5, 5), dpi=dpi)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = (
                df[metric]
                .replace([float("inf"), float("-inf")], float("nan"))
                .dropna()
                .to_numpy()
            )
            if len(values) == 0:
                ax.set_title(metric.upper())
                continue
            violin = ax.violinplot(values, positions=[0], showmedians=True, showextrema=True)
            for body in violin["bodies"]:
                body.set_facecolor(_COLORS["neutral"])
                body.set_alpha(0.5)
            violin["cmedians"].set_color("black")
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(values))
            ax.scatter(jitter, values, s=12, alpha=0.5, color=_COLORS["neutral"], zorder=3)
            ax.set_title(f"{metric.upper()}\n{values.mean():.3f} +/- {values.std():.3f}")
            ax.set_xticks([])
            ax.set_ylabel(metric)

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"Violin plot -> {output_path}")
    return output_path


def plot_metrics_boxplot(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    output_path: str | Path = "visualizations/metrics_boxplot.png",
    title: str = "Segmentation metrics",
    dpi: int = 150,
) -> Path:
    metrics = metrics or [m for m in ["dice", "hd95", "precision", "recall"] if m in df.columns]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(1, len(metrics)) * 3.5, 5), dpi=dpi)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = (
                df[metric]
                .replace([float("inf"), float("-inf")], float("nan"))
                .dropna()
                .to_numpy()
            )
            if len(values) == 0:
                ax.set_title(metric.upper())
                continue
            ax.boxplot(
                values,
                patch_artist=True,
                boxprops=dict(facecolor=_COLORS["neutral"], alpha=0.6),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(values))
            ax.scatter(1 + jitter, values, s=10, alpha=0.4, color="black", zorder=3)
            ax.set_title(f"{metric.upper()}\n{values.mean():.3f} +/- {values.std():.3f}")
            ax.set_xticks([])
            ax.set_ylabel(metric)

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"Boxplot -> {output_path}")
    return output_path


def plot_volume_scatter(
    df: pd.DataFrame,
    output_path: str | Path = "visualizations/volume_scatter.png",
    title: str = "Predicted vs ground-truth volume",
    dpi: int = 150,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "pred_volume_ml" not in df.columns or "gt_volume_ml" not in df.columns:
        logger.warning("Volume columns not found - skipping volume scatter.")
        return output_path

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        x = df["gt_volume_ml"].to_numpy()
        y = df["pred_volume_ml"].to_numpy()

        if "dice" in df.columns:
            scatter = ax.scatter(
                x,
                y,
                c=df["dice"].to_numpy(),
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                alpha=0.7,
                s=25,
                edgecolors="none",
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("DSC")
        else:
            ax.scatter(x, y, alpha=0.6, s=25, color=_COLORS["neutral"])

        limit = max(np.nanmax(x), np.nanmax(y)) * 1.05 if len(x) and len(y) else 1.0
        ax.plot([0, limit], [0, limit], "k--", linewidth=1, alpha=0.5, label="y = x")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("GT volume (ml)")
        ax.set_ylabel("Predicted volume (ml)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

        mask = ~(np.isnan(x) | np.isnan(y))
        if int(mask.sum()) > 1:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=9, va="top")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"Volume scatter -> {output_path}")
    return output_path


def plot_fold_comparison(
    fold_dfs: dict[str, pd.DataFrame],
    metric: str = "dice",
    output_path: str | Path = "visualizations/fold_comparison.png",
    dpi: int = 150,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(fold_dfs.keys())
    means: list[float] = []
    stds: list[float] = []
    for df in fold_dfs.values():
        values = (
            df[metric]
            .replace([float("inf"), float("-inf")], float("nan"))
            .dropna()
            .to_numpy()
        )
        means.append(float(np.mean(values)) if len(values) else float("nan"))
        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5), dpi=dpi)
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=_COLORS["neutral"],
            alpha=0.8,
            ecolor="black",
            error_kw=dict(linewidth=1.2),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} per fold (mean +/- std)", fontweight="bold")
        if metric in {"dice", "precision", "recall", "specificity", "nsd"}:
            ax.set_ylim(0, 1.1)

        y_offset = max(stds) * 0.05 if stds else 0.01
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"Fold comparison -> {output_path}")
    return output_path


def plot_training_curve(
    progress_csv: str | Path,
    output_path: str | Path = "visualizations/training_curve.png",
    fold: int | str | None = None,
    dpi: int = 150,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(progress_csv)
    if fold is not None and "fold" in df.columns:
        df = df[df["fold"] == int(fold)].copy()

    with plt.rc_context(_PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

        for col, label, color in [
            ("train_loss", "Train loss", "#E74C3C"),
            ("val_loss", "Val loss", "#3498DB"),
        ]:
            if col in df.columns:
                valid = df.dropna(subset=[col])
                ax1.plot(valid["epoch"], valid[col], label=label, color=color)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss curves")
        if ax1.lines:
            ax1.legend()

        dice_col = next((col for col in ["val_dice", "val_pseudo_dice"] if col in df.columns), None)
        if dice_col is not None:
            valid = df.dropna(subset=[dice_col])
            ax2.plot(valid["epoch"], valid[dice_col], color="#2ECC71", label="Val Dice")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Dice")
            ax2.set_title("Validation Dice")
            ax2.set_ylim(0, 1.05)
            if not valid.empty:
                best_idx = valid[dice_col].idxmax()
                ax2.axvline(
                    valid.loc[best_idx, "epoch"],
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
            ax2.legend(fontsize=8)

        suffix = f" - fold {fold}" if fold is not None else ""
        plt.suptitle(f"Training progress{suffix}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"Training curve -> {output_path}")
    return output_path


def plot_all_folds_training(
    all_folds_csv: str | Path,
    output_path: str | Path = "visualizations/all_folds_training.png",
    dpi: int = 150,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(all_folds_csv)
    dice_col = next((col for col in ["val_dice", "val_pseudo_dice"] if col in df.columns), None)
    if dice_col is None or "fold" not in df.columns:
        logger.warning("all_folds CSV missing 'fold' or dice column - skipping.")
        return output_path

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
        cmap = plt.get_cmap("tab10")
        for fold_idx, fold_df in df.groupby("fold"):
            valid = fold_df.dropna(subset=[dice_col])
            ax.plot(
                valid["epoch"],
                valid[dice_col],
                label=f"Fold {fold_idx}",
                color=cmap(int(fold_idx) % 10),
                alpha=0.8,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Dice")
        ax.set_title("All-fold validation Dice", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    logger.info(f"All-fold training -> {output_path}")
    return output_path
