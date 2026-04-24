"""Build SAM-Med3D-friendly prompts from coarse segmentation masks.

This module converts binary nnU-Net masks into deterministic prompt artifacts:
  - Positive points (inside lesion)
  - Negative points (near-lesion background)
  - 3D bounding boxes

The output format is JSON-serializable and intentionally model-agnostic so it
can be consumed by different SAM-Med3D forks/adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import ball, dilation


@dataclass
class ComponentPrompt:
    """Prompt package for a single connected component."""

    component_id: int
    voxel_count: int
    bbox_xyz: list[int]  # [x_min, y_min, z_min, x_max, y_max, z_max], inclusive
    bbox_xyz_norm: list[float]
    centroid_xyz: list[int]
    positive_points_xyz: list[list[int]]
    negative_points_xyz: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "voxel_count": self.voxel_count,
            "bbox_xyz": self.bbox_xyz,
            "bbox_xyz_norm": self.bbox_xyz_norm,
            "centroid_xyz": self.centroid_xyz,
            "positive_points_xyz": self.positive_points_xyz,
            "negative_points_xyz": self.negative_points_xyz,
        }


def _bbox_from_coords(coords_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = coords_xyz.min(axis=0)
    maxs = coords_xyz.max(axis=0)
    return mins, maxs


def _normalize_bbox(mins: np.ndarray, maxs: np.ndarray, shape_xyz: tuple[int, int, int]) -> list[float]:
    denom = np.array([max(s - 1, 1) for s in shape_xyz], dtype=np.float64)
    out = np.concatenate([mins / denom, maxs / denom])
    return [float(v) for v in out.tolist()]


def _centroid(coords_xyz: np.ndarray) -> np.ndarray:
    c = np.round(coords_xyz.mean(axis=0)).astype(np.int64)
    return c


def _unique_points(points: list[np.ndarray]) -> list[list[int]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[list[int]] = []
    for p in points:
        key = (int(p[0]), int(p[1]), int(p[2]))
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1], key[2]])
    return out


def _extreme_points(coords_xyz: np.ndarray) -> list[np.ndarray]:
    """Pick deterministic axis-extreme points from foreground coordinates."""
    pts: list[np.ndarray] = []
    for axis in range(3):
        min_idx = int(np.argmin(coords_xyz[:, axis]))
        max_idx = int(np.argmax(coords_xyz[:, axis]))
        pts.append(coords_xyz[min_idx])
        pts.append(coords_xyz[max_idx])
    return pts


def _clip_point(p: np.ndarray, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    clipped = np.array(
        [
            np.clip(p[0], 0, shape_xyz[0] - 1),
            np.clip(p[1], 0, shape_xyz[1] - 1),
            np.clip(p[2], 0, shape_xyz[2] - 1),
        ],
        dtype=np.int64,
    )
    return clipped


def _sample_evenly(coords_xyz: np.ndarray, n: int) -> list[np.ndarray]:
    if len(coords_xyz) == 0 or n <= 0:
        return []
    if len(coords_xyz) <= n:
        return [coords_xyz[i] for i in range(len(coords_xyz))]
    idxs = np.linspace(0, len(coords_xyz) - 1, num=n, dtype=np.int64)
    return [coords_xyz[i] for i in idxs.tolist()]


def build_component_prompts(
    mask: np.ndarray,
    max_positive_points: int = 4,
    max_negative_points: int = 4,
    min_component_voxels: int = 10,
    negative_shell_iters: int = 4,
) -> list[ComponentPrompt]:
    """Create prompts for each connected component in a binary mask.

    Parameters
    ----------
    mask:
        Binary mask in xyz array order.
    max_positive_points:
        Maximum number of positive points per component.
    max_negative_points:
        Maximum number of negative points per component.
    min_component_voxels:
        Skip tiny components below this voxel count.
    negative_shell_iters:
        Iterations for binary dilation to build a near-boundary background shell.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={mask_bin.shape}")

    labeled = label(mask_bin.astype(bool), connectivity=3)
    n_comp = int(labeled.max())
    if n_comp == 0:
        return []

    shape_xyz = tuple(int(s) for s in mask_bin.shape)
    prompts: list[ComponentPrompt] = []

    for cid in range(1, n_comp + 1):
        comp = labeled == cid
        voxel_count = int(comp.sum())
        if voxel_count < min_component_voxels:
            continue

        coords_xyz = np.argwhere(comp)
        mins, maxs = _bbox_from_coords(coords_xyz)
        centroid = _centroid(coords_xyz)

        pos_candidates: list[np.ndarray] = [centroid] + _extreme_points(coords_xyz)
        pos_unique = _unique_points(pos_candidates)
        pos_points = pos_unique[:max_positive_points]

        # Near-boundary background sampling for negative prompts.
        dil = comp.copy()
        selem = ball(1)
        for _ in range(max(1, negative_shell_iters)):
            dil = dilation(dil, footprint=selem)
        shell = np.logical_and(dil, ~comp)
        neg_coords = np.argwhere(shell)

        neg_points_raw = _sample_evenly(neg_coords, max_negative_points)
        if not neg_points_raw:
            # Fallback around expanded bbox corners.
            pad = np.array([2, 2, 2], dtype=np.int64)
            cands = [
                mins - pad,
                np.array([mins[0], mins[1], maxs[2] + 2], dtype=np.int64),
                np.array([mins[0], maxs[1] + 2, mins[2]], dtype=np.int64),
                np.array([maxs[0] + 2, mins[1], mins[2]], dtype=np.int64),
                maxs + pad,
            ]
            neg_points_raw = [_clip_point(p, shape_xyz) for p in cands[:max_negative_points]]

        neg_points = _unique_points(neg_points_raw)[:max_negative_points]

        bbox_xyz = [
            int(mins[0]), int(mins[1]), int(mins[2]),
            int(maxs[0]), int(maxs[1]), int(maxs[2]),
        ]

        prompts.append(
            ComponentPrompt(
                component_id=cid,
                voxel_count=voxel_count,
                bbox_xyz=bbox_xyz,
                bbox_xyz_norm=_normalize_bbox(mins, maxs, shape_xyz),
                centroid_xyz=[int(centroid[0]), int(centroid[1]), int(centroid[2])],
                positive_points_xyz=pos_points,
                negative_points_xyz=neg_points,
            )
        )

    prompts.sort(key=lambda p: p.voxel_count, reverse=True)
    return prompts


def build_case_prompt_payload(
    image_path: Path | None,
    mask_path: Path,
    max_positive_points: int = 4,
    max_negative_points: int = 4,
    min_component_voxels: int = 10,
    negative_shell_iters: int = 4,
) -> dict[str, object]:
    """Build a full JSON payload for one case from a coarse mask.

    If *image_path* is provided and readable, geometry fields are taken from the
    image; otherwise they are read from the mask.
    """
    mask_img = nib.load(str(mask_path))
    mask_arr = np.asarray(mask_img.dataobj)
    mask_bin = (mask_arr > 0).astype(np.uint8)

    geom_img = mask_img
    if image_path is not None and image_path.exists():
        try:
            geom_img = nib.load(str(image_path))
        except Exception:
            # Keep mask geometry as fallback.
            geom_img = mask_img

    components = build_component_prompts(
        mask=mask_bin,
        max_positive_points=max_positive_points,
        max_negative_points=max_negative_points,
        min_component_voxels=min_component_voxels,
        negative_shell_iters=negative_shell_iters,
    )

    shape_xyz = [int(s) for s in mask_bin.shape]
    affine = np.asarray(geom_img.affine, dtype=np.float64)
    spacing_xyz = [float(v) for v in geom_img.header.get_zooms()[:3]]
    direction = affine[:3, :3]
    origin = affine[:3, 3]

    case_id = mask_path.name
    if case_id.endswith(".nii.gz"):
        case_id = case_id[:-7]
    elif case_id.endswith(".nii"):
        case_id = case_id[:-4]

    return {
        "case_id": case_id,
        "image_path": str(image_path) if image_path is not None else None,
        "coarse_mask_path": str(mask_path),
        "shape_xyz": shape_xyz,
        "spacing_xyz_mm": spacing_xyz,
        "origin_xyz_mm": [float(origin[0]), float(origin[1]), float(origin[2])],
        "direction_matrix": direction.tolist(),
        "num_components": len(components),
        "components": [c.to_dict() for c in components],
        "prompt_convention": {
            "coords": "voxel indices in xyz order",
            "bbox": "[x_min, y_min, z_min, x_max, y_max, z_max], inclusive",
            "point_labels": {"positive": 1, "negative": 0},
        },
    }
