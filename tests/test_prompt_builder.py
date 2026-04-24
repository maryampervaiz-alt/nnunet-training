"""Unit tests for SAM-Med3D prompt generation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.prompt_builder import build_component_prompts


def test_empty_mask_returns_no_components() -> None:
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    prompts = build_component_prompts(mask)
    assert prompts == []


def test_single_component_has_bbox_and_points() -> None:
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[10:20, 12:24, 8:16] = 1

    prompts = build_component_prompts(
        mask,
        max_positive_points=4,
        max_negative_points=4,
        min_component_voxels=5,
    )

    assert len(prompts) == 1
    p = prompts[0]
    assert p.voxel_count == int(mask.sum())
    assert p.bbox_xyz == [10, 12, 8, 19, 23, 15]
    assert len(p.positive_points_xyz) >= 1
    assert len(p.negative_points_xyz) >= 1


def test_tiny_components_are_filtered() -> None:
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    mask[3, 3, 3] = 1
    mask[20, 20, 20] = 1

    prompts = build_component_prompts(mask, min_component_voxels=2)
    assert prompts == []
