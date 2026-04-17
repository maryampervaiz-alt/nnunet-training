"""Build nnU-Net v2 ``dataset.json`` with auto-discovered channel names.

nnU-Net v2 requires exactly this schema:

.. code-block:: json

    {
        "channel_names": {"0": "T1c"},
        "labels": {"background": 0, "GTV": 1},
        "numTraining": 500,
        "file_ending": ".nii.gz"
    }

All other fields are optional metadata.  We do NOT embed patch size,
spacing, or normalization — nnU-Net derives those automatically.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ..utils.env_utils import dataset_folder_name, get_env, get_path_env


def _discover_channel_names(images_tr: Path) -> dict[str, str]:
    """Infer channel_names from files already written to *imagesTr*.

    Looks for ``*_0000.nii.gz``, ``*_0001.nii.gz``, etc. and returns a dict
    like ``{"0": "channel_0", "1": "channel_1"}``.  If suffix metadata is
    available via a sidecar ``.channel_map.json`` (written by the converter),
    those names are used instead.
    """
    # Check for sidecar written by converter
    sidecar = images_tr.parent / ".channel_map.json"
    if sidecar.exists():
        with sidecar.open() as fh:
            raw: dict = json.load(fh)
        # raw is {suffix: channel_idx} — invert and capitalise
        names = {str(v): k.upper() for k, v in raw.items()}
        logger.info(f"Channel names from sidecar: {names}")
        return dict(sorted(names.items(), key=lambda x: int(x[0])))

    # Fallback: count *_0000, *_0001, … files
    channel_indices: set[int] = set()
    for f in images_tr.glob("*.nii.gz"):
        stem = f.name.replace(".nii.gz", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            channel_indices.add(int(parts[1]))

    if not channel_indices:
        logger.warning("imagesTr is empty — defaulting to single-channel {\"0\": \"image\"}")
        return {"0": "image"}

    names = {str(i): f"channel_{i}" for i in sorted(channel_indices)}
    logger.info(f"Channel names (auto): {names}")
    return names


def write_channel_map_sidecar(dataset_dir: Path, channel_map: dict[str, int]) -> None:
    """Write a ``.channel_map.json`` sidecar so ``dataset_json`` can recover names.

    Parameters
    ----------
    dataset_dir:
        The ``Dataset{ID}_{NAME}`` directory.
    channel_map:
        ``{suffix: channel_index}`` dict produced by the converter.
    """
    sidecar = dataset_dir / ".channel_map.json"
    with sidecar.open("w") as fh:
        json.dump(channel_map, fh, indent=2)
    logger.debug(f"Channel map sidecar → {sidecar}")


def build_dataset_json(
    dataset_dir: str | Path | None = None,
    channel_names: dict[str, str] | None = None,
    labels: dict[str, int] | None = None,
    description: str = "",
    reference: str = "",
    licence: str = "",
    release: str = "",
    file_ending: str = ".nii.gz",
    overwrite: bool = True,
) -> Path:
    """Write ``dataset.json`` into *dataset_dir*.

    Parameters
    ----------
    dataset_dir:
        Target directory (``<nnUNet_raw>/Dataset{ID}_{NAME}``).
        Auto-derived from environment variables when ``None``.
    channel_names:
        ``{"0": "T1c"}``-style mapping.  When ``None``, auto-discovered from
        files already present in ``imagesTr/`` and any ``.channel_map.json``
        sidecar.
    labels:
        Class-name → integer mapping.  When ``None``, defaults to binary
        segmentation ``{"background": 0, "GTV": 1}``.
    overwrite:
        Overwrite an existing ``dataset.json``.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    if dataset_dir is None:
        raw = get_path_env("nnUNet_raw", required=True)
        dataset_dir = raw / dataset_folder_name()
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    images_tr = dataset_dir / "imagesTr"
    images_ts = dataset_dir / "imagesTs"

    # ── Channel names ─────────────────────────────────────────────────────────
    if channel_names is None:
        channel_names = _discover_channel_names(images_tr) if images_tr.exists() else {"0": "image"}

    # ── Labels ────────────────────────────────────────────────────────────────
    if labels is None:
        labels = {"background": 0, "GTV": 1}

    # ── Count files ───────────────────────────────────────────────────────────
    # Count by channel-0 files only (each case has exactly one)
    n_train = (
        len([f for f in images_tr.glob("*_0000.nii.gz")])
        if images_tr.exists()
        else 0
    )
    n_test = (
        len([f for f in images_ts.glob("*_0000.nii.gz")])
        if images_ts.exists()
        else 0
    )

    dataset_id = int(get_env("DATASET_ID", default="001"))
    dataset_name = get_env("DATASET_NAME", default="BraTSMENRT")

    payload: dict = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": n_train,
        "file_ending": file_ending,
    }

    # Optional metadata (nnU-Net ignores unknown fields but they're useful)
    if description:
        payload["description"] = description
    if reference:
        payload["reference"] = reference
    if licence:
        payload["licence"] = licence
    if release:
        payload["release"] = release
    payload["name"] = f"Dataset{dataset_id:03d}_{dataset_name}"

    out_path = dataset_dir / "dataset.json"
    if out_path.exists() and not overwrite:
        logger.info(f"dataset.json exists, skipping: {out_path}")
        return out_path

    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        f"dataset.json written → {out_path}\n"
        f"  numTraining  : {n_train}\n"
        f"  numTest      : {n_test}\n"
        f"  channel_names: {channel_names}\n"
        f"  labels       : {labels}"
    )
    return out_path


def load_dataset_json(dataset_dir: str | Path | None = None) -> dict:
    """Load and return the existing ``dataset.json`` as a dict."""
    if dataset_dir is None:
        raw = get_path_env("nnUNet_raw", required=True)
        dataset_dir = raw / dataset_folder_name()
    path = Path(dataset_dir) / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset.json not found: {path}")
    with path.open() as fh:
        return json.load(fh)
