"""Environment variable loading and validation utilities."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_file: str | Path | None = None) -> None:
    """Load .env from *env_file* or from the project root .env (auto-discovered)."""
    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=False)
    else:
        # Walk up from CWD to find the first .env
        candidate = Path.cwd()
        for parent in [candidate, *candidate.parents]:
            dot_env = parent / ".env"
            if dot_env.exists():
                load_dotenv(dotenv_path=dot_env, override=False)
                break


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Return the value of environment variable *key*.

    Parameters
    ----------
    key:
        Variable name.
    default:
        Fallback when the variable is absent.
    required:
        Raise ``KeyError`` if the variable is absent and no *default* given.
    """
    value = os.environ.get(key, default)
    if value is None and required:
        raise KeyError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example → .env and fill in the missing values."
        )
    return value  # type: ignore[return-value]


def get_path_env(key: str, default: str | None = None, required: bool = False) -> Path:
    """Return *key* as a ``Path``, creating the directory if it doesn't exist."""
    raw = get_env(key, default=default, required=required)
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def nnunet_env() -> dict[str, str]:
    """Return a dict of the three mandatory nnU-Net environment variables."""
    return {
        "nnUNet_raw": get_env("nnUNet_raw", required=True),
        "nnUNet_preprocessed": get_env("nnUNet_preprocessed", required=True),
        "nnUNet_results": get_env("nnUNet_results", required=True),
    }


def dataset_folder_name() -> str:
    """Return the canonical nnU-Net dataset folder name, e.g. ``Dataset001_BraTSMENRT``."""
    dataset_id = get_env("DATASET_ID", default="001")
    dataset_name = get_env("DATASET_NAME", default="BraTSMENRT")
    return f"Dataset{int(dataset_id):03d}_{dataset_name}"
