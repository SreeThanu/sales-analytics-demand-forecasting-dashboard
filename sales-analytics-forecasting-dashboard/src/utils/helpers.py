"""Shared utility functions for the sales analytics pipeline."""

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration file and return as dict."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path to an absolute path."""
    return get_project_root() / relative
