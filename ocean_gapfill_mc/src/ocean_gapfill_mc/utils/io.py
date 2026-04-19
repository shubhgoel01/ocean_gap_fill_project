"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directories(paths) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
