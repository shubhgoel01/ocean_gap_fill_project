"""Compatibility CLI wrapper for the main pipeline entrypoint."""

from __future__ import annotations

from .pipeline import main as pipeline_main


def main() -> int:
    """Delegate to the canonical pipeline entrypoint."""
    return pipeline_main()
