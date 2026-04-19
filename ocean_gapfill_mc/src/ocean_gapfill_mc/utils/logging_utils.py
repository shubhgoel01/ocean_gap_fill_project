"""Logging helpers for pipeline runs."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str) -> None:
    """Configure file and console logging for the pipeline."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    return logging.getLogger(name)
