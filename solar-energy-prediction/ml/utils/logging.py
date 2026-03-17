"""Logging utilities for the application."""

from __future__ import annotations

import logging


def setup_logging(log_level: str) -> None:
    """Configure the root logger once for the whole application."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
