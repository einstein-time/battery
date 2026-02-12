"""Logging configuration utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    verbose: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        log_file: Optional file path to write logs. If None, log to console only.
        level: Logging level (default INFO).
        verbose: If True, use detailed format. If False, use minimal format.
    """
    if verbose:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    else:
        fmt = "%(levelname)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Suppress overly verbose third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    if log_file:
        logging.info(f"Logging to file: {log_file}")
