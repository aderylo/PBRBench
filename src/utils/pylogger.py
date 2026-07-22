"""Hydra/Python logger utilities."""

from __future__ import annotations

import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a python logger with standard formatting for Hydra scripts."""
    logger = logging.getLogger(name)
    return logger
