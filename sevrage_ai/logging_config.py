"""Logging setup shared by the CLI and the app.

Rule: **no patient values at INFO level or above.** Log counts, column names and ids of
configuration objects, never cell values. Use ``get_logger(__name__)`` in every module.
"""

from __future__ import annotations

import logging
from typing import Final, TextIO

__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]

LOGGER_NAME: Final = "sevrage_ai"
_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER: Final = "_sevrage_ai_handler"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``sevrage_ai`` namespace (pass ``__name__``)."""
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO, *, stream: TextIO | None = None) -> None:
    """Attach one stream handler to the package logger (idempotent) and set its level."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    logger.propagate = False
