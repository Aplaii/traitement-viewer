"""sevrage_ai: decision support for alcohol withdrawal (research tool, not a medical device).

Public contracts live in :mod:`sevrage_ai.types` and :mod:`sevrage_ai.errors`; the data
dictionary lives in :mod:`sevrage_ai.config`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("sevrage-ai")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
