"""Exception hierarchy and CLI exit codes (leader-owned contract).

Every exception carries a French, user-facing ``message_fr`` (``str(exc)`` returns it)
and optional ``details``: short technical lines such as column names, counts or config
paths. **Neither may contain patient values.**

Exit codes (design section 10): 0 = OK, 1 = unexpected error, 2 = validation or
anonymization failure, 3 = bundle integrity or compatibility failure, 4 = configuration
error. The CLI maps exceptions to exit codes through :func:`exit_code_for`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Final

__all__ = [
    "EXIT_CODES",
    "AnonymizationError",
    "BundleCompatibilityError",
    "BundleIntegrityError",
    "ConfigError",
    "ConfigWarning",
    "DataValidationError",
    "DataWarning",
    "ExitCode",
    "SevrageError",
    "SevrageWarning",
    "exit_code_for",
]


class ExitCode(IntEnum):
    """Process exit codes of the ``sevrage`` CLI."""

    OK = 0
    UNEXPECTED = 1
    DATA = 2
    BUNDLE = 3
    CONFIG = 4


class SevrageError(Exception):
    """Base class of every expected ``sevrage_ai`` error."""

    def __init__(self, message_fr: str, *, details: Iterable[str] = ()) -> None:
        super().__init__(message_fr)
        self.message_fr: str = message_fr
        self.details: tuple[str, ...] = tuple(details)

    def __str__(self) -> str:
        return self.message_fr


class ConfigError(SevrageError):
    """Invalid or unreadable configuration (schema or training YAML)."""


class DataValidationError(SevrageError):
    """Input data failed validation (missing columns, invalid values, invalid record)."""


class AnonymizationError(SevrageError):
    """Input data look identifying (forbidden column names or value patterns)."""


class BundleIntegrityError(SevrageError):
    """A model bundle is incomplete, unreadable or tampered with (hash mismatch)."""


class BundleCompatibilityError(SevrageError):
    """A model bundle is incompatible (format version, library versions, schema hash)."""


class SevrageWarning(UserWarning):
    """Base class of ``sevrage_ai`` warnings."""


class ConfigWarning(SevrageWarning):
    """Non-blocking configuration issue, e.g. a molecule without a dose equivalence."""


class DataWarning(SevrageWarning):
    """Non-blocking data issue, e.g. a schema feature the synthetic generator cannot fill."""


EXIT_CODES: Final[Mapping[type[SevrageError], ExitCode]] = MappingProxyType(
    {
        SevrageError: ExitCode.UNEXPECTED,
        ConfigError: ExitCode.CONFIG,
        DataValidationError: ExitCode.DATA,
        AnonymizationError: ExitCode.DATA,
        BundleIntegrityError: ExitCode.BUNDLE,
        BundleCompatibilityError: ExitCode.BUNDLE,
    }
)
"""Exit code for each exception class. Subclasses inherit their closest ancestor's code."""


def exit_code_for(exc: BaseException) -> ExitCode:
    """Return the CLI exit code for ``exc``: its closest mapped ancestor, else ``UNEXPECTED``."""
    for cls in type(exc).__mro__:
        if issubclass(cls, SevrageError) and cls in EXIT_CODES:
            return EXIT_CODES[cls]
    return ExitCode.UNEXPECTED
