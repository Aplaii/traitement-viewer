"""Library version snapshots and compatibility checks. Owner: WS7."""

from __future__ import annotations

import platform
import re
from collections.abc import Mapping
from importlib import metadata
from typing import Final

from sevrage_ai.errors import BundleCompatibilityError

__all__ = ["TRACKED_PACKAGES", "check_compatible", "snapshot"]

TRACKED_PACKAGES: Final = (
    "sevrage-ai",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "pydantic",
)
"""Distributions recorded in bundle metadata (plus ``python``)."""

_PYTHON: Final = "python"
_ABSENT: Final = "absent"
_BLOCKING_PARTS: Final[Mapping[str, int]] = {"scikit-learn": 2, "numpy": 1}
"""Package -> number of leading release numbers that must match (major.minor, major)."""
_RELEASE_RE: Final = re.compile(r"^(\d+)(?:\.(\d+))?")


def snapshot() -> dict[str, str]:
    """Installed versions of :data:`TRACKED_PACKAGES` plus ``"python"``."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = _ABSENT
    versions[_PYTHON] = platform.python_version()
    return versions


def _release(version: str, parts: int) -> tuple[int, ...] | None:
    """The first ``parts`` release numbers of ``version``, or ``None`` if unparseable."""
    match = _RELEASE_RE.match(version.strip())
    if match is None:
        return None
    numbers = tuple(int(group) for group in match.groups()[:parts] if group is not None)
    return numbers if len(numbers) == parts else None


def _is_blocking(name: str, saved: str, current: str) -> bool:
    parts = _BLOCKING_PARTS.get(name)
    if parts is None:
        return False
    saved_release = _release(saved, parts)
    current_release = _release(current, parts)
    if saved_release is None or current_release is None:
        return saved != current
    return saved_release != current_release


def check_compatible(
    saved: Mapping[str, str], current: Mapping[str, str], *, strict: bool
) -> tuple[str, ...]:
    """Compare version snapshots and return French mismatch descriptions.

    Blocking rule: scikit-learn major.minor or numpy major differ.

    Raises:
        BundleCompatibilityError: blocking mismatch and ``strict``.
    """
    descriptions: list[str] = []
    blocking: list[str] = []
    for name in (*TRACKED_PACKAGES, _PYTHON):
        saved_version = saved.get(name, _ABSENT)
        current_version = current.get(name, _ABSENT)
        if saved_version == current_version:
            continue
        label = "Python" if name == _PYTHON else name
        line = (
            f"{label} : version {saved_version} à l'enregistrement du modèle, "
            f"{current_version} installée"
        )
        if _is_blocking(name, saved_version, current_version):
            line += " (bloquant)"
            blocking.append(line)
        descriptions.append(line)
    if blocking and strict:
        msg = (
            "Modèle incompatible avec les bibliothèques installées : réentraîner le modèle "
            "ou réinstaller les versions d'origine (uv.lock)."
        )
        raise BundleCompatibilityError(msg, details=blocking)
    return tuple(descriptions)
