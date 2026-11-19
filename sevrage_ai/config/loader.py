"""Loading configuration files.

YAML is parsed with a subclass of :class:`yaml.SafeLoader` (no arbitrary objects) that
also rejects duplicate mapping keys, which plain ``yaml.safe_load`` silently overwrites.
Every failure becomes a :class:`~sevrage_ai.errors.ConfigError` with a French message and
the pydantic error locations in ``details``.

PyYAML reminders for config authors: quote ``"yes"``, ``"no"``, ``"on"``, ``"off"`` (else
booleans), and write floats with a dot (``1.0e+3``, not ``1e3``, which is a string).
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Hashable
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ValidationError

from sevrage_ai.config.schema import DatasetSchema
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.errors import ConfigError

__all__ = [
    "MAX_CONFIG_BYTES",
    "default_fast_training_path",
    "default_schema_path",
    "default_training_path",
    "load_schema",
    "load_training_config",
    "load_yaml",
]

MAX_CONFIG_BYTES: Final = 1_000_000
"""Configuration files larger than this are refused."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys in a mapping."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, object]:
        seen: set[Hashable] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                msg = f"clé en double « {key} » (ligne {key_node.start_mark.line + 1})"
                raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> object:
    """Parse a YAML file safely and return the resulting Python object.

    Raises:
        ConfigError: missing or oversized file, undecodable bytes, YAML syntax error or
            duplicate key.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        msg = f"Fichier de configuration introuvable : « {path.name} »."
        raise ConfigError(msg, details=[str(path)]) from exc
    if size > MAX_CONFIG_BYTES:
        msg = f"Fichier de configuration trop volumineux : « {path.name} »."
        raise ConfigError(msg, details=[f"{size} octets > {MAX_CONFIG_BYTES}"])
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"Lecture impossible (UTF-8 attendu) : « {path.name} »."
        raise ConfigError(msg, details=[str(exc)]) from exc
    loader = _UniqueKeySafeLoader(text)
    try:
        return loader.get_single_data()
    except yaml.YAMLError as exc:
        msg = f"YAML invalide dans « {path.name} »."
        raise ConfigError(msg, details=[str(exc)]) from exc
    finally:
        loader.dispose()


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    data = load_yaml(path)
    if not isinstance(data, dict):
        msg = f"« {path.name} » doit contenir un dictionnaire YAML à la racine."
        raise ConfigError(msg)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc']) or '<racine>'} : {error['msg']}"
            for error in exc.errors()
        ]
        msg = f"Configuration invalide dans « {path.name} » ({len(details)} erreur(s))."
        raise ConfigError(msg, details=details) from exc


def load_schema(path: Path) -> DatasetSchema:
    """Load and validate a dataset schema YAML file.

    May emit :class:`~sevrage_ai.errors.ConfigWarning` (e.g. molecule without dose
    equivalence).

    Raises:
        ConfigError: unreadable file or invalid schema.
    """
    return _load_model(path, DatasetSchema)


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a training configuration YAML file.

    Raises:
        ConfigError: unreadable file or invalid configuration.
    """
    return _load_model(path, TrainingConfig)


def _packaged(name: str) -> Path:
    resource = importlib.resources.files("sevrage_ai.config").joinpath("defaults", name)
    path = Path(str(resource))
    if not path.is_file():  # pragma: no cover - packaging error
        msg = f"Configuration par défaut absente du paquet : « {name} »."
        raise ConfigError(msg)
    return path


def default_schema_path() -> Path:
    """Path of the packaged default schema (``sevrage_ai/config/defaults/schema.yaml``)."""
    return _packaged("schema.yaml")


def default_training_path() -> Path:
    """Path of the packaged default training profile (``defaults/training.yaml``)."""
    return _packaged("training.yaml")


def default_fast_training_path() -> Path:
    """Path of the packaged fast profile (``defaults/training.fast.yaml``): a copy of
    ``configs/training.fast.yaml`` for installed packages and the Windows build (demo model)."""
    return _packaged("training.fast.yaml")
