"""Canonical JSON serialisation and hashing of configuration models."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

__all__ = ["canonical_json", "model_sha256"]


def canonical_json(model: BaseModel) -> str:
    """Serialise ``model`` to compact JSON with sorted keys (aliases used, e.g. ``else``)."""
    payload = model.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def model_sha256(model: BaseModel) -> str:
    """Hex sha256 of :func:`canonical_json`. Independent of YAML key order and formatting."""
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()
