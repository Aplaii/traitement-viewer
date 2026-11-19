"""Model bundle directory (D11). Owner: WS7.

Layout of ``models/<bundle_id>/``: :data:`MODEL_FILE` (joblib, compress=3, holding the
``TrainedSystem``, the ``DatasetSchema`` and the ``TrainingConfig``), :data:`METADATA_FILE`
(:class:`BundleMetadata`), :data:`METRICS_FILE` (``EvaluationReport`` JSON, optional),
:data:`MODEL_CARD_FILE` and :data:`FIGURES_DIR`.

Loading checks, **all before unpickling**: (1) ``metadata.json`` parses; (2) the format
version is supported; (3) the sha256 of ``model.joblib`` matches; (4) scikit-learn
major.minor and numpy major match (``strict_versions``), and so does the schema hash when
``expected_schema`` is given. Failures raise ``BundleIntegrityError`` (1-3) or
``BundleCompatibilityError`` (4). Pickle can run code: only load bundles from the
administrator's directory, never from an upload.

``metadata.json`` is written last, so an interrupted save never leaves a directory that
:func:`list_bundles` would report; a failed save also removes its directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import joblib
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

import sevrage_ai
from sevrage_ai.config.schema import DatasetSchema, schema_hash
from sevrage_ai.config.training import TrainingConfig, training_config_hash
from sevrage_ai.errors import BundleCompatibilityError, BundleIntegrityError
from sevrage_ai.evaluation.report import EvaluationReport
from sevrage_ai.logging_config import get_logger
from sevrage_ai.models.system import TrainedSystem
from sevrage_ai.persistence import versions
from sevrage_ai.types import ModelInfo, TargetName

__all__ = [
    "FIGURES_DIR",
    "FORMAT_VERSION",
    "METADATA_FILE",
    "METRICS_FILE",
    "MODEL_CARD_FILE",
    "MODEL_FILE",
    "BundleMetadata",
    "ModelBundle",
    "list_bundles",
    "load_bundle",
    "read_metadata",
    "save_bundle",
    "sha256_file",
    "verify_bundle",
]

FORMAT_VERSION: Final = 1
MODEL_FILE: Final = "model.joblib"
METADATA_FILE: Final = "metadata.json"
METRICS_FILE: Final = "metrics.json"
MODEL_CARD_FILE: Final = "model_card.md"
FIGURES_DIR: Final = "figures"

_LOGGER = get_logger(__name__)
_CHUNK_SIZE: Final = 1024 * 1024
_COMPRESS: Final = 3
_DEFAULT_THRESHOLD: Final = 0.5
_GIT_TIMEOUT_S: Final = 5
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_PAYLOAD_KEYS: Final = ("system", "schema", "training_config")


class BundleMetadata(BaseModel):
    """Content of ``metadata.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str = Field(min_length=1)
    format_version: int = Field(ge=1)
    created_at: AwareDatetime
    package_version: str
    python_version: str
    library_versions: dict[str, str]
    schema_hash: str
    training_config_hash: str
    data_fingerprint: str
    n_train: int = Field(ge=0)
    class_counts: dict[TargetName, dict[str, int]]
    feature_set: str
    targets: tuple[TargetName, ...]
    is_synthetic: bool
    seed: int
    git_commit: str | None = None
    artifact_sha256: str
    extra: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class ModelBundle:
    """A verified, loaded bundle."""

    path: Path
    metadata: BundleMetadata
    system: TrainedSystem
    schema: DatasetSchema
    training_config: TrainingConfig
    report: EvaluationReport | None

    def info(self) -> ModelInfo:
        """Build the :class:`~sevrage_ai.types.ModelInfo` shown by the UI and the CLI."""
        treatment = self.system.targets.get(TargetName.TREATMENT)
        threshold = None if treatment is None else treatment.threshold
        report = self.report
        return ModelInfo(
            model_id=self.metadata.bundle_id,
            created_at=self.metadata.created_at,
            is_synthetic=self.metadata.is_synthetic or self.system.is_synthetic,
            feature_set=self.metadata.feature_set,
            feature_names=tuple(feature.name for feature in self.system.features),
            targets=tuple(self.system.targets),
            n_train=self.metadata.n_train,
            schema_hash=self.metadata.schema_hash,
            package_version=self.metadata.package_version,
            alpha=self.training_config.conformal.alpha,
            treatment_threshold=_DEFAULT_THRESHOLD if threshold is None else threshold,
            grey_zone=self.training_config.treatment_threshold.grey_zone,
            dose_equivalence_to_diazepam=dict(self.schema.dose_equivalence_to_diazepam),
            metrics_summary={} if report is None else report.headline_metrics(),
            evaluation_protocol=None if report is None else report.protocol_fr,
            bundle_path=self.path,
        )


# --------------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Current UTC time (a seam for tests)."""
    return datetime.now(UTC)


def _git_commit() -> str | None:
    """``git rev-parse HEAD`` of the package's checkout, or ``None`` on any failure."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [git, "rev-parse", "HEAD"],
            shell=False,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.decode("ascii", errors="replace").strip()
    return commit if _COMMIT_RE.fullmatch(commit) else None


def _create_bundle_dir(out_dir: Path, base_id: str) -> Path:
    """Create ``out_dir/<base_id>``, or ``<base_id>_2``, ``_3``... if it already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        name = base_id if suffix == 1 else f"{base_id}_{suffix}"
        candidate = out_dir / name
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return candidate


def _write_text(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else f"{text}\n", encoding="utf-8", newline="\n")


def save_bundle(
    system: TrainedSystem,
    report: EvaluationReport | None,
    schema: DatasetSchema,
    training_config: TrainingConfig,
    meta_extra: Mapping[str, str],
    out_dir: Path,
    *,
    data_fingerprint: str,
) -> Path:
    """Write a new bundle directory under ``out_dir`` and return its path.

    The bundle id is derived from the UTC creation time and ``is_synthetic`` (for example
    ``20260925T101500Z_synthetic``). Writes the model card and the figures when ``report``
    is given.
    """
    from sevrage_ai.persistence import model_card

    schema_digest = schema_hash(schema)
    if system.schema_hash != schema_digest:
        msg = "Le système n'a pas été entraîné avec ce schéma : bundle non écrit."
        raise ValueError(msg)
    is_synthetic = system.is_synthetic or (report is not None and report.is_synthetic)
    created_at = _utcnow()
    base_id = f"{created_at:%Y%m%dT%H%M%SZ}_{'synthetic' if is_synthetic else 'real'}"
    bundle_dir = _create_bundle_dir(out_dir, base_id)
    try:
        model_path = bundle_dir / MODEL_FILE
        payload = {"system": system, "schema": schema, "training_config": training_config}
        joblib.dump(payload, model_path, compress=_COMPRESS)
        metadata = BundleMetadata(
            bundle_id=bundle_dir.name,
            format_version=FORMAT_VERSION,
            created_at=created_at,
            package_version=sevrage_ai.__version__,
            python_version=platform.python_version(),
            library_versions=versions.snapshot(),
            schema_hash=schema_digest,
            training_config_hash=training_config_hash(training_config),
            data_fingerprint=data_fingerprint,
            n_train=system.n_train,
            class_counts={t: dict(counts) for t, counts in system.class_counts.items()},
            feature_set=system.feature_set,
            targets=tuple(system.targets),
            is_synthetic=is_synthetic,
            seed=system.seed,
            git_commit=_git_commit(),
            artifact_sha256=sha256_file(model_path),
            extra=dict(meta_extra),
        )
        if report is not None:
            # Imported here: matplotlib is only needed to write figures, not to load a bundle
            # (the web version of the predictor ships without it).
            from sevrage_ai.evaluation import figures

            _write_text(bundle_dir / METRICS_FILE, report.to_json())
            figures.save_all(report, bundle_dir / FIGURES_DIR)
        card = model_card.render(metadata, report, schema, training_config)
        _write_text(bundle_dir / MODEL_CARD_FILE, card)
        _write_text(bundle_dir / METADATA_FILE, metadata.model_dump_json(indent=2))
    except BaseException:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise
    _LOGGER.info("Bundle %s écrit (%d cibles).", bundle_dir.name, len(metadata.targets))
    return bundle_dir


# --------------------------------------------------------------------------------------
# Verify and load
# --------------------------------------------------------------------------------------


def _integrity_error(message: str, path: Path, *details: str) -> BundleIntegrityError:
    return BundleIntegrityError(message, details=(f"bundle : {path}", *details))


def read_metadata(path: Path) -> BundleMetadata:
    """Checks 1-2 only (parse, format version), without reading ``model.joblib``."""
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        msg = f"Bundle incomplet ou introuvable : {METADATA_FILE} manquant."
        raise _integrity_error(msg, path)
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Bundle illisible : {METADATA_FILE} n'est pas un JSON valide."
        raise _integrity_error(msg, path, type(exc).__name__) from exc
    if not isinstance(raw, dict):
        msg = f"Bundle illisible : {METADATA_FILE} doit contenir un objet JSON."
        raise _integrity_error(msg, path)
    version = raw.get("format_version")
    if isinstance(version, int) and not isinstance(version, bool) and version != FORMAT_VERSION:
        msg = (
            f"Format de bundle non pris en charge (format {version}, "
            f"attendu {FORMAT_VERSION}) : réentraîner le modèle ou mettre à jour sevrage_ai."
        )
        raise _integrity_error(msg, path)
    try:
        return BundleMetadata.model_validate(raw)
    except ValueError as exc:
        msg = f"Bundle illisible : {METADATA_FILE} est incomplet ou invalide."
        raise _integrity_error(msg, path, type(exc).__name__) from exc


def verify_bundle(
    path: Path,
    *,
    expected_schema: DatasetSchema | None = None,
    strict_versions: bool = True,
) -> BundleMetadata:
    """Checks 1-4 without unpickling (used by ``inspect-bundle``)."""
    metadata = read_metadata(path)
    model_path = path / MODEL_FILE
    if not model_path.is_file():
        msg = f"Bundle incomplet : {MODEL_FILE} manquant."
        raise _integrity_error(msg, path)
    if sha256_file(model_path) != metadata.artifact_sha256:
        msg = f"Bundle altéré : l'empreinte sha256 de {MODEL_FILE} ne correspond pas."
        raise _integrity_error(msg, path, "sha256 différent de metadata.json")
    mismatches = versions.check_compatible(
        metadata.library_versions, versions.snapshot(), strict=strict_versions
    )
    for line in mismatches:
        _LOGGER.warning("Bundle %s : %s.", metadata.bundle_id, line)
    if expected_schema is not None and schema_hash(expected_schema) != metadata.schema_hash:
        msg = (
            "Bundle incompatible : il a été entraîné avec un autre schéma de données "
            "(empreinte du schéma différente)."
        )
        raise BundleCompatibilityError(msg, details=(f"bundle : {path}",))
    return metadata


def _unpickle(path: Path) -> tuple[TrainedSystem, DatasetSchema, TrainingConfig]:
    try:
        payload = joblib.load(path / MODEL_FILE)
    except Exception as exc:
        msg = (
            f"Bundle illisible : {MODEL_FILE} n'a pas pu être désérialisé "
            "(fichier corrompu ou code incompatible)."
        )
        raise _integrity_error(msg, path, type(exc).__name__) from exc
    if isinstance(payload, dict):
        system, schema, config = (payload.get(key) for key in _PAYLOAD_KEYS)
        if (
            isinstance(system, TrainedSystem)
            and isinstance(schema, DatasetSchema)
            and isinstance(config, TrainingConfig)
        ):
            return system, schema, config
    msg = f"Bundle invalide : contenu inattendu dans {MODEL_FILE}."
    raise _integrity_error(msg, path)


def _read_report(path: Path) -> EvaluationReport | None:
    metrics_path = path / METRICS_FILE
    if not metrics_path.is_file():
        return None
    try:
        return EvaluationReport.from_json(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"Bundle illisible : {METRICS_FILE} est invalide."
        raise _integrity_error(msg, path, type(exc).__name__) from exc


def load_bundle(
    path: Path,
    *,
    expected_schema: DatasetSchema | None = None,
    strict_versions: bool = True,
) -> ModelBundle:
    """:func:`verify_bundle`, then ``joblib.load``."""
    metadata = verify_bundle(path, expected_schema=expected_schema, strict_versions=strict_versions)
    system, schema, training_config = _unpickle(path)
    if system.schema_hash != metadata.schema_hash:
        msg = "Bundle incohérent : le modèle et metadata.json ne citent pas le même schéma."
        raise _integrity_error(msg, path)
    if system.is_synthetic and not metadata.is_synthetic:
        msg = "Bundle incohérent : modèle synthétique déclaré comme réel dans metadata.json."
        raise _integrity_error(msg, path)
    report = _read_report(path)
    _LOGGER.info("Bundle %s chargé.", metadata.bundle_id)
    return ModelBundle(
        path=path,
        metadata=metadata,
        system=system,
        schema=schema,
        training_config=training_config,
        report=report,
    )


def list_bundles(models_dir: Path) -> list[Path]:
    """Bundle directories directly under ``models_dir`` (those with metadata), newest first."""
    if not models_dir.is_dir():
        return []
    found: list[tuple[datetime, str, Path]] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            metadata = read_metadata(child)
        except BundleIntegrityError:
            _LOGGER.debug("Dossier ignoré (pas un bundle lisible) : %s", child.name)
            continue
        found.append((metadata.created_at, child.name, child))
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in found]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
