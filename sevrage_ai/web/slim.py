"""Patient-free copy of a trained model, for a public web page.

A :class:`~sevrage_ai.models.system.TrainedSystem` cannot be published as it is: the
novelty detector (``LocalOutlierFactor``) stores the scaled training rows, and every target
keeps the out-of-fold predictions of the training patients. :func:`slim_system` removes
both (the "profil global inhabituel" alert disappears with the detector). What remains is
what a prediction needs:

* the fitted fold models (scikit-learn pipelines: coefficients, trees);
* the calibration scores of the conformal sets and intervals: one number per calibration
  patient, with no feature value;
* summaries of the training data: range quantiles, medians and modes, class counts.

Residual risk, to weigh before publishing a model trained on real patients: gradient
boosting stores its bin thresholds, which are values seen in the training data (one list
per feature, not linked to patients), and any fitted model memorises a little of its
training set. Pseudonymised data remain personal data (GDPR): publishing such a model is
the data controller's decision.

The file is a pickle (joblib): only load a file you wrote yourself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import joblib
import numpy as np

from sevrage_ai.config.schema import DatasetSchema
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.errors import BundleIntegrityError
from sevrage_ai.inference.predictor import Predictor
from sevrage_ai.models.system import TrainedSystem
from sevrage_ai.models.training import CrossFit, FittedTarget
from sevrage_ai.types import ModelInfo, TargetName

__all__ = [
    "SLIM_FORMAT_VERSION",
    "load_slim_predictor",
    "patient_level_content",
    "save_slim_model",
    "slim_info",
    "slim_system",
]

SLIM_FORMAT_VERSION: Final = 1
_COMPRESS: Final = 3
_KEYS: Final = frozenset({"format", "system", "schema", "training_config", "info"})


def _empty_crossfit(crossfit: CrossFit) -> CrossFit:
    return CrossFit(
        fold_models=crossfit.fold_models,
        fold_ids=np.empty(0, dtype=np.intp),
        oof=np.empty((0, *crossfit.oof.shape[1:]), dtype=np.float64),
    )


def slim_system(system: TrainedSystem) -> TrainedSystem:
    """``system`` without the novelty detector and without out-of-fold predictions."""
    targets: dict[TargetName, FittedTarget] = {
        name: dataclasses.replace(fitted, crossfit=_empty_crossfit(fitted.crossfit))
        for name, fitted in system.targets.items()
    }
    ood = dataclasses.replace(system.ood_reference, novelty_model=None, novelty_threshold=None)
    return dataclasses.replace(system, targets=targets, ood_reference=ood)


def slim_info(info: ModelInfo) -> ModelInfo:
    """``info`` without the local bundle path."""
    return info.model_copy(update={"bundle_path": None})


def patient_level_content(system: TrainedSystem) -> tuple[str, ...]:
    """What still holds one row per training patient (empty for a slim system)."""
    found: list[str] = []
    if system.ood_reference.novelty_model is not None:
        found.append("détecteur de profils inhabituels (lignes d'apprentissage)")
    found.extend(
        f"prédictions hors échantillon de la cible {name}"
        for name, fitted in system.targets.items()
        if fitted.crossfit.oof.size or fitted.crossfit.fold_ids.size
    )
    return tuple(found)


def save_slim_model(path: Path, predictor: Predictor) -> None:
    """Write the slim copy of ``predictor`` to ``path`` (joblib, compressed)."""
    system = slim_system(predictor.system)
    payload: Mapping[str, object] = {
        "format": SLIM_FORMAT_VERSION,
        "system": system,
        "schema": predictor.schema,
        "training_config": predictor.training_config,
        "info": slim_info(predictor.info),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(payload), path, compress=_COMPRESS)


def load_slim_predictor(path: Path) -> Predictor:
    """Predictor of a file written by :func:`save_slim_model`.

    Raises:
        BundleIntegrityError: the file is not a slim model of a supported format.
    """
    payload = joblib.load(path)  # only our own file, see the module docstring
    if not isinstance(payload, dict) or set(payload) != _KEYS:
        msg = "Fichier de modèle web illisible."
        raise BundleIntegrityError(msg)
    if payload["format"] != SLIM_FORMAT_VERSION:
        msg = "Version de fichier de modèle web non prise en charge."
        raise BundleIntegrityError(msg)
    system, schema = payload["system"], payload["schema"]
    cfg, info = payload["training_config"], payload["info"]
    if not (
        isinstance(system, TrainedSystem)
        and isinstance(schema, DatasetSchema)
        and isinstance(cfg, TrainingConfig)
        and isinstance(info, ModelInfo)
    ):
        msg = "Fichier de modèle web incomplet."
        raise BundleIntegrityError(msg)
    return Predictor(system, schema, cfg, info)
