"""Confidence levels (design 7.1) and their explanation. Owner: WS3.

* HIGH: singleton conformal set containing the retained class, probability of that class
  >= ``min_top_prob_high``, no OOD flag, missing share <= ``max_missing_high``. For
  regression: relative width <= ``max_rel_width_moderate``, no OOD flag, missing share <=
  ``max_missing_high``.
* LOW: empty or full set, a critical flag, missing share > ``max_missing_low``, or relative
  interval width > ``max_rel_width_low`` (an infinite bound is LOW).
* MODERATE otherwise.

Every rule is a :class:`~sevrage_ai.types.ConfidenceCheck`, met or not: the level is the
lowest cap of the unmet checks, and the checks explain it to the clinician ("pourquoi
cette fiabilité ?"), including when it is high. Messages are French sentences with
feature labels and model outputs, never patient values. Predictions are never hidden: LOW
is displayed with "Prédiction peu fiable - se référer au protocole standard".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from sevrage_ai.config.training import AbstentionConfig
from sevrage_ai.types import ConfidenceCheck, ConfidenceLevel, Flag

__all__ = [
    "LOW_CONFIDENCE_MESSAGE",
    "Assessment",
    "assess_class",
    "assess_overall",
    "assess_regression",
    "class_confidence",
    "overall",
    "regression_confidence",
]

LOW_CONFIDENCE_MESSAGE: Final = "Prédiction peu fiable - se référer au protocole standard."
_WIDTH_EPS: Final = 1e-9
_MAX_LISTED: Final = 6
_NBSP: Final = "\u00a0"

CHECK_SET: Final = "confidence.set"
CHECK_PROBABILITY: Final = "confidence.probability"
CHECK_INTERVAL: Final = "confidence.interval"
CHECK_ALERTS: Final = "confidence.alerts"
CHECK_MISSING: Final = "confidence.missing"
CHECK_OUTPUTS: Final = "confidence.outputs"


@dataclass(frozen=True, slots=True)
class Assessment:
    """Level of one output (or of the whole prediction), its reasons and every check."""

    level: ConfidenceLevel
    reasons: tuple[str, ...]
    checks: tuple[ConfidenceCheck, ...]


# --------------------------------------------------------------------------------------
# Formatting (French)
# --------------------------------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{round(100.0 * value)}{_NBSP}%"


def _ratio(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _listing(labels: Sequence[str]) -> str:
    shown = list(labels[:_MAX_LISTED])
    more = len(labels) - len(shown)
    return ", ".join(shown) + (f" et {more} autre(s)" if more > 0 else "")


def _met(code: str, message: str) -> ConfidenceCheck:
    return ConfidenceCheck(code, met=True, message_fr=message)


def _unmet(code: str, message: str, caps_at: ConfidenceLevel) -> ConfidenceCheck:
    return ConfidenceCheck(code, met=False, message_fr=message, caps_at=caps_at)


def _assessment(checks: Sequence[ConfidenceCheck]) -> Assessment:
    """Lowest cap of the unmet checks; reasons are their messages, LOW caps first."""
    unmet = [c for c in checks if not c.met]
    low = [c.message_fr for c in unmet if c.caps_at is ConfidenceLevel.LOW]
    moderate = [c.message_fr for c in unmet if c.caps_at is ConfidenceLevel.MODERATE]
    if low:
        level = ConfidenceLevel.LOW
    elif moderate:
        level = ConfidenceLevel.MODERATE
    else:
        level = ConfidenceLevel.HIGH
    return Assessment(level=level, reasons=tuple(low + moderate), checks=tuple(checks))


# --------------------------------------------------------------------------------------
# Checks shared by every output: alerts and data completeness
# --------------------------------------------------------------------------------------


def _flag_labels(flags: Iterable[Flag], feature_labels: Mapping[str, str]) -> list[str]:
    labels: list[str] = []
    for flag in flags:
        label = (
            feature_labels.get(flag.feature, flag.feature)
            if flag.feature
            else "profil global atypique"
        )
        if label not in labels:
            labels.append(label)
    return labels


def _context_checks(
    flags: Sequence[Flag],
    missing_fraction: float,
    cfg: AbstentionConfig,
    missing_labels: Sequence[str],
    n_features: int,
    feature_labels: Mapping[str, str],
) -> list[ConfidenceCheck]:
    checks: list[ConfidenceCheck] = []
    critical = [f for f in flags if f.level == "critical"]
    unusual = [f for f in flags if f.code.startswith("ood.")]
    if critical:
        concerned = _flag_labels(critical, feature_labels)
        detail = f" ({_listing(concerned)})" if any(f.feature for f in critical) else ""
        checks.append(
            _unmet(
                CHECK_ALERTS,
                f"Alerte critique sur les données du patient{detail}.",
                ConfidenceLevel.LOW,
            )
        )
    elif unusual:
        checks.append(
            _unmet(
                CHECK_ALERTS,
                "Valeur(s) inhabituelle(s) par rapport à l'apprentissage : "
                f"{_listing(_flag_labels(unusual, feature_labels))}. Le modèle extrapole.",
                ConfidenceLevel.MODERATE,
            )
        )
    else:
        checks.append(
            _met(
                CHECK_ALERTS, "Aucune valeur inhabituelle par rapport aux patients d'apprentissage."
            )
        )

    if n_features > 0:
        count = len(missing_labels) if missing_labels else round(missing_fraction * n_features)
        detail = f" ({count} variable(s) sur {n_features}"
        detail += f" : {_listing(missing_labels)})" if missing_labels else ")"
    else:
        detail = f" ({_percent(missing_fraction)})"
    if missing_fraction > cfg.max_missing_low:
        checks.append(
            _unmet(
                CHECK_MISSING,
                f"Plus de {_percent(cfg.max_missing_low)} de données manquantes{detail}.",
                ConfidenceLevel.LOW,
            )
        )
    elif missing_fraction > cfg.max_missing_high:
        checks.append(
            _unmet(
                CHECK_MISSING,
                f"Plus de {_percent(cfg.max_missing_high)} de données manquantes{detail}.",
                ConfidenceLevel.MODERATE,
            )
        )
    elif missing_fraction == 0.0:
        checks.append(_met(CHECK_MISSING, "Toutes les variables du modèle sont renseignées."))
    else:
        checks.append(
            _met(
                CHECK_MISSING,
                f"Données suffisamment complètes{detail} : au plus "
                f"{_percent(cfg.max_missing_high)} de valeurs manquantes.",
            )
        )
    return checks


# --------------------------------------------------------------------------------------
# Per output
# --------------------------------------------------------------------------------------


def assess_class(
    proba: npt.NDArray[np.float64],
    prediction_set: npt.NDArray[np.bool_],
    *,
    missing_fraction: float,
    flags: Sequence[Flag],
    cfg: AbstentionConfig,
    label_index: int | None = None,
    missing_labels: Sequence[str] = (),
    n_features: int = 0,
    feature_labels: Mapping[str, str] | None = None,
) -> Assessment:
    """Level, reasons and checks of one classifier output (``proba``, ``prediction_set``: ``(C,)``).

    ``label_index`` is the retained class when it is not the argmax (prior-corrected or
    thresholded decision); the checks then apply to that class. ``missing_labels`` (labels
    of the missing model features), ``n_features`` and ``feature_labels`` (canonical name ->
    label, for the OOD flags) only enrich the messages.
    """
    p = np.asarray(proba, dtype=np.float64)
    members = np.asarray(prediction_set, dtype=bool)
    size = int(members.sum())
    top = int(np.argmax(p)) if label_index is None else int(label_index)
    checks: list[ConfidenceCheck] = []
    if size == 0:
        checks.append(
            _unmet(
                CHECK_SET,
                "Aucune classe n'atteint le niveau de fiabilité demandé.",
                ConfidenceLevel.LOW,
            )
        )
    elif size == p.size and p.size > 1:
        checks.append(
            _unmet(
                CHECK_SET,
                "Toutes les classes restent plausibles : le modèle ne peut pas trancher.",
                ConfidenceLevel.LOW,
            )
        )
    elif size > 1:
        checks.append(
            _unmet(
                CHECK_SET,
                f"{size} classes restent plausibles : le modèle hésite entre elles.",
                ConfidenceLevel.MODERATE,
            )
        )
    elif not members[top]:
        checks.append(
            _unmet(
                CHECK_SET,
                "La classe retenue n'est pas dans l'ensemble fiable.",
                ConfidenceLevel.MODERATE,
            )
        )
    else:
        checks.append(
            _met(CHECK_SET, "Une seule classe reste plausible, et c'est la classe retenue.")
        )
    probability = float(p[top])
    threshold = cfg.min_top_prob_high
    if probability < threshold:
        checks.append(
            _unmet(
                CHECK_PROBABILITY,
                f"Probabilité de la classe retenue inférieure à {_percent(threshold)} "
                f"({_percent(probability)}).",
                ConfidenceLevel.MODERATE,
            )
        )
    else:
        checks.append(
            _met(
                CHECK_PROBABILITY,
                f"Probabilité de la classe retenue de {_percent(probability)} "
                f"(au moins {_percent(threshold)}).",
            )
        )
    checks += _context_checks(
        flags, missing_fraction, cfg, missing_labels, n_features, feature_labels or {}
    )
    return _assessment(checks)


def assess_regression(
    point: float,
    lower: float,
    upper: float,
    *,
    missing_fraction: float,
    flags: Sequence[Flag],
    cfg: AbstentionConfig,
    missing_labels: Sequence[str] = (),
    n_features: int = 0,
    feature_labels: Mapping[str, str] | None = None,
) -> Assessment:
    """Level, reasons and checks of one regression output (canonical units)."""
    checks: list[ConfidenceCheck] = []
    if not (math.isfinite(lower) and math.isfinite(upper)):
        checks.append(_unmet(CHECK_INTERVAL, "Intervalle non borné.", ConfidenceLevel.LOW))
    else:
        width = (upper - lower) / max(abs(point), _WIDTH_EPS)
        ratio = f"{_ratio(width)} fois la valeur prédite"
        if width > cfg.max_rel_width_low:
            checks.append(
                _unmet(
                    CHECK_INTERVAL,
                    f"Intervalle très large par rapport à la valeur prédite : {ratio} "
                    f"(confiance faible au-delà de {_ratio(cfg.max_rel_width_low)} fois).",
                    ConfidenceLevel.LOW,
                )
            )
        elif width > cfg.max_rel_width_moderate:
            checks.append(
                _unmet(
                    CHECK_INTERVAL,
                    f"Intervalle large par rapport à la valeur prédite : {ratio} (confiance "
                    f"élevée jusqu'à {_ratio(cfg.max_rel_width_moderate)} fois).",
                    ConfidenceLevel.MODERATE,
                )
            )
        else:
            checks.append(
                _met(
                    CHECK_INTERVAL,
                    f"Intervalle étroit : {ratio} (au plus "
                    f"{_ratio(cfg.max_rel_width_moderate)} fois).",
                )
            )
    checks += _context_checks(
        flags, missing_fraction, cfg, missing_labels, n_features, feature_labels or {}
    )
    return _assessment(checks)


def assess_overall(
    components: Sequence[tuple[str, ConfidenceLevel]],
    *,
    flags: Sequence[Flag],
    missing_fraction: float,
    cfg: AbstentionConfig,
    missing_labels: Sequence[str] = (),
    n_features: int = 0,
    feature_labels: Mapping[str, str] | None = None,
) -> Assessment:
    """R7: the lowest level of the outputs (``(label, level)`` pairs), degraded further by
    alerts and missing data. The checks name the weakest output(s).

    Raises:
        ValueError: ``components`` is empty.
    """
    lowest = ConfidenceLevel.lowest(level for _, level in components)
    if lowest is ConfidenceLevel.HIGH:
        checks = [_met(CHECK_OUTPUTS, "Toutes les sorties ont une confiance élevée.")]
    else:
        weakest = [name for name, level in components if level is lowest]
        checks = [
            _unmet(
                CHECK_OUTPUTS,
                "La confiance globale ne peut pas dépasser celle de la sortie la moins "
                f"fiable : {_listing(weakest)} (confiance {lowest.label_fr}).",
                lowest,
            )
        ]
    checks += _context_checks(
        flags, missing_fraction, cfg, missing_labels, n_features, feature_labels or {}
    )
    return _assessment(checks)


# --------------------------------------------------------------------------------------
# Level-only API (kept for the callers that do not need the checks)
# --------------------------------------------------------------------------------------


def class_confidence(
    proba: npt.NDArray[np.float64],
    prediction_set: npt.NDArray[np.bool_],
    *,
    missing_fraction: float,
    flags: Sequence[Flag],
    cfg: AbstentionConfig,
    label_index: int | None = None,
) -> tuple[ConfidenceLevel, tuple[str, ...]]:
    """Level and reasons of one classifier output (see :func:`assess_class`)."""
    result = assess_class(
        proba,
        prediction_set,
        missing_fraction=missing_fraction,
        flags=flags,
        cfg=cfg,
        label_index=label_index,
    )
    return result.level, result.reasons


def regression_confidence(
    point: float,
    lower: float,
    upper: float,
    *,
    missing_fraction: float,
    flags: Sequence[Flag],
    cfg: AbstentionConfig,
) -> tuple[ConfidenceLevel, tuple[str, ...]]:
    """Level and reasons of one regression output (see :func:`assess_regression`)."""
    result = assess_regression(
        point, lower, upper, missing_fraction=missing_fraction, flags=flags, cfg=cfg
    )
    return result.level, result.reasons


def overall(
    levels: Iterable[ConfidenceLevel],
    *,
    flags: Sequence[Flag],
    missing_fraction: float,
    cfg: AbstentionConfig,
) -> ConfidenceLevel:
    """Level only of :func:`assess_overall` (outputs without names)."""
    components = [("", level) for level in levels]
    return assess_overall(components, flags=flags, missing_fraction=missing_fraction, cfg=cfg).level
