"""Consistency layer (rules R1-R5, R7 input): pure and deterministic. Owner: WS8a.

* R1 ``treat = P(yes) >= treatment_threshold``.
* R2 not treat: molecule ``"none"``, doses 0, duration 0. The conditional outputs are
  still shown (labelled "si un traitement est décidé").
* R3 treat: molecule = argmax of the conditional probabilities; values clipped at 0 and
  rounded per ``cfg.rounding``; duration >= 1; ``dose_day1_molecule_mg`` =
  dose / ``dose_equivalence_to_diazepam[molecule]`` (``None`` without an equivalence).
* R4 cumulative < day-1 -> cumulative = day-1, plus a warning flag.
* R5 flags: P(most severe class) >= ``severe_untreated_min_prob`` and not treated
  (critical); P(yes) in the grey zone (warning); argmax molecule not in its conformal
  set (warning).

The returned :class:`~sevrage_ai.types.FinalRecommendation` satisfies its own invariants
by construction.

Implementation notes (details the rules above leave open):

* A target contributes a value only when it is **enabled** in the schema **and** its
  output is present; otherwise the recommendation holds ``None`` for it.
* A class absent from ``probabilities`` has probability 0 (for example ``"yes"`` for a
  treatment model fitted on one class only).
* Molecule argmax: :data:`~sevrage_ai.types.NO_MOLECULE` is never a candidate. Ties go to
  the displayed ``label`` when it is among the best classes, then to the alphabetical
  order. With no candidate at all, :data:`~sevrage_ai.types.OTHER_CLASS` is used.
* Doses (mg diazepam-equivalent and mg of molecule) are rounded half-up to the nearest
  ``cfg.rounding.dose_mg``. A positive dose is never rounded to 0: below half a step it
  keeps one decimal (two significant digits if that is still 0). A molecule dose that
  overflows to infinity is ``None``.
* Durations are rounded half-up to the nearest ``cfg.rounding.duration_days`` and are at
  least 1 day when treated.
* R4 compares the model points (clipped at 0) before rounding. Rounding is monotone, so
  ``dose_cumulative_dze >= dose_day1_dze`` still holds afterwards.
* Flags come in a fixed order: severe untreated, grey zone, molecule not in set,
  cumulative below day-1, molecule without equivalence. Their messages are generic French
  sentences without any patient value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from sevrage_ai.config.schema import DatasetSchema
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.types import (
    NO_MOLECULE,
    OTHER_CLASS,
    ClassOutput,
    FinalRecommendation,
    Flag,
    FlagLevel,
    RegressionOutput,
    TargetName,
)

__all__ = [
    "CUMULATIVE_BELOW_DAY1",
    "MOLECULE_NOT_IN_SET",
    "MOLECULE_WITHOUT_EQUIVALENCE",
    "SEVERE_UNTREATED",
    "TREATMENT_GREY_ZONE",
    "TargetOutputs",
    "apply",
]

SEVERE_UNTREATED: Final = "consistency.severe_untreated"
TREATMENT_GREY_ZONE: Final = "consistency.treatment_grey_zone"
MOLECULE_NOT_IN_SET: Final = "consistency.molecule_not_in_set"
CUMULATIVE_BELOW_DAY1: Final = "consistency.cumulative_below_day1"
MOLECULE_WITHOUT_EQUIVALENCE: Final = "consistency.molecule_without_equivalence"

_MESSAGES: Final[dict[str, str]] = {
    SEVERE_UNTREATED: (
        "Probabilité notable de sevrage sévère ou compliqué alors qu'aucun traitement n'est "
        "recommandé : décision à réévaluer et surveillance renforcée."
    ),
    TREATMENT_GREY_ZONE: (
        "Probabilité de traitement proche du seuil de décision : la recommandation de "
        "traiter ou non est incertaine."
    ),
    MOLECULE_NOT_IN_SET: (
        "La molécule la plus probable ne fait pas partie de l'ensemble de prédiction "
        "conforme : choix de molécule incertain."
    ),
    CUMULATIVE_BELOW_DAY1: (
        "La dose totale prédite était inférieure à la dose du premier jour : elle a été "
        "alignée sur la dose du premier jour."
    ),
    MOLECULE_WITHOUT_EQUIVALENCE: (
        "Pas d'équivalence de dose connue pour la molécule recommandée : la dose n'est "
        "donnée qu'en mg équivalent diazépam."
    ),
}
_LEVELS: Final[dict[str, FlagLevel]] = {
    SEVERE_UNTREATED: "critical",
    TREATMENT_GREY_ZONE: "warning",
    MOLECULE_NOT_IN_SET: "warning",
    CUMULATIVE_BELOW_DAY1: "warning",
    MOLECULE_WITHOUT_EQUIVALENCE: "info",
}
_DECIMALS: Final = 9  # clean-up of step multiples (e.g. 3 * 0.1)


@dataclass(frozen=True, slots=True)
class TargetOutputs:
    """Per-target outputs of one patient, before the consistency layer."""

    severity: ClassOutput
    treatment: ClassOutput
    molecule: ClassOutput
    dose_day1: RegressionOutput | None
    dose_cumulative: RegressionOutput | None
    duration: RegressionOutput | None
    treatment_threshold: float


def apply(
    outputs: TargetOutputs, cfg: TrainingConfig, schema: DatasetSchema
) -> tuple[FinalRecommendation, list[Flag]]:
    """Apply R1-R5 and return the recommendation and the consistency flags."""
    enabled = schema.enabled_targets()

    def point(output: RegressionOutput | None, name: TargetName) -> float | None:
        if output is None or name not in enabled:
            return None
        return max(0.0, output.point)

    day1 = point(outputs.dose_day1, TargetName.DOSE_DAY1)
    cumulative = point(outputs.dose_cumulative, TargetName.DOSE_CUMULATIVE)
    duration = point(outputs.duration, TargetName.DURATION)

    positive = schema.target(TargetName.TREATMENT).positive_class or ""
    p_yes = outputs.treatment.probabilities.get(positive, 0.0)
    threshold = outputs.treatment_threshold
    treat = p_yes >= threshold  # R1

    raised: set[str] = set()
    p_high = outputs.severity.probabilities.get(schema.high_severity_class(), 0.0)
    if not treat and p_high >= cfg.consistency.severe_untreated_min_prob:
        raised.add(SEVERE_UNTREATED)
    if abs(p_yes - threshold) < cfg.treatment_threshold.grey_zone:
        raised.add(TREATMENT_GREY_ZONE)

    if not treat:  # R2
        final = FinalRecommendation(
            treat=False,
            molecule=NO_MOLECULE,
            dose_day1_dze=_zero(day1),
            dose_day1_molecule_mg=_zero(day1),
            dose_cumulative_dze=_zero(cumulative),
            duration_days=None if duration is None else 0,
            dose_cumulative_molecule_mg=_zero(cumulative),
        )
        return final, _flags(raised)

    molecule = _choose_molecule(outputs.molecule)  # R3
    if molecule not in outputs.molecule.prediction_set:
        raised.add(MOLECULE_NOT_IN_SET)
    if day1 is not None and cumulative is not None and cumulative < day1:  # R4
        cumulative = day1
        raised.add(CUMULATIVE_BELOW_DAY1)

    step = cfg.rounding.dose_mg
    day1_dze = _round_dose(day1, step)
    cumulative_dze = _round_dose(cumulative, step)
    equivalence = schema.dose_equivalence_to_diazepam.get(molecule)
    if equivalence is None and (day1_dze is not None or cumulative_dze is not None):
        raised.add(MOLECULE_WITHOUT_EQUIVALENCE)
    final = FinalRecommendation(
        treat=True,
        molecule=molecule,
        dose_day1_dze=day1_dze,
        dose_day1_molecule_mg=_molecule_mg(day1_dze, equivalence, step),
        dose_cumulative_dze=cumulative_dze,
        duration_days=_round_duration(duration, cfg.rounding.duration_days),
        dose_cumulative_molecule_mg=_molecule_mg(cumulative_dze, equivalence, step),
    )
    return final, _flags(raised)


def _flags(raised: set[str]) -> list[Flag]:
    return [
        Flag(code=code, level=level, message_fr=_MESSAGES[code])
        for code, level in _LEVELS.items()
        if code in raised
    ]


def _zero(value: float | None) -> float | None:
    return None if value is None else 0.0


def _choose_molecule(output: ClassOutput) -> str:
    candidates = [
        (probability, label)
        for label, probability in output.probabilities.items()
        if label and label != NO_MOLECULE
    ]
    if not candidates:
        return OTHER_CLASS
    best = max(probability for probability, _ in candidates)
    tied = sorted(label for probability, label in candidates if probability == best)
    return output.label if output.label in tied else tied[0]


def _round_step(value: float, step: float) -> float:
    """Half-up rounding of ``value >= 0`` to the nearest multiple of ``step``."""
    return round(math.floor(value / step + 0.5) * step, _DECIMALS)


def _round_positive(value: float, step: float) -> float:
    """:func:`_round_step`, except that a positive value never becomes 0."""
    rounded = _round_step(value, step)
    if rounded > 0.0 or value <= 0.0:
        return rounded
    one_decimal = round(value, 1)
    return one_decimal if one_decimal > 0.0 else float(f"{value:.2g}")


def _round_dose(value: float | None, step: float) -> float | None:
    return None if value is None else _round_positive(value, step)


def _molecule_mg(dose_dze: float | None, equivalence: float | None, step: float) -> float | None:
    if dose_dze is None or equivalence is None:
        return None
    dose = dose_dze / equivalence
    return _round_positive(dose, step) if math.isfinite(dose) else None


def _round_duration(value: float | None, step: int) -> int | None:
    if value is None:
        return None
    return max(1, math.floor(value / step + 0.5) * step)
