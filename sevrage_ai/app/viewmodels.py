"""Pure display helpers: turn a ``PatientPrediction`` into display structures (no Streamlit
import, unit-tested). Owner: WS10a.

Every user-facing string is French. Numbers use the French decimal comma and a narrow
no-break space as thousands separator (``format_number``). Pages and ``components`` only
render the frozen dataclasses and row dictionaries built here, so the formatting logic is
tested without Streamlit. Nothing here logs or stores patient values.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, overload

import pandas as pd
from pydantic import ValidationError

from sevrage_ai.config.dictionary import dictionary_rows
from sevrage_ai.config.schema import DatasetSchema, FeatureSpec
from sevrage_ai.data.anonymization import AnonymizationReport
from sevrage_ai.data.validation import build_record_model
from sevrage_ai.errors import ConfigError
from sevrage_ai.evaluation.report import EvaluationReport
from sevrage_ai.inference import export
from sevrage_ai.types import (
    CLASSIFICATION_TARGETS,
    DOSE_UNIT,
    NO_MOLECULE,
    OTHER_CLASS,
    TREATED,
    BatchResult,
    ClassOutput,
    ConfidenceCheck,
    ConfidenceLevel,
    FeatureKind,
    Flag,
    FlagLevel,
    MetricEstimate,
    ModelInfo,
    PatientPrediction,
    RegressionOutput,
    RowIssue,
    TargetName,
)

__all__ = [
    "ACK_LABEL",
    "DECREASE_LABEL",
    "DISCLAIMER_MD",
    "EXAMPLE_PATIENT",
    "HOW_TO_MD",
    "INCREASE_LABEL",
    "LOW_CONFIDENCE_MESSAGE",
    "METRICS_GUIDE_MD",
    "MOLECULE_SUBTITLE",
    "NO_OPTION",
    "READING_GUIDE_MD",
    "SHORT_DISCLAIMER",
    "SYNTHETIC_BANNER",
    "SYNTHETIC_EXPLANATION",
    "TOOL_PURPOSE_MD",
    "UNKNOWN_OPTION",
    "YES_OPTION",
    "Badge",
    "CheckItem",
    "CheckStatus",
    "Choice",
    "ClassCard",
    "ClassRow",
    "ConfidenceExplanation",
    "ContributionItem",
    "FinalSummary",
    "FlagGroup",
    "FlagItem",
    "FormField",
    "FormGroup",
    "MetricItem",
    "PredictionView",
    "RegressionCard",
    "SummaryFigure",
    "anonymization_rows",
    "batch_csv",
    "batch_overview",
    "batch_rows",
    "batch_summary",
    "class_label",
    "confidence_badge",
    "confidence_explanation",
    "contribution_axis_title",
    "contribution_chart_rows",
    "contribution_hint",
    "contribution_targets",
    "contributions",
    "dictionary_table",
    "dose_card",
    "duration_card",
    "equivalence_rows",
    "fallback_model_card",
    "filled_count",
    "final_summary",
    "flag_groups",
    "form_groups",
    "form_values",
    "format_feature_value",
    "format_metric",
    "format_number",
    "format_percent",
    "format_plain",
    "low_confidence_message",
    "metric_items",
    "metrics_caption",
    "metrics_rows",
    "model_info_rows",
    "molecule_card",
    "output_tab_label",
    "performance_rows",
    "prediction_json",
    "prediction_view",
    "record_from_form",
    "rejected_rows",
    "report_rows",
    "severity_card",
    "target_rows",
    "template_columns",
    "template_csv",
    "treatment_card",
    "unit_label",
    "validation_error_messages",
    "validation_messages",
    "with_context",
]

BadgeColour = Literal["green", "orange", "red"]

LOW_CONFIDENCE_MESSAGE: Final = "Prédiction peu fiable - se référer au protocole standard"
"""Shown whenever a confidence level is LOW (the prediction is never hidden)."""
SYNTHETIC_BANNER: Final = "MODÈLE ENTRAÎNÉ SUR DONNÉES SYNTHÉTIQUES"
SYNTHETIC_EXPLANATION: Final = (
    "Ce modèle n'a aucune valeur clinique : il sert uniquement aux tests et aux "
    "démonstrations de l'interface."
)
SHORT_DISCLAIMER: Final = (
    "Outil de recherche académique, pas un dispositif médical. "
    "La décision revient toujours au clinicien. Les données saisies ne sont pas enregistrées."
)
ACK_LABEL: Final = "J'ai lu et compris cet avertissement."
DISCLAIMER_MD: Final = """\
**Outil de recherche académique** développé dans le cadre d'un projet de Master 2.
**Ce n'est pas un dispositif médical** au sens du règlement (UE) 2017/745 : aucun marquage
CE, aucune évaluation clinique, aucune validation réglementaire.

- Ne jamais l'utiliser de façon autonome pour décider d'un traitement : **la décision revient
  toujours au clinicien**, qui dispose d'informations que l'outil ignore.
- Les prédictions reflètent les **pratiques passées** d'un seul service sur un petit nombre
  de patients : la molécule affichée est celle habituellement prescrite pour des profils
  similaires, pas un traitement optimal. Les doses sont converties avec une table
  d'équivalence **à valider par un clinicien**.
- Les probabilités, ensembles et intervalles sont des garanties **moyennes** sur des
  patients comparables à ceux de l'apprentissage, pas des garanties individuelles.
- Une confiance « faible » signifie : **prédiction peu fiable, se référer au protocole
  standard**.
- Un modèle entraîné sur des **données synthétiques** (bandeau rouge) n'a aucune valeur
  clinique.
- Les données saisies ou chargées sont traitées en mémoire et **ne sont pas enregistrées**.
"""
TOOL_PURPOSE_MD: Final = """\
#### À quoi sert l'outil ?
À partir des données connues **à l'admission**, il indique ce qui a été fait le plus
souvent pour des patients au profil similaire :

1. la **sévérité** probable du sevrage ;
2. si un **traitement par benzodiazépine** est habituellement instauré ;
3. si oui, la **molécule**, la **dose totale** et la **durée** habituelles.

Chaque résultat est accompagné d'un **niveau de confiance** expliqué.
"""
HOW_TO_MD: Final = """\
#### Comment l'utiliser ?
1. Lisez l'avertissement ci-dessous et cochez la case.
2. Ouvrez **Un patient** (saisie d'un formulaire) ou **Plusieurs patients** (fichier
   CSV ou Excel).
3. Lisez la **synthèse**, puis le détail de chaque résultat et sa fiabilité.

Le bouton **Exemple fictif** de la page Un patient permet d'essayer l'outil sans données.
"""
MOLECULE_SUBTITLE: Final = (
    "Molécule habituellement prescrite pour des profils similaires "
    "(probabilités conditionnelles : si un traitement est décidé)."
)
UNKNOWN_OPTION: Final = "Inconnu"
YES_OPTION: Final = "Oui"
NO_OPTION: Final = "Non"
DEFAULT_TOP_K: Final = 8

_NARROW_NBSP: Final = chr(0x202F)  # narrow no-break space (thousands separator)
_NBSP: Final = chr(0x00A0)  # no-break space (before "%" and units)
_DASH: Final = chr(0x2013)  # en dash (ranges)
_BADGES: Final[Mapping[ConfidenceLevel, tuple[BadgeColour, str]]] = {
    ConfidenceLevel.HIGH: ("green", ":material/check_circle:"),
    ConfidenceLevel.MODERATE: ("orange", ":material/info:"),
    ConfidenceLevel.LOW: ("red", ":material/warning:"),
}
_UNIT_LABELS: Final[Mapping[str, str]] = {
    DOSE_UNIT: "mg éq. diazépam",
    "jours": "jours",
    "days": "jours",
}
_FLAG_TITLES: Final[Mapping[FlagLevel, str]] = {
    "critical": "Alertes critiques",
    "warning": "Avertissements",
    "info": "Informations",
}
_FLAG_ORDER: Final[tuple[FlagLevel, ...]] = ("critical", "warning", "info")
_METRIC_LABELS: Final[Mapping[str, str]] = {
    "accuracy": "Exactitude",
    "balanced_accuracy": "Exactitude équilibrée",
    "macro_f1": "F1 macro",
    "log_loss": "Log-loss",
    "brier": "Score de Brier",
    "ece": "Erreur de calibration (ECE)",
    "roc_auc": "AUC ROC",
    "roc_auc_ovr": "AUC ROC (un contre tous)",
    "pr_auc": "AUC PR",
    "calibration_slope": "Pente de calibration",
    "calibration_intercept": "Ordonnée de calibration",
    "sensitivity": "Sensibilité",
    "specificity": "Spécificité",
    "ppv": "Valeur prédictive positive",
    "npv": "Valeur prédictive négative",
    "top2_accuracy": "Exactitude top-2",
    "qwk": "Kappa pondéré quadratique",
    "mae": "Erreur absolue moyenne",
    "median_ae": "Erreur absolue médiane",
    "rmse": "Racine de l'erreur quadratique moyenne",
    "r2": "R²",
    "coverage": "Couverture empirique",
    "mean_set_size": "Taille moyenne des ensembles",
    "singleton_rate": "Part d'ensembles à une classe",
    "mean_width": "Largeur moyenne des intervalles",
    "median_width": "Largeur médiane des intervalles",
}
_BASELINE_LABELS: Final[Mapping[str, str]] = {"dummy": "Référence naïve"}
_CLASSIFIER_DELTA_UNIT: Final = "pts"
_CLASSIFIER_TARGETS: Final = CLASSIFICATION_TARGETS
_VOWELS: Final = frozenset("aeiouyhéèêàâîôû")
_ALERT_LEVELS: Final = frozenset({"warning", "critical"})


# --------------------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------------------


def format_number(value: float, decimals: int = 0) -> str:
    """French number: decimal comma, narrow no-break space between thousands.

    ``nan`` gives ``"n.d."`` (not available), infinities give ``"∞"`` / ``"-∞"``.
    """
    if math.isnan(value):
        return "n.d."
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0  # avoid "-0"
    text = f"{rounded:,.{decimals}f}"
    return text.replace(",", _NARROW_NBSP).replace(".", ",")


def format_plain(value: float) -> str:
    """Shortest French representation of a bound or a raw value (``1.5`` -> ``"1,5"``)."""
    if math.isnan(value) or math.isinf(value):
        return format_number(value)
    return f"{value:g}".replace(".", ",")


def format_percent(probability: float, decimals: int = 0) -> str:
    """``0.724`` -> ``"72 %"`` (no-break space before the percent sign)."""
    return f"{format_number(probability * 100.0, decimals)}{_NBSP}%"


def _auto_decimals(value: float) -> int:
    magnitude = abs(value)
    if magnitude >= 100:
        return 0
    return 1 if magnitude >= 10 else 2


def format_metric(metric: MetricEstimate, decimals: int | None = None) -> str:
    """``"0,71 [0,64-0,78]"`` (en dash); ``"non disponible"`` when the value is NaN.

    ``decimals=None`` adapts to the magnitude: 2 below 10, 1 below 100, else 0.
    """
    if math.isnan(metric.value):
        return "non disponible"
    places = _auto_decimals(metric.value) if decimals is None else decimals
    value = format_number(metric.value, places)
    if math.isnan(metric.lo) or math.isnan(metric.hi):
        return value
    lo = format_number(metric.lo, places)
    hi = format_number(metric.hi, places)
    return f"{value} [{lo}{_DASH}{hi}]"


def unit_label(unit: str | None) -> str:
    """Display label of a canonical unit (``mg_diazepam_eq`` -> ``"mg éq. diazépam"``)."""
    if not unit:
        return ""
    return _UNIT_LABELS.get(unit, unit)


def _with_unit(text: str, unit: str | None) -> str:
    label = unit_label(unit)
    return f"{text}{_NBSP}{label}" if label else text


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def class_label(schema: DatasetSchema, target: TargetName, code: str) -> str:
    """French label of a class (``schema.targets[target].class_labels``), else the code."""
    if code == NO_MOLECULE:
        return "Aucune"
    spec = schema.targets.get(target)
    if spec is None:
        return code
    return spec.class_labels.get(code, code)


def _target_title(schema: DatasetSchema, target: TargetName) -> str:
    spec = schema.targets.get(target)
    return spec.label if spec is not None else target.value


def _feature_or_none(schema: DatasetSchema, name: str) -> FeatureSpec | None:
    try:
        return schema.feature(name)
    except ConfigError:
        return None


def _feature_label(schema: DatasetSchema, name: str) -> str:
    feature = _feature_or_none(schema, name)
    return feature.label if feature is not None else name


def _category_labels(feature: FeatureSpec) -> dict[str, str]:
    """Display label of each category: the first raw value mapped to it, else the code."""
    labels: dict[str, str] = {}
    for raw, mapped in (feature.value_mapping or {}).items():
        if isinstance(mapped, str) and mapped not in labels:
            labels[mapped] = raw
    return {code: labels.get(code, code) for code in feature.categories or ()}


def format_feature_value(feature: FeatureSpec, value: object) -> str:
    """Display a canonical value: ``"4000 UI/L"``, ``"Oui"``, ``"Masculin"`` or ``"Inconnu"``."""
    if _is_missing(value):
        return UNKNOWN_OPTION
    if feature.kind is FeatureKind.CATEGORICAL:
        return _category_labels(feature).get(str(value), str(value))
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return str(value)
    if feature.kind is FeatureKind.BINARY:
        return YES_OPTION if float(value) == 1.0 else NO_OPTION
    return _with_unit(format_plain(float(value)), feature.unit)


# --------------------------------------------------------------------------------------
# Badges and cards
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Badge:
    """Confidence badge: French label, Streamlit colour name and Material icon."""

    level: ConfidenceLevel
    label: str
    colour: BadgeColour
    icon: str


def confidence_badge(level: ConfidenceLevel) -> Badge:
    """``Confiance élevée`` (green), ``modérée`` (orange) or ``faible`` (red)."""
    colour, icon = _BADGES[level]
    return Badge(level=level, label=f"Confiance {level.label_fr}", colour=colour, icon=icon)


def low_confidence_message(level: ConfidenceLevel) -> str | None:
    """:data:`LOW_CONFIDENCE_MESSAGE` for LOW confidence, else ``None``."""
    return LOW_CONFIDENCE_MESSAGE if level is ConfidenceLevel.LOW else None


CheckStatus = Literal["met", "moderate", "low", "info"]
"""``met``: supports a high confidence; ``moderate`` / ``low``: caps the confidence at that
level; ``info``: context (e.g. the measured performance of the model)."""

_EXPLANATION_SUMMARY: Final[Mapping[ConfidenceLevel, str]] = {
    ConfidenceLevel.HIGH: "Confiance élevée : tous les critères ci-dessous sont remplis.",
    ConfidenceLevel.MODERATE: (
        "Confiance modérée : au moins un critère empêche une confiance élevée."
    ),
    ConfidenceLevel.LOW: "Confiance faible : au moins un critère bloquant n'est pas rempli.",
}
_STATUS_ORDER: Final[Mapping[str, int]] = {"low": 0, "moderate": 1, "met": 2, "info": 3}


@dataclass(frozen=True, slots=True)
class CheckItem:
    """One line of a confidence explanation."""

    status: CheckStatus
    text: str


@dataclass(frozen=True, slots=True)
class ConfidenceExplanation:
    """ "Pourquoi cette fiabilité ?": a summary sentence and every criterion, unmet first."""

    title: str
    summary: str
    items: tuple[CheckItem, ...]


def confidence_explanation(
    level: ConfidenceLevel,
    checks: Sequence[ConfidenceCheck],
    reasons: Sequence[str] = (),
    *,
    title: str = "Pourquoi cette fiabilité ?",
) -> ConfidenceExplanation:
    """Explanation of ``level`` from its checks (or from ``reasons`` when there are none,
    e.g. outputs of a fake predictor): criteria capping at LOW, then MODERATE, then met."""
    items: list[CheckItem] = []
    for check in checks:
        if check.met:
            items.append(CheckItem("met", check.message_fr))
        elif check.caps_at is ConfidenceLevel.LOW:
            items.append(CheckItem("low", check.message_fr))
        else:
            items.append(CheckItem("moderate", check.message_fr))
    if not checks:
        status: CheckStatus = "low" if level is ConfidenceLevel.LOW else "moderate"
        items = [CheckItem(status, reason) for reason in reasons]
    items.sort(key=lambda item: _STATUS_ORDER[item.status])
    return ConfidenceExplanation(
        title=title, summary=_EXPLANATION_SUMMARY[level], items=tuple(items)
    )


def with_context(explanation: ConfidenceExplanation, lines: Iterable[str]) -> ConfidenceExplanation:
    """``explanation`` with context lines (status ``info``) appended."""
    extra = tuple(CheckItem("info", line) for line in lines if line)
    return replace(explanation, items=explanation.items + extra)


@dataclass(frozen=True, slots=True)
class ClassRow:
    """One class of a classification card (``probability`` in [0, 1])."""

    code: str
    label: str
    probability: float
    text: str
    in_set: bool
    predicted: bool


@dataclass(frozen=True, slots=True)
class ClassCard:
    """Display structure of a classification output."""

    target: TargetName
    title: str
    subtitle: str
    decision: str
    rows: tuple[ClassRow, ...]
    set_text: str
    badge: Badge
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    explanation: ConfidenceExplanation


@dataclass(frozen=True, slots=True)
class RegressionCard:
    """Display structure of a regression output (point and conformal interval)."""

    target: TargetName
    title: str
    subtitle: str
    value_text: str
    interval_text: str
    molecule_value_text: str | None
    molecule_interval_text: str | None
    badge: Badge
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    explanation: ConfidenceExplanation


def _level_percent(alpha: float) -> str:
    return format_percent(1.0 - alpha)


def _ordered_codes(schema: DatasetSchema, output: ClassOutput) -> list[str]:
    spec = schema.targets.get(output.target)
    declared = [c for c in (spec.classes or ()) if c in output.probabilities] if spec else []
    return declared + [c for c in output.probabilities if c not in declared]


def _set_text(schema: DatasetSchema, output: ClassOutput, codes: Sequence[str]) -> str:
    prefix = (
        f"Réponses compatibles avec ce profil (ensemble de prédiction à "
        f"{_level_percent(output.alpha)}) :"
    )
    members = [class_label(schema, output.target, c) for c in codes if c in output.prediction_set]
    if not members:
        return f"{prefix} aucune, ensemble vide (prédiction incertaine)."
    text = f"{prefix} {', '.join(members)}"
    if len(members) == len(codes) and len(codes) > 1:
        return f"{text} (toutes les réponses : prédiction non informative)."
    return f"{text}."


def _class_card(
    schema: DatasetSchema,
    output: ClassOutput,
    *,
    title: str,
    subtitle: str,
    notes: Iterable[str] = (),
) -> ClassCard:
    codes = _ordered_codes(schema, output)
    rows = tuple(
        ClassRow(
            code=code,
            label=class_label(schema, output.target, code),
            probability=float(output.probabilities[code]),
            text=format_percent(output.probabilities[code]),
            in_set=code in output.prediction_set,
            predicted=code == output.label,
        )
        for code in codes
    )
    all_notes = list(notes)
    low = low_confidence_message(output.confidence)
    if low is not None:
        all_notes.append(low)
    return ClassCard(
        target=output.target,
        title=title,
        subtitle=subtitle,
        decision=class_label(schema, output.target, output.label),
        rows=rows,
        set_text=_set_text(schema, output, codes),
        badge=confidence_badge(output.confidence),
        reasons=output.reasons,
        notes=tuple(all_notes),
        explanation=confidence_explanation(output.confidence, output.checks, output.reasons),
    )


def severity_card(prediction: PatientPrediction, schema: DatasetSchema) -> ClassCard:
    """Severity: calibrated probability of each class (ordered) and the conformal set.

    The retained class may not be the most probable one (prior-corrected decision, see
    ``config.training.DecisionConfig``): a note then explains why.
    """
    output = prediction.severity
    most_probable = max(output.probabilities, key=lambda code: output.probabilities[code])
    notes = []
    if most_probable != output.label:
        retained = class_label(schema, TargetName.SEVERITY, output.label)
        usual = class_label(schema, TargetName.SEVERITY, most_probable)
        notes.append(
            f"« {retained} » est retenu car sa probabilité dépasse le plus sa fréquence "
            f"habituelle chez les patients hospitalisés, même si « {usual} » reste le plus "
            "probable."
        )
    return _class_card(
        schema,
        output,
        title=_target_title(schema, TargetName.SEVERITY),
        subtitle="Probabilité de chaque niveau de sévérité chez des patients comparables.",
        notes=notes,
    )


def _positive_class(schema: DatasetSchema) -> str:
    spec = schema.targets.get(TargetName.TREATMENT)
    return (spec.positive_class if spec is not None else None) or TREATED


def treatment_card(
    prediction: PatientPrediction, schema: DatasetSchema, info: ModelInfo
) -> ClassCard:
    """Treatment: P(oui), the decision at the model threshold and the grey-zone note.

    The grey zone is ``|P(oui) - threshold| < grey_zone`` (``TrainingConfig``).
    """
    output = prediction.treatment
    p_yes = float(output.probabilities.get(_positive_class(schema), 0.0))
    threshold = info.treatment_threshold
    notes = [
        f"Probabilité de traitement : {format_percent(p_yes)} "
        f"(seuil de décision du modèle : {format_percent(threshold)})."
    ]
    if abs(p_yes - threshold) < info.grey_zone:
        notes.append(
            "Zone grise : la probabilité est proche du seuil de décision "
            f"(± {format_percent(info.grey_zone)}). Décision à discuter au cas par cas."
        )
    return _class_card(
        schema,
        output,
        title=_target_title(schema, TargetName.TREATMENT),
        subtitle="Probabilité qu'un traitement par benzodiazépine soit instauré.",
        notes=notes,
    )


def molecule_card(prediction: PatientPrediction, schema: DatasetSchema) -> ClassCard:
    """Molecule usually prescribed for similar profiles (conditional on treatment)."""
    notes: list[str] = []
    if not prediction.final.treat:
        notes.append("Aucun traitement recommandé : molécule affichée à titre indicatif.")
    return _class_card(
        schema,
        prediction.molecule_if_treated,
        title=f"{_target_title(schema, TargetName.MOLECULE)} (si un traitement est décidé)",
        subtitle=MOLECULE_SUBTITLE,
        notes=notes,
    )


def _interval_text(output: RegressionOutput, decimals: int, unit: str | None) -> str:
    lower = format_number(output.lower, decimals)
    if math.isinf(output.upper):
        bounds = f"{lower} {_DASH} non borné"
    else:
        bounds = f"{lower} {_DASH} {format_number(output.upper, decimals)}"
    return (
        f"Fourchette plausible (intervalle à {_level_percent(output.alpha)}) : "
        f"{_with_unit(bounds, unit)}"
    )


def _molecule_name(schema: DatasetSchema, molecule: str) -> str:
    label = class_label(schema, TargetName.MOLECULE, molecule)
    return label.split(" (", 1)[0].lower()


def _of_molecule(name: str) -> str:
    return f"d'{name}" if name[:1] in _VOWELS else f"de {name}"


def _equivalence(info: ModelInfo, schema: DatasetSchema, molecule: str) -> float | None:
    factor = info.dose_equivalence_to_diazepam.get(molecule)
    if factor is None:
        factor = schema.dose_equivalence_to_diazepam.get(molecule)
    return factor if factor is not None and factor > 0 else None


def _regression_notes(output: RegressionOutput, final_treat: bool) -> list[str]:
    notes: list[str] = []
    if not final_treat:
        notes.append(
            "Aucun traitement recommandé : valeur conditionnelle affichée à titre indicatif."
        )
    if math.isinf(output.upper):
        notes.append("Intervalle non borné : l'incertitude est trop grande.")
    low = low_confidence_message(output.confidence)
    if low is not None:
        notes.append(low)
    return notes


def dose_card(
    prediction: PatientPrediction, schema: DatasetSchema, info: ModelInfo
) -> RegressionCard | None:
    """Total dose (point and interval) in mg eq. diazepam and in mg of the molecule.

    ``None`` when the cumulative dose target is disabled.
    """
    output = prediction.dose_cumulative_if_treated
    if output is None:
        return None
    molecule = prediction.molecule_if_treated.label
    factor = _equivalence(info, schema, molecule) if molecule != OTHER_CLASS else None
    molecule_value: str | None = None
    molecule_interval: str | None = None
    if factor is not None:
        of_name = _of_molecule(_molecule_name(schema, molecule))
        molecule_value = f"soit ≈ {format_number(output.point / factor)}{_NBSP}mg {of_name}"
        lower = format_number(output.lower / factor)
        upper = "non borné" if math.isinf(output.upper) else format_number(output.upper / factor)
        molecule_interval = f"Intervalle : {lower} {_DASH} {upper}{_NBSP}mg {of_name}"
    return RegressionCard(
        target=output.target,
        title=f"{_target_title(schema, output.target)} (si un traitement est décidé)",
        subtitle="Dose totale sur l'ensemble du sevrage, pour des profils similaires.",
        value_text=_with_unit(format_number(output.point), output.unit),
        interval_text=_interval_text(output, 0, output.unit),
        molecule_value_text=molecule_value,
        molecule_interval_text=molecule_interval,
        badge=confidence_badge(output.confidence),
        reasons=output.reasons,
        notes=tuple(_regression_notes(output, prediction.final.treat)),
        explanation=confidence_explanation(output.confidence, output.checks, output.reasons),
    )


def _days(value: float) -> str:
    return f"{format_number(value)}{_NBSP}{'jour' if round(value) <= 1 else 'jours'}"


def duration_card(prediction: PatientPrediction, schema: DatasetSchema) -> RegressionCard | None:
    """Treatment duration in days (point and interval); ``None`` when disabled."""
    output = prediction.duration_if_treated
    if output is None:
        return None
    return RegressionCard(
        target=output.target,
        title=f"{_target_title(schema, output.target)} (si un traitement est décidé)",
        subtitle="Nombre de jours de traitement par benzodiazépine.",
        value_text=_days(output.point),
        interval_text=_interval_text(output, 0, "jours"),
        molecule_value_text=None,
        molecule_interval_text=None,
        badge=confidence_badge(output.confidence),
        reasons=output.reasons,
        notes=tuple(_regression_notes(output, prediction.final.treat)),
        explanation=confidence_explanation(output.confidence, output.checks, output.reasons),
    )


# --------------------------------------------------------------------------------------
# Final recommendation, flags, contributions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummaryFigure:
    """One key figure of the summary (``st.metric``): label, short value, optional help."""

    label: str
    value: str
    help: str | None = None


@dataclass(frozen=True, slots=True)
class FinalSummary:
    """Recommendation after the consistency rules, with the overall confidence.

    ``figures`` are the key figures shown at a glance (molecule, total dose, duration) and
    ``figures_note`` the dose in mg of the molecule itself; ``lines`` say the same in full
    sentences (exports, tests).
    """

    treat: bool
    headline: str
    lines: tuple[str, ...]
    badge: Badge
    low_confidence: str | None
    synthetic: bool
    explanation: ConfidenceExplanation
    figures: tuple[SummaryFigure, ...] = ()
    figures_note: str | None = None


def _mg_text(dose_dze: float, molecule_mg: float | None, of_name: str) -> str:
    text = _with_unit(format_number(dose_dze), DOSE_UNIT)
    if molecule_mg is not None:
        text += f" (≈ {format_number(molecule_mg)}{_NBSP}mg {of_name})"
    return text


def final_summary(prediction: PatientPrediction, schema: DatasetSchema) -> FinalSummary:
    """Summary of ``prediction.final`` (the total dose; no day-1 dose in this project)."""
    final = prediction.final
    lines: list[str] = []
    figures: list[SummaryFigure] = []
    figures_note: str | None = None
    if final.treat:
        headline = "Traitement par benzodiazépine envisagé pour des profils similaires"
        molecule_label = class_label(schema, TargetName.MOLECULE, final.molecule)
        lines.append(f"Molécule : {molecule_label}")
        name = _molecule_name(schema, final.molecule)
        figures.append(SummaryFigure("Molécule", name[:1].upper() + name[1:], molecule_label))
        of_name = _of_molecule(name)
        if final.dose_day1_dze is not None:
            text = _mg_text(final.dose_day1_dze, final.dose_day1_molecule_mg, of_name)
            lines.append(f"Dose du premier jour : {text}")
        if final.dose_cumulative_dze is not None:
            text = _mg_text(final.dose_cumulative_dze, final.dose_cumulative_molecule_mg, of_name)
            lines.append(f"Dose totale : {text}")
            figures.append(
                SummaryFigure(
                    "Dose totale (éq. diazépam)",
                    f"{format_number(final.dose_cumulative_dze)}{_NBSP}mg",
                    "Dose cumulée sur tout le sevrage, en mg équivalent diazépam.",
                )
            )
            if final.dose_cumulative_molecule_mg is not None:
                figures_note = (
                    f"Soit ≈ {format_number(final.dose_cumulative_molecule_mg)}{_NBSP}mg "
                    f"{of_name} sur l'ensemble du sevrage."
                )
        if final.duration_days is not None:
            lines.append(f"Durée : {_days(float(final.duration_days))}")
            figures.append(SummaryFigure("Durée", _days(float(final.duration_days))))
    else:
        headline = "Pas de traitement par benzodiazépine pour des profils similaires"
        lines.append("Surveillance clinique selon le protocole du service.")
    return FinalSummary(
        treat=final.treat,
        headline=headline,
        lines=tuple(lines),
        badge=confidence_badge(prediction.overall_confidence),
        low_confidence=low_confidence_message(prediction.overall_confidence),
        synthetic=prediction.trained_on_synthetic,
        explanation=confidence_explanation(
            prediction.overall_confidence,
            prediction.overall_checks,
            title="Pourquoi cette fiabilité globale ?",
        ),
        figures=tuple(figures),
        figures_note=figures_note,
    )


_SHORT_TITLES: Final[Mapping[TargetName, str]] = {
    TargetName.SEVERITY: "Sévérité",
    TargetName.TREATMENT: "Traitement",
    TargetName.MOLECULE: "Molécule",
    TargetName.DOSE_DAY1: "Dose du 1er jour",
    TargetName.DOSE_CUMULATIVE: "Dose totale",
    TargetName.DURATION: "Durée",
}


def output_tab_label(card: ClassCard | RegressionCard) -> str:
    """Tab label of one output: coloured confidence icon and short title.

    E.g. ``":red[:material/warning:] Sévérité"``: the confidence of every output is visible
    at a glance, before opening its tab.
    """
    colour, icon = _BADGES[card.badge.level]
    return f":{colour}[{icon}] {_SHORT_TITLES.get(card.target, card.title)}"


READING_GUIDE_MD: Final = """\
- **Synthèse** : ce qui a été fait le plus souvent pour des patients au profil similaire
  (traitement, molécule, dose totale, durée). Ce n'est pas une prescription.
- **Niveau de confiance** : :green[élevée], :orange[modérée] ou :red[faible]. Il dépend des
  données renseignées, de la ressemblance du patient avec ceux de l'apprentissage et de la
  netteté de la prédiction ; « Pourquoi cette fiabilité ? » en donne le détail.
- **Probabilité** : fréquence estimée chez des patients comparables (75 % = 3 sur 4).
- **Réponses compatibles** et **fourchette plausible** : ce que le modèle ne peut pas
  exclure, avec une garantie moyenne de 90 % sur des patients comparables.
- **Variables qui ont le plus pesé** : ce qui a fait monter ou baisser la prédiction pour
  ce patient par rapport à un patient « type ». Ce n'est pas un lien de cause à effet.
"""
"""How to read the patient results, in plain French (patient page)."""


@dataclass(frozen=True, slots=True)
class FlagItem:
    """One flag message, with the label of the feature it concerns (if any)."""

    code: str
    message: str
    feature_label: str | None


@dataclass(frozen=True, slots=True)
class FlagGroup:
    """Flags of one level (``critical``, ``warning`` or ``info``)."""

    level: FlagLevel
    title: str
    items: tuple[FlagItem, ...]


def flag_groups(flags: Iterable[Flag], schema: DatasetSchema) -> tuple[FlagGroup, ...]:
    """Flags grouped by level, most severe first; empty groups are omitted."""
    materialised = tuple(flags)
    groups: list[FlagGroup] = []
    for level in _FLAG_ORDER:
        items = tuple(
            FlagItem(
                code=flag.code,
                message=flag.message_fr,
                feature_label=None
                if flag.feature is None
                else _feature_label(schema, flag.feature),
            )
            for flag in materialised
            if flag.level == level
        )
        if items:
            groups.append(FlagGroup(level=level, title=_FLAG_TITLES[level], items=items))
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class ContributionItem:
    """One local contribution, ready to display."""

    feature: str
    label: str
    value_display: str
    delta: float
    delta_text: str


def _delta_unit(schema: DatasetSchema, target: TargetName) -> str:
    if target in _CLASSIFIER_TARGETS:
        return _CLASSIFIER_DELTA_UNIT
    spec = schema.targets.get(target)
    return unit_label(spec.unit if spec is not None else None)


def contribution_targets(
    prediction: PatientPrediction, schema: DatasetSchema
) -> tuple[tuple[TargetName, str], ...]:
    """Targets that have explanations, with their French titles, in target order."""
    return tuple(
        (target, _target_title(schema, target))
        for target in TargetName
        if prediction.explanations.get(target)
    )


def contributions(
    prediction: PatientPrediction,
    schema: DatasetSchema,
    target: TargetName,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[ContributionItem, ...]:
    """The ``top_k`` largest contributions (absolute value) for ``target``.

    Classifier deltas are percentage points of P(predicted class); regressor deltas are in
    the target unit.
    """
    unit = _delta_unit(schema, target)
    ranked = sorted(
        prediction.explanations.get(target, ()), key=lambda c: abs(c.delta), reverse=True
    )
    items: list[ContributionItem] = []
    for c in ranked[:top_k]:
        text = f"{'+' if c.delta > 0 else ''}{format_number(c.delta, 1)}"
        items.append(
            ContributionItem(
                feature=c.feature,
                label=_feature_label(schema, c.feature),
                value_display=c.value_display,
                delta=float(c.delta),
                delta_text=_with_unit(text, unit),
            )
        )
    return tuple(items)


def contribution_hint(schema: DatasetSchema, target: TargetName) -> str:
    """How to read the contributions of ``target`` (unit and reference patient)."""
    reference = (
        "par rapport à un patient de référence (valeurs médianes ou les plus fréquentes "
        "de l'apprentissage). Explication locale du modèle, pas un lien de causalité."
    )
    if target in _CLASSIFIER_TARGETS:
        return f"Effet de chaque variable, en points de pourcentage de probabilité, {reference}"
    unit = _delta_unit(schema, target) or "unités de la cible"
    return f"Effet de chaque variable, en {unit}, {reference}"


INCREASE_LABEL: Final = "Fait monter"
DECREASE_LABEL: Final = "Fait baisser"


def contribution_chart_rows(items: Iterable[ContributionItem]) -> list[dict[str, str | float]]:
    """Rows of the contributions bar chart: ``Variable`` (label = value), ``Effet`` and its
    direction ``Sens`` (:data:`INCREASE_LABEL` or :data:`DECREASE_LABEL`)."""
    return [
        {
            "Variable": f"{item.label} = {item.value_display}",
            "Effet": item.delta,
            "Sens": INCREASE_LABEL if item.delta > 0 else DECREASE_LABEL,
        }
        for item in items
    ]


def contribution_axis_title(schema: DatasetSchema, target: TargetName) -> str:
    """Axis title of the contributions chart, e.g. ``"Effet (pts de probabilité)"``."""
    unit = _delta_unit(schema, target)
    if unit == _CLASSIFIER_DELTA_UNIT:
        return "Effet (pts de probabilité)"
    return f"Effet ({unit})" if unit else "Effet"


# --------------------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricItem:
    """One global performance metric with its 95 % CI, formatted."""

    key: str
    target: str
    metric: str
    text: str
    n: int | None


def _metric_label(schema: DatasetSchema, target: TargetName | None, metric: str) -> str:
    if metric in _METRIC_LABELS:
        return _METRIC_LABELS[metric]
    if metric.startswith("recall_") and target is not None:
        code = metric.removeprefix("recall_")
        return f"Rappel « {class_label(schema, target, code)} »"
    return metric


def _parse_metric_key(key: str) -> tuple[TargetName | None, str, str]:
    target_name, _, metric = key.partition(".")
    try:
        target = TargetName(target_name)
    except ValueError:
        return None, target_name, metric or key
    return target, target_name, metric


def metric_items(
    info: ModelInfo, schema: DatasetSchema, target: TargetName | None = None
) -> tuple[MetricItem, ...]:
    """``info.metrics_summary`` formatted (``"0,71 [0,64-0,78]"``), optionally one target."""
    items: list[MetricItem] = []
    for key, estimate in info.metrics_summary.items():
        parsed_target, target_name, metric = _parse_metric_key(key)
        if target is not None and parsed_target is not target:
            continue
        items.append(
            MetricItem(
                key=key,
                target=target_name
                if parsed_target is None
                else _target_title(schema, parsed_target),
                metric=_metric_label(schema, parsed_target, metric),
                text=format_metric(estimate),
                n=estimate.n,
            )
        )
    return tuple(items)


def metrics_caption(items: Sequence[MetricItem], protocol: str | None) -> str | None:
    """One-line reminder, e.g. ``"Performance globale : AUC ROC 0,80 [0,74-0,86] (n = 212)"``."""
    if not items:
        return None
    parts = [
        f"{item.metric} {item.text}" + (f" (n = {item.n})" if item.n is not None else "")
        for item in items
    ]
    suffix = f" — {protocol}" if protocol else ""
    return f"Performance globale (IC à 95 %) : {' ; '.join(parts)}{suffix}"


def metrics_rows(info: ModelInfo, schema: DatasetSchema) -> list[dict[str, str]]:
    """Headline metric table of the model page (``info.metrics_summary``)."""
    return [
        {
            "Cible": item.target,
            "Métrique": item.metric,
            "Valeur [IC 95 %]": item.text,
            "n": "" if item.n is None else str(item.n),
        }
        for item in metric_items(info, schema)
    ]


_METRIC_HINTS: Final[Mapping[str, str]] = {
    "balanced_accuracy": "Bonnes réponses, chaque classe comptant autant (hasard : 1 / nombre "
    "de classes).",
    "accuracy": "Part de bonnes réponses.",
    "roc_auc": "Capacité à distinguer les deux situations (0,5 = hasard, 1 = parfait).",
    "roc_auc_ovr": "Capacité à distinguer chaque classe des autres (0,5 = hasard).",
    "mae": "Écart moyen entre la valeur prédite et la valeur réelle, dans l'unité de la sortie.",
    "median_ae": "Écart médian entre la valeur prédite et la valeur réelle.",
    "coverage": "Part des patients dont la vraie réponse était dans les réponses compatibles "
    "ou la fourchette annoncées (visée : 90 %).",
    "brier": "Erreur des probabilités (0 = parfait ; plus bas = mieux).",
    "log_loss": "Erreur des probabilités (plus bas = mieux).",
    "sensitivity": "Part des patients concernés que le modèle repère.",
    "specificity": "Part des patients non concernés que le modèle écarte.",
}


METRICS_GUIDE_MD: Final = """
- **Modèle [IC 95 %]** : valeur mesurée par validation croisée, sur des patients que le
  modèle n'avait pas vus, avec son intervalle de confiance à 95 %.
- **Référence naïve** : ce que donnerait une règle sans intelligence (toujours la réponse
  la plus fréquente) ; le modèle doit faire mieux.
- **Exactitude équilibrée** : bonnes réponses, chaque classe comptant autant (hasard :
  1 / nombre de classes).
- **AUC ROC** : capacité à distinguer deux situations (0,5 = hasard, 1 = parfait).
- **Erreur absolue moyenne** : écart moyen entre la valeur prédite et la valeur réelle.
- **Couverture empirique** : part des patients dont la vraie réponse était dans les
  réponses compatibles ou la fourchette annoncées (visée : 90 %).
"""
"""How to read the metrics table of the model page, in plain French."""


def performance_rows(info: ModelInfo, schema: DatasetSchema) -> list[dict[str, str]]:
    """Reminder of the measured performance on the patient page: one row per output and
    metric, with a plain-French reading of the metric."""
    rows: list[dict[str, str]] = []
    for item in metric_items(info, schema):
        _target, _name, metric = _parse_metric_key(item.key)
        rows.append(
            {
                "Sortie": item.target,
                "Mesure": item.metric,
                "Valeur [IC 95 %]": item.text,
                "Comment la lire": _METRIC_HINTS.get(metric, ""),
            }
        )
    return rows


def _baseline_label(name: str) -> str:
    return f"Référence : {_BASELINE_LABELS.get(name, name)}"


def report_rows(report: EvaluationReport, schema: DatasetSchema) -> list[dict[str, str]]:
    """Full metric table of an evaluation report: the model against every baseline.

    One row per target and metric (then the conformal metrics), one column per baseline
    (``Référence : ...``, empty when the baseline does not report that metric).
    """
    baselines: list[str] = []
    for evaluation in report.targets.values():
        baselines += [name for name in evaluation.baselines if name not in baselines]
    rows: list[dict[str, str]] = []
    for target, evaluation in report.targets.items():
        title = _target_title(schema, target)
        suffix = " (patients traités)" if evaluation.conditional else ""
        for metric, estimate in evaluation.metrics.items():
            row = {
                "Cible": f"{title}{suffix}",
                "Métrique": _metric_label(schema, target, metric),
                "Modèle [IC 95 %]": format_metric(estimate),
            }
            for name in baselines:
                reference = evaluation.baselines.get(name, {}).get(metric)
                row[_baseline_label(name)] = "" if reference is None else format_metric(reference)
            row["n"] = "" if estimate.n is None else str(estimate.n)
            rows.append(row)
        for metric, estimate in evaluation.conformal.items():
            row = {
                "Cible": f"{title}{suffix}",
                "Métrique": _metric_label(schema, target, metric),
                "Modèle [IC 95 %]": format_metric(estimate),
            }
            row.update({_baseline_label(name): "" for name in baselines})
            row["n"] = "" if estimate.n is None else str(estimate.n)
            rows.append(row)
    return rows


def model_info_rows(info: ModelInfo) -> list[dict[str, str]]:
    """Key facts about the loaded model (no path beyond the bundle directory name)."""
    rows = [
        ("Identifiant", info.model_id),
        ("Date de création", info.created_at.strftime("%d/%m/%Y %H:%M %Z").strip()),
        ("Données d'entraînement", "Synthétiques" if info.is_synthetic else "Réelles"),
        ("Jeu de variables", f"{info.feature_set} ({len(info.feature_names)} variables)"),
        ("Patients à l'entraînement", str(info.n_train)),
        ("Cibles", ", ".join(t.value for t in info.targets)),
        ("Niveau des ensembles et intervalles", format_percent(1.0 - info.alpha)),
        ("Seuil de décision (traitement)", format_percent(info.treatment_threshold)),
        ("Zone grise autour du seuil", f"± {format_percent(info.grey_zone)}"),
        ("Protocole d'évaluation", info.evaluation_protocol or "non renseigné"),
        ("Version du logiciel", info.package_version),
        ("Empreinte du schéma", info.schema_hash[:12]),
        ("Dossier du modèle", info.bundle_path.name if info.bundle_path else "en mémoire"),
    ]
    return [{"Élément": key, "Valeur": value} for key, value in rows]


def equivalence_rows(schema: DatasetSchema, info: ModelInfo | None = None) -> list[dict[str, str]]:
    """Dose equivalence table: mg of diazepam for 1 mg of each molecule.

    The model's table (``info``) when given and not empty, else the schema's.
    """
    table = (info.dose_equivalence_to_diazepam if info is not None else {}) or (
        schema.dose_equivalence_to_diazepam
    )
    return [
        {
            "Molécule": class_label(schema, TargetName.MOLECULE, molecule),
            "mg de diazépam pour 1 mg": format_plain(factor),
        }
        for molecule, factor in table.items()
    ]


def fallback_model_card(info: ModelInfo, schema: DatasetSchema) -> str:
    """Short French Markdown card for a model without ``model_card.md`` (in memory, fake)."""
    lines = [f"### Modèle « {info.model_id} »", ""]
    if info.is_synthetic:
        lines += [f"**{SYNTHETIC_BANNER}.** {SYNTHETIC_EXPLANATION}", ""]
    lines += [
        "**Usage prévu :** aide à la décision de recherche pour le sevrage alcoolique, "
        "utilisée par des cliniciens. Pas un dispositif médical, jamais d'usage autonome.",
        "",
        f"- Patients à l'entraînement : {info.n_train}",
        f"- Jeu de variables : `{info.feature_set}` ({len(info.feature_names)} variables)",
        f"- Niveau des ensembles et intervalles : {format_percent(1.0 - info.alpha)}",
        "- Cibles : " + ", ".join(_target_title(schema, t) for t in info.targets),
        f"- Évaluation : {info.evaluation_protocol or 'non évalué (pas de validation croisée)'}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Whole prediction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionView:
    """Everything the patient page renders for one prediction."""

    summary: FinalSummary
    severity: ClassCard
    treatment: ClassCard
    molecule: ClassCard
    dose: RegressionCard | None
    duration: RegressionCard | None
    flags: tuple[FlagGroup, ...]
    contribution_targets: tuple[tuple[TargetName, str], ...]
    metrics: Mapping[TargetName, str | None]


def prediction_view(
    prediction: PatientPrediction, schema: DatasetSchema, info: ModelInfo
) -> PredictionView:
    """Build every display structure of one prediction."""
    metrics = {
        target: metrics_caption(metric_items(info, schema, target), None) for target in TargetName
    }
    return PredictionView(
        summary=final_summary(prediction, schema),
        severity=_with_performance(severity_card(prediction, schema), metrics),
        treatment=_with_performance(treatment_card(prediction, schema, info), metrics),
        molecule=_with_performance(molecule_card(prediction, schema), metrics),
        dose=_with_performance(dose_card(prediction, schema, info), metrics),
        duration=_with_performance(duration_card(prediction, schema), metrics),
        flags=flag_groups(prediction.flags, schema),
        contribution_targets=contribution_targets(prediction, schema),
        metrics=metrics,
    )


_PERFORMANCE_NOTE: Final = (
    "Ces critères concernent ce patient. Performance du modèle mesurée sur des patients "
    "qu'il n'avait pas vus : {caption}"
)


@overload
def _with_performance(card: ClassCard, metrics: Mapping[TargetName, str | None]) -> ClassCard: ...
@overload
def _with_performance(
    card: RegressionCard | None, metrics: Mapping[TargetName, str | None]
) -> RegressionCard | None: ...
def _with_performance(
    card: ClassCard | RegressionCard | None, metrics: Mapping[TargetName, str | None]
) -> ClassCard | RegressionCard | None:
    """Card whose explanation ends with the measured performance of its target."""
    if card is None:
        return None
    caption = metrics.get(card.target)
    if not caption:
        return card
    note = _PERFORMANCE_NOTE.format(caption=caption.removeprefix("Performance globale "))
    return replace(card, explanation=with_context(card.explanation, [note]))


def prediction_json(prediction: PatientPrediction) -> str:
    """Indented UTF-8 JSON of one prediction (``export.prediction_to_dict``), for download."""
    return json.dumps(export.prediction_to_dict(prediction), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Choice:
    """One option of a select box: display label and canonical value."""

    label: str
    value: str | float | None


@dataclass(frozen=True, slots=True)
class FormField:
    """Specification of one form widget, derived from a :class:`FeatureSpec`."""

    name: str
    label: str
    kind: FeatureKind
    unit: str | None
    help: str
    min_value: float | None
    max_value: float | None
    step: float
    choices: tuple[Choice, ...]
    required: bool

    @property
    def options(self) -> tuple[str, ...]:
        """Select box labels (``"Inconnu"`` first); empty for numeric fields."""
        return tuple(choice.label for choice in self.choices)

    @property
    def blank(self) -> str | None:
        """Widget value meaning "unknown": ``None`` (numeric) or the ``"Inconnu"`` label."""
        return self.choices[0].label if self.choices else None

    def to_canonical(self, raw: object) -> str | float | None:
        """Canonical value of a widget value (``None`` = unknown)."""
        if self.kind is FeatureKind.NUMERIC:
            if raw is None or isinstance(raw, bool) or not isinstance(raw, numbers.Real):
                return None
            number = float(raw)
            return None if math.isnan(number) else number
        for choice in self.choices:
            if choice.label == raw:
                return choice.value
        return None

    def to_widget(self, value: object) -> str | float | None:
        """Widget value showing a canonical ``value`` (inverse of :meth:`to_canonical`).

        Missing, unknown or out-of-range values give :attr:`blank`.
        """
        if self.kind is FeatureKind.NUMERIC:
            if _is_missing(value) or isinstance(value, bool):
                return None
            if not isinstance(value, numbers.Real):
                return None
            number = float(value)
            low = -math.inf if self.min_value is None else self.min_value
            high = math.inf if self.max_value is None else self.max_value
            return number if low <= number <= high else None
        for choice in self.choices:
            if choice.value is not None and choice.value == value:
                return choice.label
        return self.blank


@dataclass(frozen=True, slots=True)
class FormGroup:
    """Fields of one ``feature_groups`` entry, rendered in one expander."""

    key: str
    label: str
    fields: tuple[FormField, ...]


def _field_help(feature: FeatureSpec) -> str:
    parts = [feature.description.strip()] if feature.description.strip() else []
    if feature.unit:
        parts.append(f"Unité : {feature.unit}.")
    if feature.valid_range is not None:
        low, high = feature.valid_range
        parts.append(f"Plage valide : {format_plain(low)} {_DASH} {format_plain(high)}.")
    parts.append("Laisser vide (ou « Inconnu ») si la valeur n'est pas connue.")
    return " ".join(parts)


def _form_field(feature: FeatureSpec) -> FormField:
    label = f"{feature.label} ({feature.unit})" if feature.unit else feature.label
    low, high = feature.valid_range or (None, None)
    step = 0.1 if low is not None and high is not None and high - low <= 10 else 1.0
    choices: tuple[Choice, ...] = ()
    if feature.kind is FeatureKind.BINARY:
        choices = (Choice(UNKNOWN_OPTION, None), Choice(YES_OPTION, 1.0), Choice(NO_OPTION, 0.0))
    elif feature.kind is FeatureKind.CATEGORICAL:
        labels = _category_labels(feature)
        choices = (
            Choice(UNKNOWN_OPTION, None),
            *(Choice(labels[code], code) for code in feature.categories or ()),
        )
    return FormField(
        name=feature.name,
        label=label,
        kind=feature.kind,
        unit=feature.unit,
        help=_field_help(feature),
        min_value=None if low is None else float(low),
        max_value=None if high is None else float(high),
        step=step,
        choices=choices,
        required=feature.required,
    )


def form_groups(schema: DatasetSchema, feature_names: Sequence[str]) -> tuple[FormGroup, ...]:
    """Form fields of ``feature_names`` grouped by ``group``, in ``feature_groups`` order.

    Names unknown to the schema are skipped. Groups missing from ``feature_groups`` come
    last, labelled with their key.
    """
    by_group: dict[str, list[FormField]] = {}
    for name in feature_names:
        feature = _feature_or_none(schema, name)
        if feature is not None:
            by_group.setdefault(feature.group, []).append(_form_field(feature))
    order = [g for g in schema.feature_groups if g in by_group]
    order += [g for g in by_group if g not in schema.feature_groups]
    return tuple(
        FormGroup(key=g, label=schema.feature_groups.get(g, g), fields=tuple(by_group[g]))
        for g in order
    )


def record_from_form(
    groups: Iterable[FormGroup], values: Mapping[str, object]
) -> dict[str, str | float | None]:
    """Canonical record from widget values keyed by feature name (``None`` = unknown)."""
    return {
        field.name: field.to_canonical(values.get(field.name))
        for group in groups
        for field in group.fields
    }


EXAMPLE_PATIENT: Final[Mapping[str, str | float]] = {
    "age": 54.0,
    "sex": "male",
    "bmi": 23.5,
    "daily_alcohol_g": 160.0,
    "abstinence_days": 0.0,
    "breath_alcohol": 1.2,
    "prior_withdrawals": 2.0,
    "dsm5_withdrawal_symptoms": 1.0,
    "dsm5_criteria": 7.0,
    "fagerstrom": 4.0,
    "prior_seizure": 0.0,
    "prior_dt": 0.0,
    "cirrhosis": 0.0,
    "urine_bzd": 0.0,
    "prior_bzd_treatment": 0.0,
    "ggt": 310.0,
    "asat": 85.0,
    "platelets": 180.0,
    "sodium": 137.0,
    "potassium": 3.9,
}
"""A **fictional** patient (canonical values), to discover the tool without real data."""


def form_values(
    groups: Iterable[FormGroup], record: Mapping[str, object] | None = None
) -> dict[str, str | float | None]:
    """Widget values keyed by feature name showing ``record`` (``None``: every field blank).

    Fields absent from ``record`` (or out of range) stay blank.
    """
    source = record or {}
    return {
        field.name: field.to_widget(source.get(field.name))
        for group in groups
        for field in group.fields
    }


def filled_count(groups: Iterable[FormGroup], values: Mapping[str, object]) -> tuple[int, int]:
    """``(filled, total)``: form fields with a known value, and all fields."""
    fields = [field for group in groups for field in group.fields]
    filled = sum(1 for field in fields if field.to_canonical(values.get(field.name)) is not None)
    return filled, len(fields)


def _describe_error(schema: DatasetSchema, error: Mapping[str, object]) -> str:
    loc = error.get("loc")
    name = str(loc[0]) if isinstance(loc, tuple) and loc else ""
    label = _feature_label(schema, name) if name else "Saisie"
    kind = error.get("type")
    ctx = error.get("ctx")
    context = ctx if isinstance(ctx, Mapping) else {}
    if kind == "greater_than_equal" and isinstance(context.get("ge"), int | float):
        reason = f"la valeur doit être supérieure ou égale à {format_plain(context['ge'])}"
    elif kind == "less_than_equal" and isinstance(context.get("le"), int | float):
        reason = f"la valeur doit être inférieure ou égale à {format_plain(context['le'])}"
    elif kind == "literal_error":
        reason = "valeur non autorisée"
    elif kind == "extra_forbidden":
        reason = "variable inconnue du modèle"
    elif kind in {"float_parsing", "float_type", "finite_number"}:
        reason = "un nombre fini est attendu"
    else:
        reason = "valeur invalide"
    return f"« {label} » : {reason}."


def validation_error_messages(schema: DatasetSchema, exc: ValidationError) -> tuple[str, ...]:
    """French messages of a record validation error (labels and bounds, never values)."""
    return tuple(
        _describe_error(schema, dict(error))
        for error in exc.errors(include_input=False, include_url=False)
    )


def validation_messages(
    schema: DatasetSchema, record: Mapping[str, object], feature_set: str | None = None
) -> tuple[str, ...]:
    """Validate ``record`` with ``build_record_model``; empty tuple when it is valid."""
    selected = feature_set if feature_set in schema.feature_sets else None
    model = build_record_model(schema, "predict", feature_set=selected)
    try:
        model.model_validate(dict(record))
    except ValidationError as exc:
        return validation_error_messages(schema, exc)
    return ()


# --------------------------------------------------------------------------------------
# Tables (batch, dictionary)
# --------------------------------------------------------------------------------------


def anonymization_rows(report: AnonymizationReport) -> list[dict[str, str]]:
    """Findings of the anonymization check (column names and counts, never values)."""
    return [
        {
            "Colonne": finding.column,
            "Constat": finding.message_fr,
            "Bloquant": YES_OPTION if finding.blocking else NO_OPTION,
        }
        for finding in report.findings
    ]


def _optional_number(value: float | None, unit: str | None = None) -> str:
    return "" if value is None else _with_unit(format_number(value), unit)


def batch_rows(
    predictions: Iterable[PatientPrediction], schema: DatasetSchema
) -> list[dict[str, str]]:
    """Predictions table of the batch page: one row per predicted patient."""
    positive = _positive_class(schema)
    rows: list[dict[str, str]] = []
    for prediction in predictions:
        final = prediction.final
        alerts = sum(1 for flag in prediction.flags if flag.level in _ALERT_LEVELS)
        duration = None if final.duration_days is None else float(final.duration_days)
        rows.append(
            {
                "Ligne": prediction.row_id,
                "Sévérité": class_label(schema, TargetName.SEVERITY, prediction.severity.label),
                "P(traitement)": format_percent(
                    float(prediction.treatment.probabilities.get(positive, 0.0))
                ),
                "Recommandation": "Traitement" if final.treat else "Pas de traitement",
                "Molécule": class_label(schema, TargetName.MOLECULE, final.molecule),
                "Dose totale": _optional_number(final.dose_cumulative_dze, DOSE_UNIT),
                "Durée": "" if duration is None else _days(duration),
                "Confiance": prediction.overall_confidence.label_fr,
                "Alertes": str(alerts),
            }
        )
    return rows


def batch_summary(result: BatchResult) -> str:
    """``"12 ligne(s) prédite(s), 2 ligne(s) rejetée(s)."``"""
    rejected = len({issue.row_id for issue in result.rejected})
    return f"{len(result.predictions)} ligne(s) prédite(s), {rejected} ligne(s) rejetée(s)."


def batch_csv(result: BatchResult) -> bytes:
    """Escaped CSV of a batch (``export.to_frame`` + ``export.to_csv_bytes``), for download."""
    return export.to_csv_bytes(export.to_frame(result))


def batch_overview(result: BatchResult) -> tuple[SummaryFigure, ...]:
    """Key figures of a batch: predicted patients, rejected rows, low confidence, alerts."""
    rejected = len({issue.row_id for issue in result.rejected})
    low = sum(1 for p in result.predictions if p.overall_confidence is ConfidenceLevel.LOW)
    alerts = sum(
        1 for p in result.predictions if any(flag.level in _ALERT_LEVELS for flag in p.flags)
    )
    return (
        SummaryFigure("Patients prédits", str(len(result.predictions))),
        SummaryFigure(
            "Lignes rejetées", str(rejected), "Lignes non prédites : voir leurs motifs plus bas."
        ),
        SummaryFigure(
            "Confiance faible",
            str(low),
            "Prédictions peu fiables : se référer au protocole standard.",
        ),
        SummaryFigure("Avec alertes", str(alerts), "Patients avec au moins une alerte."),
    )


def template_columns(schema: DatasetSchema, feature_names: Sequence[str]) -> list[str]:
    """Source columns a batch file needs for ``feature_names``: the id column (if any), then
    each feature's column (and its NON column for OUI / NON / NON CONNU questions)."""
    columns = [schema.dataset.id_column] if schema.dataset.id_column else []
    for name in feature_names:
        feature = _feature_or_none(schema, name)
        if feature is None:
            continue
        columns.append(feature.source_column)
        if feature.negative_column:
            columns.append(feature.negative_column)
    return columns


def template_csv(schema: DatasetSchema, feature_names: Sequence[str]) -> bytes:
    """Empty CSV (header line only) with :func:`template_columns`, to prepare a batch file."""
    return export.to_csv_bytes(pd.DataFrame(columns=template_columns(schema, feature_names)))


def rejected_rows(issues: Iterable[RowIssue], schema: DatasetSchema) -> list[dict[str, str]]:
    """Rejected rows of a batch: row id, feature label (or raw column), code and reason."""
    return [
        {
            "Ligne": issue.row_id,
            "Variable": "" if issue.column is None else _feature_label(schema, issue.column),
            "Code": issue.code,
            "Motif": issue.message_fr,
        }
        for issue in issues
    ]


def dictionary_table(
    schema: DatasetSchema, model_features: Iterable[str] = ()
) -> list[dict[str, str]]:
    """The data dictionary (``config.dictionary.dictionary_rows``) plus the column
    ``Utilisée par le modèle``."""
    used = set(model_features)
    return [
        {**row, "Utilisée par le modèle": YES_OPTION if row["Variable"] in used else NO_OPTION}
        for row in dictionary_rows(schema)
    ]


def target_rows(schema: DatasetSchema) -> list[dict[str, str]]:
    """The prediction targets of the schema, with their classes or unit."""
    rows: list[dict[str, str]] = []
    for name, spec in schema.targets.items():
        if spec.classes:
            detail = ", ".join(spec.class_labels.get(c, c) for c in spec.classes)
        elif spec.class_labels:
            detail = ", ".join(spec.class_labels.values())
        else:
            detail = unit_label(spec.unit)
        rows.append(
            {
                "Cible": name.value,
                "Libellé": spec.label,
                "Type": spec.kind.value,
                "Classes ou unité": detail,
                "Active": YES_OPTION if spec.enabled else NO_OPTION,
                "Description": " ".join(spec.description.split()),
            }
        )
    return rows
