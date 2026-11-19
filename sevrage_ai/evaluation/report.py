"""Evaluation report: JSON-serialisable (``metrics.json``) and rendered to Markdown. Owner: WS5a."""

from __future__ import annotations

import math
from typing import Final, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from sevrage_ai.types import MetricEstimate, TargetName, TaskKind

__all__ = ["EvaluationReport", "ReliabilityCurve", "TargetEvaluation"]

_HEADLINE_METRIC: Final[dict[TargetName, str]] = {
    TargetName.SEVERITY: "balanced_accuracy",
    TargetName.TREATMENT: "roc_auc",
    TargetName.MOLECULE: "balanced_accuracy",
    TargetName.DOSE_DAY1: "mae",
    TargetName.DOSE_CUMULATIVE: "mae",
    TargetName.DURATION: "mae",
}
"""Main metric shown per target in :meth:`EvaluationReport.headline_metrics`."""

_HEADLINE_FALLBACK: Final[dict[TargetName, str]] = {TargetName.MOLECULE: "accuracy"}
"""Fallback metric used when the main one of :data:`_HEADLINE_METRIC` is absent."""

_METRIC_LABELS_FR: Final[dict[str, str]] = {
    "accuracy": "Exactitude",
    "balanced_accuracy": "Exactitude équilibrée",
    "macro_f1": "F1 macro",
    "log_loss": "Perte logarithmique",
    "brier": "Score de Brier",
    "ece": "ECE (erreur de calibration)",
    "roc_auc": "AUC ROC",
    "pr_auc": "AUC PR",
    "roc_auc_ovr": "AUC ROC (un contre tous)",
    "top2_accuracy": "Exactitude top-2",
    "qwk": "Kappa quadratique pondéré",
    "calibration_slope": "Pente de calibration",
    "calibration_intercept": "Ordonnée de calibration",
    "sensitivity": "Sensibilité",
    "specificity": "Spécificité",
    "ppv": "VPP",
    "npv": "VPN",
    "mae": "Erreur absolue moyenne",
    "median_ae": "Erreur absolue médiane",
    "rmse": "RMSE",
    "r2": "R²",
}
"""French label of a well-known metric key, used by :meth:`EvaluationReport.to_markdown`."""


def _fmt(value: float, decimals: int = 2) -> str:
    """``value`` with ``decimals`` digits and a comma separator, or ``"n/d"`` when ``NaN``."""
    if math.isnan(value):
        return "n/d"
    return f"{value:.{decimals}f}".replace(".", ",")


def _fmt_ci(estimate: MetricEstimate, decimals: int = 2) -> str:
    """``"value [lo ; hi]"``, or ``"n/d"`` when the point estimate is undefined."""
    if math.isnan(estimate.value):
        return "n/d"
    return (
        f"{_fmt(estimate.value, decimals)} "
        f"[{_fmt(estimate.lo, decimals)} ; {_fmt(estimate.hi, decimals)}]"
    )


def _metric_label(key: str) -> str:
    if key.startswith("recall_"):
        return f"Rappel ({key.removeprefix('recall_')})"
    if key.startswith("severe_vs_rest_"):
        suffix = key.removeprefix("severe_vs_rest_")
        return f"{_METRIC_LABELS_FR.get(suffix, suffix)} (sévère vs reste)"
    return _METRIC_LABELS_FR.get(key, key)


class _Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


class ReliabilityCurve(_Report):
    """Reliability-diagram points for one class (one-vs-rest), from outer-fold predictions."""

    label: str
    mean_predicted: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    counts: tuple[int, ...]


class TargetEvaluation(_Report):
    """Evaluation of one target over the outer CV.

    ``metrics`` and ``conformal`` use the keys of :mod:`sevrage_ai.evaluation.metrics`.
    ``baselines`` maps a baseline name (``"dummy"`` or a ``schema.baselines`` name) to its
    metrics. ``selected_candidates`` lists the candidate chosen in each outer fold.
    ``global_importance`` is the mean permutation importance per canonical feature.
    ``conditional`` is ``True`` for targets evaluated on treated patients only.
    """

    target: TargetName
    kind: TaskKind
    n: int = Field(ge=0)
    conditional: bool = False
    classes: tuple[str, ...] | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, MetricEstimate] = Field(default_factory=dict)
    baselines: dict[str, dict[str, MetricEstimate]] = Field(default_factory=dict)
    conformal: dict[str, MetricEstimate] = Field(default_factory=dict)
    selected_candidates: tuple[str, ...] = ()
    confusion_matrix: tuple[tuple[int, ...], ...] | None = None
    reliability: tuple[ReliabilityCurve, ...] = ()
    global_importance: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class EvaluationReport(_Report):
    """Whole evaluation (repeated nested CV, or an external hold-out).

    ``protocol_fr`` describes the protocol in French, e.g. ``"CV imbriquée répétée 5x5"``.
    ``end_to_end`` holds whole-system metrics, e.g. ``molecule.accuracy_with_none`` or
    ``dose_day1.mae_end_to_end`` (0 for untreated patients).
    """

    report_version: Literal["1"] = "1"
    created_at: AwareDatetime
    is_synthetic: bool
    n_rows: int = Field(ge=0)
    feature_set: str
    protocol_fr: str
    seed: int
    outer_splits: int = Field(ge=0)
    outer_repeats: int = Field(ge=0)
    inner_splits: int = Field(ge=0)
    crossfit_splits: int = Field(ge=0)
    alpha: float = Field(gt=0.0, lt=1.0)
    targets: dict[TargetName, TargetEvaluation] = Field(default_factory=dict)
    end_to_end: dict[str, MetricEstimate] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_json(self) -> str:
        """Indented JSON (``NaN`` kept as ``NaN``)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Inverse of :meth:`to_json`."""
        return cls.model_validate_json(text)

    def headline_metrics(self) -> dict[str, MetricEstimate]:
        """Main metric per target, keyed ``"<target>.<metric>"`` (for ``ModelInfo``).

        The main metric is ``balanced_accuracy`` for ``severity`` and ``molecule`` (falling
        back to ``accuracy`` when absent), ``roc_auc`` for ``treatment`` and ``mae`` for the
        regression targets (see :data:`_HEADLINE_METRIC`). A key is only included when the
        metric is present in ``targets[target].metrics``. ``"<target>.coverage"`` is also
        included when ``targets[target].conformal`` has a ``"coverage"`` key.
        """
        result: dict[str, MetricEstimate] = {}
        for name, target in self.targets.items():
            key = _HEADLINE_METRIC.get(name)
            if key is not None:
                estimate = target.metrics.get(key)
                if estimate is None:
                    fallback = _HEADLINE_FALLBACK.get(name)
                    if fallback is not None:
                        estimate = target.metrics.get(fallback)
                        key = fallback
                if estimate is not None:
                    result[f"{name.value}.{key}"] = estimate
            coverage = target.conformal.get("coverage")
            if coverage is not None:
                result[f"{name.value}.coverage"] = coverage
        return result

    def to_markdown(self) -> str:
        """French Markdown summary: tables with CIs, baselines, coverage and warnings.

        Self-contained (no access to the dataset schema): targets and classes are shown by
        their canonical (English) names, not their French display labels -- see
        ``sevrage_ai.persistence.model_card`` for the schema-aware model card. Every number
        is rounded to 2-3 decimals; an undefined (``NaN``) value is shown as ``"n/d"``. Holds
        aggregates only, never a per-patient value.
        """
        lines: list[str] = [
            "# Rapport d'évaluation",
            "",
            f"- Protocole : {self.protocol_fr}",
            f"- Échantillon : n = {self.n_rows} ({'synthétique' if self.is_synthetic else 'réel'})",
            f"- Jeu de variables : `{self.feature_set}`",
            f"- Créé le : {self.created_at.isoformat()}",
            f"- Graine aléatoire : {self.seed}",
            "",
        ]
        if self.is_synthetic:
            lines.append(
                "> **Évaluation sur données synthétiques** : ces chiffres n'ont aucune "
                "valeur clinique."
            )
            lines.append("")
        for name, target in self.targets.items():
            lines.extend(self._target_markdown(name, target))
        if self.end_to_end:
            lines.append("## Métriques de bout en bout")
            lines.append("")
            for key, estimate in self.end_to_end.items():
                lines.append(f"- `{key}` : {_fmt_ci(estimate)}")
            lines.append("")
        if self.warnings:
            lines.append("## Avertissements")
            lines.append("")
            lines.extend(f"- {warning}" for warning in self.warnings)
            lines.append("")
        text = "\n".join(lines)
        return text if text.endswith("\n") else f"{text}\n"

    def _target_markdown(self, name: TargetName, target: TargetEvaluation) -> list[str]:
        lines = [f"## {name.value} (n = {target.n})", ""]
        if target.conditional:
            lines.append("Cible conditionnelle : évaluée uniquement sur les patients traités.")
            lines.append("")
        if target.class_counts:
            counts = ", ".join(f"{cls} : {n}" for cls, n in target.class_counts.items())
            lines.append(f"Répartition des classes : {counts}.")
            lines.append("")
        if target.metrics:
            lines.extend(self._metrics_table(target))
            lines.append("")
        if target.conformal:
            lines.append("**Couverture conforme :**")
            for key, estimate in target.conformal.items():
                lines.append(f"- `{key}` : {_fmt_ci(estimate)}")
            lines.append("")
        classes, confusion = target.classes, target.confusion_matrix
        if classes is not None and confusion is not None:
            lines.extend(self._confusion_markdown(classes, confusion))
            lines.append("")
        if target.selected_candidates:
            counts_by_candidate: dict[str, int] = {}
            for candidate in target.selected_candidates:
                counts_by_candidate[candidate] = counts_by_candidate.get(candidate, 0) + 1
            frequency = ", ".join(
                f"`{candidate}` ({n} pli{'s' if n > 1 else ''})"
                for candidate, n in counts_by_candidate.items()
            )
            lines.append(f"Candidats sélectionnés : {frequency}.")
            lines.append("")
        if target.global_importance:
            top = sorted(target.global_importance.items(), key=lambda item: -item[1])[:10]
            lines.append("**Importance globale (top 10) :**")
            lines.extend(f"- `{feature}` : {_fmt(value, 3)}" for feature, value in top)
            lines.append("")
        if target.warnings:
            lines.extend(f"- {warning}" for warning in target.warnings)
            lines.append("")
        return lines

    def _metrics_table(self, target: TargetEvaluation) -> list[str]:
        dummy = target.baselines.get("dummy", {})
        rules = {name: values for name, values in target.baselines.items() if name != "dummy"}
        lines = [
            "| Métrique | Valeur [IC 95 %] | Référence naïve | Règle clinique |",
            "|---|---|---|---|",
        ]
        for key, estimate in target.metrics.items():
            naive = _fmt_ci(dummy[key]) if key in dummy else "n/d"
            rule_cells = [
                f"{_fmt_ci(values[key])} ({rule_name})"
                for rule_name, values in rules.items()
                if key in values
            ]
            rule = "; ".join(rule_cells) if rule_cells else "n/d"
            lines.append(f"| {_metric_label(key)} | {_fmt_ci(estimate)} | {naive} | {rule} |")
        return lines

    def _confusion_markdown(
        self, classes: tuple[str, ...], confusion: tuple[tuple[int, ...], ...]
    ) -> list[str]:
        header = "| Réel \\ Prédit | " + " | ".join(classes) + " |"
        separator = "|---|" + "---|" * len(classes)
        rows = [
            "| " + classes[i] + " | " + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(confusion)
        ]
        return ["**Matrice de confusion :**", header, separator, *rows]
