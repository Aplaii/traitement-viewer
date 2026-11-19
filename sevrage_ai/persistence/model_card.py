"""Model card rendering (TRIPOD+AI-aligned sections, French Markdown). Owner: WS7.

Sections: SYNTHETIC banner (when relevant); purpose and intended users; out of scope (not
a medical device under EU MDR 2017/745, no autonomous use); data (centre, period, n,
inclusion criteria); target definitions; features; model per target; performance with
CIs against the baselines; calibration and conformal coverage; limitations (small n,
single centre, confounding by indication, dose protocol history); fairness by sex and age
band where n allows; version information.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Final

from sevrage_ai.config.conditions import AllOf, AnyOf, Comparison, ElseRule, Membership, WhenRule
from sevrage_ai.config.schema import DatasetSchema, DerivedSource, FeatureSpec, TargetSpec
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.errors import ConfigError
from sevrage_ai.evaluation.report import EvaluationReport, TargetEvaluation
from sevrage_ai.persistence.bundle import BundleMetadata
from sevrage_ai.types import MetricEstimate, TargetName

__all__ = ["render"]

_BANNER: Final = "MODÈLE ENTRAÎNÉ SUR DONNÉES SYNTHÉTIQUES"

_SECTIONS: Final = (
    "## 1. Objectif et utilisateurs visés",
    "## 2. Hors du champ d'utilisation",
    "## 3. Données",
    "## 4. Définition des cibles",
    "## 5. Variables prédictives",
    "## 6. Modèle par cible",
    "## 7. Performances",
    "## 8. Calibration et couverture conforme",
    "## 9. Limites",
    "## 10. Équité",
    "## 11. Informations de version",
)

_NO_EVALUATION: Final = (
    "Évaluation non réalisée : ce modèle a été entraîné sans évaluation (option `--no-eval`)."
)

_OP_FR: Final[dict[str, str]] = {
    "eq": "=",
    "ne": "≠",
    "gt": ">",
    "gte": "≥",
    "lt": "<",
    "lte": "≤",
}

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
    "mae": "Erreur absolue moyenne (MAE)",
    "median_ae": "Erreur absolue médiane",
    "rmse": "RMSE",
    "r2": "R²",
}


# --------------------------------------------------------------------------------------
# Number and metric formatting
# --------------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Fixed 2-decimal, comma-separated formatting, or ``"n/d"`` when ``NaN``."""
    if math.isnan(value):
        return "n/d"
    return f"{value:.2f}".replace(".", ",")


def _fmt_ci(estimate: MetricEstimate) -> str:
    """``"value [lo ; hi]"``, or ``"non défini"`` when the point estimate is undefined."""
    if math.isnan(estimate.value):
        return "non défini"
    return f"{_fmt(estimate.value)} [{_fmt(estimate.lo)} ; {_fmt(estimate.hi)}]"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.0f} %"


def _fmt_trim(value: float) -> str:
    """2-decimal formatting with trailing zeros stripped (``1.0`` -> ``"1"``, comma decimal)."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _metric_label(key: str, spec: TargetSpec | None) -> str:
    if key.startswith("recall_"):
        cls = key.removeprefix("recall_")
        label = cls if spec is None else spec.class_labels.get(cls, cls)
        return f"Rappel « {label} »"
    if key.startswith("severe_vs_rest_"):
        suffix = key.removeprefix("severe_vs_rest_")
        return f"{_METRIC_LABELS_FR.get(suffix, suffix)} (sévère vs reste)"
    return _METRIC_LABELS_FR.get(key, key)


# --------------------------------------------------------------------------------------
# Derivation rules -> French text
# --------------------------------------------------------------------------------------


def _feature_label(schema: DatasetSchema, name: str) -> str:
    try:
        return schema.feature(name).label
    except ConfigError:
        return name


def _scalar_fr(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".replace(".", ",")


def _leaf_fr(schema: DatasetSchema, leaf: Comparison | Membership) -> str:
    label = _feature_label(schema, leaf.feature)
    if isinstance(leaf, Membership):
        values = ", ".join(_scalar_fr(v) for v in leaf.value)
        return f"{label} ∈ {{{values}}}"
    return f"{label} {_OP_FR[leaf.op]} {_scalar_fr(leaf.value)}"


def _condition_fr(schema: DatasetSchema, cond: Comparison | Membership | AnyOf | AllOf) -> str:
    if isinstance(cond, AnyOf):
        return " ou ".join(f"({_condition_fr(schema, child)})" for child in cond.any)
    if isinstance(cond, AllOf):
        return " et ".join(f"({_condition_fr(schema, child)})" for child in cond.all)
    return _leaf_fr(schema, cond)


def _rule_lines(schema: DatasetSchema, spec: TargetSpec) -> list[str]:
    source = spec.source
    if not isinstance(source, DerivedSource):  # pragma: no cover - guarded by the caller
        return []
    lines = ["Règles de dérivation (première correspondance) :"]
    for rule in source.derived:
        if isinstance(rule, ElseRule):
            label = spec.class_labels.get(rule.otherwise, rule.otherwise)
            lines.append(f"- Sinon : **{label}**.")
        else:
            when: WhenRule = rule
            label = spec.class_labels.get(when.then, when.then)
            lines.append(f"- Si {_condition_fr(schema, when.when)} → **{label}**.")
    if source.on_missing == "else":
        lines.append(
            "Donnée manquante empêchant de vérifier une règle : la règle est considérée "
            "comme non vérifiée et la classe retombe alors sur la règle par défaut (Sinon)."
        )
    else:
        lines.append(
            "Donnée manquante empêchant de vérifier une règle plus sévère : la classe est "
            "laissée indéterminée plutôt que d'être classée par une règle moins sévère."
        )
    lines.append("Règles à valider par un clinicien avant tout usage.")
    return lines


# --------------------------------------------------------------------------------------
# Sections 1-2: purpose and scope
# --------------------------------------------------------------------------------------


def _purpose_lines() -> list[str]:
    return [
        "Cet outil est un outil de **recherche** en **addictologie** : à partir des variables "
        "recueillies à l'admission, il estime la sévérité probable du sevrage alcoolique, "
        "l'intérêt d'un traitement par benzodiazépine puis, si un traitement est décidé, la "
        "molécule, la dose totale et la durée. Il est destiné aux cliniciens du service dans "
        "un cadre de recherche, pour éclairer et non remplacer le jugement clinique."
    ]


def _scope_lines() -> list[str]:
    return [
        "Ce logiciel n'est **pas un dispositif médical** au sens du règlement (UE) 2017/745 : "
        "il n'a pas de marquage CE et n'a pas vocation à un usage **autonome**. Il ne doit "
        "jamais se substituer au jugement d'un **clinicien** ni être utilisé sans supervision "
        "médicale ; toute recommandation doit être vérifiée avant application."
    ]


# --------------------------------------------------------------------------------------
# Section 3: data
# --------------------------------------------------------------------------------------


def _data_lines(schema: DatasetSchema, metadata: BundleMetadata) -> list[str]:
    lines = [
        " ".join(schema.dataset.description.split()),
        f"Effectif d'entraînement de ce modèle : n = {metadata.n_train}.",
    ]
    if metadata.class_counts:
        lines.append("Répartition des classes dans l'ensemble d'apprentissage :")
        for name, counts in metadata.class_counts.items():
            spec = schema.targets.get(name)
            label = name.value if spec is None else spec.label
            class_labels = {} if spec is None else spec.class_labels
            parts = ", ".join(f"{class_labels.get(c, c)} : {n}" for c, n in counts.items())
            lines.append(f"- **{label}** — {parts}.")
    return lines


# --------------------------------------------------------------------------------------
# Section 4: target definitions
# --------------------------------------------------------------------------------------


def _target_definitions_lines(schema: DatasetSchema) -> list[str]:
    lines: list[str] = []
    for spec in schema.enabled_targets().values():
        lines.append(f"### {spec.label}")
        if spec.description:
            lines.append(" ".join(spec.description.split()))
        if isinstance(spec.source, DerivedSource):
            lines.extend(_rule_lines(schema, spec))
        elif spec.classes is not None:
            labels = ", ".join(spec.class_labels.get(c, c) for c in spec.classes)
            lines.append(f"Classes : {labels}.")
        elif spec.unit:
            lines.append(f"Unité : {spec.unit}.")
        if spec.conditional_on is not None:
            lines.append("Cible évaluée uniquement sur les patients traités.")
        lines.append("")
    molecule_spec = schema.targets.get(TargetName.MOLECULE)
    molecule_labels = {} if molecule_spec is None else molecule_spec.class_labels
    if schema.dose_equivalence_to_diazepam:
        lines.append("**Table d'équivalence des doses (mg équivalent diazépam pour 1 mg) :**")
        lines.append("| Molécule | mg éq. diazépam |")
        lines.append("|---|---|")
        for molecule, factor in schema.dose_equivalence_to_diazepam.items():
            label = molecule_labels.get(molecule, molecule)
            lines.append(f"| {label} | {_fmt_trim(factor)} |")
        lines.append(
            "Table d'équivalence à valider par un clinicien (les tables publiées diffèrent)."
        )
    return lines


# --------------------------------------------------------------------------------------
# Section 5: features
# --------------------------------------------------------------------------------------


def _features_lines(schema: DatasetSchema, metadata: BundleMetadata) -> list[str]:
    try:
        features: tuple[FeatureSpec, ...] = tuple(schema.predictors(metadata.feature_set))
    except ConfigError:
        return [
            f"Jeu de variables : `{metadata.feature_set}` — absent du schéma (configuration "
            "incohérente : voir metrics.json et la configuration d'entraînement)."
        ]
    lines = [f"Jeu de variables : `{metadata.feature_set}` ({len(features)} variables)."]
    by_group: dict[str, list[FeatureSpec]] = {}
    for feature in features:
        by_group.setdefault(feature.group, []).append(feature)
    for group, members in by_group.items():
        group_label = schema.feature_groups.get(group, group)
        labels = ", ".join(feature.label for feature in members)
        lines.append(f"- **{group_label}** : {labels}.")
    return lines


# --------------------------------------------------------------------------------------
# Section 6: model per target
# --------------------------------------------------------------------------------------


def _candidates_line(spec: TargetSpec, name: TargetName, report: EvaluationReport | None) -> str:
    target_eval = None if report is None else report.targets.get(name)
    candidates = () if target_eval is None else target_eval.selected_candidates
    if not candidates:
        return f"- **{spec.label}** : voir metrics.json."
    counts = Counter(candidates)
    parts = [
        f"`{candidate}` ({n} pli)" if n == 1 else f"`{candidate}` ({n} plis)"
        for candidate, n in counts.items()
    ]
    return f"- **{spec.label}** : " + ", ".join(parts) + "."


def _model_lines(
    schema: DatasetSchema, training_config: TrainingConfig, report: EvaluationReport | None
) -> list[str]:
    cv = training_config.cv
    lines = [
        f"Familles de candidats : régression logistique / Ridge et gradient boosting peu "
        f"profond, sélectionnés par validation croisée interne ({cv.inner_splits} plis) avec "
        f"la règle « 1 écart-type ».",
        f"Cross-fit (calibration, seuil de traitement, ensembles conformes) : "
        f"{cv.crossfit_splits} plis.",
        "",
    ]
    for name, spec in schema.enabled_targets().items():
        lines.append(_candidates_line(spec, name, report))
    return lines


# --------------------------------------------------------------------------------------
# Section 7: performance
# --------------------------------------------------------------------------------------


def _performance_table(
    schema: DatasetSchema, name: TargetName, target_eval: TargetEvaluation
) -> list[str]:
    spec = schema.targets.get(name)
    dummy = target_eval.baselines.get("dummy", {})
    rules = {
        rule_name: values
        for rule_name, values in target_eval.baselines.items()
        if rule_name != "dummy"
    }
    lines = [
        "| Métrique | Valeur [IC 95 %] | Référence naïve | Règle clinique |",
        "|---|---|---|---|",
    ]
    for key, estimate in target_eval.metrics.items():
        label = _metric_label(key, spec)
        naive = _fmt_ci(dummy[key]) if key in dummy else "n/d"
        rule_cells = [
            f"{_fmt_ci(values[key])} ({rule_name})"
            for rule_name, values in rules.items()
            if key in values
        ]
        rule = "; ".join(rule_cells) if rule_cells else "n/d"
        lines.append(f"| {label} | {_fmt_ci(estimate)} | {naive} | {rule} |")
    return lines


def _end_to_end_lines(report: EvaluationReport) -> list[str]:
    items = [
        (key, estimate)
        for key, estimate in report.end_to_end.items()
        if not key.startswith("fairness.")
    ]
    if not items:
        return []
    lines = ["", "**Métriques de bout en bout :**"]
    lines.extend(f"- `{key}` : {_fmt_ci(estimate)}" for key, estimate in items)
    return lines


def _performance_lines(schema: DatasetSchema, report: EvaluationReport | None) -> list[str]:
    if report is None:
        return [_NO_EVALUATION]
    lines = [
        f"Protocole d'évaluation : {report.protocol_fr}.",
        "Les intervalles de confiance à 95 % sont estimés par bootstrap percentile sur les "
        "patients ; ils sous-estiment la variabilité liée à l'échantillon d'apprentissage "
        "(Bengio et Grandvalet, 2004).",
        "",
    ]
    for name, target_eval in report.targets.items():
        spec = schema.targets.get(name)
        label = name.value if spec is None else spec.label
        lines.append(f"### {label} (n = {target_eval.n})")
        lines.extend(_performance_table(schema, name, target_eval))
        lines.append("")
    lines.extend(_end_to_end_lines(report))
    return lines


# --------------------------------------------------------------------------------------
# Section 8: calibration and conformal coverage
# --------------------------------------------------------------------------------------


def _calibration_lines(schema: DatasetSchema, report: EvaluationReport | None) -> list[str]:
    if report is None:
        return [_NO_EVALUATION]
    lines = [
        f"Niveau de couverture visé (1 - alpha) : {_fmt_percent(1.0 - report.alpha)}.",
        "La couverture conforme est **marginale** (moyenne sur l'ensemble des patients "
        "échangeables avec ceux de l'apprentissage), pas garantie individuellement pour "
        "chaque patient.",
        "",
    ]
    for name, target_eval in report.targets.items():
        spec = schema.targets.get(name)
        label = name.value if spec is None else spec.label
        bits: list[str] = []
        conformal = target_eval.conformal
        if "coverage" in conformal:
            bits.append(f"Couverture empirique : {_fmt_ci(conformal['coverage'])}")
        if "mean_set_size" in conformal:
            bits.append(f"Taille moyenne de l'ensemble : {_fmt_ci(conformal['mean_set_size'])}")
        if "mean_width" in conformal:
            bits.append(f"Largeur moyenne de l'intervalle : {_fmt_ci(conformal['mean_width'])}")
        metrics_ = target_eval.metrics
        if "calibration_slope" in metrics_:
            bits.append(f"Pente de calibration : {_fmt_ci(metrics_['calibration_slope'])}")
        if "calibration_intercept" in metrics_:
            bits.append(f"Ordonnée de calibration : {_fmt_ci(metrics_['calibration_intercept'])}")
        if "brier" in metrics_:
            bits.append(f"Score de Brier : {_fmt_ci(metrics_['brier'])}")
        if bits:
            lines.append(f"- **{label}** : " + " ; ".join(bits) + ".")
    return lines


# --------------------------------------------------------------------------------------
# Section 9: limitations
# --------------------------------------------------------------------------------------


def _limits_lines(metadata: BundleMetadata, report: EvaluationReport | None) -> list[str]:
    lines = [
        f"**Petit effectif** (n = {metadata.n_train} dans cet entraînement) et étude "
        "**monocentrique** : les performances peuvent être optimistes ou instables ; une "
        "validation externe est indispensable.",
        "**Biais de confusion par indication** : la molécule et les doses reflètent les "
        "habitudes de prescription du service au moment du recueil, pas un traitement "
        "optimal validé ; un patient non traité aurait peut-être dû l'être.",
        "**Cibles construites** à partir de texte libre (molécule, dose totale) et de scores "
        "de suivi : erreurs de saisie possibles, table d'équivalence des doses à valider par "
        "un clinicien.",
        "**Valeurs manquantes** imputées à l'intérieur de chaque pli d'apprentissage ; une "
        "forte proportion de données manquantes abaisse le niveau de confiance affiché.",
    ]
    if TargetName.DOSE_DAY1 not in metadata.targets:
        lines.append(
            "Il n'existe pas de dose du **premier jour** dans les données source "
            "(`dose_day1`) : cette cible est absente du modèle."
        )
    if report is not None:
        for target_eval in report.targets.values():
            lines.extend(f"- {warning}" for warning in target_eval.warnings)
        lines.extend(
            f"- {warning}" for warning in report.warnings if "synthé" not in warning.lower()
        )
        if metadata.is_synthetic:
            lines.append("Évaluation sur données synthétiques.")
    return lines


# --------------------------------------------------------------------------------------
# Section 10: fairness
# --------------------------------------------------------------------------------------


def _fairness_lines(report: EvaluationReport | None) -> list[str]:
    if report is None:
        return [_NO_EVALUATION]
    items = [
        (key.removeprefix("fairness."), estimate)
        for key, estimate in report.end_to_end.items()
        if key.startswith("fairness.")
    ]
    if not items:
        return [
            "Évaluation par sous-groupe non réalisée : l'effectif ne permet pas d'évaluer "
            "finement les performances par sexe ou tranche d'âge."
        ]
    lines = ["Performances par sous-groupe (effectif restreint, à interpréter avec prudence) :"]
    lines.extend(f"- `{key}` : {_fmt_ci(estimate)}" for key, estimate in items)
    return lines


# --------------------------------------------------------------------------------------
# Section 11: version information
# --------------------------------------------------------------------------------------


def _version_lines(metadata: BundleMetadata) -> list[str]:
    commit = metadata.git_commit or "non disponible"
    return [
        f"- Identifiant du bundle : `{metadata.bundle_id}`",
        f"- Version du paquet `sevrage_ai` : {metadata.package_version}",
        f"- Python : {metadata.python_version}",
        f"- scikit-learn : {metadata.library_versions.get('scikit-learn', 'n/d')}",
        f"- Empreinte du schéma : `{metadata.schema_hash}`",
        f"- Empreinte de la configuration d'entraînement : `{metadata.training_config_hash}`",
        f"- Empreinte des données d'entraînement : `{metadata.data_fingerprint}`",
        f"- Commit Git : {commit}",
        f"- Créé le : {metadata.created_at.isoformat()}",
    ]


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def render(
    metadata: BundleMetadata,
    report: EvaluationReport | None,
    schema: DatasetSchema,
    training_config: TrainingConfig,
) -> str:
    """Render the French Markdown model card. (Takes parts, not a bundle, so that
    ``save_bundle`` can render the card before the bundle exists.)"""
    lines: list[str] = [f"# Fiche modèle — {metadata.bundle_id}", ""]
    if metadata.is_synthetic:
        lines.append(
            f"> **{_BANNER}** : ce modèle a été entraîné sur des données synthétiques "
            "générées automatiquement ; aucune conclusion clinique ne doit en être tirée."
        )
        lines.append("")
    sections = (
        _purpose_lines(),
        _scope_lines(),
        _data_lines(schema, metadata),
        _target_definitions_lines(schema),
        _features_lines(schema, metadata),
        _model_lines(schema, training_config, report),
        _performance_lines(schema, report),
        _calibration_lines(schema, report),
        _limits_lines(metadata, report),
        _fairness_lines(report),
        _version_lines(metadata),
    )
    for heading, body in zip(_SECTIONS, sections, strict=True):
        lines.append(heading)
        lines.append("")
        lines.extend(body)
        lines.append("")
    text = "\n".join(lines)
    return text if text.endswith("\n") else f"{text}\n"
