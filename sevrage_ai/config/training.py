"""Training configuration (``configs/training.yaml``). **Leader-owned.**

Every model is frozen and rejects unknown keys. Candidate grids are typed per estimator
family; ``models/candidates.py`` turns them into the cartesian product of candidates.
"""

from __future__ import annotations

from typing import Annotated, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from sevrage_ai.config._canonical import model_sha256
from sevrage_ai.types import TargetName

__all__ = [
    "AbstentionConfig",
    "BootstrapConfig",
    "CVConfig",
    "CalibrationConfig",
    "CandidatesConfig",
    "ClassificationCandidates",
    "ConformalConfig",
    "ConsistencyConfig",
    "DecisionConfig",
    "EvaluationConfig",
    "ExplainConfig",
    "ForestGrid",
    "HGBClassifierGrid",
    "HGBRegressorGrid",
    "LogisticGrid",
    "OODConfig",
    "RegressionCandidates",
    "RidgeGrid",
    "RoundingConfig",
    "SelectionConfig",
    "ThresholdConfig",
    "TrainingConfig",
    "TwoStageConfig",
    "training_config_hash",
]

Probability = Annotated[float, Field(gt=0.0, lt=1.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
ClassWeight = Literal["balanced"] | None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _unique(values: tuple[object, ...], what: str) -> None:
    if len(set(values)) != len(values):
        _fail(f"{what} : valeurs en double")


class CVConfig(_Frozen):
    """Cross-validation sizes. ``n_splits`` are clamped at runtime to the smallest class."""

    outer_splits: int = Field(ge=2)
    outer_repeats: PositiveInt
    inner_splits: int = Field(ge=2)
    crossfit_splits: int = Field(ge=2)


class LogisticGrid(_Frozen):
    """L2 logistic regression (multinomial for more than 2 classes)."""

    C: tuple[PositiveFloat, ...] = Field(min_length=1)
    class_weight: tuple[ClassWeight, ...] = ("balanced",)
    max_iter: int = Field(default=2000, ge=100)


class HGBClassifierGrid(_Frozen):
    """Shallow ``HistGradientBoostingClassifier`` (no early stopping)."""

    max_depth: tuple[PositiveInt, ...] = Field(min_length=1)
    learning_rate: tuple[PositiveFloat, ...] = Field(min_length=1)
    max_iter: tuple[PositiveInt, ...] = Field(min_length=1)
    min_samples_leaf: tuple[PositiveInt, ...] = Field(min_length=1)
    l2_regularization: tuple[NonNegativeFloat, ...] = (1.0,)
    class_weight: tuple[ClassWeight, ...] = ("balanced",)


class ForestGrid(_Frozen):
    """Random forest: fully grown trees, limited by ``min_samples_leaf``."""

    n_estimators: tuple[PositiveInt, ...] = (300,)
    min_samples_leaf: tuple[PositiveInt, ...] = Field(min_length=1)
    max_features: tuple[Literal["sqrt", "log2"], ...] = ("sqrt",)
    class_weight: tuple[Literal["balanced", "balanced_subsample"] | None, ...] = (
        "balanced_subsample",
    )


class RidgeGrid(_Frozen):
    """Ridge regression on the (transformed) target."""

    alpha: tuple[PositiveFloat, ...] = Field(min_length=1)


class HGBRegressorGrid(_Frozen):
    """Shallow ``HistGradientBoostingRegressor`` (no early stopping)."""

    max_depth: tuple[PositiveInt, ...] = Field(min_length=1)
    loss: tuple[Literal["absolute_error", "squared_error", "poisson"], ...] = ("absolute_error",)
    min_samples_leaf: tuple[PositiveInt, ...] = Field(min_length=1)
    learning_rate: tuple[PositiveFloat, ...] = (0.05,)
    max_iter: tuple[PositiveInt, ...] = (150,)
    l2_regularization: tuple[NonNegativeFloat, ...] = (1.0,)


class ClassificationCandidates(_Frozen):
    """Candidate families for classification targets (at least one)."""

    logistic: LogisticGrid | None = None
    hgb: HGBClassifierGrid | None = None
    forest: ForestGrid | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> ClassificationCandidates:
        if self.logistic is None and self.hgb is None and self.forest is None:
            _fail("au moins une famille de classifieurs doit être déclarée")
        return self


class RegressionCandidates(_Frozen):
    """Candidate families for regression targets (at least one)."""

    ridge: RidgeGrid | None = None
    hgb: HGBRegressorGrid | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> RegressionCandidates:
        if self.ridge is None and self.hgb is None:
            _fail("au moins une famille de régresseurs doit être déclarée")
        return self


class CandidatesConfig(_Frozen):
    """Candidate grids per task type."""

    classification: ClassificationCandidates
    regression: RegressionCandidates


class CalibrationConfig(_Frozen):
    """Calibration methods compared by inner-CV log-loss.

    ``isotonic`` is accepted here but refused at runtime when n < 1000 (design 3.2). With
    fewer than ``cv`` members in the smallest class, the candidate falls back to ``none``.
    """

    methods: tuple[Literal["none", "sigmoid", "isotonic"], ...] = Field(min_length=1)
    cv: int = Field(default=3, ge=2)

    @field_validator("methods")
    @classmethod
    def _distinct(
        cls, methods: tuple[Literal["none", "sigmoid", "isotonic"], ...]
    ) -> tuple[Literal["none", "sigmoid", "isotonic"], ...]:
        _unique(tuple(methods), "calibration.methods")
        return methods


class SelectionConfig(_Frozen):
    """Model selection metric (scikit-learn scorer names, higher is better)."""

    classification: Literal["neg_log_loss", "neg_brier_score"] = "neg_log_loss"
    regression: Literal["neg_mean_absolute_error", "neg_root_mean_squared_error"] = (
        "neg_mean_absolute_error"
    )
    one_se_rule: bool = True


class ConformalConfig(_Frozen):
    """Cross-conformal sets (classification) and CV+ intervals (regression)."""

    alpha: float = Field(gt=0.0, le=0.5)
    class_conditional: bool = False
    score: Literal["lac"] = "lac"


class ThresholdConfig(_Frozen):
    """Treatment decision threshold chosen on out-of-fold probabilities.

    ``target_sensitivity``: highest threshold whose OOF sensitivity >= ``value``.
    ``youden``: maximises sensitivity + specificity - 1 (``value`` ignored).
    ``fixed``: threshold = ``value``. ``grey_zone``: |P(yes) - threshold| < grey_zone lowers
    confidence and raises a warning flag.
    """

    strategy: Literal["target_sensitivity", "youden", "fixed"] = "target_sensitivity"
    value: Probability = 0.90
    grey_zone: float = Field(default=0.10, ge=0.0, lt=0.5)


class AbstentionConfig(_Frozen):
    """Thresholds for HIGH / MODERATE / LOW confidence (design 7.1).

    Relative interval width = (upper - lower) / max(point, epsilon).
    """

    min_top_prob_high: Probability = 0.70
    max_rel_width_moderate: PositiveFloat = 1.5
    max_rel_width_low: PositiveFloat = 3.0
    max_missing_high: Fraction = 0.10
    max_missing_low: Fraction = 0.30

    @model_validator(mode="after")
    def _ordered(self) -> AbstentionConfig:
        if self.max_rel_width_moderate > self.max_rel_width_low:
            _fail("abstention : max_rel_width_moderate doit être <= max_rel_width_low")
        if self.max_missing_high > self.max_missing_low:
            _fail("abstention : max_missing_high doit être <= max_missing_low")
        return self


class OODConfig(_Frozen):
    """Out-of-distribution detection settings.

    Per-feature training quantile range ``range_quantiles``, plus an optional
    LocalOutlierFactor novelty score. A patient whose score falls below the training
    ``novelty_quantile`` quantile gets a warning flag.
    """

    range_quantiles: tuple[Fraction, Fraction] = (0.01, 0.99)
    novelty: Literal["lof", "none"] = "lof"
    lof_n_neighbors: int = Field(default=20, ge=2)
    novelty_quantile: float = Field(default=0.01, gt=0.0, lt=0.5)

    @field_validator("range_quantiles")
    @classmethod
    def _ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not value[0] < value[1]:
            _fail("ood.range_quantiles doit vérifier bas < haut")
        return value


class BootstrapConfig(_Frozen):
    """Percentile bootstrap for metric confidence intervals."""

    n_resamples: int = Field(default=1000, ge=10)
    ci: Probability = 0.95


class RoundingConfig(_Frozen):
    """Display rounding of the final recommendation (doses to the nearest ``dose_mg``)."""

    dose_mg: PositiveFloat = 5.0
    duration_days: PositiveInt = 1


class ConsistencyConfig(_Frozen):
    """Consistency layer thresholds (rule R5)."""

    severe_untreated_min_prob: Probability = 0.30
    """P(most severe class) at or above which "no treatment" raises a critical flag."""


class DecisionConfig(_Frozen):
    """How the predicted class of a multiclass target is chosen (probabilities never change).

    For the targets in ``prior_corrected``, the predicted class is the argmax of
    ``p_k / pi_k`` (``pi`` = class prevalence in the training data, Laplace-smoothed): the
    class whose probability exceeds its base rate the most. This is the Bayes decision for
    balanced accuracy; a plain argmax of calibrated probabilities almost always returns the
    majority class ("léger") and misses the rare severe withdrawals. ``treatment`` uses its
    own threshold (:class:`ThresholdConfig`) and cannot be listed here.
    """

    prior_corrected: tuple[TargetName, ...] = (TargetName.SEVERITY,)

    @field_validator("prior_corrected")
    @classmethod
    def _no_treatment(cls, value: tuple[TargetName, ...]) -> tuple[TargetName, ...]:
        allowed = {TargetName.SEVERITY, TargetName.MOLECULE}
        if not set(value) <= allowed:
            _fail("decision.prior_corrected n'accepte que severity et molecule")
        return value


class TwoStageConfig(_Frozen):
    """Two-stage model of a class target (see ``models.staged``).

    ``P(class) = P(gate) * P(class | gate)``: the gate is "target ``gate`` equals
    ``equals``" (the patient is treated) and the remaining ``1 - P(gate)`` goes to class
    ``otherwise`` (mild withdrawal). Both models use the candidate chosen for the target.
    """

    gate: TargetName = TargetName.TREATMENT
    equals: str = "yes"
    otherwise: str = "low"


class ExplainConfig(_Frozen):
    """Explanation settings."""

    top_k: PositiveInt = 8
    permutation_repeats: PositiveInt = 10


class EvaluationConfig(_Frozen):
    """Evaluation settings."""

    min_class_count_warning: PositiveInt = 10


class TrainingConfig(_Frozen):
    """Whole training profile. ``seed`` is propagated to every random operation."""

    seed: int = Field(ge=0)
    feature_set: str = Field(min_length=1)
    n_jobs: int = -1
    cv: CVConfig
    candidates: CandidatesConfig
    calibration: CalibrationConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    conformal: ConformalConfig
    treatment_threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    ood: OODConfig = Field(default_factory=OODConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    rounding: RoundingConfig = Field(default_factory=RoundingConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    two_stage: dict[TargetName, TwoStageConfig] = Field(default_factory=dict)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("n_jobs")
    @classmethod
    def _valid_n_jobs(cls, value: int) -> int:
        if value == 0 or value < -1:
            _fail("n_jobs doit valoir -1 (tous les cœurs) ou un entier >= 1")
        return value

    @field_validator("two_stage")
    @classmethod
    def _valid_two_stage(
        cls, value: dict[TargetName, TwoStageConfig]
    ) -> dict[TargetName, TwoStageConfig]:
        for target, stage in value.items():
            if target is not TargetName.SEVERITY or stage.gate is target:
                _fail("two_stage n'accepte que severity, avec une autre cible comme porte")
        return value


def training_config_hash(config: TrainingConfig) -> str:
    """Hex sha256 of the training config's canonical JSON. Stored in bundle metadata."""
    return model_sha256(config)
