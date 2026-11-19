"""Cross-module contracts of ``sevrage_ai``.

**Frozen after WS0 -- leader-owned.** Every workstream codes against these types.
A change goes through a *contract change request* in the workstream hand-off report.

Canonical data conventions
==========================

All modules exchange patient data in *canonical* form (:class:`CanonicalData`), produced
by :func:`sevrage_ai.data.mapping.to_canonical` from raw tables. The invariants below are
checked when a :class:`CanonicalData` is constructed.

Index
    ``X``, ``y``, ``groups`` and ``ids`` share the same ``0..n-1`` positional index
    (``pd.RangeIndex(n)``). Use :meth:`CanonicalData.subset` to take row subsets: it selects
    by position and re-indexes every part.

``X`` -- features
    One column per ``admission`` feature of the schema, named by the feature's canonical
    ``name``, in schema declaration order, **including** features whose source column was
    absent from the raw table (then all missing). ``post_admission`` features are never in
    ``X``: they only feed target derivation.

    * ``numeric``: ``float64``, missing = ``NaN``, canonical units (``scale`` already
      applied, e.g. grams/day rather than standard drinks/day).
    * ``binary``: ``float64`` restricted to ``{0.0, 1.0}``, missing = ``NaN``.
    * ``categorical``: :data:`STRING_DTYPE` (``pd.StringDtype("python")``), missing =
      ``pd.NA``. Values are canonical category labels (after ``value_mapping``); values
      outside the declared ``categories`` are kept verbatim and reported by validation.

    Why ``StringDtype("python")`` and not ``object``: it is explicit and identical under
    pandas 2 and 3. The ``"string"`` alias silently switches to pyarrow storage when pyarrow
    is installed (streamlit installs it), and pandas 3 infers its own ``str`` dtype, whose
    missing value is ``NaN``. Two caveats, both verified with pandas 3.0 and scikit-learn 1.9:

    * scikit-learn cannot impute ``pd.NA`` in string columns
      (``TypeError: boolean value of NA is ambiguous``). The preprocessing pipeline
      (``features/preprocessing.py``) must first convert categorical columns to ``object``
      with ``np.nan``, e.g. ``s.astype(object).where(s.notna(), np.nan)``.
    * ``Series.map`` and similar operations return pandas' default ``str`` dtype. Re-cast
      with ``.astype(STRING_DTYPE)`` before building a :class:`CanonicalData`.

``y`` -- targets (``None`` for prediction inputs)
    One column per *enabled* target, named by the :class:`TargetName` value:

    * ``severity``: :data:`STRING_DTYPE`, one of the schema classes (ordered from least to
      most severe; the **last** class is the "high" class), ``<NA>`` = undetermined.
    * ``treatment``: :data:`STRING_DTYPE`, :data:`TREATED` (``"yes"``) or
      :data:`UNTREATED` (``"no"``), ``<NA>`` = unknown.
    * ``molecule``: :data:`STRING_DTYPE`, canonical molecule name (e.g. ``"diazepam"``),
      *before* rare-class merging, ``<NA>`` when untreated or unknown.
    * ``dose_day1``, ``dose_cumulative``: ``float64``, mg **diazepam-equivalent**
      (:data:`DOSE_UNIT`), ``NaN`` when untreated or unknown.
    * ``duration_days``: ``float64`` days, ``NaN`` when untreated or unknown.

    Conditional targets (``conditional_on`` in the schema) are **always missing** on rows
    where their condition does not hold. ``0`` never means "not applicable".

``ids``
    :data:`STRING_DTYPE`, never missing. The pseudonymous id column when
    ``dataset.id_column`` is set, otherwise the 1-based data row number as a string
    (``"1"``, ``"2"``, ...). Never a feature. Uniqueness is checked by validation, not here.

``groups``
    :data:`STRING_DTYPE` patient key used for grouped CV when ``dataset.group_column`` is
    set, else ``None``.

``is_synthetic``
    ``True`` when the data come from the synthetic generator (``__synthetic__`` marker
    column). It must propagate to bundles, reports, the CLI and the UI.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from sevrage_ai.config.schema import DatasetSchema

__all__ = [
    "CLASSIFICATION_TARGETS",
    "DOSE_UNIT",
    "FLOAT_DTYPE",
    "NO_MOLECULE",
    "OTHER_CLASS",
    "REGRESSION_TARGETS",
    "STRING_DTYPE",
    "STRING_TARGETS",
    "SYNTHETIC_MARKER_COLUMN",
    "TREATED",
    "UNTREATED",
    "BatchResult",
    "CanonicalData",
    "ClassOutput",
    "ConfidenceCheck",
    "ConfidenceLevel",
    "Contribution",
    "FeatureKind",
    "FinalRecommendation",
    "Flag",
    "FlagLevel",
    "MetricEstimate",
    "ModelFamily",
    "ModelInfo",
    "PatientPrediction",
    "PredictorProtocol",
    "RegressionOutput",
    "RowIssue",
    "TargetName",
    "TaskKind",
    "Timing",
    "has_string_dtype",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

STRING_DTYPE: Final = pd.StringDtype(storage="python")
"""Dtype of every canonical string column (categorical features, class targets, ids)."""

FLOAT_DTYPE: Final = np.dtype(np.float64)
"""Dtype of every canonical numeric or binary column."""

SYNTHETIC_MARKER_COLUMN: Final = "__synthetic__"
"""Metadata column present (and truthy) in every synthetic file."""

TREATED: Final = "yes"
"""Canonical ``treatment`` value for treated patients (the positive class)."""

UNTREATED: Final = "no"
"""Canonical ``treatment`` value for untreated patients."""

NO_MOLECULE: Final = "none"
"""``FinalRecommendation.molecule`` when no treatment is recommended."""

OTHER_CLASS: Final = "other"
"""Label of the class that merges rare molecule classes (``min_class_count``)."""

DOSE_UNIT: Final = "mg_diazepam_eq"
"""Unit of the canonical dose targets."""


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class TargetName(StrEnum):
    """Names of the prediction targets (also the column names of ``CanonicalData.y``)."""

    SEVERITY = "severity"
    TREATMENT = "treatment"
    MOLECULE = "molecule"
    DOSE_DAY1 = "dose_day1"
    DOSE_CUMULATIVE = "dose_cumulative"
    DURATION = "duration_days"


CLASSIFICATION_TARGETS: Final[frozenset[TargetName]] = frozenset(
    {TargetName.SEVERITY, TargetName.TREATMENT, TargetName.MOLECULE}
)
"""Targets predicted by classifiers (their ``y`` column has :data:`STRING_DTYPE`)."""

STRING_TARGETS: Final[frozenset[TargetName]] = CLASSIFICATION_TARGETS
"""Alias of :data:`CLASSIFICATION_TARGETS`, used for dtype checks."""

REGRESSION_TARGETS: Final[frozenset[TargetName]] = frozenset(
    {TargetName.DOSE_DAY1, TargetName.DOSE_CUMULATIVE, TargetName.DURATION}
)
"""Targets predicted by regressors (their ``y`` column is ``float64``)."""


class TaskKind(StrEnum):
    """Kind of supervised task attached to a target."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    ORDINAL = "ordinal"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        """``True`` for binary, multiclass and ordinal tasks."""
        return self is not TaskKind.REGRESSION


class ConfidenceLevel(StrEnum):
    """Per-output reliability level (see design section 7.1). Ordered LOW < MODERATE < HIGH."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Ordering key: LOW=0, MODERATE=1, HIGH=2."""
        return _CONFIDENCE_RANK[self]

    @property
    def label_fr(self) -> str:
        """French display label."""
        return _CONFIDENCE_LABEL_FR[self]

    @classmethod
    def lowest(cls, levels: Iterable[ConfidenceLevel]) -> ConfidenceLevel:
        """Return the least confident level of ``levels`` (which must not be empty)."""
        materialised = tuple(levels)
        if not materialised:
            msg = "ConfidenceLevel.lowest() needs at least one level"
            raise ValueError(msg)
        return min(materialised, key=lambda level: level.rank)


_CONFIDENCE_RANK: Final[Mapping[ConfidenceLevel, int]] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MODERATE: 1,
    ConfidenceLevel.HIGH: 2,
}
_CONFIDENCE_LABEL_FR: Final[Mapping[ConfidenceLevel, str]] = {
    ConfidenceLevel.LOW: "faible",
    ConfidenceLevel.MODERATE: "modérée",
    ConfidenceLevel.HIGH: "élevée",
}


class FeatureKind(StrEnum):
    """Kind of a clinical feature, which fixes its canonical dtype."""

    NUMERIC = "numeric"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class Timing(StrEnum):
    """When a feature is measured. Only ``admission`` features may be predictors (D9)."""

    ADMISSION = "admission"
    POST_ADMISSION = "post_admission"


class ModelFamily(StrEnum):
    """Preprocessing family: scaled/one-hot for linear models, raw/ordinal for trees."""

    LINEAR = "linear"
    TREE = "tree"


FlagLevel = Literal["info", "warning", "critical"]
"""Severity of a :class:`Flag`. ``critical`` flags force LOW confidence."""

_FLAG_LEVELS: Final = frozenset({"info", "warning", "critical"})
_FLAG_CODE_RE: Final = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
_PROBA_SUM_TOLERANCE: Final = 1e-6


def has_string_dtype(series: pd.Series[str]) -> bool:
    """Return ``True`` if ``series`` uses the canonical :data:`STRING_DTYPE`."""
    dtype = series.dtype
    return (
        isinstance(dtype, pd.StringDtype) and dtype.storage == "python" and dtype.na_value is pd.NA
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_finite_non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


# --------------------------------------------------------------------------------------
# Canonical data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalData:
    """Canonical patient table shared by every module (conventions in the module docstring).

    Construction checks index alignment and dtypes and raises ``ValueError`` on a violation.
    That is a programming error, not a user-data error: user data problems are reported by
    ``data.validation``.
    """

    X: pd.DataFrame
    y: pd.DataFrame | None
    groups: pd.Series[str] | None
    ids: pd.Series[str]
    is_synthetic: bool

    def __post_init__(self) -> None:
        n = len(self.X)
        expected = pd.RangeIndex(n)
        _require(self.X.index.equals(expected), "CanonicalData.X must be indexed 0..n-1")
        _require(self.X.columns.is_unique, "CanonicalData.X column names must be unique")
        for column in self.X.columns:
            series = self.X[column]
            _require(
                series.dtype == FLOAT_DTYPE or has_string_dtype(series),
                f"CanonicalData.X[{column!r}] must be float64 or STRING_DTYPE",
            )
        _require(self.ids.index.equals(expected), "CanonicalData.ids must be indexed 0..n-1")
        _require(has_string_dtype(self.ids), "CanonicalData.ids must use STRING_DTYPE")
        _require(not bool(self.ids.isna().any()), "CanonicalData.ids must not be missing")
        if self.groups is not None:
            _require(
                self.groups.index.equals(expected), "CanonicalData.groups must be indexed 0..n-1"
            )
            _require(has_string_dtype(self.groups), "CanonicalData.groups must use STRING_DTYPE")
        if self.y is not None:
            self._check_targets(self.y, expected)

    @staticmethod
    def _check_targets(y: pd.DataFrame, expected: pd.RangeIndex) -> None:
        _require(y.index.equals(expected), "CanonicalData.y must be indexed 0..n-1")
        valid_names = {name.value for name in TargetName}
        for column in y.columns:
            _require(column in valid_names, f"CanonicalData.y has unknown target {column!r}")
            series = y[column]
            if TargetName(column) in STRING_TARGETS:
                _require(
                    has_string_dtype(series), f"CanonicalData.y[{column!r}] must use STRING_DTYPE"
                )
            else:
                _require(
                    series.dtype == FLOAT_DTYPE, f"CanonicalData.y[{column!r}] must be float64"
                )

    @property
    def n_rows(self) -> int:
        """Number of rows (patients or episodes)."""
        return len(self.X)

    def subset(self, positions: npt.ArrayLike) -> CanonicalData:
        """Return the rows at ``positions`` (positional), re-indexed ``0..k-1``."""
        index = np.asarray(positions, dtype=np.intp)
        return CanonicalData(
            X=self.X.iloc[index].reset_index(drop=True),
            y=None if self.y is None else self.y.iloc[index].reset_index(drop=True),
            groups=None if self.groups is None else self.groups.iloc[index].reset_index(drop=True),
            ids=self.ids.iloc[index].reset_index(drop=True),
            is_synthetic=self.is_synthetic,
        )


# --------------------------------------------------------------------------------------
# Prediction outputs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Flag:
    """A message attached to a prediction.

    ``code`` is namespaced ``<area>.<name>``: ``ood.*`` (out of distribution),
    ``consistency.*``, ``safety.<rule_id>``, ``abstention.*``, ``model.*`` (e.g. a constant
    molecule model), ``data.*``. ``message_fr`` is shown to clinicians. ``feature`` is a
    canonical feature name. **Never** put patient values in ``code``, ``message_fr`` or
    ``feature``.
    """

    code: str
    level: FlagLevel
    message_fr: str
    feature: str | None = None

    def __post_init__(self) -> None:
        _require(self.level in _FLAG_LEVELS, f"Flag.level must be one of {sorted(_FLAG_LEVELS)}")
        _require(bool(_FLAG_CODE_RE.match(self.code)), "Flag.code must look like 'area.name'")
        _require(bool(self.message_fr.strip()), "Flag.message_fr must not be empty")


@dataclass(frozen=True, slots=True)
class ConfidenceCheck:
    """One criterion behind a confidence level (the "why this reliability?" explanation).

    ``met`` is ``True`` when the criterion supports a high confidence. Otherwise
    ``caps_at`` is the highest level the output can reach because of it (``MODERATE`` or
    ``LOW``). ``code`` is namespaced ``confidence.<name>``. ``message_fr`` explains the
    criterion for this patient with feature labels and model outputs (probabilities,
    interval widths), **never** a patient's feature value.
    """

    code: str
    met: bool
    message_fr: str
    caps_at: ConfidenceLevel | None = None

    def __post_init__(self) -> None:
        _require(bool(self.code) and bool(self.message_fr), "ConfidenceCheck needs a message")
        _require(
            self.met == (self.caps_at is None),
            "ConfidenceCheck: caps_at is set if and only if the criterion is not met",
        )
        _require(
            self.caps_at is not ConfidenceLevel.HIGH,
            "ConfidenceCheck: caps_at must be MODERATE or LOW",
        )


@dataclass(frozen=True, slots=True)
class ClassOutput:
    """Output of one classification target for one patient.

    ``probabilities`` maps every class label to its probability (they sum to 1).
    ``label`` is the displayed decision: the argmax, except for ``treatment`` (thresholded
    decision) and the targets of ``decision.prior_corrected`` (argmax of p / prevalence).
    ``prediction_set`` is the conformal set at level ``1 - alpha``. ``reasons`` are French
    explanations of ``confidence`` (the criteria that are not met); ``checks`` lists every
    criterion, met or not (empty for outputs built without it, e.g. by a fake predictor).
    Treat ``probabilities`` as read-only.
    """

    target: TargetName
    label: str
    probabilities: Mapping[str, float]
    prediction_set: frozenset[str]
    alpha: float
    confidence: ConfidenceLevel
    reasons: tuple[str, ...]
    checks: tuple[ConfidenceCheck, ...] = ()

    def __post_init__(self) -> None:
        _require(self.target in CLASSIFICATION_TARGETS, "ClassOutput.target must be a class target")
        _require(len(self.probabilities) > 0, "ClassOutput.probabilities must not be empty")
        _require(
            all(0.0 <= p <= 1.0 for p in self.probabilities.values()),
            "ClassOutput.probabilities must lie in [0, 1]",
        )
        _require(
            abs(math.fsum(self.probabilities.values()) - 1.0) <= _PROBA_SUM_TOLERANCE,
            "ClassOutput.probabilities must sum to 1",
        )
        _require(self.label in self.probabilities, "ClassOutput.label must be a known class")
        _require(
            self.prediction_set <= self.probabilities.keys(),
            "ClassOutput.prediction_set must only contain known classes",
        )
        _require(0.0 < self.alpha < 1.0, "ClassOutput.alpha must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class RegressionOutput:
    """Output of one regression target for one patient, in canonical units.

    Doses are in :data:`DOSE_UNIT`. ``[lower, upper]`` is the conformal interval at level
    ``1 - alpha``. ``upper`` may be ``+inf`` when the interval is unbounded, which forces
    LOW confidence. Invariant: ``0 <= lower <= point <= upper``.
    """

    target: TargetName
    point: float
    lower: float
    upper: float
    alpha: float
    unit: str
    confidence: ConfidenceLevel
    reasons: tuple[str, ...]
    checks: tuple[ConfidenceCheck, ...] = ()

    def __post_init__(self) -> None:
        _require(
            self.target in REGRESSION_TARGETS, "RegressionOutput.target must be a regression target"
        )
        _require(
            _is_finite_non_negative(self.lower) and _is_finite_non_negative(self.point),
            "RegressionOutput.lower and point must be finite and >= 0",
        )
        _require(not math.isnan(self.upper), "RegressionOutput.upper must not be NaN")
        _require(
            self.lower <= self.point <= self.upper,
            "RegressionOutput requires lower <= point <= upper",
        )
        _require(0.0 < self.alpha < 1.0, "RegressionOutput.alpha must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class Contribution:
    """Local explanation of one feature (occlusion against the reference patient).

    ``delta`` is in percentage points of P(predicted class) for classifiers, or in target
    units for regressors. ``value_display`` is the formatted value of the patient being
    displayed (e.g. ``"4000 UI/L"`` or ``"Inconnu"``).
    """

    feature: str
    value_display: str
    delta: float


@dataclass(frozen=True, slots=True)
class FinalRecommendation:
    """Recommendation produced by the consistency layer (rules R1 to R4 of the design).

    * ``treat is False`` implies ``molecule == NO_MOLECULE`` and every dose or duration
      equal to ``0`` (or ``None`` when that target is disabled).
    * ``treat is True`` implies a real molecule and ``duration_days >= 1`` (or ``None``).
    * ``dose_day1_molecule_mg`` / ``dose_cumulative_molecule_mg`` (the same doses in mg of
      the recommended molecule) are ``None`` when the molecule has no dose equivalence (for
      example :data:`OTHER_CLASS`) or the target is disabled.
    * Every value is finite and ``>= 0``. When both doses are present,
      ``dose_cumulative_dze >= dose_day1_dze``.
    """

    treat: bool
    molecule: str
    dose_day1_dze: float | None
    dose_day1_molecule_mg: float | None
    dose_cumulative_dze: float | None
    duration_days: int | None
    dose_cumulative_molecule_mg: float | None = None

    def __post_init__(self) -> None:
        numbers = (
            self.dose_day1_dze,
            self.dose_day1_molecule_mg,
            self.dose_cumulative_dze,
            self.dose_cumulative_molecule_mg,
            None if self.duration_days is None else float(self.duration_days),
        )
        _require(
            all(v is None or _is_finite_non_negative(v) for v in numbers),
            "FinalRecommendation values must be finite and >= 0",
        )
        if self.treat:
            _require(
                bool(self.molecule) and self.molecule != NO_MOLECULE,
                "FinalRecommendation: treat=True requires a molecule",
            )
            _require(
                self.duration_days is None or self.duration_days >= 1,
                "FinalRecommendation: treat=True requires duration_days >= 1",
            )
        else:
            _require(
                self.molecule == NO_MOLECULE,
                "FinalRecommendation: treat=False requires molecule == 'none'",
            )
            _require(
                all(v is None or v == 0.0 for v in numbers),
                "FinalRecommendation: treat=False requires zero doses and duration",
            )
        if self.dose_day1_dze is not None and self.dose_cumulative_dze is not None:
            _require(
                self.dose_cumulative_dze >= self.dose_day1_dze,
                "FinalRecommendation: cumulative dose must be >= day-1 dose",
            )


@dataclass(frozen=True, slots=True)
class PatientPrediction:
    """Everything shown for one patient.

    The ``*_if_treated`` outputs are conditional predictions ("si un traitement est
    décidé"), always computed and shown, even when ``final.treat`` is ``False``. They are
    ``None`` only when the target is disabled in the schema. ``explanations`` may be empty
    when explanations were not requested. ``overall_checks`` explains
    ``overall_confidence`` (weakest output, data completeness, alerts).
    """

    row_id: str
    severity: ClassOutput
    treatment: ClassOutput
    molecule_if_treated: ClassOutput
    dose_day1_if_treated: RegressionOutput | None
    dose_cumulative_if_treated: RegressionOutput | None
    duration_if_treated: RegressionOutput | None
    final: FinalRecommendation
    flags: tuple[Flag, ...]
    explanations: Mapping[TargetName, tuple[Contribution, ...]]
    overall_confidence: ConfidenceLevel
    model_id: str
    trained_on_synthetic: bool
    overall_checks: tuple[ConfidenceCheck, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.row_id), "PatientPrediction.row_id must not be empty")
        pairs: tuple[tuple[ClassOutput | RegressionOutput | None, TargetName], ...] = (
            (self.severity, TargetName.SEVERITY),
            (self.treatment, TargetName.TREATMENT),
            (self.molecule_if_treated, TargetName.MOLECULE),
            (self.dose_day1_if_treated, TargetName.DOSE_DAY1),
            (self.dose_cumulative_if_treated, TargetName.DOSE_CUMULATIVE),
            (self.duration_if_treated, TargetName.DURATION),
        )
        for output, expected in pairs:
            _require(
                output is None or output.target is expected,
                f"PatientPrediction: output for {expected.value} has the wrong target",
            )


@dataclass(frozen=True, slots=True)
class RowIssue:
    """A per-row problem (the row is rejected in batch prediction).

    ``row_id`` is the canonical row id. ``column`` is a canonical feature name when known,
    or the raw column name before mapping. ``code`` is namespaced (``mapping.*``,
    ``validation.*``). ``message_fr`` must not contain the offending value.
    """

    row_id: str
    column: str | None
    code: str
    message_fr: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Result of a batch prediction: one prediction or one or more issues per input row."""

    predictions: tuple[PatientPrediction, ...]
    rejected: tuple[RowIssue, ...]


# --------------------------------------------------------------------------------------
# Serialisable models
# --------------------------------------------------------------------------------------


class MetricEstimate(BaseModel):
    """A performance metric with its percentile-bootstrap confidence interval.

    ``value`` is the point estimate. ``[lo, hi]`` is the ``level`` CI. ``sd_repeats`` is the
    SD across repeated outer-CV repeats (``None`` if not applicable). ``n`` is the number of
    observations used. ``NaN`` is allowed (undefined metric) and round-trips through JSON.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    value: float
    lo: float
    hi: float
    sd_repeats: float | None = None
    level: float = Field(default=0.95, gt=0.0, lt=1.0)
    n: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> MetricEstimate:
        if self.lo > self.hi:  # comparisons with NaN are False, so NaN bounds pass
            msg = "MetricEstimate requires lo <= hi"
            raise ValueError(msg)
        return self


class ModelInfo(BaseModel):
    """Summary of a loaded model, everything the UI and CLI need to display.

    ``metrics_summary`` keys are ``"<target>.<metric>"``, e.g.
    ``"severity.balanced_accuracy"`` or ``"treatment.roc_auc"``. ``bundle_path`` is the
    bundle directory, or ``None`` for in-memory or fake predictors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    model_id: str = Field(min_length=1)
    created_at: AwareDatetime
    is_synthetic: bool
    feature_set: str
    feature_names: tuple[str, ...]
    targets: tuple[TargetName, ...]
    n_train: int = Field(ge=0)
    schema_hash: str
    package_version: str
    alpha: float = Field(gt=0.0, lt=1.0)
    treatment_threshold: float = Field(ge=0.0, le=1.0)
    grey_zone: float = Field(ge=0.0, lt=0.5)
    dose_equivalence_to_diazepam: dict[str, float] = Field(default_factory=dict)
    metrics_summary: dict[str, MetricEstimate] = Field(default_factory=dict)
    evaluation_protocol: str | None = None
    bundle_path: Path | None = None


# --------------------------------------------------------------------------------------
# Predictor protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class PredictorProtocol(Protocol):
    """What the UI and CLI need from a predictor (the real one or ``FakePredictor``)."""

    @property
    def info(self) -> ModelInfo:
        """Model summary."""
        ...

    @property
    def schema(self) -> DatasetSchema:
        """Schema the model was trained with. It drives the patient form and the dictionary."""
        ...

    def predict_frame(
        self, df: pd.DataFrame, *, raw_columns: bool = True, explain: bool = False
    ) -> BatchResult:
        """Predict a batch.

        With ``raw_columns=True``, ``df`` uses raw source column names and raw values: the
        anonymization check and the mapping run first. With ``raw_columns=False``, ``df``
        already holds the canonical ``X`` columns. Invalid rows go to
        ``BatchResult.rejected``. Raises ``AnonymizationError`` when the table fails the
        anonymization check, which blocks the whole file. Per-patient explanations are
        costly and absent from batch exports: they are only computed with ``explain=True``.
        """
        ...

    def predict_record(self, record: Mapping[str, object]) -> PatientPrediction:
        """Predict one patient from a mapping of canonical feature name to canonical value.

        Missing keys or ``None`` mean unknown. Raises ``DataValidationError`` if the record
        is invalid.
        """
        ...
