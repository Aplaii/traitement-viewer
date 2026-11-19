"""Repeated nested cross-validation of the whole system (design section 8). Owner: WS5b.

One outer split set is shared by all targets (stratified on severity x treatment). In
each outer training fold: :func:`sevrage_ai.models.system.train_all` (inner selection,
cross-fit, conformal, threshold). The outer test fold then goes through the **production
inference path**, ``Predictor.predict_canonical``, so the evaluation covers gating,
conformal, abstention and consistency.

Aggregation
    Each row is predicted once per repeat. Probabilities / predictions are averaged over
    repeats per row; metrics are computed on these averages, with a percentile bootstrap
    CI over rows (all metrics of one resample computed together) and the SD of the metric
    across repeats. Conformal metrics are per-row averages over repeats. The treatment
    decision metrics use the mean of the fold thresholds. Baselines are computed on the
    same rows: "dummy" (class prevalence or median) and the clinical rules of
    ``schema.baselines``. Global importance is the permutation importance on the outer
    test folds of the first repeat, averaged over folds.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.confidence.calibration import reliability_bins
from sevrage_ai.config.schema import BaselineSpec, DatasetSchema
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.errors import DataValidationError
from sevrage_ai.evaluation import baselines, metrics
from sevrage_ai.evaluation.report import EvaluationReport, ReliabilityCurve, TargetEvaluation
from sevrage_ai.explain.global_ import PredictFn, permutation_batched
from sevrage_ai.inference.predictor import Predictor
from sevrage_ai.logging_config import get_logger
from sevrage_ai.models.splits import outer_stratification_key, repeated_splits
from sevrage_ai.models.system import TrainedSystem, train_all
from sevrage_ai.models.tasks import (
    TargetTask,
    build_tasks,
    encode_labels,
    predicted_indices,
    regression_values,
)
from sevrage_ai.models.training import score_candidate
from sevrage_ai.types import (
    NO_MOLECULE,
    REGRESSION_TARGETS,
    TREATED,
    CanonicalData,
    ClassOutput,
    MetricEstimate,
    PatientPrediction,
    RegressionOutput,
    TargetName,
    TaskKind,
)

__all__ = ["evaluate_predictor", "run"]

_LOGGER = get_logger(__name__)
_SEVERE_PREFIX: Final = "severe_vs_rest_"
_RELIABILITY_BINS: Final = 10

Progress = Callable[[str], None]


# --------------------------------------------------------------------------------------
# Collection of outer-fold predictions
# --------------------------------------------------------------------------------------


@dataclass
class _Store:
    """Predictions of every row for every repeat (NaN when not predicted)."""

    n_rows: int
    n_repeats: int
    classes: Mapping[TargetName, tuple[str, ...]]
    proba: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    sets: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    point: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    lower: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    upper: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    treat: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    final_molecule: npt.NDArray[np.object_] = field(default_factory=lambda: np.zeros(0, object))
    final_values: dict[TargetName, npt.NDArray[np.float64]] = field(default_factory=dict)
    thresholds: list[float] = field(default_factory=list)
    candidates: dict[TargetName, list[str]] = field(default_factory=lambda: defaultdict(list))
    importance: dict[TargetName, list[dict[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def init(self, targets: Sequence[TargetName]) -> None:
        shape = (self.n_repeats, self.n_rows)
        for name in targets:
            if name in self.classes:
                width = len(self.classes[name])
                self.proba[name] = np.full((*shape, width), np.nan)
                self.sets[name] = np.full((*shape, width), np.nan)
            else:
                self.point[name] = np.full(shape, np.nan)
                self.lower[name] = np.full(shape, np.nan)
                self.upper[name] = np.full(shape, np.nan)
                self.final_values[name] = np.full(shape, np.nan)
        self.treat = np.full(shape, np.nan)
        self.final_molecule = np.full(shape, None, dtype=object)

    def record(self, repeat: int, row: int, prediction: PatientPrediction) -> None:
        outputs: dict[TargetName, ClassOutput | RegressionOutput | None] = {
            TargetName.SEVERITY: prediction.severity,
            TargetName.TREATMENT: prediction.treatment,
            TargetName.MOLECULE: prediction.molecule_if_treated,
            TargetName.DOSE_DAY1: prediction.dose_day1_if_treated,
            TargetName.DOSE_CUMULATIVE: prediction.dose_cumulative_if_treated,
            TargetName.DURATION: prediction.duration_if_treated,
        }
        for name, output in outputs.items():
            if isinstance(output, ClassOutput) and name in self.proba:
                classes = self.classes[name]
                values = np.array([output.probabilities.get(c, 0.0) for c in classes])
                total = values.sum()
                self.proba[name][repeat, row] = values / total if total > 0 else values
                self.sets[name][repeat, row] = [c in output.prediction_set for c in classes]
            elif isinstance(output, RegressionOutput) and name in self.point:
                self.point[name][repeat, row] = output.point
                self.lower[name][repeat, row] = output.lower
                self.upper[name][repeat, row] = output.upper
        final = prediction.final
        self.treat[repeat, row] = float(final.treat)
        self.final_molecule[repeat, row] = final.molecule
        for name, value in (
            (TargetName.DOSE_DAY1, final.dose_day1_dze),
            (TargetName.DOSE_CUMULATIVE, final.dose_cumulative_dze),
            (
                TargetName.DURATION,
                None if final.duration_days is None else float(final.duration_days),
            ),
        ):
            if name in self.final_values and value is not None:
                self.final_values[name][repeat, row] = value


def _fold_importance(
    system: TrainedSystem,
    test: CanonicalData,
    schema: DatasetSchema,
    cfg: TrainingConfig,
) -> dict[TargetName, dict[str, float]]:
    """Permutation importance of every non-constant target on one outer test fold."""
    if test.y is None:
        return {}
    result: dict[TargetName, dict[str, float]] = {}
    tasks = {t.name: t for t in build_tasks(schema, test.y)}
    names = [f.name for f in system.features]
    for name, fitted in system.targets.items():
        task = tasks.get(name)
        fold_task = fitted.task
        if task is None or fold_task.is_constant or task.n_rows < 2:
            continue
        X = test.X.loc[task.row_mask, names].reset_index(drop=True)
        labels = test.y[name.value][task.row_mask]
        if fold_task.classes is not None:
            index = {c: i for i, c in enumerate(fold_task.classes)}
            merged = [fold_task.merged_classes.get(str(v), str(v)) for v in labels.tolist()]
            keep = np.array([m in index for m in merged], dtype=bool)
            if keep.sum() < 2:
                continue
            X = X.loc[keep].reset_index(drop=True)
            y_true: npt.NDArray[np.float64] | npt.NDArray[np.intp] = np.array(
                [index[m] for m, k in zip(merged, keep, strict=True) if k], dtype=np.intp
            )

            predict_fn: PredictFn = fitted.predict_proba

        else:
            y_true = labels.to_numpy(dtype=np.float64)

            predict_fn = fitted.predict

        def score_fn(
            predictions: npt.NDArray[Any],
            _task: TargetTask = fold_task,
            _y: npt.NDArray[np.intp] | npt.NDArray[np.float64] = y_true,
        ) -> float:
            return score_candidate(_task, cfg, _y, predictions)

        # One prediction call per target and fold (the permuted copies are stacked).
        result[name] = permutation_batched(
            predict_fn,
            score_fn,
            X,
            list(system.features),
            n_repeats=cfg.explain.permutation_repeats,
            seed=cfg.seed,
        )
    return result


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def _resample(
    n: int, rng: np.random.Generator, strata: npt.NDArray[np.intp] | None
) -> npt.NDArray[np.intp]:
    if strata is None:
        return rng.integers(0, n, size=n).astype(np.intp)
    parts = [
        rng.choice(np.flatnonzero(strata == s), size=int((strata == s).sum()), replace=True)
        for s in np.unique(strata)
    ]
    return np.concatenate(parts).astype(np.intp)


def _bootstrap(
    fn: Callable[[npt.NDArray[np.intp]], Mapping[str, float]],
    n: int,
    cfg: TrainingConfig,
    strata: npt.NDArray[np.intp] | None = None,
) -> dict[str, tuple[float, float]]:
    """Percentile CI of every metric returned by ``fn`` (one call per resample)."""
    rng = np.random.default_rng(cfg.seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.bootstrap.n_resamples):
        for key, value in fn(_resample(n, rng, strata)).items():
            if math.isfinite(value):
                samples[key].append(value)
    alpha = (1.0 - cfg.bootstrap.ci) / 2.0
    return {
        key: (float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha)))
        for key, values in samples.items()
    }


def _estimates(
    values: Mapping[str, float],
    cis: Mapping[str, tuple[float, float]],
    per_repeat: Sequence[Mapping[str, float]],
    n: int,
    cfg: TrainingConfig,
) -> dict[str, MetricEstimate]:
    out: dict[str, MetricEstimate] = {}
    for key, value in values.items():
        lo, hi = cis.get(key, (math.nan, math.nan))
        repeats = [r[key] for r in per_repeat if key in r and math.isfinite(r[key])]
        sd = float(np.std(repeats, ddof=1)) if len(repeats) > 1 else None
        if math.isfinite(value) and math.isfinite(lo) and math.isfinite(hi):
            lo, hi = min(lo, value), max(hi, value)
        out[key] = MetricEstimate(
            value=float(value), lo=lo, hi=hi, sd_repeats=sd, level=cfg.bootstrap.ci, n=n
        )
    return out


def _labels(data: CanonicalData) -> pd.DataFrame:
    if data.y is None:
        msg = "Évaluation impossible : données sans cibles."
        raise DataValidationError(msg)
    return data.y


def _mean_over_repeats(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-wise mean over the repeat axis (axis 0), ignoring NaN (all-NaN rows stay NaN)."""
    counts = np.sum(~np.isnan(values), axis=0)
    totals = np.nansum(values, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: npt.NDArray[np.float64] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return mean


# --------------------------------------------------------------------------------------
# Per-target evaluation
# --------------------------------------------------------------------------------------


def _class_rows(
    task: TargetTask, store: _Store, y: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Positions (in the data) of the task rows that were predicted, and their labels."""
    rows = np.flatnonzero(task.row_mask)
    predicted = ~np.isnan(store.proba[task.name][:, rows, 0]).all(axis=0)
    return rows[predicted], y[predicted]


def _evaluate_class_target(
    task: TargetTask,
    data: CanonicalData,
    store: _Store,
    schema: DatasetSchema,
    cfg: TrainingConfig,
) -> TargetEvaluation:
    classes = task.classes or ()
    y_all = encode_labels(task, _labels(data)[task.name.value])
    rows, y = _class_rows(task, store, y_all)
    proba_rep = store.proba[task.name][:, rows, :]
    proba = _mean_over_repeats(proba_rep)
    ordinal = task.kind is TaskKind.ORDINAL
    threshold = float(np.mean(store.thresholds)) if task.name is TargetName.TREATMENT else None

    prior_corrected = task.name in cfg.decision.prior_corrected

    def decide(p: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        return predicted_indices(p, task, threshold=threshold, prior_corrected=prior_corrected)

    def compute(p: npt.NDArray[np.float64], labels: npt.NDArray[np.intp]) -> dict[str, float]:
        predicted = decide(p)
        result = metrics.classification(
            labels, p, classes, threshold=threshold, ordinal=ordinal, predicted=predicted
        )
        result.update(_severe_vs_rest(labels, predicted, classes, schema, task.name))
        return result

    values = compute(proba, y)
    per_repeat = [
        compute(rep[~np.isnan(rep).any(axis=1)], y[~np.isnan(rep).any(axis=1)])
        for rep in proba_rep
        if (~np.isnan(rep).any(axis=1)).sum() > 1
    ]
    cis = _bootstrap(lambda idx: compute(proba[idx], y[idx]), len(y), cfg, strata=y)
    sets_rep = store.sets[task.name][:, rows, :]
    covered = np.take_along_axis(sets_rep, y[np.newaxis, :, np.newaxis], axis=2)[:, :, 0]
    sizes = sets_rep.sum(axis=2)  # NaN for repeats where the row was not predicted
    per_row = {
        "coverage": _mean_over_repeats(covered),
        "mean_set_size": _mean_over_repeats(sizes),
        "singleton_rate": _mean_over_repeats(np.where(np.isnan(sizes), np.nan, sizes == 1.0)),
    }
    conformal = _row_average_estimates(per_row, cfg)
    predicted = decide(proba)
    confusion = tuple(
        tuple(int(((y == i) & (predicted == j)).sum()) for j in range(len(classes)))
        for i in range(len(classes))
    )
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    warnings = [
        f"Classe « {c} » : seulement {k} cas (< {cfg.evaluation.min_class_count_warning})."
        for c, k in counts.items()
        if k < cfg.evaluation.min_class_count_warning
    ]
    return TargetEvaluation(
        target=task.name,
        kind=task.kind,
        n=len(y),
        conditional=schema.enabled_targets()[task.name].conditional_on is not None,
        classes=classes,
        class_counts=counts,
        metrics=_estimates(values, cis, per_repeat, len(y), cfg),
        baselines=_class_baselines(task, data, rows, y, classes, schema, cfg),
        conformal=conformal,
        selected_candidates=tuple(store.candidates[task.name]),
        confusion_matrix=confusion,
        reliability=_reliability(y, proba, classes),
        global_importance=_mean_importance(store.importance[task.name]),
        warnings=tuple(warnings),
    )


def _severe_vs_rest(
    y: npt.NDArray[np.intp],
    predicted: npt.NDArray[np.intp],
    classes: Sequence[str],
    schema: DatasetSchema,
    name: TargetName,
) -> dict[str, float]:
    """Model metrics comparable with the "most severe class vs rest" clinical rules."""
    if name is not TargetName.SEVERITY or len(y) == 0:
        return {}
    high = classes.index(schema.high_severity_class())
    return _binary_rates(y == high, predicted == high, prefix=_SEVERE_PREFIX)


def _binary_rates(
    truth: npt.NDArray[np.bool_], predicted: npt.NDArray[np.bool_], *, prefix: str = ""
) -> dict[str, float]:
    tp = float((truth & predicted).sum())
    tn = float((~truth & ~predicted).sum())
    fp = float((~truth & predicted).sum())
    fn = float((truth & ~predicted).sum())

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else math.nan

    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        f"{prefix}sensitivity": sensitivity,
        f"{prefix}specificity": specificity,
        f"{prefix}ppv": ratio(tp, tp + fp),
        f"{prefix}npv": ratio(tn, tn + fn),
        f"{prefix}balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def _class_baselines(
    task: TargetTask,
    data: CanonicalData,
    rows: npt.NDArray[np.intp],
    y: npt.NDArray[np.intp],
    classes: Sequence[str],
    schema: DatasetSchema,
    cfg: TrainingConfig,
) -> dict[str, dict[str, MetricEstimate]]:
    out: dict[str, dict[str, MetricEstimate]] = {}
    prevalence = baselines.dummy_class_proba(y, len(classes), len(y))

    def dummy(idx: npt.NDArray[np.intp]) -> dict[str, float]:
        result = metrics.classification(y[idx], prevalence[idx], classes)
        majority = np.argmax(prevalence[idx], axis=1).astype(np.intp)
        result.update(_severe_vs_rest(y[idx], majority, classes, schema, task.name))
        return result

    values = dummy(np.arange(len(y)))
    out["dummy"] = _estimates(values, _bootstrap(dummy, len(y), cfg, strata=y), [], len(y), cfg)
    X = data.X.iloc[rows].reset_index(drop=True)
    for spec in schema.baselines:
        if spec.target is not task.name:
            continue
        positive_label = spec.positive_class or schema.targets[spec.target].positive_class
        if positive_label not in classes:
            continue
        truth = y == classes.index(positive_label)
        rule = _rule_estimates(spec, X, truth, cfg)
        if len(classes) > 2:
            # A rule is a "this class vs the rest" decision: its metrics line up with the
            # model's ``severe_vs_rest_*`` metrics, never with the multiclass ones.
            rule = {f"{_SEVERE_PREFIX}{key}": value for key, value in rule.items()}
        out[spec.name] = rule
    return out


def _rule_estimates(
    spec: BaselineSpec, X: pd.DataFrame, truth: npt.NDArray[np.bool_], cfg: TrainingConfig
) -> dict[str, MetricEstimate]:
    def compute(idx: npt.NDArray[np.intp]) -> dict[str, float]:
        return baselines.evaluate_rule(spec, X.iloc[idx].reset_index(drop=True), truth[idx])

    n = len(truth)
    values = compute(np.arange(n))
    return _estimates(values, _bootstrap(compute, n, cfg), [], n, cfg)


def _reliability(
    y: npt.NDArray[np.intp], proba: npt.NDArray[np.float64], classes: Sequence[str]
) -> tuple[ReliabilityCurve, ...]:
    labels = [classes[1]] if len(classes) == 2 else list(classes)
    curves = []
    for label in labels:
        index = classes.index(label)
        bins = reliability_bins(
            (y == index).astype(np.intp), proba[:, index], n_bins=_RELIABILITY_BINS
        )
        curves.append(
            ReliabilityCurve(
                label=label,
                mean_predicted=tuple(float(v) for v in bins.mean_predicted),
                observed_frequency=tuple(float(v) for v in bins.observed_frequency),
                counts=tuple(int(v) for v in bins.counts),
            )
        )
    return tuple(curves)


def _mean_importance(folds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not folds:
        return {}
    keys = list(folds[0])
    return {k: float(np.mean([f.get(k, 0.0) for f in folds])) for k in keys}


def _row_average_estimates(
    per_row: Mapping[str, npt.NDArray[np.float64]], cfg: TrainingConfig
) -> dict[str, MetricEstimate]:
    cleaned = {k: v[~np.isnan(v)] for k, v in per_row.items()}
    n = min((v.size for v in cleaned.values()), default=0)
    values = {k: float(v.mean()) if v.size else math.nan for k, v in cleaned.items()}

    def compute(idx: npt.NDArray[np.intp]) -> dict[str, float]:
        return {k: float(v[idx].mean()) for k, v in cleaned.items() if v.size == n}

    cis = _bootstrap(compute, n, cfg) if n else {}
    return _estimates(values, cis, [], n, cfg)


def _evaluate_regression_target(
    task: TargetTask,
    data: CanonicalData,
    store: _Store,
    schema: DatasetSchema,
    cfg: TrainingConfig,
) -> TargetEvaluation:
    y_all = regression_values(task, _labels(data)[task.name.value])
    rows = np.flatnonzero(task.row_mask)
    point_rep = store.point[task.name][:, rows]
    predicted = ~np.isnan(point_rep).all(axis=0)
    rows, y = rows[predicted], y_all[predicted]
    point_rep = point_rep[:, predicted]
    point = _mean_over_repeats(point_rep)
    values = metrics.regression(y, point)
    per_repeat = [
        metrics.regression(y[~np.isnan(rep)], rep[~np.isnan(rep)])
        for rep in point_rep
        if (~np.isnan(rep)).sum() > 1
    ]
    cis = _bootstrap(lambda idx: metrics.regression(y[idx], point[idx]), len(y), cfg)
    lower_rep = store.lower[task.name][:, rows]
    upper_rep = store.upper[task.name][:, rows]
    inside = ((lower_rep <= y) & (y <= upper_rep)).astype(np.float64)
    inside[np.isnan(lower_rep)] = np.nan
    width = upper_rep - lower_rep
    per_row = {
        "coverage": _mean_over_repeats(inside),
        "mean_width": _mean_over_repeats(np.where(np.isfinite(width), width, np.nan)),
    }
    conformal = _row_average_estimates(per_row, cfg)
    row_width = per_row["mean_width"]
    finite_width = row_width[np.isfinite(row_width)]
    median_width = {
        "median_width": float(np.median(finite_width)) if finite_width.size else math.nan
    }
    median_cis = (
        _bootstrap(
            lambda idx: {"median_width": float(np.median(finite_width[idx]))},
            finite_width.size,
            cfg,
        )
        if finite_width.size
        else {}
    )
    conformal |= _estimates(median_width, median_cis, [], int(finite_width.size), cfg)
    median = baselines.dummy_regression(y, len(y))
    dummy_values = metrics.regression(y, median)
    dummy_cis = _bootstrap(lambda idx: metrics.regression(y[idx], median[idx]), len(y), cfg)
    return TargetEvaluation(
        target=task.name,
        kind=task.kind,
        n=len(y),
        conditional=schema.enabled_targets()[task.name].conditional_on is not None,
        metrics=_estimates(values, cis, per_repeat, len(y), cfg),
        baselines={"dummy": _estimates(dummy_values, dummy_cis, [], len(y), cfg)},
        conformal=conformal,
        selected_candidates=tuple(store.candidates[task.name]),
        global_importance=_mean_importance(store.importance[task.name]),
    )


# --------------------------------------------------------------------------------------
# End-to-end metrics
# --------------------------------------------------------------------------------------


def _end_to_end(
    data: CanonicalData,
    tasks: Mapping[TargetName, TargetTask],
    store: _Store,
    cfg: TrainingConfig,
) -> dict[str, MetricEstimate]:
    y_frame = _labels(data)
    out: dict[str, MetricEstimate] = {}
    treatment = y_frame[TargetName.TREATMENT.value]
    known = treatment.notna().to_numpy(dtype=bool)
    treated = (treatment == TREATED).fillna(value=False).to_numpy(dtype=bool)
    molecule_task = tasks.get(TargetName.MOLECULE)
    if molecule_task is not None:
        truth = np.full(data.n_rows, None, dtype=object)
        truth[known & ~treated] = NO_MOLECULE
        labels = y_frame[TargetName.MOLECULE.value].tolist()
        for i in np.flatnonzero(treated):
            if isinstance(labels[i], str):
                truth[i] = molecule_task.merged_classes.get(labels[i], labels[i])
        rows = np.array([t is not None for t in truth]) & ~pd.isna(store.final_molecule).all(axis=0)
        correct = np.full((store.n_repeats, data.n_rows), np.nan)
        for r in range(store.n_repeats):
            for i in np.flatnonzero(rows):
                if store.final_molecule[r, i] is not None:
                    correct[r, i] = float(store.final_molecule[r, i] == truth[i])
        estimate = _row_average_estimates(
            {"accuracy_with_none": _mean_over_repeats(correct[:, rows])}, cfg
        )
        out["molecule.accuracy_with_none"] = estimate["accuracy_with_none"]
    for name in REGRESSION_TARGETS:
        if name not in store.final_values or name.value not in y_frame:
            continue
        values = y_frame[name.value].to_numpy(dtype=np.float64)
        truth_values = np.where(known & ~treated, 0.0, values)
        rows = known & ~np.isnan(truth_values)
        error = np.abs(store.final_values[name][:, rows] - truth_values[rows])
        estimate = _row_average_estimates({"mae_end_to_end": _mean_over_repeats(error)}, cfg)
        out[f"{name.value}.mae_end_to_end"] = estimate["mae_end_to_end"]
    return out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def _report(
    data: CanonicalData,
    schema: DatasetSchema,
    cfg: TrainingConfig,
    store: _Store,
    *,
    protocol_fr: str,
    outer_splits: int,
    outer_repeats: int,
    model_tasks: Mapping[TargetName, TargetTask] | None = None,
    warnings: Sequence[str] = (),
) -> EvaluationReport:
    tasks = _tasks(data, schema, model_tasks)
    evaluations: dict[TargetName, TargetEvaluation] = {}
    for name, task in tasks.items():
        if task.is_constant:
            continue
        if task.classes is not None:
            evaluations[name] = _evaluate_class_target(task, data, store, schema, cfg)
        else:
            evaluations[name] = _evaluate_regression_target(task, data, store, schema, cfg)
    return EvaluationReport(
        created_at=dt.datetime.now(dt.UTC),
        is_synthetic=data.is_synthetic,
        n_rows=data.n_rows,
        feature_set=cfg.feature_set,
        protocol_fr=protocol_fr,
        seed=cfg.seed,
        outer_splits=outer_splits,
        outer_repeats=outer_repeats,
        inner_splits=cfg.cv.inner_splits,
        crossfit_splits=cfg.cv.crossfit_splits,
        alpha=cfg.conformal.alpha,
        targets=evaluations,
        end_to_end=_end_to_end(data, tasks, store, cfg),
        warnings=tuple(warnings),
    )


def _tasks(
    data: CanonicalData,
    schema: DatasetSchema,
    model_tasks: Mapping[TargetName, TargetTask] | None,
) -> dict[TargetName, TargetTask]:
    """Evaluation tasks: those of ``data``, or aligned on a trained model's classes (rows
    whose label the model cannot predict are left out)."""
    y = _labels(data)
    tasks = {t.name: t for t in build_tasks(schema, y)}
    if model_tasks is None:
        return tasks
    aligned: dict[TargetName, TargetTask] = {}
    for name, task in tasks.items():
        model = model_tasks.get(name)
        if model is None:
            continue
        if model.classes is None or task.classes is None:
            aligned[name] = task
            continue
        merged = [model.merged_classes.get(str(v), str(v)) for v in y[name.value].tolist()]
        known = np.array([m in model.classes for m in merged], dtype=bool)
        aligned[name] = replace(
            task,
            classes=model.classes,
            merged_classes=model.merged_classes,
            row_mask=task.row_mask & known,
            constant_class=model.constant_class,
        )
    return aligned


def _new_store(
    data: CanonicalData,
    schema: DatasetSchema,
    repeats: int,
    model_tasks: Mapping[TargetName, TargetTask] | None = None,
) -> _Store:
    tasks = _tasks(data, schema, model_tasks)
    classes = {n: t.classes for n, t in tasks.items() if t.classes is not None}
    store = _Store(n_rows=data.n_rows, n_repeats=repeats, classes=classes)
    store.init(list(tasks))
    return store


def run(
    data: CanonicalData,
    schema: DatasetSchema,
    cfg: TrainingConfig,
    *,
    progress: Progress | None = None,
) -> EvaluationReport:
    """Evaluate the system on ``data`` (labelled) and return the full report.

    Raises:
        DataValidationError: ``data`` has no labels, or an outer training fold cannot be
            trained (too few labelled rows).
    """
    if data.y is None:
        msg = "Évaluation impossible : données sans cibles."
        raise DataValidationError(msg)
    key = outer_stratification_key(data.y, schema, min_cell=cfg.cv.outer_splits)
    groups = None if data.groups is None else data.groups.to_numpy(dtype=str)
    repeats = cfg.cv.outer_repeats
    splits = repeated_splits(key, groups, cfg.cv.outer_splits, repeats, cfg.seed)
    per_repeat = len(splits) // repeats
    store = _new_store(data, schema, repeats)
    for index, (train, test) in enumerate(splits):
        repeat = index // per_repeat
        message = f"CV externe : pli {index + 1}/{len(splits)} (répétition {repeat + 1})."
        _LOGGER.info(message)
        if progress is not None:
            progress(message)
        system = train_all(data.subset(train), schema, cfg)
        predictor = Predictor.from_system(system, schema, cfg)
        test_data = data.subset(test)
        result = predictor.predict_canonical(test_data, explain=False)
        position = {row_id: int(test[i]) for i, row_id in enumerate(test_data.ids.tolist())}
        for prediction in result.predictions:
            store.record(repeat, position[prediction.row_id], prediction)
        treatment = system.targets.get(TargetName.TREATMENT)
        if treatment is not None and treatment.threshold is not None:
            store.thresholds.append(treatment.threshold)
        for name, fitted in system.targets.items():
            store.candidates[name].append(fitted.candidate.name)
        if repeat == 0:
            for name, values in _fold_importance(system, test_data, schema, cfg).items():
                store.importance[name].append(values)
    protocol = (
        f"CV imbriquée répétée {per_repeat}x{repeats} "
        f"(sélection {cfg.cv.inner_splits} plis, cross-fit {cfg.cv.crossfit_splits} plis), "
        f"n={data.n_rows}"
    )
    return _report(
        data,
        schema,
        cfg,
        store,
        protocol_fr=protocol,
        outer_splits=per_repeat,
        outer_repeats=repeats,
    )


def evaluate_predictor(
    predictor: Predictor, data: CanonicalData, *, protocol_fr: str | None = None
) -> EvaluationReport:
    """External (or temporal) validation of a trained predictor on new labelled data.

    Raises:
        DataValidationError: ``data`` has no labels.
    """
    if data.y is None:
        msg = "Évaluation impossible : données sans cibles."
        raise DataValidationError(msg)
    schema, cfg = predictor.schema, predictor.training_config
    model_tasks = {name: fitted.task for name, fitted in predictor.system.targets.items()}
    store = _new_store(data, schema, 1, model_tasks)
    result = predictor.predict_canonical(data, explain=False)
    position = {row_id: i for i, row_id in enumerate(data.ids.tolist())}
    for prediction in result.predictions:
        store.record(0, position[prediction.row_id], prediction)
    treatment = predictor.system.targets.get(TargetName.TREATMENT)
    if treatment is not None and treatment.threshold is not None:
        store.thresholds.append(treatment.threshold)
    for name, fitted in predictor.system.targets.items():
        store.candidates[name].append(fitted.candidate.name)
    protocol = (
        protocol_fr or f"Validation externe du modèle {predictor.info.model_id}, n={data.n_rows}"
    )
    return _report(
        data,
        schema,
        cfg,
        store,
        protocol_fr=protocol,
        outer_splits=0,
        outer_repeats=1,
        model_tasks=model_tasks,
    )
