"""Predictor: the production inference path, implementing ``PredictorProtocol``. Owner: WS8b.

``predict_frame`` runs, in order:

1. anonymization check (raw columns only; failure raises ``AnonymizationError``);
2. ``to_canonical`` (or ``from_canonical_features``);
3. per-row validation (invalid rows go to ``rejected``);
4. OOD detection;
5. each target model;
6. conformal sets and intervals;
7. abstention (confidence levels);
8. ``consistency.apply``;
9. ``safety_rules.evaluate``;
10. explanations.

``predict_canonical`` starts at step 3 and is what the nested CV calls on outer test folds.
Model outputs are computed for the whole batch at once (vectorised over rows); only the
per-patient assembly and the explanations loop over rows.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import ValidationError

from sevrage_ai import __version__
from sevrage_ai.confidence import abstention
from sevrage_ai.confidence.conformal import CrossConformalClassifier, CVPlusRegressor
from sevrage_ai.confidence.ood import detect, missing_fraction
from sevrage_ai.config.schema import DatasetSchema, FeatureSpec
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.data.anonymization import check, ensure_anonymized
from sevrage_ai.data.mapping import MappingReport, from_canonical_features, to_canonical
from sevrage_ai.data.validation import build_record_model, validate
from sevrage_ai.errors import DataValidationError
from sevrage_ai.evaluation.report import EvaluationReport
from sevrage_ai.explain.local import occlusion
from sevrage_ai.inference import consistency, safety_rules
from sevrage_ai.models.system import TrainedSystem
from sevrage_ai.models.tasks import predicted_indices
from sevrage_ai.models.training import FittedTarget, original_units
from sevrage_ai.persistence.bundle import load_bundle
from sevrage_ai.types import (
    CLASSIFICATION_TARGETS,
    BatchResult,
    CanonicalData,
    ClassOutput,
    Contribution,
    Flag,
    ModelInfo,
    PatientPrediction,
    RegressionOutput,
    TargetName,
)

__all__ = ["DATA_VALUE_SET_MISSING", "Predictor", "TargetBatch"]

DATA_VALUE_SET_MISSING: Final = "data.value_set_missing"
_RECORD_ID: Final = "patient"


@dataclass(frozen=True, slots=True, eq=False)
class TargetBatch:
    """Raw model outputs of one target for a batch (before per-patient assembly).

    Classification: ``proba`` ``(n, C)`` and ``sets`` ``(n, C)``. Regression: ``point``,
    ``lower``, ``upper`` ``(n,)`` in original units (``lower >= 0``, ``lower <= point <=
    upper``).
    """

    fitted: FittedTarget
    proba: npt.NDArray[np.float64] | None = None
    sets: npt.NDArray[np.bool_] | None = None
    point: npt.NDArray[np.float64] | None = None
    lower: npt.NDArray[np.float64] | None = None
    upper: npt.NDArray[np.float64] | None = None


def _target_batch(fitted: FittedTarget, X: pd.DataFrame) -> TargetBatch:
    folds = fitted.fold_predictions(X)
    if fitted.task.classes is not None:
        proba = np.clip(folds.mean(axis=0), 0.0, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        if isinstance(fitted.conformal, CrossConformalClassifier):
            sets = fitted.conformal.predict_set(folds)
        else:  # constant task: the single class is certain
            sets = np.ones_like(proba, dtype=bool)
        return TargetBatch(fitted=fitted, proba=proba, sets=sets)
    mean = folds.mean(axis=0)
    point = np.maximum(original_units(fitted.task, mean), 0.0)
    if isinstance(fitted.conformal, CVPlusRegressor):
        low_model, high_model = fitted.conformal.predict_interval(folds)
        lower = np.maximum(original_units(fitted.task, low_model), 0.0)
        upper = original_units(fitted.task, high_model)
    else:  # pragma: no cover - regression tasks always have a conformal model
        lower, upper = point.copy(), point.copy()
    lower = np.where(np.isfinite(lower), lower, 0.0)
    lower = np.minimum(lower, point)
    upper = np.maximum(np.where(np.isnan(upper), np.inf, upper), point)
    return TargetBatch(fitted=fitted, point=point, lower=lower, upper=upper)


@dataclass(frozen=True, slots=True)
class _RowContext:
    """Data completeness of one patient, shared by the confidence of every output."""

    missing_fraction: float
    missing_labels: tuple[str, ...]


class Predictor:
    """Real predictor built from a trained system (in memory or loaded from a bundle)."""

    def __init__(
        self,
        system: TrainedSystem,
        schema: DatasetSchema,
        training_config: TrainingConfig,
        info: ModelInfo,
    ) -> None:
        self._system = system
        self._schema = schema
        self._cfg = training_config
        self._info = info
        self._features: tuple[FeatureSpec, ...] = system.features
        self._record_model = build_record_model(schema, "predict")
        self._feature_labels = {f.name: f.label for f in schema.features}

    @classmethod
    def from_bundle(cls, path: Path) -> Self:
        """Load and verify a bundle (``persistence.load_bundle``), then wrap it."""
        bundle = load_bundle(path)
        return cls(bundle.system, bundle.schema, bundle.training_config, bundle.info())

    @classmethod
    def from_system(
        cls,
        system: TrainedSystem,
        schema: DatasetSchema,
        training_config: TrainingConfig,
        *,
        report: EvaluationReport | None = None,
        model_id: str = "en-memoire",
    ) -> Self:
        """Wrap an in-memory trained system (demo model of the app, nested CV, tests)."""
        treatment = system.targets.get(TargetName.TREATMENT)
        threshold = 0.5 if treatment is None or treatment.threshold is None else treatment.threshold
        info = ModelInfo(
            model_id=model_id,
            created_at=dt.datetime.now(dt.UTC),
            is_synthetic=system.is_synthetic,
            feature_set=system.feature_set,
            feature_names=tuple(f.name for f in system.features),
            targets=tuple(system.targets),
            n_train=system.n_train,
            schema_hash=system.schema_hash,
            package_version=__version__,
            alpha=training_config.conformal.alpha,
            treatment_threshold=float(threshold),
            grey_zone=training_config.treatment_threshold.grey_zone,
            dose_equivalence_to_diazepam=dict(schema.dose_equivalence_to_diazepam),
            metrics_summary={} if report is None else report.headline_metrics(),
            evaluation_protocol=None if report is None else report.protocol_fr,
            bundle_path=None,
        )
        return cls(system, schema, training_config, info)

    @property
    def info(self) -> ModelInfo:
        """Model summary."""
        return self._info

    @property
    def schema(self) -> DatasetSchema:
        """Schema the model was trained with."""
        return self._schema

    @property
    def system(self) -> TrainedSystem:
        """The trained system (fitted targets, references)."""
        return self._system

    @property
    def training_config(self) -> TrainingConfig:
        """Training profile of the model (thresholds used at inference)."""
        return self._cfg

    # ------------------------------------------------------------------ public API

    def predict_frame(
        self, df: pd.DataFrame, *, raw_columns: bool = True, explain: bool = False
    ) -> BatchResult:
        """See :meth:`sevrage_ai.types.PredictorProtocol.predict_frame`."""
        if raw_columns:
            exempt = [
                c for c in (self._schema.dataset.id_column, self._schema.dataset.group_column) if c
            ]
            ensure_anonymized(check(df, self._schema.anonymization, exempt_columns=exempt))
            mapped = to_canonical(df, self._schema, with_targets=False)
        else:
            mapped = from_canonical_features(df, self._schema)
        return self.predict_canonical(mapped.data, explain=explain, mapping_report=mapped.report)

    def predict_record(self, record: Mapping[str, object]) -> PatientPrediction:
        """See :meth:`sevrage_ai.types.PredictorProtocol.predict_record`."""
        cleaned = {key: _none_if_missing(value) for key, value in record.items()}
        try:
            self._record_model.model_validate(cleaned)
        except ValidationError as exc:
            details = [
                f"{'.'.join(str(p) for p in error['loc']) or '<racine>'} : {error['msg']}"
                for error in exc.errors(include_input=False, include_url=False)
            ]
            msg = "Saisie invalide : corrigez les champs signalés."
            raise DataValidationError(msg, details=details) from exc
        frame = pd.DataFrame([dict(cleaned)])
        mapped = from_canonical_features(frame, self._schema, ids=[_RECORD_ID])
        result = self.predict_canonical(mapped.data, mapping_report=mapped.report)
        if not result.predictions:
            msg = "Saisie invalide : corrigez les champs signalés."
            raise DataValidationError(msg, details=[i.message_fr for i in result.rejected])
        return result.predictions[0]

    def predict_canonical(
        self,
        data: CanonicalData,
        *,
        explain: bool = True,
        mapping_report: MappingReport | None = None,
    ) -> BatchResult:
        """Steps 3-10 on already canonical data (``explain=False`` skips step 10)."""
        report = validate(data, self._schema, "predict")
        rejected_ids = report.rejected_row_ids()
        keep = np.array([row_id not in rejected_ids for row_id in data.ids.tolist()], dtype=bool)
        if not keep.any():
            return BatchResult(predictions=(), rejected=report.row_issues)
        subset = data.subset(np.flatnonzero(keep))
        X = subset.X.loc[:, [f.name for f in self._features]]
        batches = {name: _target_batch(fitted, X) for name, fitted in self._system.targets.items()}
        ood_flags = detect(X, self._system.ood_reference)
        missing = missing_fraction(X, self._features)
        absent = X.isna().to_numpy(dtype=bool)
        data_flags = self._data_flags(subset, mapping_report)
        predictions = [
            self._assemble(
                i,
                subset,
                X,
                batches,
                ood_flags[i] + data_flags[i],
                _RowContext(
                    missing_fraction=float(missing[i]),
                    missing_labels=tuple(
                        self._feature_labels.get(f.name, f.name)
                        for f, is_absent in zip(self._features, absent[i], strict=True)
                        if is_absent
                    ),
                ),
                explain,
            )
            for i in range(subset.n_rows)
        ]
        return BatchResult(predictions=tuple(predictions), rejected=report.row_issues)

    # ------------------------------------------------------------------ assembly

    def _data_flags(
        self, data: CanonicalData, mapping_report: MappingReport | None
    ) -> list[list[Flag]]:
        flags: list[list[Flag]] = [[] for _ in range(data.n_rows)]
        if mapping_report is None or not mapping_report.row_issues:
            return flags
        position = {row_id: i for i, row_id in enumerate(data.ids.tolist())}
        labels = {f.name: f.label for f in self._schema.features}
        for issue in mapping_report.row_issues:
            row = position.get(issue.row_id)
            if row is None or issue.column not in labels:
                continue
            flags[row].append(
                Flag(
                    DATA_VALUE_SET_MISSING,
                    "info",
                    f"« {labels[issue.column]} » : valeur illisible ou hors bornes, "
                    "considérée comme manquante.",
                    issue.column,
                )
            )
        return flags

    def _class_output(
        self,
        name: TargetName,
        batch: TargetBatch,
        row: int,
        flags: Sequence[Flag],
        context: _RowContext,
    ) -> ClassOutput:
        classes = batch.fitted.task.classes
        if classes is None or batch.proba is None or batch.sets is None:
            msg = f"{name.value} is not a classification output"
            raise TypeError(msg)
        proba = batch.proba[row]
        members = batch.sets[row]
        threshold = None
        if name is TargetName.TREATMENT:
            threshold = batch.fitted.threshold if batch.fitted.threshold is not None else 0.5
        index = int(
            predicted_indices(
                proba[np.newaxis, :],
                batch.fitted.task,
                threshold=threshold,
                prior_corrected=name in self._cfg.decision.prior_corrected,
            )[0]
        )
        label = classes[index]
        assessment = abstention.assess_class(
            proba,
            members,
            flags=flags,
            label_index=index,
            missing_fraction=context.missing_fraction,
            missing_labels=context.missing_labels,
            n_features=len(self._features),
            feature_labels=self._feature_labels,
            cfg=self._cfg.abstention,
        )
        return ClassOutput(
            target=name,
            label=label,
            probabilities={c: float(p) for c, p in zip(classes, proba, strict=True)},
            prediction_set=frozenset(c for c, m in zip(classes, members, strict=True) if m),
            alpha=self._cfg.conformal.alpha,
            confidence=assessment.level,
            reasons=assessment.reasons,
            checks=assessment.checks,
        )

    def _regression_output(
        self,
        name: TargetName,
        batch: TargetBatch,
        row: int,
        flags: Sequence[Flag],
        context: _RowContext,
    ) -> RegressionOutput:
        if batch.point is None or batch.lower is None or batch.upper is None:
            msg = f"{name.value} is not a regression output"
            raise TypeError(msg)
        point = float(batch.point[row])
        lower = float(batch.lower[row])
        upper = float(batch.upper[row])
        assessment = abstention.assess_regression(
            point,
            lower,
            upper,
            flags=flags,
            missing_fraction=context.missing_fraction,
            missing_labels=context.missing_labels,
            n_features=len(self._features),
            feature_labels=self._feature_labels,
            cfg=self._cfg.abstention,
        )
        return RegressionOutput(
            target=name,
            point=point,
            lower=lower,
            upper=upper,
            alpha=self._cfg.conformal.alpha,
            unit=batch.fitted.task.unit or "",
            confidence=assessment.level,
            reasons=assessment.reasons,
            checks=assessment.checks,
        )

    def _assemble(
        self,
        row: int,
        data: CanonicalData,
        X: pd.DataFrame,
        batches: Mapping[TargetName, TargetBatch],
        input_flags: list[Flag],
        context: _RowContext,
        explain: bool,
    ) -> PatientPrediction:
        task_flags = [
            flag
            for batch in batches.values()
            for flag in batch.fitted.task.flags
            if flag.code == "model.constant_task"
        ]
        flags = input_flags + task_flags
        outputs: dict[TargetName, ClassOutput | RegressionOutput] = {}
        for name, batch in batches.items():
            if name in CLASSIFICATION_TARGETS:
                outputs[name] = self._class_output(name, batch, row, flags, context)
            else:
                outputs[name] = self._regression_output(name, batch, row, flags, context)
        severity = _as_class(outputs[TargetName.SEVERITY])
        treatment = _as_class(outputs[TargetName.TREATMENT])
        molecule = _as_class(outputs[TargetName.MOLECULE])
        regressions = {n: o for n, o in outputs.items() if isinstance(o, RegressionOutput)}
        treatment_batch = batches[TargetName.TREATMENT]
        threshold = treatment_batch.fitted.threshold
        target_outputs = consistency.TargetOutputs(
            severity=severity,
            treatment=treatment,
            molecule=molecule,
            dose_day1=regressions.get(TargetName.DOSE_DAY1),
            dose_cumulative=regressions.get(TargetName.DOSE_CUMULATIVE),
            duration=regressions.get(TargetName.DURATION),
            treatment_threshold=0.5 if threshold is None else threshold,
        )
        final, consistency_flags = consistency.apply(target_outputs, self._cfg, self._schema)
        x_row = data.X.iloc[[row]].reset_index(drop=True)
        safety_flags = safety_rules.evaluate(x_row, final, self._schema.safety_rules)
        all_flags = tuple(flags + consistency_flags + safety_flags)
        overall = abstention.assess_overall(
            [(self._output_label(name), o.confidence) for name, o in outputs.items()],
            flags=all_flags,
            missing_fraction=context.missing_fraction,
            missing_labels=context.missing_labels,
            n_features=len(self._features),
            feature_labels=self._feature_labels,
            cfg=self._cfg.abstention,
        )
        explanations = (
            self._explain(X.iloc[[row]].reset_index(drop=True), outputs) if explain else {}
        )
        return PatientPrediction(
            row_id=str(data.ids.iloc[row]),
            severity=severity,
            treatment=treatment,
            molecule_if_treated=molecule,
            dose_day1_if_treated=regressions.get(TargetName.DOSE_DAY1),
            dose_cumulative_if_treated=regressions.get(TargetName.DOSE_CUMULATIVE),
            duration_if_treated=regressions.get(TargetName.DURATION),
            final=final,
            flags=all_flags,
            explanations=explanations,
            overall_confidence=overall.level,
            model_id=self._info.model_id,
            trained_on_synthetic=self._info.is_synthetic,
            overall_checks=overall.checks,
        )

    def _output_label(self, name: TargetName) -> str:
        """Label of an output in the overall explanation (conditional ones say so)."""
        spec = self._schema.targets.get(name)
        if spec is None:
            return name.value
        if spec.conditional_on is not None:
            return f"{spec.label} (si traitement)"
        return spec.label

    # ------------------------------------------------------------------ explanations

    def _explain(
        self, x: pd.DataFrame, outputs: Mapping[TargetName, ClassOutput | RegressionOutput]
    ) -> dict[TargetName, tuple[Contribution, ...]]:
        explanations: dict[TargetName, tuple[Contribution, ...]] = {}
        reference = self._system.reference_patient
        top_k = self._cfg.explain.top_k
        for name, fitted in self._system.targets.items():
            if fitted.task.is_constant:
                continue
            output = outputs[name]
            if isinstance(output, ClassOutput):
                classes = fitted.task.classes or ()
                focus = (
                    fitted.task.positive_class
                    if name is TargetName.TREATMENT and fitted.task.positive_class
                    else output.label
                )
                index = classes.index(focus)

                def predict_fn(
                    frame: pd.DataFrame, _f: FittedTarget = fitted, _i: int = index
                ) -> npt.NDArray[np.float64]:
                    result: npt.NDArray[np.float64] = 100.0 * _f.predict_proba(frame)[:, _i]
                    return result

            else:

                def predict_fn(
                    frame: pd.DataFrame, _f: FittedTarget = fitted, _i: int = 0
                ) -> npt.NDArray[np.float64]:
                    del _i
                    return _f.predict(frame)

            explanations[name] = occlusion(predict_fn, x, reference, self._features, top_k=top_k)
        return explanations


def _as_class(output: ClassOutput | RegressionOutput) -> ClassOutput:
    if not isinstance(output, ClassOutput):
        msg = f"{output.target.value} is not a classification output"
        raise TypeError(msg)
    return output


def _none_if_missing(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
