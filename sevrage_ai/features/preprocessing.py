"""Preprocessing inside the model pipelines (never fitted outside a CV fold). Owner: WS4.

``build_preprocessor`` returns a ``ColumnTransformer`` that selects the given features by
canonical name from ``CanonicalData.X`` (other columns are dropped), with
``verbose_feature_names_out=False`` and ``set_output(transform="pandas")``:

* ``linear`` family: numeric -> median imputation (``add_indicator=True``) +
  ``StandardScaler``; binary -> most-frequent imputation with indicator; categorical ->
  most-frequent imputation + ``OneHotEncoder(handle_unknown="infrequent_if_exist",
  min_frequency=5, sparse_output=False)``.
* ``tree`` family: numeric -> median imputation with indicator (no scaling); binary as
  above; categorical -> ``OrdinalEncoder`` (unknown and missing encoded as ``NaN``, which
  HGB treats as missing), used with the HGB categorical mask.

Imputers keep features that are entirely missing in a training fold
(``keep_empty_features=True``): small folds of sparse variables must not change the
output columns. Categorical columns arrive as ``StringDtype("python")`` with ``pd.NA``,
which scikit-learn cannot impute: they are converted to ``object`` with ``np.nan`` first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from sevrage_ai.config.schema import FeatureSpec
from sevrage_ai.types import FeatureKind, ModelFamily

__all__ = [
    "build_preprocessor",
    "categorical_feature_names",
    "monotonic_constraints",
    "to_object_frame",
]


def to_object_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as ``object`` with ``np.nan`` for missing (picklable helper)."""
    converted = frame.astype(object)
    return converted.where(frame.notna(), np.nan)


def _by_kind(features: Sequence[FeatureSpec], kind: FeatureKind) -> list[str]:
    return [f.name for f in features if f.kind is kind]


def build_preprocessor(features: Sequence[FeatureSpec], family: ModelFamily) -> ColumnTransformer:
    """Unfitted preprocessor for ``features`` (canonical admission features) and ``family``."""
    numeric = _by_kind(features, FeatureKind.NUMERIC)
    binary = _by_kind(features, FeatureKind.BINARY)
    categorical = _by_kind(features, FeatureKind.CATEGORICAL)
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        steps: list[tuple[str, object]] = [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            )
        ]
        if family is ModelFamily.LINEAR:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if binary:
        imputer = SimpleImputer(
            strategy="most_frequent", add_indicator=True, keep_empty_features=True
        )
        transformers.append(("binary", imputer, binary))
    if categorical:
        to_object = FunctionTransformer(to_object_frame, feature_names_out="one-to-one")
        encoder: object
        if family is ModelFamily.LINEAR:
            encoder = Pipeline(
                [
                    ("to_object", to_object),
                    (
                        "impute",
                        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="infrequent_if_exist",
                            min_frequency=5,
                            sparse_output=False,
                        ),
                    ),
                ]
            )
        else:
            encoder = Pipeline(
                [
                    ("to_object", to_object),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                            encoded_missing_value=np.nan,
                        ),
                    ),
                ]
            )
        transformers.append(("categorical", encoder, categorical))
    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def categorical_feature_names(features: Sequence[FeatureSpec]) -> list[str]:
    """Output column names of the tree preprocessor that HGB must treat as categorical."""
    return _by_kind(features, FeatureKind.CATEGORICAL)


def monotonic_constraints(features: Sequence[FeatureSpec]) -> dict[str, int]:
    """HGB ``monotonic_cst`` keyed by output column name (only non-zero ``monotonic``)."""
    return {f.name: int(f.monotonic) for f in features if f.monotonic != 0}
