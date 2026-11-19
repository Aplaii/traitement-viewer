"""Synthetic data generator (design section 7.4). Owner: WS2.

Produces a **raw-format** table, as if exported from the hospital, so that the tests
exercise the whole mapping path. Everything is derived from the schema, so the generator
follows any column mapping:

* columns named after the schema's ``source_column`` values (features and target
  sources), the id column, and the ``__synthetic__`` marker column (always ``1``);
* raw encodings obtained by inverting ``value_mapping`` (``Checked``/``Unchecked``,
  ``OUI``/``NON``, ``Masculin``/``Féminin``...), raw units (``scale`` undone), a few
  censored lab values (``"<5"``) and unreadable cells (``"DM"``);
* targets written the way the schema reads them: a class label through the inverse
  ``value_mapping``; free-text prescriptions (``"Valium 250 mg"``) for the
  ``prescription_*`` parsers; ``"7 jours"`` for ``leading_number``; doses in mg of the
  prescribed molecule for ``convert_with: molecule``;
* untreated patients: empty prescription and dose cells.

Latent structural model: a dependence score ``z`` drives consumption, history and labs
(signs in ``_PLAUSIBLE``); a withdrawal propensity ``s`` (correlated with ``z``) drives
the post-admission outcomes (hence the derived severity), the treatment and the doses;
the dependence also lengthens the taper. Oxazepam becomes more likely with age and with
cirrhosis (data fact: oxazepam patients are older, with more cirrhosis). Every target keeps a
signal a model can learn from n ~ 240 rows. Missing values: MCAR ``missing_rate``.

Deterministic: same ``SyntheticSpec`` and schema give identical output, byte for byte once
written with :func:`write_csv`. Each column draws from its own random stream (seed + column
name) and the outcomes from a dedicated one: adding a feature to the schema leaves the other
columns, the treatment and the doses unchanged.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.config.schema import ColumnSource, DatasetSchema, FeatureSpec
from sevrage_ai.types import (
    SYNTHETIC_MARKER_COLUMN,
    TREATED,
    UNTREATED,
    FeatureKind,
    TargetName,
    Timing,
)

__all__ = ["SYNTHETIC_PREFIX", "SyntheticSpec", "default_filename", "generate", "write_csv"]

SYNTHETIC_PREFIX: Final = "SYNTHETIC_"
"""Every synthetic file name starts with this prefix."""

# canonical name -> (mean, sd, loading on z) in canonical units; unknown numeric features
# are drawn around the middle of their valid range.
_PLAUSIBLE: Final[dict[str, tuple[float, float, float]]] = {
    "age": (47.0, 11.0, 0.2),
    "bmi": (24.5, 4.5, -0.2),
    "education_years": (11.0, 3.0, -0.2),
    "daily_alcohol_g": (190.0, 90.0, 0.8),
    "drinking_days_4w": (26.0, 5.0, 0.5),
    "units_per_drinking_day": (18.0, 9.0, 0.7),
    "heavy_drinking_days": (26.0, 6.0, 0.5),
    "abstinence_days": (1.5, 2.0, -0.2),
    "breath_alcohol": (0.6, 0.8, 0.3),
    "age_first_drink": (15.0, 3.0, -0.3),
    "dsm5_criteria": (9.0, 2.0, 0.6),
    "audit_score": (30.0, 6.0, 0.6),
    "prior_withdrawals": (1.5, 2.0, 0.7),
    "fagerstrom": (4.0, 1.5, 0.2),
    "had_anxiety": (10.0, 4.0, 0.3),
    "had_depression": (9.0, 4.0, 0.3),
    "sodium": (139.0, 3.0, -0.3),
    "potassium": (3.9, 0.45, -0.3),
    "chloride": (103.0, 4.0, -0.2),
    "bicarbonate": (25.0, 3.0, 0.0),
    "creatinine": (65.0, 18.0, 0.0),
    "urea": (4.0, 1.5, -0.1),
    "crp": (6.0, 8.0, 0.2),
    "tsh": (1.5, 1.0, 0.0),
    "magnesium": (0.79, 0.1, -0.3),
    "phosphorus": (0.98, 0.2, -0.2),
    "albumin": (41.0, 5.0, -0.3),
    "alp": (100.0, 40.0, 0.3),
    "ggt": (220.0, 250.0, 0.7),
    "asat": (70.0, 60.0, 0.6),
    "alat": (55.0, 45.0, 0.4),
    "asat_alat_ratio": (1.4, 0.6, 0.4),
    "bilirubin": (11.0, 9.0, 0.3),
    "folate": (14.0, 8.0, -0.2),
    "vitamin_b12": (360.0, 150.0, 0.1),
    "vitamin_d": (18.0, 10.0, -0.1),
    "prothrombin": (95.0, 15.0, -0.3),
    "hemoglobin": (14.0, 1.5, -0.2),
    "mcv": (96.0, 6.0, 0.5),
    "platelets": (230.0, 80.0, -0.4),
    "leukocytes": (7.0, 2.0, 0.1),
    "nlr": (2.5, 1.5, 0.2),
    "fib4": (2.2, 2.0, 0.4),
    "peth": (1500.0, 1000.0, 0.6),
    "homocysteine": (18.0, 10.0, 0.3),
}
# binary canonical name -> (intercept, loading on z)
_BINARY: Final[dict[str, tuple[float, float]]] = {
    "prior_seizure": (-1.6, 0.9),
    "prior_dt": (-1.9, 0.9),
    "cirrhosis": (-2.6, 0.6),
    "urine_bzd": (-0.3, 0.4),
    "prior_bzd_treatment": (-0.1, 0.4),
    "psychiatric_disorder": (0.6, 0.3),
    "family_aud": (0.5, 0.3),
}
_INTEGER_FEATURES: Final = frozenset(
    {
        "age",
        "education_years",
        "drinking_days_4w",
        "heavy_drinking_days",
        "abstinence_days",
        "age_first_drink",
        "dsm5_criteria",
        "audit_score",
        "prior_withdrawals",
        "fagerstrom",
        "had_anxiety",
        "had_depression",
        "platelets",
    }
)
_CENSORED_LOW: Final = frozenset({"crp", "folate"})
_UNREADABLE: Final = "DM"


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Generator controls."""

    n: int = 300
    seed: int = 42
    missing_rate: float = 0.06
    male_fraction: float = 0.75


def _sigmoid(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-values))
    return result


def _inverse_mapping(
    mapping: dict[str, float | str] | dict[str, str | None] | None,
) -> dict[object, str]:
    """First raw key for each mapped value (e.g. ``1.0 -> "Checked"``)."""
    inverse: dict[object, str] = {}
    for key, value in (mapping or {}).items():
        if value is not None:
            inverse.setdefault(float(value) if isinstance(value, int | float) else value, key)
    return inverse


def _numeric(
    feature: FeatureSpec, driver: npt.NDArray[np.float64], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    low, high = feature.valid_range or (0.0, 100.0)
    default = ((low + high) / 2.0, (high - low) / 8.0, 0.0)
    mean, sd, loading = _PLAUSIBLE.get(feature.name, default)
    noise = rng.normal(size=driver.size)
    values = mean + sd * (loading * driver + np.sqrt(max(1.0 - loading**2, 0.1)) * noise)
    if feature.name in _INTEGER_FEATURES:
        values = np.round(values)
    clipped: npt.NDArray[np.float64] = np.clip(values, low, high)
    return clipped


def _raw_number(value: float, feature: FeatureSpec) -> str:
    raw = value / feature.scale
    return str(round(raw)) if float(raw).is_integer() or abs(raw) >= 100 else f"{raw:.2f}"


def _feature_column(
    feature: FeatureSpec,
    driver: npt.NDArray[np.float64],
    context: dict[str, npt.NDArray[np.float64]],
    spec: SyntheticSpec,
    rng: np.random.Generator,
) -> list[str | None]:
    n = driver.size
    if feature.kind is FeatureKind.CATEGORICAL:
        categories = list(feature.categories or ())
        weights = np.full(len(categories), 1.0 / len(categories))
        if feature.name == "sex" and "male" in categories and len(categories) == 2:
            weights = np.where(
                np.asarray(categories) == "male", spec.male_fraction, 1.0 - spec.male_fraction
            )
        drawn = rng.choice(np.asarray(categories, dtype=object), size=n, p=weights)
        inverse = _inverse_mapping(feature.value_mapping)
        return [inverse.get(c, str(c)) for c in drawn]
    if feature.kind is FeatureKind.BINARY:
        intercept, loading = _BINARY.get(feature.name, (-1.0, 0.3))
        values = (rng.random(n) < _sigmoid(intercept + loading * driver)).astype(np.float64)
        context[feature.name] = values
        inverse = _inverse_mapping(feature.value_mapping)
        return [inverse.get(v, str(int(v))) for v in values]
    values = _numeric(feature, driver, rng)
    context[feature.name] = values
    return [_raw_number(v, feature) for v in values]


def _stream(spec: SyntheticSpec, key: str) -> np.random.Generator:
    """Dedicated random stream for one column or step, stable across schema changes
    (``zlib.crc32``: the built-in ``hash`` of a string changes between processes)."""
    return np.random.default_rng([spec.seed, zlib.crc32(key.encode("utf-8"))])


def _negative_column(
    positive: list[str | None], feature: FeatureSpec, spec: SyntheticSpec
) -> list[str | None]:
    """NON box of an OUI / NON / NON CONNU question: ticked for half of the "not OUI" rows,
    the other half staying NON CONNU (neither box ticked)."""
    inverse = _inverse_mapping(feature.value_mapping)
    ticked, blank = inverse.get(1.0, "1"), inverse.get(0.0, "0")
    known = _stream(spec, f"{feature.name}:no").random(len(positive)) < 0.5
    return [
        None if cell is None else ticked if cell == blank and k else blank
        for cell, k in zip(positive, known, strict=True)
    ]


def _inject_missing(
    column: list[str | None], feature: FeatureSpec, spec: SyntheticSpec, rng: np.random.Generator
) -> list[str | None]:
    n = len(column)
    out = list(column)
    if feature.required:
        return out
    for i in np.flatnonzero(rng.random(n) < spec.missing_rate):
        out[i] = None
    if feature.kind is FeatureKind.NUMERIC and n >= 50:
        out[int(rng.integers(n))] = _UNREADABLE
        if feature.name in _CENSORED_LOW and feature.valid_range is not None:
            out[int(rng.integers(n))] = f"<{max(feature.valid_range[0], 1.0):g}"
    return out


def _outcome_column(
    feature: FeatureSpec, s: npt.NDArray[np.float64], rng: np.random.Generator
) -> list[str | None]:
    n = s.size
    if feature.kind is FeatureKind.BINARY:
        values = (rng.random(n) < _sigmoid(-4.2 + 1.3 * s)).astype(np.float64)
        inverse = _inverse_mapping(feature.value_mapping)
        return [inverse.get(v, str(int(v))) for v in values]
    if feature.kind is FeatureKind.CATEGORICAL:
        return [None] * n
    low, high = feature.valid_range or (0.0, 20.0)
    share = _sigmoid(1.1 * s - 1.1) * 0.6
    values = np.round(low + (high - low) * share + rng.normal(scale=0.6, size=n))
    return [_raw_number(v, feature) for v in np.clip(values, low, high)]


def _molecules(
    schema: DatasetSchema, context: dict[str, npt.NDArray[np.float64]], rng: np.random.Generator
) -> npt.NDArray[np.object_]:
    n = next(iter(context.values())).size if context else 0
    known = schema.dose_equivalence_to_diazepam
    main = "diazepam" if "diazepam" in known else sorted(known)[0]
    alternative = "oxazepam" if "oxazepam" in known else main
    rare = "lorazepam" if "lorazepam" in known else alternative
    age = context.get("age", np.full(n, 45.0))
    cirrhosis = context.get("cirrhosis", np.zeros(n))
    hepatic_logit = -2.75 + 0.25 * (age - 47.0) + 4.0 * (cirrhosis == 1.0)
    hepatic = rng.random(n) < _sigmoid(hepatic_logit)
    molecule = np.where(hepatic, alternative, main).astype(object)
    # A rare molecule, below min_class_count at test sizes: it exercises the rare-class
    # path without adding a pure-noise class to the molecule task.
    molecule[rng.random(n) < 0.02] = rare
    return molecule


def _prescription_text(molecule: str, dose_mg: float, schema: DatasetSchema) -> str:
    aliases = schema.molecule_aliases.get(molecule, ())
    brand = next((a for a in reversed(aliases) if a.isalpha() and a != molecule), molecule)
    return f"{brand.capitalize()} : {round(dose_mg)} mg"


def _target_column(
    name: TargetName,
    source: ColumnSource,
    values: dict[str, npt.NDArray[np.object_] | npt.NDArray[np.float64]],
    schema: DatasetSchema,
) -> list[str | None]:
    treated = values["treated"].astype(bool)
    molecule = values["molecule"]
    dose = values["dose_dze"].astype(np.float64)
    if name is TargetName.TREATMENT:
        inverse = _inverse_mapping(source.value_mapping)
        return [
            inverse.get(TREATED if t else UNTREATED, TREATED if t else UNTREATED) for t in treated
        ]
    if source.parser in {"prescription_molecule", "prescription_dose"}:
        equivalence = schema.dose_equivalence_to_diazepam
        return [
            _prescription_text(str(m), d / equivalence[str(m)], schema) if t else None
            for t, m, d in zip(treated, molecule, dose, strict=True)
        ]
    if name is TargetName.MOLECULE:
        inverse = _inverse_mapping(source.value_mapping)
        return [
            inverse.get(m, str(m)) if t else None for t, m in zip(treated, molecule, strict=True)
        ]
    raw = {
        TargetName.DOSE_DAY1: values["dose_day1_dze"],
        TargetName.DOSE_CUMULATIVE: values["dose_dze"],
        TargetName.DURATION: values["duration"],
    }[name].astype(np.float64)
    if source.convert_with == "molecule":
        equivalence = schema.dose_equivalence_to_diazepam
        raw = np.array(
            [r / equivalence.get(str(m), 1.0) for r, m in zip(raw, molecule, strict=True)]
        )
    raw = raw / source.scale
    if source.parser == "leading_number":
        return [f"{round(r)} jours" if t else "3" for t, r in zip(treated, raw, strict=True)]
    return [f"{r:.0f}" if t else None for t, r in zip(treated, raw, strict=True)]


def generate(spec: SyntheticSpec, schema: DatasetSchema) -> pd.DataFrame:
    """Generate ``spec.n`` raw rows (see module docstring). Pure and deterministic."""
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    z = rng.normal(size=n)
    s = 0.8 * z + 0.6 * rng.normal(size=n)
    columns: dict[str, list[str | None]] = {}
    context: dict[str, npt.NDArray[np.float64]] = {}
    if schema.dataset.id_column is not None:
        columns[schema.dataset.id_column] = [f"S{i + 1:05d}" for i in range(n)]
    for feature in schema.features:
        feature_rng = _stream(spec, feature.name)
        if feature.timing is Timing.POST_ADMISSION:
            columns[feature.source_column] = _outcome_column(feature, s, feature_rng)
            continue
        column = _feature_column(feature, z, context, spec, feature_rng)
        columns[feature.source_column] = _inject_missing(column, feature, spec, feature_rng)
        if feature.negative_column is not None:
            columns[feature.negative_column] = _negative_column(
                columns[feature.source_column], feature, spec
            )

    # Feature names are identifiers (no leading underscore): no clash with this key.
    rng = _stream(spec, "__outcomes__")
    abstinence = context.get("abstinence_days", np.ones(n))
    propensity = 1.6 * s - 0.15 * np.clip(abstinence, 0, 10) + 0.8 + 0.5 * rng.normal(size=n)
    treated = rng.random(n) < _sigmoid(propensity)
    molecule = _molecules(schema, context, rng)
    dose_dze = np.exp(np.log(180.0) + 0.35 * s + 0.35 * z + 0.25 * rng.normal(size=n))
    taper = 3.0 + dose_dze / 35.0 + z + rng.normal(scale=1.0, size=n)
    duration = np.clip(np.round(taper), 2, 40)
    dose_day1 = np.minimum(dose_dze / np.maximum(duration / 2.5, 1.0), dose_dze)
    values: dict[str, npt.NDArray[np.object_] | npt.NDArray[np.float64]] = {
        "treated": treated.astype(np.float64),
        "molecule": molecule,
        "dose_dze": np.round(dose_dze / 5.0) * 5.0,
        "dose_day1_dze": np.round(dose_day1 / 5.0) * 5.0,
        "duration": duration,
    }
    for name, target in schema.enabled_targets().items():
        source = target.source
        if not isinstance(source, ColumnSource) or source.column in columns:
            continue
        columns[source.column] = _target_column(name, source, values, schema)
    columns[SYNTHETIC_MARKER_COLUMN] = ["1"] * n
    return pd.DataFrame(columns, dtype=object)


def default_filename(spec: SyntheticSpec) -> str:
    """Default file name, e.g. ``"SYNTHETIC_300_seed42.csv"``."""
    return f"{SYNTHETIC_PREFIX}{spec.n}_seed{spec.seed}.csv"


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` like a French Excel export: ``;`` separator, decimal comma, UTF-8 with BOM,
    ``\\n`` line endings. Creates parent directories. Returns ``path``.

    Raises:
        ValueError: ``path.name`` does not start with :data:`SYNTHETIC_PREFIX`.
    """
    if not path.name.startswith(SYNTHETIC_PREFIX):
        msg = f"synthetic files must be named {SYNTHETIC_PREFIX}*: {path.name}"
        raise ValueError(msg)
    frame = df.map(lambda v: v.replace(".", ",") if isinstance(v, str) and _is_decimal(v) else v)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, sep=";", lineterminator="\n")
    path.write_bytes(text.encode("utf-8-sig"))
    return path


def _is_decimal(text: str) -> bool:
    head, _, tail = text.partition(".")
    return bool(head.lstrip("-").isdigit() and tail.isdigit())
