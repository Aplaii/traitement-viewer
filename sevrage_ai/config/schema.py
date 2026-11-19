"""Dataset schema: the single source of truth about columns, targets and rules (D2, D9).

**Leader-owned.** Every model is frozen and rejects unknown keys. The whole schema is
validated when it is loaded, so the rest of the code can trust it. Validation messages are
in French because the people who edit the YAML are the site's data managers.

Raw values and columns (contract for ``data/mapping.py``)
    * A ``source_column`` matches a raw header after :func:`normalize_column_name`
      (surrounding whitespace stripped, inner runs of whitespace collapsed to one space,
      **case-insensitive**). REDCap exports put line breaks and double spaces in their
      headers; the YAML can use the single-spaced form. Uniqueness of source columns is
      enforced on the normalised form.
    * ``value_mapping`` keys match the raw cell converted to text with
      :func:`normalize_column_name` (same normalisation). Keys are unique after
      normalisation. Integer YAML keys (``1:``) are accepted as strings; boolean keys are
      rejected (quote ``"yes"``/``"no"``).
    * Values listed in ``dataset.na_values`` (compared the same way) are missing in every
      column.
    * ``valid_range`` and ``scale``: canonical value = raw number x ``scale``, and
      ``valid_range`` applies to the canonical value. In a **raw** table, a value outside
      the range (a typing error such as a potassium of 401) becomes missing and is
      reported; in a **canonical** record (form, ``raw_columns=False``) it is an error.
    * Regular expressions of the schema (``missing_patterns``, ``molecule_aliases``) are
      searched in text folded by :func:`fold_text` (lower case, accents removed), so they
      are written without accents: ``"diazepam|valium"`` matches ``"Diazépam"``.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Iterable, Mapping
from typing import Annotated, Final, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator, model_validator

from sevrage_ai.config._canonical import model_sha256
from sevrage_ai.config.conditions import (
    Comparison,
    Condition,
    DerivationRule,
    ElseRule,
    Membership,
    OnMissing,
    iter_leaves,
)
from sevrage_ai.errors import ConfigError, ConfigWarning
from sevrage_ai.types import (
    DOSE_UNIT,
    OTHER_CLASS,
    REGRESSION_TARGETS,
    SYNTHETIC_MARKER_COLUMN,
    TREATED,
    UNTREATED,
    FeatureKind,
    FlagLevel,
    TargetName,
    TaskKind,
    Timing,
)

__all__ = [
    "ALL_FEATURES",
    "AnonymizationSpec",
    "BaselineSpec",
    "ColumnSource",
    "ConditionalOn",
    "CsvSpec",
    "DatasetSchema",
    "DatasetSpec",
    "DerivedSource",
    "FeatureSpec",
    "SafetyRule",
    "TargetParser",
    "TargetSpec",
    "fold_text",
    "normalize_column_name",
    "schema_hash",
]

ALL_FEATURES: Final = "*"
"""Feature-set value meaning "every admission feature, in declaration order"."""

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
ColumnName = Annotated[str, Field(min_length=1, max_length=512)]
MappedValue = float | str
TargetParser = Literal["leading_number", "prescription_molecule", "prescription_dose"]
"""Text parsers for target columns typed as free text (see :class:`ColumnSource`)."""

_DOSE_TARGETS: Final = frozenset({TargetName.DOSE_DAY1, TargetName.DOSE_CUMULATIVE})
_CONDITIONAL_TARGETS: Final = frozenset({TargetName.MOLECULE} | REGRESSION_TARGETS)
_REQUIRED_TARGETS: Final = (TargetName.SEVERITY, TargetName.TREATMENT, TargetName.MOLECULE)
_BINARY_VALUES: Final = frozenset({0.0, 1.0})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def normalize_column_name(name: str) -> str:
    """Matching key of a raw header or raw cell text: whitespace collapsed, case folded."""
    return " ".join(name.split()).casefold()


def fold_text(text: str) -> str:
    """Lower-case ``text`` and remove accents (``"Diazépam"`` -> ``"diazepam"``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _check_patterns(patterns: Iterable[str]) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            _fail(f"expression régulière invalide « {pattern} » : {exc}")


def _check_column_name(value: str) -> str:
    if value != value.strip():
        _fail(f"le nom de colonne « {value} » ne doit pas commencer ni finir par un espace")
    return value


def _normalise_mapping_keys(value: object) -> object:
    """Accept str/int YAML keys (as stripped str), reject bool keys and bool values."""
    if not isinstance(value, Mapping):
        return value
    normalised: dict[str, object] = {}
    for key, mapped in value.items():
        if isinstance(key, bool) or not isinstance(key, str | int):
            _fail(f"clé de value_mapping invalide ({key!r}) : utiliser du texte entre guillemets")
        if isinstance(mapped, bool):
            _fail(f"valeur booléenne pour la clé « {key} » : utiliser 1/0 ou du texte")
        normalised[str(key).strip()] = mapped
    return normalised


def _check_unique_casefold(keys: Iterable[str], what: str) -> None:
    seen: dict[str, str] = {}
    for key in keys:
        folded = normalize_column_name(key)
        if folded in seen:
            _fail(f"{what} en double (sans tenir compte de la casse) : « {seen[folded]} »")
        seen[folded] = key


def _check_range(valid_range: tuple[float, float] | None, owner: str) -> None:
    if valid_range is not None and not valid_range[0] < valid_range[1]:
        _fail(f"{owner} : valid_range doit vérifier borne basse < borne haute")


# --------------------------------------------------------------------------------------
# Dataset-level settings
# --------------------------------------------------------------------------------------


class CsvSpec(_Frozen):
    """How to read delimited files. ``auto`` means "sniff it" (French Excel exports)."""

    sep: Literal["auto", ",", ";", "\t", "|"] = "auto"
    decimal: Literal["auto", ".", ","] = "auto"
    encoding: Literal["auto", "utf-8", "utf-8-sig", "cp1252", "latin-1"] = "auto"
    sheet: str | int | None = None
    """XLSX only: sheet name or 0-based index; ``None`` = first sheet."""


class DatasetSpec(_Frozen):
    """Description of the source table."""

    description: str
    id_column: ColumnName | None = None
    """Pseudonymous id: kept to join outputs, never a feature."""
    group_column: ColumnName | None = None
    """Patient key when several episodes per patient exist (grouped CV)."""
    metadata_columns: tuple[ColumnName, ...] = (SYNTHETIC_MARKER_COLUMN,)
    """Columns silently removed before mapping (never reported as undeclared)."""
    csv: CsvSpec = Field(default_factory=CsvSpec)
    na_values: tuple[str, ...] = ("", "NA", "ND", "NR", "?", "-", "inconnu")
    censored_numbers: Literal["bound", "missing"] = "bound"
    """How ``"<5"`` or ``"> 120"`` (values beyond a lab's measuring range) are read: as the
    bound (5, 120) or as missing."""

    @field_validator("id_column", "group_column")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return None if value is None else _check_column_name(value)


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------


class FeatureSpec(_Frozen):
    """One clinical variable.

    ``timing="post_admission"`` marks outcomes (for example the peak CIWA score): they may
    feed target derivation but can never be predictors. ``required`` features must be
    present as columns (validation error otherwise), and a missing value in a row forces
    LOW confidence. ``monotonic`` is the expected direction of the effect (+1/-1) and maps
    to HGB monotonic constraints. ``group`` refers to ``DatasetSchema.feature_groups``.

    ``negative_column`` (binary only) handles the REDCap OUI / NON / NON CONNU questions
    exported as checkboxes: ``source_column`` is the OUI box and ``negative_column`` the NON
    box. The feature is 1 when OUI is ticked, 0 when NON is ticked and **missing** otherwise:
    NON CONNU means "not known yet", neither true nor false.
    """

    name: Identifier
    label: str = Field(min_length=1)
    source_column: ColumnName
    kind: FeatureKind
    unit: str | None = None
    description: str = ""
    valid_range: tuple[float, float] | None = None
    categories: tuple[str, ...] | None = None
    value_mapping: dict[str, MappedValue] | None = None
    scale: float = Field(default=1.0, gt=0.0)
    monotonic: Literal[-1, 0, 1] = 0
    timing: Timing = Timing.ADMISSION
    group: Identifier = "other"
    required: bool = False
    negative_column: ColumnName | None = None

    @field_validator("source_column")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        return _check_column_name(value)

    @field_validator("negative_column")
    @classmethod
    def _strip_negative(cls, value: str | None) -> str | None:
        return None if value is None else _check_column_name(value)

    @field_validator("value_mapping", mode="before")
    @classmethod
    def _normalise_mapping(cls, value: object) -> object:
        return _normalise_mapping_keys(value)

    @model_validator(mode="after")
    def _check_kind(self) -> FeatureSpec:
        owner = f"variable « {self.name} »"
        is_numeric = self.kind is FeatureKind.NUMERIC
        if self.kind is FeatureKind.CATEGORICAL:
            if not self.categories:
                _fail(f"{owner} : une variable catégorielle doit déclarer ses categories")
            else:
                _check_unique_casefold(self.categories, f"{owner} : catégorie")
        elif self.categories is not None:
            _fail(f"{owner} : categories n'est permis que pour une variable catégorielle")
        if self.valid_range is not None and not is_numeric:
            _fail(f"{owner} : valid_range n'est permis que pour une variable numérique")
        _check_range(self.valid_range, owner)
        if self.scale != 1.0 and not is_numeric:
            _fail(f"{owner} : scale n'est permis que pour une variable numérique")
        if self.monotonic != 0 and self.kind is FeatureKind.CATEGORICAL:
            _fail(f"{owner} : monotonic est interdit pour une variable catégorielle")
        if self.value_mapping is not None:
            self._check_mapping(self.value_mapping, owner)
        if self.negative_column is not None and self.kind is not FeatureKind.BINARY:
            _fail(f"{owner} : negative_column n'est permis que pour une variable binaire")
        return self

    def _check_mapping(self, mapping: Mapping[str, MappedValue], owner: str) -> None:
        _check_unique_casefold(mapping.keys(), f"{owner} : clé de value_mapping")
        for key, mapped in mapping.items():
            if self.kind is FeatureKind.CATEGORICAL:
                if not isinstance(mapped, str) or mapped not in (self.categories or ()):
                    _fail(f"{owner} : « {key} » doit correspondre à une des categories")
            elif isinstance(mapped, str):
                _fail(f"{owner} : « {key} » doit correspondre à un nombre")
            elif self.kind is FeatureKind.BINARY and mapped not in _BINARY_VALUES:
                _fail(f"{owner} : « {key} » doit correspondre à 0 ou 1")


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------


class ConditionalOn(_Frozen):
    """The target only exists when another (binary) target equals ``equals``.

    On rows where the condition does not hold, the conditional target is set to missing
    during mapping, whatever the raw column contains.
    """

    target: TargetName
    equals: str = Field(min_length=1)


class ColumnSource(_Frozen):
    """Target read from one raw column.

    ``value_mapping`` (classification only) maps raw values to canonical classes; ``null``
    maps to missing. Raw values that already equal a canonical class are kept; any other
    value becomes missing and is reported. ``convert_with: molecule`` converts a dose in
    mg of the patient's molecule to mg diazepam-equivalent with
    ``dose_equivalence_to_diazepam`` (regression only). ``scale`` multiplies the raw number.

    Free-text columns (``parser``), e.g. « Valium 390mg : (20X3)X2+... »:

    * ``leading_number`` (regression): the number at the start of the text (``"7 jours"``
      -> 7); text that does not start with a number is missing.
    * ``prescription_molecule`` (molecule): the **first** molecule named in the text,
      recognised with ``DatasetSchema.molecule_aliases`` (the initial prescribing choice);
      ``value_mapping`` may then regroup molecules (``prazepam: oxazepam``: a treatment
      continued from before the admission, switched to its oxazepam equivalent).
    * ``prescription_dose`` (dose targets): total dose in mg diazepam-equivalent, summed
      over every molecule named, each with its own amount: the number after ``=`` when the
      text details a sum (``"20mg + 30 + 30 = 80"``), else the first ``<number> mg``. Any
      molecule without an amount or without a dose equivalence makes the total missing.

    ``missing_patterns`` (regexes, see the module docstring) mark texts that must be read
    as missing whatever they contain, e.g. ``"mettre dose"`` (not filled in), ``"sortie
    anticipee"`` (incomplete, censored withdrawal) or ``"ttt habituel"`` (usual treatment,
    not a withdrawal protocol). Several targets may read the same column.
    """

    column: ColumnName
    value_mapping: dict[str, str | None] | None = None
    convert_with: Literal["molecule"] | None = None
    scale: float = Field(default=1.0, gt=0.0)
    parser: TargetParser | None = None
    missing_patterns: tuple[str, ...] = ()

    @field_validator("column")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _check_column_name(value)

    @field_validator("value_mapping", mode="before")
    @classmethod
    def _normalise_mapping(cls, value: object) -> object:
        return _normalise_mapping_keys(value)

    @field_validator("value_mapping")
    @classmethod
    def _unique_keys(cls, value: dict[str, str | None] | None) -> dict[str, str | None] | None:
        if value is not None:
            _check_unique_casefold(value.keys(), "clé de value_mapping")
        return value

    @field_validator("missing_patterns")
    @classmethod
    def _compile(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        _check_patterns(patterns)
        return patterns

    @model_validator(mode="after")
    def _check_parser(self) -> ColumnSource:
        if (
            self.parser is not None
            and self.parser != "prescription_molecule"
            and self.value_mapping is not None
        ):
            _fail("parser et value_mapping sont incompatibles (sauf prescription_molecule)")
        if self.parser == "prescription_dose" and self.convert_with is not None:
            _fail("parser prescription_dose convertit déjà les doses : retirer convert_with")
        return self


class DerivedSource(_Frozen):
    """Target derived from outcome features with first-match-wins rules (no ``eval``).

    The rule list must end with exactly one ``{else: <class>}``. ``on_missing`` controls
    rows where a rule cannot be decided because of missing values (see
    :func:`sevrage_ai.config.conditions.derive_labels`).
    """

    derived: tuple[DerivationRule, ...] = Field(min_length=1)
    on_missing: OnMissing = "undetermined"

    @model_validator(mode="after")
    def _else_last(self) -> DerivedSource:
        else_positions = [i for i, rule in enumerate(self.derived) if isinstance(rule, ElseRule)]
        if else_positions != [len(self.derived) - 1]:
            _fail("la liste derived doit se terminer par une unique règle « else »")
        return self

    def labels(self) -> tuple[str, ...]:
        """Every label the rules can produce, in rule order."""
        return tuple(
            rule.otherwise if isinstance(rule, ElseRule) else rule.then for rule in self.derived
        )


class TargetSpec(_Frozen):
    """One prediction target.

    ``classes`` is ordered: for an ordinal target, from least to most severe. For a
    multiclass target it may be ``None``: the classes are then inferred from the data and
    frozen in the model bundle. Classes rarer than ``min_class_count`` are merged into
    ``"other"``. ``class_labels`` gives French display labels. ``transform`` applies to
    regression targets before fitting. ``valid_range`` (regression only) is a HARD bound
    on the canonical value.
    """

    enabled: bool = True
    kind: TaskKind
    label: str = Field(min_length=1)
    description: str = ""
    source: ColumnSource | DerivedSource
    classes: tuple[str, ...] | None = None
    class_labels: dict[str, str] = Field(default_factory=dict)
    positive_class: str | None = None
    conditional_on: ConditionalOn | None = None
    min_class_count: int = Field(default=1, ge=1)
    unit: str | None = None
    transform: Literal["none", "log1p"] = "none"
    valid_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> TargetSpec:
        if self.classes is not None:
            if len(set(self.classes)) != len(self.classes) or len(self.classes) < 2:
                _fail("classes doit contenir au moins 2 valeurs distinctes")
            unknown = set(self.class_labels) - set(self.classes) - {OTHER_CLASS}
            if unknown:
                _fail(f"class_labels contient des classes inconnues : {sorted(unknown)}")
        if self.kind is TaskKind.REGRESSION:
            self._check_regression()
        else:
            self._check_classification()
        return self

    def _check_regression(self) -> None:
        if self.classes is not None or self.positive_class is not None or self.class_labels:
            _fail("une cible de régression n'a ni classes, ni positive_class, ni class_labels")
        if not self.unit:
            _fail("une cible de régression doit déclarer son unit")
        if isinstance(self.source, DerivedSource):
            _fail("une cible de régression ne peut pas être dérivée par règles")
        elif self.source.value_mapping is not None:
            _fail("value_mapping n'est pas permis pour une cible de régression")
        elif self.source.parser == "prescription_molecule":
            _fail("parser prescription_molecule est réservé à la cible molecule")
        _check_range(self.valid_range, "cible")

    def _check_classification(self) -> None:
        if self.transform != "none" or self.valid_range is not None:
            _fail("transform et valid_range sont réservés aux cibles de régression")
        if self.kind in {TaskKind.ORDINAL, TaskKind.BINARY} and self.classes is None:
            _fail(f"une cible {self.kind.value} doit déclarer ses classes")
        if self.kind is TaskKind.BINARY:
            if self.classes is None or len(self.classes) != 2:
                _fail("une cible binaire doit avoir exactement 2 classes")
            if self.positive_class is None or self.positive_class not in self.classes:
                _fail("positive_class doit être une des classes de la cible binaire")
        elif self.positive_class is not None:
            _fail("positive_class n'est permis que pour une cible binaire")
        if isinstance(self.source, DerivedSource):
            if self.classes is None:
                _fail("une cible dérivée doit déclarer ses classes")
            unknown = set(self.source.labels()) - set(self.classes or ())
            if unknown:
                _fail(f"les règles produisent des classes non déclarées : {sorted(unknown)}")
            return
        if self.source.convert_with is not None or self.source.scale != 1.0:
            _fail("convert_with et scale sont réservés aux cibles de régression")
        if self.source.parser in {"leading_number", "prescription_dose"}:
            _fail(f"parser {self.source.parser} est réservé aux cibles de régression")
        for key, mapped in (self.source.value_mapping or {}).items():
            if mapped is None:
                continue
            if self.classes is not None and mapped not in self.classes:
                _fail(f"value_mapping : « {key} » ne correspond à aucune classe déclarée")
            if self.classes is None and not re.fullmatch(r"[a-z][a-z0-9_]*", mapped):
                _fail(f"value_mapping : « {mapped} » doit être un identifiant en minuscules")


# --------------------------------------------------------------------------------------
# Baselines, anonymization, safety rules
# --------------------------------------------------------------------------------------


class BaselineSpec(_Frozen):
    """A clinical-score baseline: predict ``positive_class`` when ``feature op threshold``.

    ``positive_class`` defaults to the target's ``positive_class`` for binary targets and
    is required otherwise (for example ``severity`` / ``high`` = "high vs rest").
    """

    name: str = Field(min_length=1)
    target: TargetName
    positive_class: str | None = None
    feature: Identifier
    op: Literal["gt", "gte", "lt", "lte"] = "gte"
    threshold: float


class AnonymizationSpec(_Frozen):
    """Heuristic identification checks run before any training or batch prediction.

    ``forbidden_column_patterns`` are Python regexes searched in column names.
    ``scan_values`` enables NIR/email/phone/full-date regexes on text cells. A text column
    whose share of distinct non-missing values exceeds ``high_cardinality_ratio`` (with at
    least ``min_rows_for_cardinality`` non-missing values) is flagged as free text.
    """

    forbidden_column_patterns: tuple[str, ...] = Field(min_length=1)
    scan_values: bool = True
    high_cardinality_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    min_rows_for_cardinality: int = Field(default=20, ge=1)

    @field_validator("forbidden_column_patterns")
    @classmethod
    def _compile(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                _fail(f"expression régulière invalide « {pattern} » : {exc}")
        return patterns

    def compiled_patterns(self) -> tuple[re.Pattern[str], ...]:
        """The column-name patterns, compiled."""
        return tuple(re.compile(pattern) for pattern in self.forbidden_column_patterns)


class SafetyRule(_Frozen):
    """Informational annotation. It never changes model outputs (R6).

    The rule fires for a patient when ``when`` is ``True`` on the canonical admission
    features, the recommendation matches ``applies_to`` (``treated`` / ``untreated`` /
    ``all``) and, if ``molecule_in`` is set, the recommended molecule is in that list. It
    then produces ``Flag(code="safety.<id>", level=level, message_fr=message_fr)``.
    """

    id: Identifier
    level: FlagLevel = "warning"
    applies_to: Literal["treated", "untreated", "all"] = "treated"
    when: Condition
    molecule_in: Annotated[tuple[str, ...], Field(min_length=1)] | None = None
    message_fr: str = Field(min_length=1)

    @model_validator(mode="after")
    def _molecule_needs_treatment(self) -> SafetyRule:
        if self.molecule_in is not None and self.applies_to == "untreated":
            _fail(f"règle « {self.id} » : molecule_in est incompatible avec applies_to=untreated")
        return self


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


class DatasetSchema(_Frozen):
    """Whole data dictionary + mapping + targets + rules. See ``configs/schema.yaml``."""

    schema_version: Literal["1.0"] = "1.0"
    dataset: DatasetSpec
    feature_groups: dict[Identifier, str] = Field(min_length=1)
    """Group id -> French label, in display order."""
    feature_sets: dict[Identifier, tuple[Identifier, ...] | Literal["*"]] = Field(min_length=1)
    features: tuple[FeatureSpec, ...] = Field(min_length=1)
    targets: dict[TargetName, TargetSpec]
    dose_equivalence_to_diazepam: dict[Identifier, PositiveFloat]
    """mg diazepam per 1 mg of molecule. MUST be validated by a clinician."""
    molecule_aliases: dict[Identifier, tuple[str, ...]] = Field(default_factory=dict)
    """Canonical molecule -> regexes of its names in free text (brand names, spellings).
    The canonical name itself always matches. Used by the ``prescription_*`` parsers."""
    baselines: tuple[BaselineSpec, ...] = ()
    anonymization: AnonymizationSpec
    safety_rules: tuple[SafetyRule, ...] = ()

    # ---------------------------------------------------------------- validation

    @model_validator(mode="after")
    def _validate_schema(self) -> DatasetSchema:
        self._check_features()
        self._check_feature_sets()
        self._check_targets_present()
        self._check_target_kinds()
        self._check_conditional_on()
        self._check_targets_not_predictors()
        self._check_parsers()
        self._check_unique_columns()
        self._check_conditions()
        self._check_baselines()
        self._check_safety_rules()
        self._check_dose_equivalence()
        return self

    def _check_features(self) -> None:
        names = [feature.name for feature in self.features]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            _fail(f"noms de variables en double : {duplicates}")
        for feature in self.features:
            if feature.group not in self.feature_groups:
                _fail(f"variable « {feature.name} » : groupe « {feature.group} » non déclaré")
            if feature.required and feature.timing is not Timing.ADMISSION:
                _fail(f"variable « {feature.name} » : seule une variable d'admission est requise")

    def _check_feature_sets(self) -> None:
        by_name = {feature.name: feature for feature in self.features}
        for set_name, members in self.feature_sets.items():
            if members == ALL_FEATURES:
                continue
            if not members:
                _fail(f"le jeu de variables « {set_name} » est vide")
            if len(set(members)) != len(members):
                _fail(f"le jeu de variables « {set_name} » contient des doublons")
            for member in members:
                feature = by_name.get(member)
                if feature is None:
                    _fail(f"jeu « {set_name} » : variable inconnue « {member} »")
                elif feature.timing is not Timing.ADMISSION:
                    _fail(
                        f"jeu « {set_name} » : « {member} » est mesurée après l'admission "
                        "et ne peut pas être un prédicteur"
                    )

    def _check_targets_present(self) -> None:
        for name in _REQUIRED_TARGETS:
            spec = self.targets.get(name)
            if spec is None or not spec.enabled:
                _fail(f"la cible « {name.value} » doit être déclarée et activée")
        if not any(
            (spec := self.targets.get(name)) is not None and spec.enabled for name in _DOSE_TARGETS
        ):
            _fail("au moins une cible de dose (dose_day1 ou dose_cumulative) doit être activée")

    def _check_target_kinds(self) -> None:
        allowed: dict[TargetName, frozenset[TaskKind]] = {
            TargetName.SEVERITY: frozenset({TaskKind.ORDINAL, TaskKind.MULTICLASS}),
            TargetName.TREATMENT: frozenset({TaskKind.BINARY}),
            TargetName.MOLECULE: frozenset({TaskKind.MULTICLASS}),
        }
        for name, spec in self.targets.items():
            kinds = allowed.get(name, frozenset({TaskKind.REGRESSION}))
            if spec.kind not in kinds:
                _fail(f"la cible « {name.value} » doit être de type {sorted(kinds)}")
            if name in _DOSE_TARGETS and spec.unit != DOSE_UNIT:
                _fail(f"la cible « {name.value} » doit avoir unit: {DOSE_UNIT}")
        treatment = self.targets[TargetName.TREATMENT]
        if set(treatment.classes or ()) != {TREATED, UNTREATED} or (
            treatment.positive_class != TREATED
        ):
            _fail('la cible treatment doit avoir classes ["no", "yes"] et positive_class "yes"')

    def _check_conditional_on(self) -> None:
        for name, spec in self.targets.items():
            condition = spec.conditional_on
            if condition is not None:
                parent = self.targets.get(condition.target)
                if parent is None or not parent.enabled or parent.kind is not TaskKind.BINARY:
                    _fail(
                        f"cible « {name.value} » : conditional_on doit viser une cible "
                        "binaire déclarée et activée"
                    )
                elif condition.equals not in (parent.classes or ()):
                    _fail(f"cible « {name.value} » : conditional_on.equals n'est pas une classe")
            if name in _CONDITIONAL_TARGETS and (
                condition is None
                or condition.target is not TargetName.TREATMENT
                or condition.equals != TREATED
            ):
                _fail(
                    f"la cible « {name.value} » doit avoir "
                    'conditional_on: {target: treatment, equals: "yes"}'
                )

    def _check_targets_not_predictors(self) -> None:
        predictor_columns = {
            feature.source_column.casefold(): feature.name
            for feature in self.features
            if feature.timing is Timing.ADMISSION
        }
        by_name = {feature.name: feature for feature in self.features}
        for name, spec in self.targets.items():
            source = spec.source
            if isinstance(source, ColumnSource):
                clash = predictor_columns.get(source.column.casefold())
                if clash is not None:
                    _fail(
                        f"la colonne source de la cible « {name.value} » est aussi celle du "
                        f"prédicteur « {clash} » (fuite de données)"
                    )
                continue
            for rule in source.derived:
                if isinstance(rule, ElseRule):
                    continue
                for leaf in iter_leaves(rule.when):
                    feature = by_name.get(leaf.feature)
                    if feature is None:
                        _fail(f"cible « {name.value} » : variable inconnue « {leaf.feature} »")
                    elif feature.timing is Timing.ADMISSION:
                        _fail(
                            f"cible « {name.value} » : la dérivation utilise la variable "
                            f"d'admission « {leaf.feature} » (fuite de données)"
                        )

    def _check_parsers(self) -> None:
        for patterns in self.molecule_aliases.values():
            _check_patterns(patterns)
        uses_prescription = False
        for name, spec in self.targets.items():
            parser = spec.source.parser if isinstance(spec.source, ColumnSource) else None
            if parser == "prescription_molecule" and name is not TargetName.MOLECULE:
                _fail(f"cible « {name.value} » : prescription_molecule est réservé à molecule")
            if parser == "prescription_dose" and name not in _DOSE_TARGETS:
                _fail(f"cible « {name.value} » : prescription_dose est réservé aux doses")
            uses_prescription |= parser in {"prescription_molecule", "prescription_dose"}
        if uses_prescription and not self.molecule_aliases:
            _fail("les parsers prescription_* exigent molecule_aliases")

    def _check_unique_columns(self) -> None:
        columns = [feature.source_column for feature in self.features]
        columns += [f.negative_column for f in self.features if f.negative_column]
        target_columns = {
            normalize_column_name(spec.source.column): spec.source.column
            for spec in self.targets.values()
            if isinstance(spec.source, ColumnSource)
        }
        columns += list(target_columns.values())  # several targets may read one column
        columns += [c for c in (self.dataset.id_column, self.dataset.group_column) if c]
        columns += list(self.dataset.metadata_columns)
        _check_unique_casefold(columns, "colonne source")

    def _check_conditions(self) -> None:
        by_name = {feature.name: feature for feature in self.features}
        conditions: list[Condition] = [rule.when for rule in self.safety_rules]
        for spec in self.targets.values():
            if isinstance(spec.source, DerivedSource):
                conditions += [r.when for r in spec.source.derived if not isinstance(r, ElseRule)]
        for condition in conditions:
            for leaf in iter_leaves(condition):
                feature = by_name.get(leaf.feature)
                if feature is None:
                    _fail(f"condition : variable inconnue « {leaf.feature} »")
                else:
                    _check_leaf_type(leaf, feature)

    def _check_baselines(self) -> None:
        by_name = {feature.name: feature for feature in self.features}
        names = [baseline.name for baseline in self.baselines]
        if len(set(names)) != len(names):
            _fail("noms de baselines en double")
        for baseline in self.baselines:
            owner = f"baseline « {baseline.name} »"
            spec = self.targets.get(baseline.target)
            if spec is None or not spec.enabled or not spec.kind.is_classification:
                _fail(f"{owner} : la cible doit être une classification activée")
            positive = baseline.positive_class or spec.positive_class
            if positive is None:
                _fail(f"{owner} : positive_class est requis pour cette cible")
            elif spec.classes is not None and positive not in spec.classes:
                _fail(f"{owner} : positive_class n'est pas une classe de la cible")
            feature = by_name.get(baseline.feature)
            if feature is None or feature.timing is not Timing.ADMISSION:
                _fail(f"{owner} : la variable doit être une variable d'admission déclarée")
            elif feature.kind is FeatureKind.CATEGORICAL:
                _fail(f"{owner} : la variable doit être numérique ou binaire")

    def _check_safety_rules(self) -> None:
        ids = [rule.id for rule in self.safety_rules]
        if len(set(ids)) != len(ids):
            _fail("identifiants de règles de sécurité en double")
        by_name = {feature.name: feature for feature in self.features}
        known_molecules = self.molecule_names() | {OTHER_CLASS}
        for rule in self.safety_rules:
            for leaf in iter_leaves(rule.when):
                feature = by_name.get(leaf.feature)
                if feature is not None and feature.timing is not Timing.ADMISSION:
                    _fail(
                        f"règle « {rule.id} » : « {leaf.feature} » n'est pas connue à l'admission"
                    )
            unknown = set(rule.molecule_in or ()) - known_molecules
            if unknown:
                _fail(f"règle « {rule.id} » : molécules inconnues {sorted(unknown)}")

    def _check_dose_equivalence(self) -> None:
        uses_conversion = any(
            isinstance(spec.source, ColumnSource)
            and (spec.source.convert_with is not None or spec.source.parser == "prescription_dose")
            for spec in self.targets.values()
            if spec.enabled
        )
        if not uses_conversion:
            return
        missing = sorted(self.molecule_names() - set(self.dose_equivalence_to_diazepam))
        if missing:
            warnings.warn(
                f"molécules sans équivalence de dose (doses non converties) : {missing}",
                ConfigWarning,
                stacklevel=2,
            )

    # ---------------------------------------------------------------- accessors

    def feature(self, name: str) -> FeatureSpec:
        """Return the feature called ``name``.

        Raises:
            ConfigError: unknown feature.
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        msg = f"Variable inconnue dans le schéma : « {name} »."
        raise ConfigError(msg)

    def admission_features(self) -> list[FeatureSpec]:
        """Every ``admission`` feature, in declaration order (the columns of ``X``)."""
        return [f for f in self.features if f.timing is Timing.ADMISSION]

    def post_admission_features(self) -> list[FeatureSpec]:
        """Every ``post_admission`` (outcome) feature, in declaration order."""
        return [f for f in self.features if f.timing is Timing.POST_ADMISSION]

    def predictors(self, feature_set: str) -> list[FeatureSpec]:
        """Features of ``feature_set``, in declaration order. All are ``admission`` features.

        Raises:
            ConfigError: unknown feature set.
        """
        members = self.feature_sets.get(feature_set)
        if members is None:
            msg = (
                f"Jeu de variables inconnu : « {feature_set} » "
                f"(disponibles : {', '.join(self.feature_sets)})."
            )
            raise ConfigError(msg)
        if members == ALL_FEATURES:
            return self.admission_features()
        wanted = set(members)
        return [f for f in self.features if f.name in wanted]

    def enabled_targets(self) -> dict[TargetName, TargetSpec]:
        """Enabled targets, in :class:`TargetName` declaration order."""
        return {
            name: self.targets[name]
            for name in TargetName
            if name in self.targets and self.targets[name].enabled
        }

    def target(self, name: TargetName) -> TargetSpec:
        """Return the spec of target ``name`` (enabled or not).

        Raises:
            ConfigError: the target is not declared.
        """
        spec = self.targets.get(name)
        if spec is None:
            msg = f"Cible non déclarée dans le schéma : « {name.value} »."
            raise ConfigError(msg)
        return spec

    def high_severity_class(self) -> str:
        """The most severe ``severity`` class (last declared class)."""
        classes = self.targets[TargetName.SEVERITY].classes
        if classes is None:  # pragma: no cover - excluded by validation
            msg = "La cible severity doit déclarer ses classes."
            raise ConfigError(msg)
        return classes[-1]

    def molecule_names(self) -> frozenset[str]:
        """Canonical molecule names declared by the molecule target's mapping or classes."""
        spec = self.targets.get(TargetName.MOLECULE)
        if spec is None:
            return frozenset()
        names = set(spec.classes or ()) - {OTHER_CLASS}
        if isinstance(spec.source, ColumnSource) and spec.source.value_mapping:
            names |= {value for value in spec.source.value_mapping.values() if value is not None}
        names |= set(self.molecule_aliases)
        return frozenset(names)

    def source_columns(self) -> dict[str, str]:
        """Every declared raw column, keyed by :func:`normalize_column_name`: features, target
        sources, id, group and metadata columns (value = the column name as declared)."""
        columns = [feature.source_column for feature in self.features]
        columns += [f.negative_column for f in self.features if f.negative_column]
        columns += [
            spec.source.column
            for spec in self.targets.values()
            if isinstance(spec.source, ColumnSource)
        ]
        columns += [c for c in (self.dataset.id_column, self.dataset.group_column) if c]
        columns += list(self.dataset.metadata_columns)
        return {normalize_column_name(column): column for column in columns}


def _check_leaf_type(leaf: Comparison | Membership, feature: FeatureSpec) -> None:
    values = leaf.value if isinstance(leaf, Membership) else (leaf.value,)
    owner = f"condition sur « {feature.name} »"
    if feature.kind is FeatureKind.CATEGORICAL:
        if leaf.op in {"gt", "gte", "lt", "lte"}:
            _fail(f"{owner} : opérateur d'ordre interdit sur une variable catégorielle")
        if not all(isinstance(v, str) and v in (feature.categories or ()) for v in values):
            _fail(f"{owner} : les valeurs doivent être des categories déclarées")
        return
    if any(isinstance(v, str) for v in values):
        _fail(f"{owner} : les valeurs doivent être numériques")
    if (
        feature.kind is FeatureKind.BINARY
        and leaf.op in {"eq", "ne", "in"}
        and not all(v in _BINARY_VALUES for v in values)
    ):
        _fail(f"{owner} : une variable binaire se compare à 0 ou 1")


def schema_hash(schema: DatasetSchema) -> str:
    """Hex sha256 of the schema's canonical JSON (sorted keys, aliases). Stored in bundles."""
    return model_sha256(schema)
