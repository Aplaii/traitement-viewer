"""Data dictionary rendered from the schema (``sevrage docs-dictionary``, app page)."""

from __future__ import annotations

from sevrage_ai.config.schema import ALL_FEATURES, ColumnSource, DatasetSchema, DerivedSource
from sevrage_ai.types import Timing

__all__ = ["dictionary_rows", "render_data_dictionary"]

_KINDS = {"numeric": "numérique", "binary": "binaire", "categorical": "catégorielle"}


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def dictionary_rows(schema: DatasetSchema) -> list[dict[str, str]]:
    """One row per feature: French column headers, ready for a table."""
    rows = []
    for feature in schema.features:
        low_high = (
            ""
            if feature.valid_range is None
            else f"{feature.valid_range[0]:g} à {feature.valid_range[1]:g}"
        )
        sets = [
            name
            for name, members in schema.feature_sets.items()
            if (members == ALL_FEATURES and feature.timing is Timing.ADMISSION)
            or (members != ALL_FEATURES and feature.name in members)
        ]
        rows.append(
            {
                "Variable": feature.name,
                "Libellé": feature.label,
                "Groupe": schema.feature_groups[feature.group],
                "Type": _KINDS[feature.kind.value],
                "Unité": feature.unit or "",
                "Bornes": low_high,
                "Modalités": ", ".join(feature.categories or ()),
                "Moment": "admission"
                if feature.timing is Timing.ADMISSION
                else "après l'admission",
                "Jeux": ", ".join(sets),
                "Colonne source": feature.source_column
                if feature.negative_column is None
                else f"{feature.source_column} ; NON : {feature.negative_column}",
            }
        )
    return rows


def render_data_dictionary(schema: DatasetSchema) -> str:
    """French Markdown data dictionary: dataset, features, targets and dose equivalences."""
    lines = [
        "# Dictionnaire des données",
        "",
        "Généré automatiquement depuis le schéma (`sevrage docs-dictionary`) : "
        "ne pas modifier à la main.",
        "",
        f"**Jeu de données :** {_escape(schema.dataset.description)}",
        "",
        "## Variables",
        "",
    ]
    rows = dictionary_rows(schema)
    headers = list(rows[0]) if rows else []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    lines += ["| " + " | ".join(_escape(row[h]) for h in headers) + " |" for row in rows]
    lines += ["", "## Cibles", ""]
    for name, spec in schema.targets.items():
        state = "" if spec.enabled else " (désactivée)"
        lines.append(f"### {spec.label} (`{name.value}`){state}")
        lines.append("")
        if spec.description:
            lines += [_escape(spec.description), ""]
        if spec.classes:
            labels = ", ".join(f"{c} = {spec.class_labels.get(c, c)}" for c in spec.classes)
            lines += [f"- Classes : {labels}"]
        if spec.unit:
            lines += [f"- Unité : {spec.unit}"]
        if isinstance(spec.source, ColumnSource):
            parser = f", lecture `{spec.source.parser}`" if spec.source.parser else ""
            lines += [f"- Colonne source : « {spec.source.column} »{parser}"]
        elif isinstance(spec.source, DerivedSource):
            lines += ["- Dérivée par règles à partir des variables d'évolution."]
        if spec.conditional_on is not None:
            lines += [
                f"- Uniquement si `{spec.conditional_on.target.value}` = "
                f"`{spec.conditional_on.equals}`"
            ]
        lines.append("")
    lines += [
        "## Équivalences de dose (mg de diazépam pour 1 mg) — à valider par un clinicien",
        "",
        "| Molécule | Équivalence |",
        "|---|---|",
    ]
    lines += [f"| {m} | {v:g} |" for m, v in sorted(schema.dose_equivalence_to_diazepam.items())]
    return "\n".join(lines) + "\n"
