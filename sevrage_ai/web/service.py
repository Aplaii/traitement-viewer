"""Plain-French JSON for the web page: form description and prediction display.

The page is a public demonstration: :data:`PUBLIC_WARNING` asks never to enter a real
patient. Values sent by the page are reduced to the model features (unknown names and
invalid values are dropped) and are never logged or stored; they come back only in the
response, to the browser that sent them.

The texts reuse the view models of the Streamlit app (:mod:`sevrage_ai.app.viewmodels`)
for the numbers and labels, with shorter, friendlier wording for a non-specialist reader.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from sevrage_ai.app import viewmodels as vm
from sevrage_ai.errors import SevrageError
from sevrage_ai.inference import export
from sevrage_ai.types import (
    ConfidenceLevel,
    FeatureKind,
    PatientPrediction,
    PredictorProtocol,
    RegressionOutput,
    TargetName,
)
from sevrage_ai.web.slim import load_slim_predictor

__all__ = [
    "CONFIG_ROUTE",
    "JSON_TYPE",
    "MAX_BODY_BYTES",
    "PREDICT_ROUTE",
    "PUBLIC_WARNING",
    "RequestError",
    "Response",
    "WebService",
    "handle",
    "json_response",
]

Json = dict[str, object]

CONFIG_ROUTE: Final = "/api/config"
PREDICT_ROUTE: Final = "/api/predict"
MAX_BODY_BYTES: Final = 16_384
JSON_TYPE: Final = "application/json; charset=utf-8"

TITLE: Final = "Anticiper le sevrage d'alcool"
INTRO: Final = (
    "Décrivez un patient imaginaire tel qu'il arrive dans le service et regardez ce que le "
    "modèle en pense : la gravité probable du sevrage, le besoin d'un traitement, la "
    "molécule, la dose et la durée."
)
PUBLIC_WARNING: Final = (
    "Démonstration publique : n'entrez jamais les informations d'un vrai patient. Ce que "
    "vous saisissez part vers un serveur le temps du calcul, puis est oublié : rien n'est "
    "enregistré."
)
NOT_A_DEVICE: Final = (
    "Outil de recherche d'un projet de Master 2, pas un dispositif médical. Il montre ce "
    "qui a été fait le plus souvent pour des patients au profil proche : la décision "
    "revient toujours au médecin."
)
SYNTHETIC_NOTE: Final = (
    "Ce modèle a appris sur des patients inventés : ses résultats n'ont aucune valeur "
    "médicale, ils servent seulement à montrer l'outil."
)
REAL_NOTE: Final = (
    "Ce modèle a appris sur {n} sevrages d'un service d'addictologie. Le site ne contient "
    "aucune de leurs données, seulement ce que le modèle en a retenu."
)
UNKNOWN_LABEL: Final = "Je ne sais pas"
GUIDE: Final[tuple[tuple[str, str], ...]] = (
    (
        "Le résumé",
        "Ce qui a été fait le plus souvent pour des patients qui ressemblent à celui-ci. "
        "Ce n'est pas une prescription.",
    ),
    (
        "La confiance",
        "Plutôt sûr, moyennement sûr ou pas sûr. Elle baisse quand des informations manquent, "
        "quand le patient ne ressemble pas à ceux de l'apprentissage ou quand le modèle hésite.",
    ),
    (
        "Les pourcentages",
        "Sur 100 patients comparables, combien étaient dans ce cas : 75 % veut dire 3 sur 4.",
    ),
    (
        "Les réponses possibles",
        "Celles que le modèle ne peut pas écarter. La bonne réponse en fait partie 9 fois "
        "sur 10 chez des patients comparables.",
    ),
    (
        "Ce qui a pesé",
        "Les informations qui ont fait monter ou baisser le résultat, comparé à un patient "
        "« type ». Ce n'est pas un lien de cause à effet.",
    ),
)
_TRUST: Final[Mapping[ConfidenceLevel, tuple[str, str]]] = {
    ConfidenceLevel.HIGH: ("Plutôt sûr", "Le modèle est plutôt sûr de ce résultat."),
    ConfidenceLevel.MODERATE: (
        "Moyennement sûr",
        "Le modèle hésite un peu : à prendre avec prudence.",
    ),
    ConfidenceLevel.LOW: (
        "Pas sûr",
        "Le modèle n'est pas sûr de lui : mieux vaut s'en tenir au protocole habituel du service.",
    ),
}
_OUTPUTS: Final[Mapping[TargetName, tuple[str, str]]] = {
    TargetName.SEVERITY: ("Gravité du sevrage", "Le sevrage risque-t-il d'être sévère ?"),
    TargetName.TREATMENT: ("Traitement", "Faudra-t-il une benzodiazépine ?"),
    TargetName.MOLECULE: ("Molécule", "Si on traite, avec quelle molécule ?"),
    TargetName.DOSE_DAY1: ("Dose du premier jour", "Si on traite, quelle dose le premier jour ?"),
    TargetName.DOSE_CUMULATIVE: ("Dose totale", "Si on traite, quelle dose en tout ?"),
    TargetName.DURATION: ("Durée", "Si on traite, pendant combien de jours ?"),
}
_FLAG_TITLES: Final[Mapping[str, str]] = {
    "critical": "À regarder de près",
    "warning": "À savoir",
    "info": "Pour information",
}
WEIGHTS_HINT: Final = (
    "Ce qui a fait monter ou baisser ce résultat pour ce patient, comparé à un patient « type » "
    "(les valeurs les plus courantes). Ce n'est pas un lien de cause à effet."
)
_TREATED_HEADLINE: Final = (
    "Pour des patients qui lui ressemblent, on a le plus souvent prescrit une benzodiazépine."
)
_UNTREATED_HEADLINE: Final = (
    "Pour des patients qui lui ressemblent, on n'a le plus souvent pas prescrit de "
    "benzodiazépine : surveillance selon le protocole du service."
)
_INVALID: Final = "Certaines valeurs ne conviennent pas : corrigez-les puis relancez."
_NBSP: Final = chr(0x00A0)


class RequestError(Exception):
    """A request the page can fix (French ``message`` and ``details``)."""

    def __init__(self, status: int, message: str, details: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = tuple(details)


@dataclass(frozen=True, slots=True)
class Response:
    """HTTP status, body and content type of an answer."""

    status: int
    body: bytes
    content_type: str = JSON_TYPE


def _finite(value: object) -> object:
    """``value`` with every non-finite float replaced by ``None`` (strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def json_response(status: int, payload: Mapping[str, object]) -> Response:
    """Strict UTF-8 JSON answer."""
    text = json.dumps(_finite(payload), ensure_ascii=False, allow_nan=False)
    return Response(status=status, body=text.encode("utf-8"))


def _trust(level: ConfidenceLevel) -> Json:
    label, sentence = _TRUST[level]
    return {"level": level.value, "label": label, "sentence": sentence}


def _why(explanation: vm.ConfidenceExplanation) -> list[Json]:
    return [{"status": item.status, "text": item.text} for item in explanation.items]


def _human_note(note: str, threshold: float) -> str | None:
    """Plain wording of a card note, ``None`` when the page already says it."""
    if note == vm.LOW_CONFIDENCE_MESSAGE or note.startswith(("Intervalle non borné", "« ")):
        return None
    if note.startswith("Probabilité de traitement"):
        return f"Le modèle répond « oui » à partir de {round(100 * threshold)} % de probabilité."
    if note.startswith("Zone grise"):
        return "C'est proche de la limite : à discuter au cas par cas."
    if note.startswith("Aucun traitement recommandé"):
        return "Le modèle ne conseille pas de traitement : valeur donnée à titre indicatif."
    return note


def _notes(notes: Sequence[str], threshold: float) -> list[str]:
    return [text for note in notes if (text := _human_note(note, threshold)) is not None]


def _level(alpha: float) -> int:
    return round(100 * (1.0 - alpha))


def _possible_text(card: vm.ClassCard, alpha: float) -> str:
    possible = [row.label for row in card.rows if row.in_set]
    if not possible:
        return "Aucune réponse ne ressort nettement : le modèle hésite."
    if len(possible) == len(card.rows) > 1:
        return "Le modèle ne peut écarter aucune réponse : il hésite."
    head = "Seule réponse possible" if len(possible) == 1 else "Réponses possibles"
    return (
        f"{head} : {', '.join(possible)}. La bonne réponse en fait partie {_level(alpha)} fois "
        "sur 100 chez des patients comparables."
    )


def _class_output(card: vm.ClassCard, alpha: float, threshold: float) -> Json:
    title, question = _OUTPUTS.get(card.target, (card.title, card.subtitle))
    notes = _notes(card.notes, threshold)
    lead, tag = "Réponse", "réponse retenue"
    top = max(card.rows, key=lambda row: row.probability)
    if not top.predicted:
        # Prior-corrected decision (severity): the flagged class is not the most probable.
        lead, tag = "Niveau à anticiper", "niveau signalé"
        retained = next(row for row in card.rows if row.predicted)
        notes.insert(
            0,
            f"Ce niveau est signalé car, chez ce patient, sa probabilité "
            f"({round(100 * retained.probability)} %) dépasse le plus sa fréquence habituelle, "
            f"même si « {top.label} » reste le plus probable.",
        )
    return {
        "key": card.target.value,
        "kind": "class",
        "title": title,
        "question": question,
        "lead": lead,
        "tag": tag,
        "answer": card.decision,
        "rows": [
            {
                "label": row.label,
                "percent": round(100 * row.probability),
                "answer": row.predicted,
                "possible": row.in_set,
            }
            for row in card.rows
        ],
        "possible": _possible_text(card, alpha),
        "trust": _trust(card.badge.level),
        "why": _why(card.explanation),
        "notes": notes,
    }


def _range_text(output: RegressionOutput, unit: str) -> str:
    if math.isinf(output.upper):
        return "L'incertitude est trop grande pour donner une fourchette."
    low, high = vm.format_number(output.lower), vm.format_number(output.upper)
    return (
        f"Pour {_level(output.alpha)} patients comparables sur 100 : entre {low} et "
        f"{high}{_NBSP}{unit}."
    )


def _number_output(card: vm.RegressionCard, output: RegressionOutput, threshold: float) -> Json:
    title, question = _OUTPUTS.get(card.target, (card.title, card.subtitle))
    unit = vm.unit_label(output.unit) or output.unit
    return {
        "key": card.target.value,
        "kind": "number",
        "title": title,
        "question": question,
        "answer": card.value_text,
        "answer_note": card.molecule_value_text,
        "point": output.point,
        "lower": output.lower,
        "upper": None if math.isinf(output.upper) else output.upper,
        "unit": unit,
        "range": _range_text(output, unit),
        "trust": _trust(card.badge.level),
        "why": _why(card.explanation),
        "notes": _notes(card.notes, threshold),
    }


def _summary(view: vm.PredictionView, level: ConfidenceLevel) -> Json:
    summary = view.summary
    return {
        "treat": summary.treat,
        "headline": _TREATED_HEADLINE if summary.treat else _UNTREATED_HEADLINE,
        "figures": [
            {"label": figure.label, "value": figure.value, "help": figure.help}
            for figure in summary.figures
        ],
        "note": summary.figures_note,
        "trust": _trust(level),
        "why": _why(summary.explanation),
    }


def _canonical(field: vm.FormField, raw: object) -> str | float | None:
    """Canonical value of one sent value, ``None`` when unknown or invalid."""
    if field.kind is FeatureKind.NUMERIC:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return None
        number = float(raw)
        return number if math.isfinite(number) else None
    for choice in field.choices:
        if choice.value is not None and raw == choice.value:
            return choice.value
    return None


def _field(field: vm.FormField) -> Json:
    hint = None
    if field.min_value is not None and field.max_value is not None:
        low, high = vm.format_plain(field.min_value), vm.format_plain(field.max_value)
        hint = f"Entre {low} et {high}" + (f"{_NBSP}{field.unit}" if field.unit else "")
    return {
        "name": field.name,
        "label": field.label,
        "kind": field.kind.value,
        "hint": hint,
        "min": field.min_value,
        "max": field.max_value,
        "choices": [
            {
                "label": UNKNOWN_LABEL if choice.value is None else choice.label,
                "value": choice.value,
            }
            for choice in field.choices
        ],
    }


class WebService:
    """What the web page asks of a predictor: the form and the readable predictions."""

    def __init__(self, predictor: PredictorProtocol) -> None:
        self._predictor = predictor
        self._schema = predictor.schema
        self._info = predictor.info
        self._groups = vm.form_groups(self._schema, self._info.feature_names)
        self._fields = {field.name: field for group in self._groups for field in group.fields}

    @classmethod
    def from_model_file(cls, path: Path) -> WebService:
        """Service of a slim model file (:func:`sevrage_ai.web.slim.save_slim_model`)."""
        return cls(load_slim_predictor(path))

    def config(self) -> Json:
        """Texts, model summary and form of the page."""
        info = self._info
        performance = [
            {
                "output": row["Sortie"],
                "measure": row["Mesure"],
                "value": row["Valeur [IC 95 %]"],
                "hint": row["Comment la lire"],
            }
            for row in vm.performance_rows(info, self._schema)
        ]
        return {
            "title": TITLE,
            "intro": INTRO,
            "warnings": [PUBLIC_WARNING, NOT_A_DEVICE],
            "model": {
                "synthetic": info.is_synthetic,
                "note": SYNTHETIC_NOTE if info.is_synthetic else REAL_NOTE.format(n=info.n_train),
                "features": len(info.feature_names),
                "performance": performance,
                "protocol": info.evaluation_protocol,
            },
            "groups": [
                {"label": group.label, "fields": [_field(field) for field in group.fields]}
                for group in self._groups
            ],
            "example": {k: v for k, v in vm.EXAMPLE_PATIENT.items() if k in self._fields},
            "guide": [{"title": title, "text": text} for title, text in GUIDE],
            "weights_hint": WEIGHTS_HINT,
        }

    def record(self, values: Mapping[str, object]) -> dict[str, str | float | None]:
        """Canonical record of the sent values: model features only, invalid ones unknown."""
        return {name: _canonical(field, values.get(name)) for name, field in self._fields.items()}

    def predict(self, values: Mapping[str, object]) -> Json:
        """Readable prediction of the sent values.

        Raises:
            RequestError: invalid values (422), with French details.
        """
        record = self.record(values)
        errors = vm.validation_messages(self._schema, record, self._info.feature_set)
        if errors:
            raise RequestError(422, _INVALID, errors)
        try:
            prediction = self._predictor.predict_record(record)
        except SevrageError as exc:
            raise RequestError(422, _INVALID, exc.details) from exc
        filled = sum(value is not None for value in record.values())
        return self.prediction_json(prediction, filled=filled)

    def prediction_json(self, prediction: PatientPrediction, *, filled: int) -> Json:
        """Display structure of ``prediction`` (``filled`` = number of known features)."""
        view = vm.prediction_view(prediction, self._schema, self._info)
        alpha, threshold = self._info.alpha, self._info.treatment_threshold
        outputs: list[Json] = [
            _class_output(view.severity, alpha, threshold),
            _class_output(view.treatment, alpha, threshold),
            _class_output(view.molecule, alpha, threshold),
        ]
        numbers = (
            (view.dose, prediction.dose_cumulative_if_treated),
            (view.duration, prediction.duration_if_treated),
        )
        outputs += [
            _number_output(c, o, threshold) for c, o in numbers if c is not None and o is not None
        ]
        weights = [
            {
                "key": target.value,
                "title": _OUTPUTS.get(target, (title, ""))[0],
                "axis": vm.contribution_axis_title(self._schema, target),
                "items": [
                    {
                        "label": item.label,
                        "value": item.value_display,
                        "delta": item.delta,
                        "text": item.delta_text,
                    }
                    for item in vm.contributions(prediction, self._schema, target)
                ],
            }
            for target, title in view.contribution_targets
        ]
        return {
            "synthetic": prediction.trained_on_synthetic,
            "filled": filled,
            "total": len(self._fields),
            "summary": _summary(view, prediction.overall_confidence),
            "outputs": outputs,
            "alerts": [
                {
                    "level": group.level,
                    "title": _FLAG_TITLES.get(group.level, group.title),
                    "items": [item.message for item in group.items],
                }
                for group in view.flags
            ],
            "weights": weights,
            "raw": export.prediction_to_dict(prediction),
        }


def handle(service: WebService | None, method: str, target: str, body: bytes) -> Response:
    """Answer one request of the page (``target`` = request path, query string allowed)."""
    path = urlsplit(target).path.rstrip("/")
    if path not in (CONFIG_ROUTE, PREDICT_ROUTE):
        return json_response(404, {"error": "Page introuvable."})
    if service is None:
        return json_response(503, {"error": "Le modèle n'est pas disponible pour le moment."})
    if path == CONFIG_ROUTE:
        if method != "GET":
            return json_response(405, {"error": "Méthode non autorisée."})
        return json_response(200, service.config())
    if method != "POST":
        return json_response(405, {"error": "Méthode non autorisée."})
    if len(body) > MAX_BODY_BYTES:
        return json_response(413, {"error": "Requête trop volumineuse."})
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return json_response(400, {"error": "Requête illisible."})
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return json_response(400, {"error": "Requête illisible."})
    try:
        return json_response(200, service.predict(values))
    except RequestError as exc:
        return json_response(exc.status, {"error": exc.message, "details": list(exc.details)})
