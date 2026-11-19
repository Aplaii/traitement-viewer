"""HTTP handler of the web version, on the standard library only.

Vercel's Python functions expect a ``BaseHTTPRequestHandler`` subclass named ``handler``:
``vercel/api/predict.py`` sets ``handler = make_handler(<model file>)``. The local preview
(``vercel/dev_server.py``) uses the same class. The model is loaded on the first request
and kept for the following ones. Request lines and bodies are never logged.
"""

from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import ClassVar, Final

from sevrage_ai.logging_config import get_logger
from sevrage_ai.web.service import MAX_BODY_BYTES, Response, WebService, handle, json_response

__all__ = ["SECURITY_HEADERS", "ApiHandler", "make_handler", "service_loader"]

_LOGGER = get_logger(__name__)
SECURITY_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)


def service_loader(model_path: Path) -> Callable[[], WebService | None]:
    """Lazy, cached loader of the service of ``model_path`` (``None`` if unreadable)."""
    cache: dict[str, WebService | None] = {}

    def load() -> WebService | None:
        if "service" not in cache:
            try:
                cache["service"] = WebService.from_model_file(model_path)
            except Exception as exc:  # unreadable model: answered 503, retried next time
                _LOGGER.error("Modèle web illisible (%s).", type(exc).__name__)
                return None
        return cache["service"]

    return load


class ApiHandler(BaseHTTPRequestHandler):
    """``GET /api/config`` and ``POST /api/predict`` (see :func:`service.handle`)."""

    server_version = "sevrage-web"
    sys_version = ""
    load_service: ClassVar[Callable[[], WebService | None]]

    def do_GET(self) -> None:
        """Answer a GET request."""
        self.answer(self._dispatch("GET", b""))

    def do_POST(self) -> None:
        """Answer a POST request (body of at most ``MAX_BODY_BYTES``)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY_BYTES:
            self.answer(json_response(413, {"error": "Requête trop volumineuse."}))
            return
        self.answer(self._dispatch("POST", self.rfile.read(length)))

    def _dispatch(self, method: str, body: bytes) -> Response:
        try:
            return handle(type(self).load_service(), method, self.path, body)
        except Exception as exc:  # never show internals (or values) to the page
            _LOGGER.error("Erreur interne de la version web (%s).", type(exc).__name__)
            return json_response(500, {"error": "Erreur interne : réessayez plus tard."})

    def answer(self, response: Response) -> None:
        """Send ``response`` with the security headers."""
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the access log: it would print request lines."""
        del format, args


def make_handler(model_path: Path) -> type[ApiHandler]:
    """Handler class answering with the model of ``model_path``."""
    loader = service_loader(model_path)

    class Handler(ApiHandler):
        load_service = staticmethod(loader)

    return Handler
