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

__all__ = ["SECURITY_HEADERS", "ApiHandler", "ServiceLoader", "make_handler", "service_loader"]

_LOGGER = get_logger(__name__)
SECURITY_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)


class ServiceLoader:
    """Lazy, cached loader of the service of ``model_path`` (``None`` while unreadable).

    ``failure`` is the type of the last loading error (no message): the page shows it, to
    help diagnose a deployment. The full message goes to the server log (no patient data:
    the web model holds none).
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.failure: str | None = None
        self._service: WebService | None = None

    def __call__(self) -> WebService | None:
        """The service, loaded on the first call (retried while it fails)."""
        if self._service is None:
            try:
                self._service = WebService.from_model_file(self.model_path)
                self.failure = None
            except Exception as exc:  # unreadable model: answered 503, retried next time
                self.failure = type(exc).__name__
                _LOGGER.error("Modèle web illisible (%s : %s).", self.failure, str(exc)[:300])
        return self._service


def service_loader(model_path: Path) -> ServiceLoader:
    """Lazy, cached loader of the service of ``model_path``."""
    return ServiceLoader(model_path)


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
            loader = type(self).load_service
            service = loader()
            return handle(service, method, self.path, body, cause=getattr(loader, "failure", None))
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
