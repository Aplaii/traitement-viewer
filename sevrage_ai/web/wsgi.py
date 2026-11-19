"""WSGI application of the web version: the page and the API in a single app.

Vercel's Python "entrypoint" mode runs one application (a top-level ``app`` in
``app.py``) and routes every request to it, so this app serves both the static page of
``public/`` and ``/api/*`` (see :func:`sevrage_ai.web.service.handle`). The local preview
uses the same app. Request lines, bodies and values are never logged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from typing import Final
from wsgiref.types import StartResponse, WSGIEnvironment

from sevrage_ai.logging_config import get_logger
from sevrage_ai.web.server import SECURITY_HEADERS, service_loader
from sevrage_ai.web.service import MAX_BODY_BYTES, Response, handle, json_response

__all__ = ["CONTENT_SECURITY_POLICY", "CONTENT_TYPES", "WsgiApp", "make_app"]

WsgiApp = Callable[[WSGIEnvironment, StartResponse], Iterable[bytes]]

_LOGGER = get_logger(__name__)
CONTENT_SECURITY_POLICY: Final = (
    "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
CONTENT_TYPES: Final[dict[str, str]] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
_NOT_FOUND: Final = Response(404, b"Page introuvable.", "text/plain; charset=utf-8")


def _static(root: Path, path: str) -> Response:
    """File of ``root`` for ``path`` (``/`` = ``index.html``), 404 outside ``root``."""
    relative = path.lstrip("/") or "index.html"
    target = (root / relative).resolve()
    content_type = CONTENT_TYPES.get(target.suffix.lower())
    if content_type is None or not target.is_relative_to(root) or not target.is_file():
        return _NOT_FOUND
    return Response(200, target.read_bytes(), content_type)


def _body(environ: WSGIEnvironment) -> bytes | None:
    """Request body, ``None`` when its length is missing, invalid or too large."""
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        return None
    if not 0 <= length <= MAX_BODY_BYTES:
        return None
    body: bytes = environ["wsgi.input"].read(length) if length else b""
    return body


def make_app(model_path: Path, public: Path) -> WsgiApp:
    """WSGI app serving ``public/`` and the API of the model of ``model_path``."""
    load = service_loader(model_path)
    root = public.resolve()

    def answer(environ: WSGIEnvironment) -> Response:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO") or "/")
        if not path.startswith("/api/"):
            if method not in ("GET", "HEAD"):
                return json_response(405, {"error": "Méthode non autorisée."})
            return _static(root, path)
        body = _body(environ) if method == "POST" else b""
        if body is None:
            return json_response(413, {"error": "Requête trop volumineuse."})
        try:
            return handle(load(), method, path, body)
        except Exception as exc:  # never show internals (or values) to the page
            _LOGGER.error("Erreur interne de la version web (%s).", type(exc).__name__)
            return json_response(500, {"error": "Erreur interne : réessayez plus tard."})

    def app(environ: WSGIEnvironment, start_response: StartResponse) -> Iterable[bytes]:
        response = answer(environ)
        headers = [
            ("Content-Type", response.content_type),
            ("Content-Length", str(len(response.body))),
            ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
            *SECURITY_HEADERS,
        ]
        start_response(f"{response.status} {HTTPStatus(response.status).phrase}", headers)
        if str(environ.get("REQUEST_METHOD", "GET")).upper() == "HEAD":
            return [b""]
        return [response.body]

    return app
