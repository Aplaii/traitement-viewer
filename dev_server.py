"""Aperçu local de la version web, comme sur Vercel : page ``public/`` et ``/api/predict``.

    .venv/Scripts/python.exe vercel/dev_server.py --port 8765

Lancer d'abord ``scripts/export_vercel.py`` (modèle, paquet copié, ``public/config.json``).
Écoute uniquement sur 127.0.0.1.
"""

from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Final, override
from urllib.parse import urlsplit

ROOT: Final = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sevrage_ai.web.server import ApiHandler, service_loader  # noqa: E402 - after the path set-up
from sevrage_ai.web.service import Response  # noqa: E402

PUBLIC: Final = ROOT / "public"
CONTENT_TYPES: Final = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}
CSP: Final = (
    "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


class DevHandler(ApiHandler):
    """``/api/*`` comme sur Vercel, le reste lu dans ``public/``."""

    load_service = staticmethod(service_loader(ROOT / "model" / "model.joblib"))

    @override
    def do_GET(self) -> None:
        """Static file, or the API for ``/api/*``."""
        if self.path.startswith("/api/"):
            super().do_GET()
            return
        self.answer(self._static())

    def _static(self) -> Response:
        relative = urlsplit(self.path).path.lstrip("/") or "index.html"
        target = (PUBLIC / relative).resolve()
        content_type = CONTENT_TYPES.get(target.suffix)
        if not target.is_relative_to(PUBLIC.resolve()) or not target.is_file() or not content_type:
            return Response(404, b"Introuvable", "text/plain; charset=utf-8")
        return Response(200, target.read_bytes(), content_type)

    @override
    def end_headers(self) -> None:
        """Same Content-Security-Policy as ``vercel.json``."""
        self.send_header("Content-Security-Policy", CSP)
        super().end_headers()


def main() -> None:
    """Serve until interrupted (Ctrl+C)."""
    parser = argparse.ArgumentParser(description="Aperçu local de la version web.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DevHandler)
    print(f"Aperçu : http://127.0.0.1:{args.port}/  (Ctrl+C pour arrêter)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
