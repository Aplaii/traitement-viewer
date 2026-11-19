"""Aperçu local de la version web, avec la même application que sur Vercel (``app.py``).

    .venv/Scripts/python.exe vercel/dev_server.py --port 8765

Le modèle, le paquet copié et ``public/config.json`` doivent être présents (générés par
``scripts/export_vercel.py``). Écoute uniquement sur 127.0.0.1.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Final, override
from wsgiref.simple_server import WSGIRequestHandler, make_server

ROOT: Final = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402 - after the path set-up


class QuietHandler(WSGIRequestHandler):
    """No access log: it would print request lines."""

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


def main() -> None:
    """Serve until interrupted (Ctrl+C)."""
    parser = argparse.ArgumentParser(description="Aperçu local de la version web.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with make_server("127.0.0.1", args.port, app, handler_class=QuietHandler) as server:
        print(f"Aperçu : http://127.0.0.1:{args.port}/  (Ctrl+C pour arrêter)", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
