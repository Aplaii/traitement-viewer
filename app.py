"""Point d'entrée de Vercel : une application WSGI qui sert la page et l'API.

Vercel charge la variable ``app`` de ce fichier et lui envoie toutes les requêtes
(``/`` et les fichiers de ``public/``, ``/api/predict``). Voir ``sevrage_ai.web.wsgi``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sevrage_ai.web.wsgi import make_app  # noqa: E402 - after the path set-up

app = make_app(ROOT / "model" / "model.joblib", ROOT / "public")
