"""POST /api/predict : prédiction pour un patient imaginaire (voir ``sevrage_ai.web``).

Fonction Python de Vercel : la classe ``handler`` répond aux requêtes. Le modèle
(``model/model.joblib``) et le paquet (``sevrage_ai/``) sont copiés à côté par
``scripts/export_vercel.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sevrage_ai.web.server import make_handler  # noqa: E402 - after the path set-up

handler = make_handler(ROOT / "model" / "model.joblib")
