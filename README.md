# Sevrage IA · démonstration web

Page de démonstration d'un outil de recherche (projet Challenge IA, Master 2). À partir d'un
patient **imaginaire**, elle montre ce que le modèle anticipe à l'admission : la gravité du
sevrage d'alcool, le besoin d'un traitement par benzodiazépine, la molécule, la dose et la
durée.

**Ce dépôt ne contient aucune donnée de patient.** Le modèle (`model/model.joblib`) a été
entraîné sur des patients inventés : ses résultats n'ont aucune valeur médicale. Il a été
produit par `scripts/export_vercel.py` du projet principal, qui retire du modèle toute ligne
de patient.

## Contenu

- `public/` : la page (HTML, CSS, JavaScript) et `config.json` (textes et formulaire).
- `app.py` : le point d'entrée de Vercel, une application Python (WSGI) qui sert la page et
  calcule la prédiction (`/api/predict`).
- `api/predict.py` : la même API pour l'ancien mode « un fichier par fonction » de Vercel.
- `model/` : le modèle de démonstration.
- `sevrage_ai/` : la partie du paquet nécessaire à la prédiction (copie générée).
- `requirements.txt` et `.python-version` : les versions exactes, nécessaires pour relire le
  modèle.

## Mettre en ligne sur Vercel

Importez ce dépôt dans Vercel, avec la racine du dépôt comme dossier du projet et sans
commande de build. Vous pouvez aussi lancer `vercel deploy --prod` depuis ce dossier.

## Essayer en local

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows : .venv\Scripts\pip
.venv/bin/python dev_server.py                 # puis http://127.0.0.1:8765/
```

## Mettre à jour

Ne modifiez pas à la main `sevrage_ai/`, `model/` ni `public/config.json`. Ces fichiers sont
régénérés dans le projet principal par `uv run python -m scripts.export_vercel`, puis
recopiés ici.

## Avertissement

Outil de recherche, pas un dispositif médical : la décision revient toujours au médecin.
Ne saisissez jamais les informations d'un vrai patient.
