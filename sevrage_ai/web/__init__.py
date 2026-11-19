"""Web version of the predictor (Vercel): patient-free model file and JSON service.

:mod:`~sevrage_ai.web.slim` writes a copy of a trained model without any training row,
:mod:`~sevrage_ai.web.service` turns predictions into plain-French JSON for the page and
:mod:`~sevrage_ai.web.server` serves it with the standard library (``BaseHTTPRequestHandler``,
the interface of Vercel's Python functions).
"""
