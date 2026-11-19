"""Safety-rule annotations (R6): informational flags that never change outputs. Owner: WS8a.

A rule fires when ``conditions.evaluate(rule.when, x_row)`` is ``True``, ``applies_to``
matches ``final.treat``, and ``final.molecule`` is in ``molecule_in`` (when set). It yields
``Flag(code=f"safety.{rule.id}", level=rule.level, message_fr=rule.message_fr)``.

Rules never change ``final``; they are pure, order-preserving annotations evaluated on the
canonical admission features of one patient (``x_row``, exactly one row).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from sevrage_ai.config.conditions import evaluate as evaluate_condition
from sevrage_ai.config.schema import SafetyRule
from sevrage_ai.types import FinalRecommendation, Flag

__all__ = ["evaluate"]


def _applies(rule: SafetyRule, final: FinalRecommendation) -> bool:
    """Whether ``rule.applies_to`` and ``rule.molecule_in`` match ``final``."""
    if rule.applies_to == "treated" and not final.treat:
        return False
    if rule.applies_to == "untreated" and final.treat:
        return False
    return rule.molecule_in is None or final.molecule in rule.molecule_in


def evaluate(
    x_row: pd.DataFrame, final: FinalRecommendation, rules: Sequence[SafetyRule]
) -> list[Flag]:
    """Flags of the rules that fire for the one-row canonical frame ``x_row``, in rule order."""
    flags: list[Flag] = []
    for rule in rules:
        if not _applies(rule, final):
            continue
        if not bool(evaluate_condition(rule.when, x_row).iloc[0]):
            continue
        flags.append(Flag(code=f"safety.{rule.id}", level=rule.level, message_fr=rule.message_fr))
    return flags
