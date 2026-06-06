"""Eval metrics: base-vs-tuned signal lives in field accuracy + hallucination.

JSON validity is ~100% under guided decoding, so it is a guardrail, not the
headline. Ingredients are aligned by position (model is asked to preserve input
order). See docs/learning/05-eval-metrics.md.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import jsonschema  # type: ignore[import-untyped]

from pour_decisions.schema import Cocktail, Ingredient


def _aligned(
    pred: list[Cocktail], gold: list[Cocktail]
) -> Generator[tuple[Ingredient, Ingredient], None, None]:
    for p, g in zip(pred, gold, strict=True):
        yield from zip(p.ingredients, g.ingredients, strict=False)  # position-aligned


def field_accuracy(pred: list[Cocktail], gold: list[Cocktail]) -> dict[str, float]:
    totals = {"quantity": 0, "unit": 0, "ingredient": 0}
    correct = {"quantity": 0, "unit": 0, "ingredient": 0}
    for pi, gi in _aligned(pred, gold):
        totals["quantity"] += 1
        totals["unit"] += 1
        totals["ingredient"] += 1
        correct["quantity"] += int(pi.quantity == gi.quantity)
        correct["unit"] += int(pi.unit == gi.unit)
        correct["ingredient"] += int(pi.ingredient.strip().lower() == gi.ingredient.strip().lower())
    return {k: (correct[k] / totals[k] if totals[k] else 0.0) for k in totals}


def hallucination_rate(pred: list[Cocktail], gold: list[Cocktail]) -> float:
    invented = 0
    null_slots = 0
    for pi, gi in _aligned(pred, gold):
        for field in ("quantity", "unit"):
            if getattr(gi, field) is None:
                null_slots += 1
                invented += int(getattr(pi, field) is not None)
    return invented / null_slots if null_slots else 0.0


def json_validity(raw_outputs: list[str], schema: dict[str, object]) -> float:
    if not raw_outputs:
        return 0.0
    ok = 0
    for raw in raw_outputs:
        try:
            jsonschema.validate(json.loads(raw), schema)
            ok += 1
        except (json.JSONDecodeError, jsonschema.ValidationError):
            pass
    return ok / len(raw_outputs)
