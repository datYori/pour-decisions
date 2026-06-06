"""Target JSON schema for cocktail extraction (single source of truth).

Consumed by data_prep (training targets), metrics (eval), and serve_client
(vLLM `guided_json`). See docs/learning/05-eval-metrics.md for why we keep
`unit` a free string normalized in code rather than a strict enum.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Controlled vocabulary the model is trained toward. `unit` stays `str | None`
# (not an Enum) so guided decoding never hard-fails on an unseen token; the
# canonical set is enforced by normalization + measured in eval.
CANONICAL_UNITS: set[str] = {
    "ml",
    "cl",
    "oz",
    "dash",
    "tsp",
    "tbsp",
    "barspoon",
    "splash",
    "drop",
    "twist",
    "piece",
    "part",
    "cube",
    "slice",
    "wedge",
    "leaf",
    "shot",
    "pinch",
}

_UNIT_ALIASES: dict[str, str] = {
    "dashes": "dash",
    "drops": "drop",
    "twists": "twist",
    "twist of": "twist",
    "pieces": "piece",
    "parts": "part",
    "cubes": "cube",
    "slices": "slice",
    "wedges": "wedge",
    "leaves": "leaf",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "bar spoon": "barspoon",
    "centilitre": "cl",
    "millilitre": "ml",
    "milliliter": "ml",
    "ounce": "oz",
    "ounces": "oz",
    "shots": "shot",
    "tblsp": "tbsp",
}


def normalize_unit(raw: str | None) -> str | None:
    """Map a raw unit token to the canonical vocabulary, or None if empty/unknown-blank."""
    if raw is None:
        return None
    token = raw.strip().lower()
    if not token:
        return None
    if token in CANONICAL_UNITS:
        return token
    return _UNIT_ALIASES.get(token)  # None when not mappable


class Ingredient(BaseModel):
    quantity: float | None = Field(default=None, description="Numeric amount, or null if none.")
    unit: str | None = Field(default=None, description="Canonical unit, or null.")
    ingredient: str = Field(description="Ingredient name with quantity/unit stripped.")


class Cocktail(BaseModel):
    name: str
    ingredients: list[Ingredient]
    category: str | None = None
    method: str | None = None
    garnish: str | None = None
