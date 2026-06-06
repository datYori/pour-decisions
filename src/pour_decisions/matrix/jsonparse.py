"""Lenient extraction of a JSON object from free-form model output, and tolerant parse."""

from __future__ import annotations

from pydantic import ValidationError

from pour_decisions.schema import Cocktail


def extract_json_block(raw: str) -> str | None:
    """Return the first balanced {...} object as a string, or None. Ignores code fences/prose."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def parse_cocktail_lenient(raw: str) -> Cocktail | None:
    block = extract_json_block(raw)
    if block is None:
        return None
    try:
        return Cocktail.model_validate_json(block)
    except ValidationError:
        return None
