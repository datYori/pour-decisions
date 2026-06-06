"""Rule-based SILVER parser for CocktailDB raw strings ("1 oz  Coconut rum").

SILVER = machine-generated labels (not human gold). Used for TRAIN augmentation
ONLY. Eval always uses gold IBA — see docs/learning/05-eval-metrics.md.
"""

from __future__ import annotations

import re

from pour_decisions.schema import Ingredient, normalize_unit

_NULL_QTY_PREFIXES = ("fill with", "top with", "splash of", "dash of")
_QTY_RE = re.compile(r"^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s+(.*)$")


def _to_float(token: str) -> float:
    token = token.strip()
    if " " in token:  # mixed fraction "1 1/2"
        whole, frac = token.split()
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)
    if "/" in token:
        num, den = token.split("/")
        return float(num) / float(den)
    return float(token)


def slug_to_name(link: str | None) -> str | None:
    if not link:
        return None
    tail = link.rstrip("/").split("/")[-1]  # "135-Coconut-rum"
    tail = re.sub(r"^\d+-", "", tail)  # "Coconut-rum"
    return tail.replace("-", " ").strip() or None


def parse_silver_line(raw: str, link: str | None = None) -> Ingredient:
    text = re.sub(r"\s+", " ", raw).strip()
    lower = text.lower()
    canonical_name = slug_to_name(link)

    if any(lower.startswith(p) for p in _NULL_QTY_PREFIXES):
        name = canonical_name or re.sub(r"(?i)^(fill with|top with|splash of|dash of)\s*", "", text)
        return Ingredient(quantity=None, unit=None, ingredient=name.strip())

    m = _QTY_RE.match(text)
    if not m:  # no leading quantity, e.g. "Soda Water"
        return Ingredient(quantity=None, unit=None, ingredient=canonical_name or text)

    qty = _to_float(m.group(1))
    rest = m.group(2)
    # try "<unit>  <name>" (double space) else "<unit> <name>"
    unit, name = _split_unit_name(rest)
    return Ingredient(quantity=qty, unit=normalize_unit(unit), ingredient=canonical_name or name)


def _split_unit_name(rest: str) -> tuple[str | None, str]:
    if "  " in rest:
        unit, name = rest.split("  ", 1)
        return unit.strip(), name.strip()
    parts = rest.split(" ", 1)
    if len(parts) == 2 and normalize_unit(parts[0]) is not None:
        return parts[0], parts[1].strip()
    # handle "twist of X"
    m = re.match(r"^(twist of|dash of|piece of)\s+(.*)$", rest, re.I)
    if m:
        return m.group(1).split()[0], m.group(2).strip()
    return None, rest.strip()
