"""Load the IBA (MIT) gold set: raw ingredient_direction + parsed quantity/unit/ingredient."""

from __future__ import annotations

import csv
from pathlib import Path

from pour_decisions.schema import Cocktail, Ingredient, normalize_unit


def _clean(value: str) -> str | None:
    v = value.strip()
    return None if v in ("", "NA") else v


def _parse_qty(raw: str) -> float | None:
    v = raw.strip()
    if not v or v == "NA":
        return None
    try:
        return float(v)
    except ValueError:
        if v.count("/") == 1:  # simple fraction "1/2","1/4"
            num, den = v.split("/")
            try:
                return float(num) / float(den)
            except ValueError:
                return None
        return None  # ranges "2-3", words "few" -> None


def load_iba_gold(ingredients_csv: Path, cocktails_csv: Path) -> list[Cocktail]:
    meta: dict[str, dict[str, str | None]] = {}
    with open(cocktails_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["name"]] = {
                "category": _clean(row.get("category", "")),
                "method": _clean(row.get("method", "")),
                "garnish": _clean(row.get("garnish", "")),
            }

    grouped: dict[str, list[Ingredient]] = {}
    order: list[str] = []
    with open(ingredients_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name not in grouped:
                grouped[name] = []
                order.append(name)
            grouped[name].append(
                Ingredient(
                    quantity=_parse_qty(row["quantity"]),
                    unit=normalize_unit(row["unit"]),
                    ingredient=row["ingredient"].strip(),
                )
            )

    cocktails: list[Cocktail] = []
    for name in order:
        m = meta.get(name, {})
        cocktails.append(
            Cocktail(
                name=name,
                ingredients=grouped[name],
                category=m.get("category"),
                method=m.get("method"),
                garnish=m.get("garnish"),
            )
        )
    return cocktails
