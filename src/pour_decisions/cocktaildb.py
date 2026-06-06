"""Load CocktailDB (CC0) as SILVER cocktails for TRAIN augmentation only."""

from __future__ import annotations

import csv
from pathlib import Path

from pour_decisions.schema import Cocktail, Ingredient
from pour_decisions.silver import parse_silver_line


def load_cocktaildb_silver(drinks_csv: Path, ingredients_csv: Path) -> list[Cocktail]:
    names: dict[str, str] = {}
    with open(drinks_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[row["id"]] = row["name"]

    grouped: dict[str, list[Ingredient]] = {}
    order: list[str] = []
    with open(ingredients_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["id"]
            if cid not in grouped:
                grouped[cid] = []
                order.append(cid)
            grouped[cid].append(
                parse_silver_line(row["ingredient_name"], row.get("ingredient_link"))
            )

    return [
        Cocktail(name=names.get(cid, f"drink-{cid}"), ingredients=grouped[cid])
        for cid in order
        if grouped[cid]
    ]
