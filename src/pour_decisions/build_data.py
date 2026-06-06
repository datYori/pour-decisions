"""Build data/prepared/{train,val,test}.jsonl from the committed gold IBA set."""

import sys
from pathlib import Path

from pour_decisions.cocktaildb import load_cocktaildb_silver
from pour_decisions.dataset import build_record, split_by_cocktail, write_jsonl
from pour_decisions.iba import load_iba_gold

RAW = Path("data/raw/iba")
OUT = Path("data/prepared")


def main() -> None:
    cocktails = load_iba_gold(
        RAW / "iba-cocktails-ingredients-web.csv", RAW / "iba-cocktails-web.csv"
    )
    train, val, test = split_by_cocktail(cocktails, ratios=(0.7, 0.15, 0.15), seed=0)
    n_train, n_val, n_test = len(train), len(val), len(test)
    print(f"gold IBA: {len(cocktails)} cocktails -> train {n_train} / val {n_val} / test {n_test}")

    if "--augment" in sys.argv:
        silver = load_cocktaildb_silver(
            Path("data/raw/cocktaildb/drinks.csv"), Path("data/raw/cocktaildb/ingredients.csv")
        )
        train = train + silver  # silver added to TRAIN only; val/test stay gold IBA
        print(f"augmented train with {len(silver)} silver CocktailDB cocktails (train-only)")

    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        write_jsonl([build_record(c) for c in split], OUT / f"{name}.jsonl")


if __name__ == "__main__":
    main()
