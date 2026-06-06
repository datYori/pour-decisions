# Data sources & licenses

Raw data in this folder is redistributable under the licenses below (verified 2026-05-31).
Generated/derived data lives in `../prepared/` and is gitignored.

## `cocktaildb/` — License: **CC0-1.0 (Public Domain)**
- Source: Kaggle `pxxthik/the-cocktail-db-recipe-collection`, scraped from <https://www.thecocktaildb.com/>.
- License verified on the dataset card and via the Kaggle metadata API (`"licenses": [{"name": "CC0-1.0"}]`).
- Files:
  - `drinks.csv` — **621 cocktails** — columns: `id, name, link, image_link` (names only; no category/glass/alcoholic/method).
  - `ingredients.csv` — **2,509 rows** — columns: `id` (FK → `drinks.id`), `ingredient_name` (raw combined string, e.g. `"1 oz  Coconut rum"`, `"1 twist of  Lemon peel"`), `ingredient_link` (slug → canonical ingredient), `ingredient_image`.
- Role: **train-only SILVER volume** for the measure-parse (quantity/unit are not pre-parsed; the canonical ingredient name is recoverable from the link slug). US-bar units (`oz`, `twist of`, `dash`).
- CC0 waives attribution; credited here as a courtesy.

## `iba/` — License: **MIT** (Copyright (c) 2023 Rasmus Bååth — see `iba/LICENSE`)
- Source: GitHub `rasmusab/iba-cocktails`, **`iba-web/` folder only**. The upstream `wikipedia/` folder is CC-BY-SA 3.0 and was **deliberately excluded** to avoid share-alike.
- Files:
  - `iba-cocktails-ingredients-web.csv` — **356 rows** — columns: `category, name, ingredient_direction` (raw), `quantity, unit, ingredient, note` ← **GOLD parse labels**.
  - `iba-cocktails-web.csv` — **90 cocktails** — columns: `category, name, ingredients, method, garnish`.
  - `iba-cocktails-web.json` — combined form.
  - `LICENSE` — the MIT license text (retained as required).
- Role: **GOLD** — the eval set + train core. All-metric (`ml`).

## Combined-corpus rationale
- **Eval held-out = gold IBA only** (label integrity — never grade against self-made silver labels).
- **Train = gold IBA + silver CocktailDB** (~8× volume, and imperial↔metric unit diversity → cross-unit normalization).
