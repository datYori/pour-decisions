from pathlib import Path

from pour_decisions.iba import load_iba_gold

FIX = Path(__file__).parent / "fixtures"


def test_load_iba_gold_groups_by_cocktail():
    cocktails = load_iba_gold(FIX / "iba_ingredients_tiny.csv", FIX / "iba_cocktails_tiny.csv")
    by_name = {c.name: c for c in cocktails}
    assert set(by_name) == {"Bellini", "Negroni"}
    negroni = by_name["Negroni"]
    assert len(negroni.ingredients) == 5
    assert negroni.ingredients[0].quantity == 30.0
    assert negroni.ingredients[0].unit == "ml"
    assert negroni.ingredients[0].ingredient == "Gin"
    assert negroni.method == "Stir over ice."
    assert negroni.garnish == "Orange peel"
    assert negroni.category == "Unforgettables"


def test_garnish_na_becomes_none():
    cocktails = load_iba_gold(FIX / "iba_ingredients_tiny.csv", FIX / "iba_cocktails_tiny.csv")
    bellini = next(c for c in cocktails if c.name == "Bellini")
    assert bellini.garnish is None


def test_parse_fractional_and_word_quantities():
    cocktails = load_iba_gold(FIX / "iba_ingredients_tiny.csv", FIX / "iba_cocktails_tiny.csv")
    negroni = next(c for c in cocktails if c.name == "Negroni")
    by_ingredient = {i.ingredient: i for i in negroni.ingredients}
    assert by_ingredient["Orange Juice"].quantity == 0.5
    assert by_ingredient["Bitters"].quantity is None
