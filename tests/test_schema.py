import json

from pour_decisions.schema import CANONICAL_UNITS, Cocktail, Ingredient, normalize_unit


def test_ingredient_accepts_null_quantity_and_unit():
    ing = Ingredient(quantity=None, unit=None, ingredient="Soda Water")
    assert ing.quantity is None and ing.unit is None


def test_cocktail_optional_fields_default_none():
    c = Cocktail(name="Negroni", ingredients=[Ingredient(quantity=30, unit="ml", ingredient="Gin")])
    assert c.category is None and c.method is None and c.garnish is None


def test_normalize_unit_maps_variants():
    assert normalize_unit("dashes") == "dash"
    assert normalize_unit("OZ") == "oz"
    assert normalize_unit("cl") == "cl"
    assert normalize_unit("") is None
    assert normalize_unit("twist of") == "twist"


def test_json_schema_exports_and_has_required_fields():
    schema = Cocktail.model_json_schema()
    assert schema["type"] == "object"
    assert "ingredients" in schema["properties"]
    # round-trips as JSON (used by vLLM guided decoding)
    json.dumps(schema)


def test_canonical_units_is_nonempty_set():
    assert "ml" in CANONICAL_UNITS and "oz" in CANONICAL_UNITS
