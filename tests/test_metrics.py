from pour_decisions.metrics import field_accuracy, hallucination_rate, json_validity
from pour_decisions.schema import Cocktail, Ingredient


def _c(items):
    return Cocktail(name="X", ingredients=[Ingredient(**d) for d in items])


def test_field_accuracy_perfect():
    gold = [_c([{"quantity": 30, "unit": "ml", "ingredient": "Gin"}])]
    pred = [_c([{"quantity": 30, "unit": "ml", "ingredient": "gin"}])]  # case-insensitive name
    acc = field_accuracy(pred, gold)
    assert acc["quantity"] == 1.0 and acc["unit"] == 1.0 and acc["ingredient"] == 1.0


def test_field_accuracy_partial():
    gold = [_c([{"quantity": 30, "unit": "ml", "ingredient": "Gin"}])]
    pred = [_c([{"quantity": 45, "unit": "ml", "ingredient": "Gin"}])]  # wrong qty
    acc = field_accuracy(pred, gold)
    assert acc["quantity"] == 0.0 and acc["unit"] == 1.0


def test_hallucination_rate_counts_invented_values():
    gold = [_c([{"quantity": None, "unit": None, "ingredient": "Soda"}])]
    pred = [_c([{"quantity": 1, "unit": "oz", "ingredient": "Soda"}])]  # invented qty+unit
    assert hallucination_rate(pred, gold) == 1.0


def test_json_validity():
    schema = Cocktail.model_json_schema()
    good = '{"name":"X","ingredients":[{"quantity":1,"unit":"oz","ingredient":"Rum"}]}'
    bad = '{"name":"X"}'  # missing required ingredients
    assert json_validity([good, bad], schema) == 0.5
