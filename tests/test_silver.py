from pour_decisions.silver import parse_silver_line, slug_to_name


def test_parses_simple_oz():
    ing = parse_silver_line("1 oz  Coconut rum", "/ingredient/135-Coconut-rum")
    assert ing.quantity == 1.0 and ing.unit == "oz" and ing.ingredient == "Coconut rum"


def test_parses_fraction():
    ing = parse_silver_line("1/2 oz  Amaretto", None)
    assert ing.quantity == 0.5 and ing.unit == "oz"


def test_parses_mixed_fraction():
    ing = parse_silver_line("1 1/2 oz Tequila", None)
    assert ing.quantity == 1.5 and ing.unit == "oz"


def test_twist_unit_and_null_quantity_phrases():
    ing = parse_silver_line("1 twist of  Lemon peel", "/ingredient/294-Lemon-peel")
    assert ing.unit == "twist" and ing.ingredient == "Lemon peel"
    fill = parse_silver_line("Fill with  Soda Water", None)
    assert fill.quantity is None and fill.unit is None and fill.ingredient == "Soda Water"


def test_slug_to_name():
    assert slug_to_name("/ingredient/135-Coconut-rum") == "Coconut rum"
