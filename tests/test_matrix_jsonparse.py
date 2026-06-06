from pour_decisions.matrix.jsonparse import extract_json_block, parse_cocktail_lenient

VALID = '{"name":"X","ingredients":[{"quantity":30.0,"unit":"ml","ingredient":"Gin"}]}'


def test_plain_json():
    assert extract_json_block(VALID) == VALID


def test_code_fenced():
    raw = f"```json\n{VALID}\n```"
    assert extract_json_block(raw) == VALID


def test_prose_prefix_and_trailing_text():
    raw = f"Sure! Here is the JSON:\n{VALID}\nHope that helps."
    assert extract_json_block(raw) == VALID


def test_nested_braces_balanced():
    raw = f"junk {VALID} more junk"
    assert extract_json_block(raw) == VALID


def test_no_json_returns_none():
    assert extract_json_block("there is no json here") is None


def test_parse_lenient_valid():
    c = parse_cocktail_lenient(f"```\n{VALID}\n```")
    assert c is not None and c.name == "X" and c.ingredients[0].unit == "ml"


def test_parse_lenient_invalid_returns_none():
    assert parse_cocktail_lenient("not json") is None
    assert parse_cocktail_lenient('{"name": 123}') is None  # schema-invalid
