from pour_decisions.schema import Cocktail
from pour_decisions.serve_client import build_request, parse_cocktail


def test_build_request_uses_json_schema_response_format():
    req = build_request("Negroni", ["30 ml Gin", "30 ml Campari"], model="m")
    assert req["response_format"]["type"] == "json_schema"
    assert "schema" in req["response_format"]["json_schema"]
    assert req["messages"][0]["role"] == "system"
    assert "Negroni" in req["messages"][1]["content"]


def test_parse_cocktail_roundtrip():
    raw = Cocktail(name="Negroni", ingredients=[]).model_dump_json()
    c = parse_cocktail(raw)
    assert isinstance(c, Cocktail) and c.name == "Negroni"
