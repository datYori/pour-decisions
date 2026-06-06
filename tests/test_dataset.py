import json
from pathlib import Path

from pour_decisions.dataset import build_record, split_by_cocktail, write_jsonl
from pour_decisions.schema import Cocktail, Ingredient


def _c(name: str) -> Cocktail:
    return Cocktail(name=name, ingredients=[Ingredient(quantity=30, unit="ml", ingredient="Gin")])


def test_build_record_has_user_and_assistant_with_valid_json_target():
    rec = build_record(_c("Negroni"))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    parsed = json.loads(rec["messages"][-1]["content"])  # assistant target is valid JSON
    assert parsed["name"] == "Negroni"
    assert parsed["ingredients"][0]["unit"] == "ml"


def test_split_by_cocktail_is_disjoint_and_deterministic():
    cocktails = [_c(f"C{i}") for i in range(10)]
    train, val, test = split_by_cocktail(cocktails, ratios=(0.6, 0.2, 0.2), seed=0)

    def names(xs: list) -> set[str]:
        return {c.name for c in xs}

    assert names(train) & names(val) == set()
    assert names(train) & names(test) == set()
    assert names(val) & names(test) == set()
    assert len(train) + len(val) + len(test) == 10
    # deterministic
    train2, _, _ = split_by_cocktail(cocktails, ratios=(0.6, 0.2, 0.2), seed=0)
    assert names(train) == names(train2)


def test_write_jsonl_roundtrips(tmp_path: Path):
    recs = [build_record(_c("A")), build_record(_c("B"))]
    out = tmp_path / "train.jsonl"
    write_jsonl(recs, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["messages"][0]["role"] == "system"
