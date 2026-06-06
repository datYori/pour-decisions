from pour_decisions.matrix.infer import predict_records, score
from pour_decisions.matrix.registry import by_key

GOLD = '{"name":"X","ingredients":[{"quantity":30.0,"unit":"ml","ingredient":"Gin"}]}'
REC = {
    "messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Name: X\nIngredients:\n- 30 ml Gin"},
        {"role": "assistant", "content": GOLD},
    ]
}


class FakePredictor:
    def __init__(self, out: str) -> None:
        self.out = out
        self.seen: list[str] = []

    def generate(self, prompt: str, *, constrained: bool) -> str:
        self.seen.append(prompt)
        return self.out


def test_predict_uses_base_text_prompt_for_base_model():
    spec = by_key("qwen2.5-0.5b")
    fp = FakePredictor(GOLD)
    preds = predict_records(fp, [REC], spec, constrained=False)
    assert preds[0].valid is True and preds[0].cocktail is not None
    assert fp.seen[0].endswith("### JSON:\n")  # used render.base_text_prompt


def test_predict_marks_unparseable_invalid():
    spec = by_key("qwen2.5-0.5b")
    preds = predict_records(FakePredictor("no json at all"), [REC], spec, constrained=False)
    assert preds[0].valid is False and preds[0].cocktail is None


def test_score_perfect_match():
    spec = by_key("qwen2.5-0.5b")
    preds = predict_records(FakePredictor(GOLD), [REC], spec, constrained=False)
    s = score(preds, [REC])
    assert s.quantity == 1.0 and s.unit == 1.0 and s.ingredient == 1.0 and s.json_validity == 1.0
