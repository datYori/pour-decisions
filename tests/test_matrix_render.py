from pour_decisions.matrix.render import (
    SEP,
    base_text_prompt,
    base_text_train,
    messages_prompt,
    target_json,
)

REC = {
    "messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Name: X\nIngredients:\n- 30 ml Gin"},
        {"role": "assistant", "content": '{"name":"X","ingredients":[]}'},
    ]
}


def test_messages_prompt_drops_assistant():
    msgs = messages_prompt(REC)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_target_json_is_assistant_content():
    assert target_json(REC) == '{"name":"X","ingredients":[]}'


def test_base_text_parity():
    # training text == eval prompt prefix + the target. This is the anti-skew guarantee.
    assert base_text_train(REC) == base_text_prompt(REC) + target_json(REC)


def test_base_prompt_ends_with_separator():
    assert base_text_prompt(REC).endswith(SEP)
