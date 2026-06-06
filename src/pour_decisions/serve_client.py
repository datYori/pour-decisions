"""Build vLLM structured-output requests and parse responses into Cocktail.

The OpenAI client itself is only constructed in `extract_cocktail` (lazy import)
so unit tests never need the `openai` package or a live server.
"""

from __future__ import annotations

from pour_decisions.prompts import SYSTEM_PROMPT, format_user
from pour_decisions.schema import Cocktail


def build_request(name: str, raw_lines: list[str], model: str) -> dict:  # type: ignore[type-arg]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user(name, raw_lines)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cocktail",
                "strict": True,
                "schema": Cocktail.model_json_schema(),
            },
        },
        "temperature": 0.0,
    }


def parse_cocktail(raw_json: str) -> Cocktail:
    return Cocktail.model_validate_json(raw_json)


def extract_cocktail(name: str, raw_lines: list[str], *, model: str, base_url: str) -> Cocktail:
    from openai import OpenAI  # type: ignore[import-not-found]  # lazy; gpu extra only

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    req = build_request(name, raw_lines, model=model)
    resp = client.chat.completions.create(**req)
    return parse_cocktail(resp.choices[0].message.content)
