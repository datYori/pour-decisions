"""Turn Cocktail objects into mistral-finetune `messages` jsonl, split by cocktail."""

from __future__ import annotations

import json
import random
from pathlib import Path

from pour_decisions.prompts import SYSTEM_PROMPT, format_user
from pour_decisions.schema import Cocktail


def build_record(cocktail: Cocktail) -> dict[str, object]:
    raw_lines = [
        f"{_fmt_qty(i.quantity)}{(i.unit + ' ') if i.unit else ''}{i.ingredient}".strip()
        for i in cocktail.ingredients
    ]
    target = cocktail.model_dump_json(exclude_none=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user(cocktail.name, raw_lines)},
            {"role": "assistant", "content": target},
        ]
    }


def _fmt_qty(q: float | None) -> str:
    if q is None:
        return ""
    s = f"{q:g}"  # 30.0 -> "30", 1.5 -> "1.5"
    return f"{s} "


def split_by_cocktail(
    cocktails: list[Cocktail], ratios: tuple[float, float, float], seed: int
) -> tuple[list[Cocktail], list[Cocktail], list[Cocktail]]:
    shuffled = list(cocktails)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
