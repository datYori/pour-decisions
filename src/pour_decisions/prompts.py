"""The ONE prompt, imported by both training-data generation and inference.

Keeping train-time and serve-time prompts identical is the single most common
fine-tuning footgun to avoid: a tuned model is sensitive to the exact wording.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a bartender's data assistant. Given a cocktail name and its raw "
    "ingredient lines, output ONLY a JSON object with: name, ingredients "
    "(each with quantity as a number or null, unit as a short canonical string "
    "or null, and ingredient as the name with amount/unit removed), and "
    "optionally category, method, garnish. Do not invent quantities or units "
    "that are not present; use null when absent."
)


def format_user(name: str, raw_lines: list[str]) -> str:
    lines = "\n".join(f"- {ln}" for ln in raw_lines)
    return f"Name: {name}\nIngredients:\n{lines}"
