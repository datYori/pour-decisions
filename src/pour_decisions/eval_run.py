"""Eval runner: compare base vs tuned on gold test set, gate on field accuracy.

`compare_and_gate` is pure (used by CI on a fixture). `run_against_server` hits a
live vLLM endpoint (used on the GPU box via `make eval`) and logs to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

from pour_decisions.metrics import field_accuracy, hallucination_rate
from pour_decisions.schema import Cocktail


def _as_cocktails(items: list[dict]) -> list[Cocktail]:  # type: ignore[type-arg]
    return [Cocktail.model_validate(d) for d in items]


def compare_and_gate(base: list, tuned: list, gold: list) -> dict:  # type: ignore[type-arg]
    base_c, tuned_c, gold_c = _as_cocktails(base), _as_cocktails(tuned), _as_cocktails(gold)
    base_acc = field_accuracy(base_c, gold_c)
    tuned_acc = field_accuracy(tuned_c, gold_c)
    passed = tuned_acc["ingredient"] >= base_acc["ingredient"] and (
        tuned_acc["quantity"] + tuned_acc["unit"] >= base_acc["quantity"] + base_acc["unit"]
    )
    return {
        "base": base_acc,
        "tuned": tuned_acc,
        "passed": passed,
        "tuned_hallucination": hallucination_rate(tuned_c, gold_c),
    }


def render_report(result: dict) -> str:  # type: ignore[type-arg]
    b, t = result["base"], result["tuned"]
    rows = "\n".join(
        f"| {k} | {b[k]:.3f} | {t[k]:.3f} | {t[k] - b[k]:+.3f} |"
        for k in ("quantity", "unit", "ingredient")
    )
    return (
        "# Eval: base vs tuned (gold IBA test)\n\n"
        "| field | base | tuned | Δ |\n|---|---|---|---|\n"
        + rows
        + f"\n\nTuned hallucination rate: {result['tuned_hallucination']:.3f}\n"
        f"\n**Gate: {'PASS' if result['passed'] else 'FAIL'}**\n"
    )


def run_against_server(
    test_jsonl: Path, *, base_model: str, tuned_model: str, base_url: str, tuned_url: str
) -> dict:  # type: ignore[type-arg]
    """Hit a live vLLM server for base+tuned over the gold test set. Run on the GPU box."""
    import mlflow  # type: ignore[import-not-found]

    from pour_decisions.serve_client import extract_cocktail

    gold, base_pred, tuned_pred = [], [], []
    for line in test_jsonl.read_text().strip().splitlines():
        rec = json.loads(line)
        gold_c = Cocktail.model_validate_json(rec["messages"][-1]["content"])
        name, raw = _name_and_lines(rec)
        gold.append(gold_c.model_dump())
        base_pred.append(
            extract_cocktail(name, raw, model=base_model, base_url=base_url).model_dump()
        )
        tuned_pred.append(
            extract_cocktail(name, raw, model=tuned_model, base_url=tuned_url).model_dump()
        )

    result = compare_and_gate(base_pred, tuned_pred, gold)
    with mlflow.start_run(run_name="eval-base-vs-tuned"):
        for split in ("base", "tuned"):
            for field, val in result[split].items():
                mlflow.log_metric(f"{split}_{field}_acc", val)
        mlflow.log_metric("tuned_hallucination", result["tuned_hallucination"])
    return result


def _name_and_lines(rec: dict) -> tuple[str, list[str]]:  # type: ignore[type-arg]
    user = rec["messages"][1]["content"]
    name = user.split("\n", 1)[0].removeprefix("Name: ").strip()
    # Assumes ingredient lines are prefixed "- " — the format prompts.format_user produces.
    lines = [ln[2:] for ln in user.splitlines() if ln.startswith("- ")]
    return name, lines
