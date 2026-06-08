"""Backend-agnostic eval inference + scoring. Reuses metrics.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pour_decisions.matrix.jsonparse import parse_cocktail_lenient
from pour_decisions.matrix.registry import ModelSpec
from pour_decisions.matrix.render import base_text_prompt, messages_prompt
from pour_decisions.matrix.report import EvalScores
from pour_decisions.metrics import field_accuracy, hallucination_rate
from pour_decisions.schema import Cocktail

Record = dict[str, list[dict[str, str]]]


class Predictor(Protocol):
    def generate(self, prompt: str, *, constrained: bool) -> str: ...


@dataclass
class Prediction:
    raw: str
    cocktail: Cocktail | None
    valid: bool


def _prompt_for(rec: Record, spec: ModelSpec) -> str:
    if spec.has_chat_template:
        # Predictor applies the chat template; pass a sentinel the predictor understands.
        # For base-only v1 this branch is unused; kept for the instruct follow-up.
        msgs = messages_prompt(rec)
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    return base_text_prompt(rec)


def predict_records(
    predictor: Predictor, records: list[Record], spec: ModelSpec, *, constrained: bool
) -> list[Prediction]:
    preds: list[Prediction] = []
    for rec in records:
        raw = predictor.generate(_prompt_for(rec, spec), constrained=constrained)
        cocktail = parse_cocktail_lenient(raw)
        preds.append(Prediction(raw=raw, cocktail=cocktail, valid=cocktail is not None))
    return preds


def _gold(records: list[Record]) -> list[Cocktail]:
    return [Cocktail.model_validate_json(r["messages"][-1]["content"]) for r in records]


_EMPTY = Cocktail(name="", ingredients=[])


def score(preds: list[Prediction], gold_records: list[Record]) -> EvalScores:
    gold = _gold(gold_records)
    pred_c = [p.cocktail if p.cocktail is not None else _EMPTY for p in preds]
    acc = field_accuracy(pred_c, gold)
    validity = sum(p.valid for p in preds) / len(preds) if preds else 0.0
    return EvalScores(
        quantity=acc["quantity"],
        unit=acc["unit"],
        ingredient=acc["ingredient"],
        json_validity=validity,
    )


def tuned_hallucination(preds: list[Prediction], gold_records: list[Record]) -> float:
    gold = _gold(gold_records)
    pred_c = [p.cocktail if p.cocktail is not None else _EMPTY for p in preds]
    return hallucination_rate(pred_c, gold)


class MLXPredictor:
    """Lazy MLX + Outlines predictor. Integration-only (needs Apple Silicon + the `mlx` extra)."""

    def __init__(self, hf_id: str, adapter_path: str | None = None) -> None:
        import mlx_lm

        loaded: Any = mlx_lm.load(hf_id, adapter_path=adapter_path)
        self._model: Any = loaded[0]
        self._tok: Any = loaded[1]
        self._hf_id = hf_id
        self._adapter = adapter_path
        self._json_model: Any = None  # built lazily for constrained mode

    def generate(self, prompt: str, *, constrained: bool) -> str:
        if constrained:
            import outlines

            if self._json_model is None:
                self._json_model = outlines.from_mlxlm(self._model, self._tok)
            return str(self._json_model(prompt, output_type=Cocktail))
        import mlx_lm

        return str(
            mlx_lm.generate(self._model, self._tok, prompt=prompt, max_tokens=512, verbose=False)
        )
