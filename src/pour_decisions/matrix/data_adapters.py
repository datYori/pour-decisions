"""Convert canonical `messages` records into per-backend training files."""

from __future__ import annotations

import json
from pathlib import Path

from pour_decisions.matrix.registry import ModelSpec
from pour_decisions.matrix.render import Record, base_text_train, messages_prompt, target_json


def _row(rec: Record, spec: ModelSpec) -> dict[str, object]:
    if spec.has_chat_template:
        return {
            "messages": messages_prompt(rec) + [{"role": "assistant", "content": target_json(rec)}]
        }
    return {"text": base_text_train(rec)}


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_mlx_data(train: list[Record], val: list[Record], out_dir: Path, spec: ModelSpec) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl([_row(r, spec) for r in train], out_dir / "train.jsonl")
    _write_jsonl([_row(r, spec) for r in val], out_dir / "valid.jsonl")
    return out_dir


def build_peft_records(train: list[Record], spec: ModelSpec) -> list[dict[str, object]]:
    return [_row(r, spec) for r in train]
