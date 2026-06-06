"""One renderer for both training data and eval prompts. Prevents train/serve skew."""

from __future__ import annotations

Record = dict[str, list[dict[str, str]]]

SEP = "\n\n### JSON:\n"


def _system(rec: Record) -> str:
    return rec["messages"][0]["content"]


def _user(rec: Record) -> str:
    return rec["messages"][1]["content"]


def messages_prompt(rec: Record) -> list[dict[str, str]]:
    """System + user only (the assistant target is supplied by the trainer / is the label)."""
    return [rec["messages"][0], rec["messages"][1]]


def target_json(rec: Record) -> str:
    return rec["messages"][2]["content"]


def base_text_prompt(rec: Record) -> str:
    """Flat prompt prefix for a base model (no chat template)."""
    return f"{_system(rec)}\n\n{_user(rec)}{SEP}"


def base_text_train(rec: Record) -> str:
    """Full training string for a base model: prompt prefix + target. MUST equal prompt+target."""
    return base_text_prompt(rec) + target_json(rec)
