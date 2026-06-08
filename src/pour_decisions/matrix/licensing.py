"""Publish-compliance for fine-tuned derivatives. Drive from the registry license field."""

from __future__ import annotations

from pour_decisions.matrix.registry import ModelSpec

GEMMA_NOTICE = (
    "Gemma is provided under and subject to the Gemma Terms of Use "
    "found at ai.google.dev/gemma/terms"
)
LLAMA_NOTICE = (
    "Llama 3.2 is licensed under the Llama 3.2 Community License, "
    "Copyright (c) Meta Platforms, Inc. All Rights Reserved. Built with Llama."
)


def notice_for(spec: ModelSpec) -> str | None:
    if spec.license == "gemma":
        return GEMMA_NOTICE
    if spec.license == "llama3.2":
        return LLAMA_NOTICE
    return None  # apache-2.0 / mit: attribution only


def enforce_derivative_name(spec: ModelSpec, name: str) -> str:
    """Llama derivatives must begin with 'Llama'."""
    if spec.license == "llama3.2" and not name.startswith("Llama"):
        return f"Llama-{name}"
    return name
