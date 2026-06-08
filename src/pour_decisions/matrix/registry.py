"""The model matrix as typed, declarative data. Single source for which models we tune."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LicenseTier = Literal["free", "open-weight"]


@dataclass(frozen=True)
class LoraCfg:
    rank: int = 16
    iters: int = 300
    batch_size: int = 1
    num_layers: int = 8
    learning_rate: float = 1e-4
    max_seq_len: int = 2048


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: str
    license: str
    license_tier: LicenseTier
    gated: bool
    has_chat_template: bool
    lora: LoraCfg = field(default_factory=LoraCfg)


# LoRA LR is headroom-sensitive: small/undertrained bases (<=1.1B) learn cleanly at the 1e-4
# default; saturated >=1.5B bases mode-collapse at 1e-4, so they use mlx-lm's 1e-5 default
# (validated on ministral-3-3b-base; see docs/runbooks/local-matrix.md).
MATRIX: list[ModelSpec] = [
    ModelSpec(
        "ministral-3-3b-base",
        "mistralai/Ministral-3-3B-Base-2512",
        "3B",
        "apache-2.0",
        "free",
        False,
        False,
        # bf16 LoRA on 24GB Mac; lr 1e-5 (1e-4 mode-collapses saturated bases, see runbook).
        LoraCfg(num_layers=8, max_seq_len=2048, learning_rate=1e-5),
    ),
    ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", "0.5B", "apache-2.0", "free", False, False),
    ModelSpec(
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B",
        "1.5B",
        "apache-2.0",
        "free",
        False,
        False,
        LoraCfg(learning_rate=1e-5),
    ),
    ModelSpec(
        "smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "135M", "apache-2.0", "free", False, False
    ),
    ModelSpec(
        "smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "360M", "apache-2.0", "free", False, False
    ),
    ModelSpec(
        "smollm2-1.7b",
        "HuggingFaceTB/SmolLM2-1.7B",
        "1.7B",
        "apache-2.0",
        "free",
        False,
        False,
        LoraCfg(learning_rate=1e-5),
    ),
    ModelSpec(
        "smollm3-3b-base",
        "HuggingFaceTB/SmolLM3-3B-Base",
        "3B",
        "apache-2.0",
        "free",
        False,
        False,
        # bf16 LoRA on 24GB Mac; lr 1e-5 (1e-4 mode-collapses saturated bases, see runbook).
        LoraCfg(num_layers=8, max_seq_len=2048, learning_rate=1e-5),
    ),
    ModelSpec(
        "tinyllama-1.1b",
        "TinyLlama/TinyLlama_v1.1",
        "1.1B",
        "apache-2.0",
        "free",
        False,
        False,
    ),
    ModelSpec("pythia-410m", "EleutherAI/pythia-410m", "410M", "apache-2.0", "free", False, False),
    ModelSpec("pythia-1b", "EleutherAI/pythia-1b", "1B", "apache-2.0", "free", False, False),
    ModelSpec("gemma-3-270m", "google/gemma-3-270m", "270M", "gemma", "open-weight", True, False),
    ModelSpec("gemma-3-1b-pt", "google/gemma-3-1b-pt", "1B", "gemma", "open-weight", True, False),
    ModelSpec(
        "llama-3.2-1b",
        "meta-llama/Llama-3.2-1B",
        "1B",
        "llama3.2",
        "open-weight",
        True,
        False,
    ),
]


def by_key(key: str) -> ModelSpec:
    for s in MATRIX:
        if s.key == key:
            return s
    raise KeyError(key)


def select(*, tiers: set[str] | None = None, include_gated: bool = True) -> list[ModelSpec]:
    out = []
    for s in MATRIX:
        if tiers is not None and s.license_tier not in tiers:
            continue
        if not include_gated and s.gated:
            continue
        out.append(s)
    return out
