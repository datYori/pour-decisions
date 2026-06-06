"""Trainer protocol + MLX (Apple Silicon) and PEFT (CUDA) LoRA backends.

MLXTrainer trains in-process via mlx_lm and emits training telemetry
(metrics.jsonl, summary.json, training_dashboard.png, tb/). Heavy libs are
imported lazily. The actual training is exercised on hardware, not in unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pour_decisions.matrix.registry import ModelSpec


class Trainer(Protocol):
    def train(self, spec: ModelSpec, data_dir: Path, out_dir: Path) -> Path: ...


def count_examples(data_dir: Path) -> int:
    p = data_dir / "train.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


class MLXTrainer:
    def train(self, spec: ModelSpec, data_dir: Path, out_dir: Path) -> Path:
        from pour_decisions.matrix import telemetry
        from pour_decisions.matrix.mlxtrain import run_lora_with_metrics
        from pour_decisions.matrix.plots import render_dashboard

        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        tm = run_lora_with_metrics(
            spec, data_dir, out_dir, count_examples(data_dir), tb_dir=run_dir / "tb"
        )
        telemetry.write(tm, run_dir)
        render_dashboard(tm, run_dir / "training_dashboard.png")
        return out_dir


class PEFTTrainer:
    """CUDA LoRA via transformers+peft+trl. Written per spec; not run on the Mac."""

    def train(self, spec: ModelSpec, data_dir: Path, out_dir: Path) -> Path:
        import torch  # noqa: F401
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        out_dir.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype="auto", device_map="auto"
        )
        ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
        peft_cfg = LoraConfig(
            r=spec.lora.rank, lora_alpha=spec.lora.rank * 2, task_type="CAUSAL_LM"
        )
        sft = SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=spec.lora.batch_size,
            max_steps=spec.lora.iters,
            learning_rate=spec.lora.learning_rate,
            max_seq_length=spec.lora.max_seq_len,
            dataset_text_field="text",
            logging_steps=10,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft,
            train_dataset=ds,
            peft_config=peft_cfg,
            tokenizer=tok,
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        return out_dir
