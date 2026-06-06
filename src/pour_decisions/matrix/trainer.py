"""Trainer protocol + MLX (Apple Silicon) and PEFT (CUDA) LoRA backends.

The argv builder `mlx_lora_command` is pure and unit-tested. `.train()` runs the backend and is
exercised only on hardware (see the validation run). Heavy libs are imported lazily.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol

from pour_decisions.matrix.registry import ModelSpec


class Trainer(Protocol):
    def train(self, spec: ModelSpec, data_dir: Path, out_dir: Path) -> Path: ...


def mlx_lora_command(spec: ModelSpec, data_dir: Path, out_dir: Path) -> list[str]:
    return [
        "mlx_lm.lora",
        "--model",
        spec.hf_id,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(out_dir),
        "--iters",
        str(spec.lora.iters),
        "--batch-size",
        str(spec.lora.batch_size),
        "--num-layers",
        str(spec.lora.num_layers),
        "--learning-rate",
        str(spec.lora.learning_rate),
        "--max-seq-length",
        str(spec.lora.max_seq_len),
    ]


class MLXTrainer:
    def train(self, spec: ModelSpec, data_dir: Path, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = mlx_lora_command(spec, data_dir, out_dir)
        t0 = time.monotonic()
        subprocess.run(cmd, check=True)
        (out_dir / ".train_seconds").write_text(str(time.monotonic() - t0))
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
