"""In-process mlx-lm LoRA training that captures standard training telemetry.

Mirrors mlx_lm.lora.run() but passes our own callback (run() discards it and
substitutes its wandb/swanlab reporter). Heavy mlx imports are local so this
module imports cleanly without the `mlx` extra; `build_overrides` is pure.
"""

from __future__ import annotations

import gc
import time
import types
from pathlib import Path
from typing import Any

from pour_decisions.matrix.callbacks import CompositeCallback, MetricsRecorder, TensorBoardSink
from pour_decisions.matrix.registry import ModelSpec
from pour_decisions.matrix.telemetry import TrainMetrics


def build_overrides(
    spec: ModelSpec,
    data_dir: Path,
    out_dir: Path,
    *,
    steps_per_report: int,
    steps_per_eval: int,
) -> dict[str, object]:
    """The args we override on top of mlx_lm.lora.CONFIG_DEFAULTS. Pure (no mlx import)."""
    return {
        "model": spec.hf_id,
        "data": str(data_dir),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": spec.lora.num_layers,
        "batch_size": spec.lora.batch_size,
        "iters": spec.lora.iters,
        "learning_rate": spec.lora.learning_rate,
        "max_seq_length": spec.lora.max_seq_len,
        "adapter_path": str(out_dir),
        "steps_per_report": steps_per_report,
        "steps_per_eval": steps_per_eval,
    }


def _trainable_pct(model: Any) -> float:
    # bf16-only: .size on packed quantized weights would undercount; revisit (mlx-lm
    # get_total_parameters) if a quantized/QLoRA base is ever added to the registry.
    from mlx.utils import tree_flatten

    params: Any = tree_flatten(model.parameters())
    trainable_params: Any = tree_flatten(model.trainable_parameters())
    total: int = sum(v.size for _, v in params)
    trainable: int = sum(v.size for _, v in trainable_params)
    return float(100.0 * trainable / total) if total else 0.0


def run_lora_with_metrics(
    spec: ModelSpec,
    data_dir: Path,
    out_dir: Path,
    n_train_examples: int,
    *,
    steps_per_report: int = 10,
    steps_per_eval: int = 50,
    tensorboard: bool = True,
    tb_dir: Path | None = None,
) -> TrainMetrics:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        **CONFIG_DEFAULTS,
        **build_overrides(
            spec,
            data_dir,
            out_dir,
            steps_per_report=steps_per_report,
            steps_per_eval=steps_per_eval,
        ),
    }
    args = types.SimpleNamespace(**merged)
    np.random.seed(args.seed)  # mirror mlx_lm.lora.run() for reproducible data shuffling

    loaded: Any = load(spec.hf_id, tokenizer_config={"trust_remote_code": True})
    model: Any = loaded[0]
    tokenizer: Any = loaded[1]
    datasets: Any = load_dataset(args, tokenizer)
    train_set: Any = datasets[0]
    valid_set: Any = datasets[1]

    recorder = MetricsRecorder()
    sinks: list[Any] = [recorder]
    if tensorboard:
        sinks.append(TensorBoardSink(tb_dir if tb_dir is not None else out_dir / "tb"))
    callback: Any = CompositeCallback(sinks)

    t0 = time.monotonic()
    try:
        train_model(args, model, train_set, valid_set, training_callback=callback)
    finally:
        callback.close()
    elapsed = time.monotonic() - t0
    pct = _trainable_pct(model)

    del model, tokenizer, train_set, valid_set
    gc.collect()
    mx.clear_cache()

    return TrainMetrics(
        model_key=spec.key,
        hf_id=spec.hf_id,
        learning_rate=spec.lora.learning_rate,
        iters=spec.lora.iters,
        batch_size=spec.lora.batch_size,
        num_layers=spec.lora.num_layers,
        n_train_examples=n_train_examples,
        train_series=recorder.train_series,
        val_series=recorder.val_series,
        trainable_pct=pct,
        total_train_seconds=elapsed,
    )
