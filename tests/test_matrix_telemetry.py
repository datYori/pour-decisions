# tests/test_matrix_telemetry.py
import math
from pathlib import Path

from pour_decisions.matrix.telemetry import (
    IterMetric,
    TrainMetrics,
    ValMetric,
    load,
    write,
)


def _tm() -> TrainMetrics:
    return TrainMetrics(
        model_key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        learning_rate=1e-4,
        iters=300,
        batch_size=1,
        num_layers=8,
        n_train_examples=62,
        train_series=[
            IterMetric(10, 1.0, 1e-4, 9.0, 2400.0, 2500, 2.0),
            IterMetric(20, 0.5, 1e-4, 9.0, 2400.0, 5000, 2.1),
        ],
        val_series=[ValMetric(1, 1.7, 0.8), ValMetric(200, 0.4, 0.9)],
        trainable_pct=0.3,
        total_train_seconds=36.0,
    )


def test_derived_metrics():
    tm = _tm()
    assert tm.final_train_loss == 0.5
    assert math.isclose(tm.final_perplexity, math.exp(0.5))
    assert tm.best_val_loss == 0.4
    assert tm.best_val_iter == 200
    assert math.isclose(tm.train_val_gap, 0.5 - 0.4)
    assert math.isclose(tm.epochs, 300 * 1 / 62)
    assert math.isclose(tm.mean_tokens_per_sec, 2400.0)
    assert tm.peak_memory_gb == 2.1


def test_derived_metrics_empty_val():
    tm = _tm()
    tm.val_series = []
    assert tm.best_val_loss is None
    assert tm.train_val_gap is None


def test_write_and_load_round_trip(tmp_path: Path):
    tm = _tm()
    write(tm, tmp_path)
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    back = load(tmp_path)
    assert back.train_series == tm.train_series
    assert back.val_series == tm.val_series
    assert back.final_train_loss == tm.final_train_loss
    assert back.trainable_pct == tm.trainable_pct
