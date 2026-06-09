# tests/test_matrix_plots.py
import importlib.util
from pathlib import Path

import pytest

from pour_decisions.matrix.telemetry import IterMetric, TrainMetrics, ValMetric


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed"
)
def test_render_dashboard_writes_png(tmp_path: Path):
    from pour_decisions.matrix.plots import render_dashboard

    tm = TrainMetrics(
        model_key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        learning_rate=1e-4,
        iters=300,
        batch_size=1,
        num_layers=8,
        n_train_examples=62,
        train_series=[IterMetric(10, 1.0, 1e-4, 9.0, 2400.0, 2500, 2.0)],
        val_series=[ValMetric(1, 1.7, 0.8)],
        trainable_pct=0.3,
        total_train_seconds=36.0,
    )
    out = render_dashboard(tm, tmp_path / "dash.png")
    assert out is not None and out.exists() and out.stat().st_size > 0
