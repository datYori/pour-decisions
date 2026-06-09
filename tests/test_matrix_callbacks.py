# tests/test_matrix_callbacks.py
import importlib.util
from pathlib import Path

import pytest

from pour_decisions.matrix.callbacks import CompositeCallback, MetricsRecorder

TRAIN_INFO = {
    "iteration": 10,
    "train_loss": 1.23,
    "learning_rate": 1e-4,
    "iterations_per_second": 9.0,
    "tokens_per_second": 2400.0,
    "trained_tokens": 2500,
    "peak_memory": 2.0,
}
VAL_INFO = {"iteration": 1, "val_loss": 1.7, "val_time": 0.8}


def test_recorder_collects_train_and_val():
    r = MetricsRecorder()
    r.on_train_loss_report(TRAIN_INFO)
    r.on_val_loss_report(VAL_INFO)
    assert len(r.train_series) == 1 and len(r.val_series) == 1
    m = r.train_series[0]
    assert m.iteration == 10 and m.train_loss == 1.23 and m.tokens_per_sec == 2400.0
    assert m.trained_tokens == 2500 and m.peak_memory_gb == 2.0
    assert r.val_series[0].val_loss == 1.7


def test_composite_fans_out():
    a, b = MetricsRecorder(), MetricsRecorder()
    c = CompositeCallback([a, b])
    c.on_train_loss_report(TRAIN_INFO)
    c.on_val_loss_report(VAL_INFO)
    assert len(a.train_series) == 1 and len(b.train_series) == 1
    assert len(a.val_series) == 1 and len(b.val_series) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("tensorboardX") is None, reason="tensorboardX not installed"
)
def test_tensorboard_sink_writes_events(tmp_path: Path):
    from pour_decisions.matrix.callbacks import TensorBoardSink

    sink = TensorBoardSink(tmp_path / "tb")
    sink.on_train_loss_report(TRAIN_INFO)
    sink.on_val_loss_report(VAL_INFO)
    sink.close()
    events = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert events, "no events file written"
    assert events[0].stat().st_size > 200, "events file too small (scalars probably not flushed)"
    # idempotent: second close must not raise
    sink.close()
