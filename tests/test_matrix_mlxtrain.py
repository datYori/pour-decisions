# tests/test_matrix_mlxtrain.py
from pathlib import Path

from pour_decisions.matrix.mlxtrain import build_overrides
from pour_decisions.matrix.registry import by_key


def test_build_overrides_maps_spec_and_paths():
    spec = by_key("ministral-3-3b-base")
    ov = build_overrides(
        spec, Path("/d/data"), Path("/d/adapter"), steps_per_report=10, steps_per_eval=50
    )
    assert ov["model"] == "mistralai/Ministral-3-3B-Base-2512"
    assert ov["data"] == "/d/data"
    assert ov["adapter_path"] == "/d/adapter"
    assert ov["train"] is True
    assert ov["fine_tune_type"] == "lora"
    assert ov["learning_rate"] == spec.lora.learning_rate
    assert ov["iters"] == spec.lora.iters
    assert ov["num_layers"] == spec.lora.num_layers
    assert ov["batch_size"] == spec.lora.batch_size
    assert ov["max_seq_length"] == spec.lora.max_seq_len
    assert ov["steps_per_report"] == 10 and ov["steps_per_eval"] == 50
