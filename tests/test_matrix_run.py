from pathlib import Path

from pour_decisions.matrix.registry import by_key
from pour_decisions.matrix.run_matrix import run_one

GOLD = '{"name":"X","ingredients":[{"quantity":30.0,"unit":"ml","ingredient":"Gin"}]}'
REC = {
    "messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Name: X\nIngredients:\n- 30 ml Gin"},
        {"role": "assistant", "content": GOLD},
    ]
}


class FakeTrainer:
    def train(self, spec, data_dir: Path, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


class FakePredictor:
    def __init__(self, out: str) -> None:
        self.out = out

    def generate(self, prompt: str, *, constrained: bool) -> str:
        return self.out


def test_run_one_produces_ok_result(tmp_path: Path):
    spec = by_key("qwen2.5-0.5b")

    def factory(hf_id: str, adapter_path: str | None = None) -> FakePredictor:
        return FakePredictor("junk" if adapter_path is None else GOLD)

    result = run_one(
        spec,
        FakeTrainer(),
        factory,
        [REC],
        [REC],
        [REC],
        modes=["unconstrained"],
        work_dir=tmp_path,
    )
    assert result.status == "ok"
    m = result.modes["unconstrained"]
    assert m.untuned.json_validity == 0.0 and m.tuned.json_validity == 1.0
    assert m.tuned.quantity == 1.0


def test_run_one_records_error_on_train_failure(tmp_path: Path):
    spec = by_key("qwen2.5-0.5b")

    class BoomTrainer:
        def train(self, spec, data_dir, out_dir):
            raise RuntimeError("arch unsupported on MLX")

    def factory(hf_id: str, adapter_path: str | None = None) -> FakePredictor:
        return FakePredictor(GOLD)

    result = run_one(
        spec,
        BoomTrainer(),
        factory,
        [REC],
        [REC],
        [REC],
        modes=["unconstrained"],
        work_dir=tmp_path,
    )
    assert result.status == "error" and "arch unsupported" in result.reason


def test_read_training_summary(tmp_path):
    import json

    from pour_decisions.matrix.run_matrix import _read_training_summary

    assert _read_training_summary(tmp_path) == (None, None)
    (tmp_path / "summary.json").write_text(
        json.dumps({"final_train_loss": 0.1, "best_val_loss": 0.4})
    )
    assert _read_training_summary(tmp_path) == (0.1, 0.4)
