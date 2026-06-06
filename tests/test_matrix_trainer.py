from pathlib import Path

from pour_decisions.matrix.trainer import MLXTrainer, PEFTTrainer, count_examples


def test_count_examples(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n\n')
    assert count_examples(tmp_path) == 2


def test_count_examples_missing(tmp_path: Path):
    assert count_examples(tmp_path) == 0


def test_trainers_expose_train():
    # Structural: both expose a callable .train(spec, data_dir, out_dir) -> Path
    assert callable(MLXTrainer().train)
    assert callable(PEFTTrainer().train)
