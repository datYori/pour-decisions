import json
from pathlib import Path

from pour_decisions.matrix.data_adapters import build_peft_records, write_mlx_data
from pour_decisions.matrix.registry import by_key

REC = {
    "messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Name: X\nIngredients:\n- 30 ml Gin"},
        {"role": "assistant", "content": '{"name":"X","ingredients":[]}'},
    ]
}
BASE = by_key("qwen2.5-0.5b")  # has_chat_template False


def test_write_mlx_data_creates_named_files(tmp_path: Path):
    write_mlx_data([REC], [REC], tmp_path, BASE)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()  # note: 'valid', not 'val'


def test_write_mlx_data_base_uses_text_key(tmp_path: Path):
    write_mlx_data([REC], [REC], tmp_path, BASE)
    line = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert "text" in line and "messages" not in line
    assert line["text"].endswith('{"name":"X","ingredients":[]}')


def test_build_peft_records_base_text(tmp_path: Path):
    recs = build_peft_records([REC], BASE)
    assert recs[0]["text"].endswith('{"name":"X","ingredients":[]}')
