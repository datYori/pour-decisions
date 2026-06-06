import json
from pathlib import Path

from pour_decisions.eval_run import compare_and_gate

FIX = Path(__file__).parent / "fixtures" / "eval_fixture.json"


def test_gate_passes_when_tuned_beats_base():
    data = json.loads(FIX.read_text())
    result = compare_and_gate(data["base"], data["tuned"], data["gold"])
    assert result["passed"] is True
    assert result["tuned"]["quantity"] > result["base"]["quantity"]


def test_gate_fails_when_tuned_regresses():
    data = json.loads(FIX.read_text())
    result = compare_and_gate(data["tuned"], data["base"], data["gold"])  # swapped
    assert result["passed"] is False
