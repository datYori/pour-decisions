import json
from pathlib import Path

from pour_decisions.matrix.report import (
    EvalScores,
    ModelResult,
    ModeResult,
    load_results,
    rebuild_results,
    render_markdown,
    write_reports,
)


def _scores(q: float, u: float, ing: float, valid: float) -> EvalScores:
    return EvalScores(quantity=q, unit=u, ingredient=ing, json_validity=valid)


def _result() -> ModelResult:
    unc = ModeResult(
        untuned=_scores(0.2, 0.2, 0.5, 0.3),
        tuned=_scores(0.9, 0.9, 1.0, 1.0),
        tuned_hallucination=0.0,
    )
    return ModelResult(
        key="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        params="135M",
        license="apache-2.0",
        backend="mlx",
        status="ok",
        train_seconds=42.0,
        modes={"unconstrained": unc},
    )


def test_render_markdown_has_header_and_delta():
    md = render_markdown([_result()])
    assert "| model |" in md
    assert "smollm2-135m" in md
    assert "+0.700" in md  # quantity delta 0.9 - 0.2 in unconstrained mode


def test_render_marks_skipped():
    r = ModelResult(
        key="ministral-3-3b-base",
        hf_id="x",
        params="3B",
        license="apache-2.0",
        backend="mlx",
        status="skipped",
        train_seconds=0.0,
        modes={},
        reason="arch unsupported",
    )
    md = render_markdown([r])
    assert "skipped" in md and "arch unsupported" in md


def test_write_reports_emits_md_and_json(tmp_path: Path):
    md_path, json_path = tmp_path / "m.md", tmp_path / "m.json"
    write_reports([_result()], md_path, json_path)
    assert md_path.exists() and json_path.exists()
    data = json.loads(json_path.read_text())
    assert data[0]["key"] == "smollm2-135m"
    assert data[0]["modes"]["unconstrained"]["tuned"]["json_validity"] == 1.0


def test_rebuild_results_round_trip(tmp_path: Path):
    original = _result()
    md_path, json_path = tmp_path / "m.md", tmp_path / "m.json"
    write_reports([original], md_path, json_path)
    rebuilt = rebuild_results(load_results(json_path))
    assert render_markdown(rebuilt) == render_markdown([original])
