"""Result dataclasses + matrix report rendering (markdown + json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "skipped", "error"]


@dataclass
class EvalScores:
    quantity: float
    unit: float
    ingredient: float
    json_validity: float


@dataclass
class ModeResult:
    untuned: EvalScores
    tuned: EvalScores
    tuned_hallucination: float


@dataclass
class ModelResult:
    key: str
    hf_id: str
    params: str
    license: str
    backend: str
    status: Status
    train_seconds: float
    modes: dict[str, ModeResult]
    reason: str = ""


_HEADER = (
    "| model | params | license | backend | mode "
    "| qty u->t (d) | unit u->t (d) | ingr u->t (d) | valid u->t | halluc | train s |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _cell(u: float, t: float) -> str:
    return f"{u:.3f}->{t:.3f} ({t - u:+.3f})"


def render_markdown(results: list[ModelResult]) -> str:
    lines = [
        "# Local fine-tune matrix: per-model untuned -> tuned delta\n",
        "Eval on the 15 held-out gold IBA cocktails (n=15; deltas are noisy at this size).\n",
        _HEADER.rstrip("\n"),
    ]
    for r in results:
        if r.status != "ok" or not r.modes:
            lines.append(
                f"| {r.key} | {r.params} | {r.license} | {r.backend} | - "
                f"| {r.status}: {r.reason} | | | | | {r.train_seconds:.0f} |"
            )
            continue
        for mode, m in r.modes.items():
            lines.append(
                f"| {r.key} | {r.params} | {r.license} | {r.backend} | {mode} "
                f"| {_cell(m.untuned.quantity, m.tuned.quantity)} "
                f"| {_cell(m.untuned.unit, m.tuned.unit)} "
                f"| {_cell(m.untuned.ingredient, m.tuned.ingredient)} "
                f"| {m.untuned.json_validity:.3f}->{m.tuned.json_validity:.3f} "
                f"| {m.tuned_hallucination:.3f} | {r.train_seconds:.0f} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(results: list[ModelResult], md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(results))
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))


def load_results(json_path: Path) -> list[dict[str, object]]:
    return list(json.loads(json_path.read_text()))


def rebuild_results(data: list[dict[str, object]]) -> list[ModelResult]:
    out: list[ModelResult] = []
    for d in data:
        raw_modes = d.get("modes") or {}
        assert isinstance(raw_modes, dict)
        modes: dict[str, ModeResult] = {}
        for mode, mr in raw_modes.items():
            assert isinstance(mr, dict)
            u = mr["untuned"]
            t = mr["tuned"]
            assert isinstance(u, dict) and isinstance(t, dict)
            modes[mode] = ModeResult(
                untuned=EvalScores(
                    quantity=float(u["quantity"]),
                    unit=float(u["unit"]),
                    ingredient=float(u["ingredient"]),
                    json_validity=float(u["json_validity"]),
                ),
                tuned=EvalScores(
                    quantity=float(t["quantity"]),
                    unit=float(t["unit"]),
                    ingredient=float(t["ingredient"]),
                    json_validity=float(t["json_validity"]),
                ),
                tuned_hallucination=float(mr["tuned_hallucination"]),
            )
        out.append(
            ModelResult(
                key=str(d["key"]),
                hf_id=str(d["hf_id"]),
                params=str(d["params"]),
                license=str(d["license"]),
                backend=str(d["backend"]),
                status=d["status"],  # type: ignore[arg-type]
                train_seconds=float(d["train_seconds"]),  # type: ignore[arg-type]
                modes=modes,
                reason=str(d.get("reason", "")),
            )
        )
    return out
