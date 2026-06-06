"""Orchestrate download -> data -> train -> eval(both modes) -> result, per model."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

from pour_decisions.matrix.data_adapters import write_mlx_data
from pour_decisions.matrix.infer import (
    Prediction,
    Predictor,
    predict_records,
    score,
    tuned_hallucination,
)
from pour_decisions.matrix.registry import ModelSpec, by_key, select
from pour_decisions.matrix.report import ModelResult, ModeResult, write_reports
from pour_decisions.matrix.trainer import MLXTrainer, Trainer

Record = dict[str, list[dict[str, str]]]
PredictorFactory = Callable[..., Predictor]

REPORTS = Path("reports")


def read_records(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_one(
    spec: ModelSpec,
    trainer: Trainer,
    predictor_factory: PredictorFactory,
    train_records: list[Record],
    val_records: list[Record],
    test_records: list[Record],
    *,
    modes: list[str],
    work_dir: Path,
    backend: str = "mlx",
) -> ModelResult:
    data_dir = work_dir / spec.key / "data"
    adapter_dir = work_dir / spec.key / "adapter"
    base = ModelResult(
        key=spec.key,
        hf_id=spec.hf_id,
        params=spec.params,
        license=spec.license,
        backend=backend,
        status="ok",
        train_seconds=0.0,
        modes={},
    )
    try:
        write_mlx_data(train_records, val_records, data_dir, spec)
        t0 = time.monotonic()
        trainer.train(spec, data_dir, adapter_dir)
        base.train_seconds = time.monotonic() - t0

        up_by_mode: dict[str, list[Prediction]] = {}
        untuned = predictor_factory(spec.hf_id, adapter_path=None)
        for mode in modes:
            up_by_mode[mode] = predict_records(
                untuned, test_records, spec, constrained=(mode == "constrained")
            )
        del untuned
        gc.collect()

        tp_by_mode: dict[str, list[Prediction]] = {}
        tuned = predictor_factory(spec.hf_id, adapter_path=str(adapter_dir))
        for mode in modes:
            tp_by_mode[mode] = predict_records(
                tuned, test_records, spec, constrained=(mode == "constrained")
            )
        del tuned
        gc.collect()

        for mode in modes:
            base.modes[mode] = ModeResult(
                untuned=score(up_by_mode[mode], test_records),
                tuned=score(tp_by_mode[mode], test_records),
                tuned_hallucination=tuned_hallucination(tp_by_mode[mode], test_records),
            )
    except Exception as exc:  # one bad model must not sink the matrix
        base.status = "error"
        base.reason = str(exc)
    return base


def _make_mlx_predictor(hf_id: str, adapter_path: str | None = None) -> Predictor:
    from pour_decisions.matrix.infer import MLXPredictor

    return MLXPredictor(hf_id, adapter_path=adapter_path)


def select_specs(keys: list[str] | None, include_gated: bool) -> list[ModelSpec]:
    if keys:
        return [by_key(k) for k in keys]
    return select(include_gated=include_gated)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local fine-tune + delta matrix")
    p.add_argument("--backend", choices=["mlx", "peft"], default="mlx")
    p.add_argument("--key", action="append", help="run only these model keys (repeatable)")
    p.add_argument("--modes", default="unconstrained,constrained")
    p.add_argument("--no-gated", action="store_true")
    p.add_argument("--data-dir", default="data/prepared")
    p.add_argument("--work-dir", default="runs/local")
    args = p.parse_args(argv)

    data = Path(args.data_dir)
    train = read_records(data / "train.jsonl")
    val = read_records(data / "val.jsonl")
    test = read_records(data / "test.jsonl")
    modes = [m for m in args.modes.split(",") if m]

    if args.backend == "mlx":
        trainer: Trainer = MLXTrainer()
        factory: PredictorFactory = _make_mlx_predictor
    else:
        from pour_decisions.matrix.trainer import PEFTTrainer

        def _peft_predictor(hf_id: str, adapter_path: str | None = None) -> Predictor:
            raise NotImplementedError("PEFT predictor runs on the CUDA workstation only")

        trainer = PEFTTrainer()
        factory = _peft_predictor

    specs = select_specs(args.key, include_gated=not args.no_gated)
    results = [
        run_one(
            s,
            trainer,
            factory,
            train,
            val,
            test,
            modes=modes,
            work_dir=Path(args.work_dir),
            backend=args.backend,
        )
        for s in specs
    ]
    write_reports(results, REPORTS / "local-matrix.md", REPORTS / "local-matrix.json")
    ok = sum(r.status == "ok" for r in results)
    print(f"matrix done: {ok}/{len(results)} ok -> reports/local-matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
