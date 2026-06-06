"""Training telemetry: standard fine-tuning metrics, persistence, derived stats.

Backend-agnostic. Populated by callbacks (callbacks.py) from mlx-lm's training
reports; the same schema is reusable for the PEFT/HF backend later. Pure stdlib.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IterMetric:
    iteration: int
    train_loss: float
    learning_rate: float
    it_per_sec: float
    tokens_per_sec: float
    trained_tokens: int
    peak_memory_gb: float


@dataclass(frozen=True)
class ValMetric:
    iteration: int
    val_loss: float
    val_time_s: float


@dataclass
class TrainMetrics:
    model_key: str
    hf_id: str
    learning_rate: float
    iters: int
    batch_size: int
    num_layers: int
    n_train_examples: int
    train_series: list[IterMetric] = field(default_factory=list)
    val_series: list[ValMetric] = field(default_factory=list)
    trainable_pct: float | None = None
    total_train_seconds: float | None = None

    @property
    def final_train_loss(self) -> float | None:
        return self.train_series[-1].train_loss if self.train_series else None

    @property
    def final_perplexity(self) -> float | None:
        v = self.final_train_loss
        return math.exp(v) if v is not None else None

    @property
    def best_val(self) -> ValMetric | None:
        return min(self.val_series, key=lambda v: v.val_loss) if self.val_series else None

    @property
    def best_val_loss(self) -> float | None:
        bv = self.best_val
        return bv.val_loss if bv is not None else None

    @property
    def best_val_iter(self) -> int | None:
        bv = self.best_val
        return bv.iteration if bv is not None else None

    @property
    def best_val_perplexity(self) -> float | None:
        v = self.best_val_loss
        return math.exp(v) if v is not None else None

    @property
    def train_val_gap(self) -> float | None:
        ftl, bvl = self.final_train_loss, self.best_val_loss
        return (ftl - bvl) if (ftl is not None and bvl is not None) else None

    @property
    def epochs(self) -> float:
        return (
            self.iters * self.batch_size / self.n_train_examples if self.n_train_examples else 0.0
        )

    @property
    def mean_tokens_per_sec(self) -> float | None:
        xs = [m.tokens_per_sec for m in self.train_series]
        return sum(xs) / len(xs) if xs else None

    @property
    def peak_memory_gb(self) -> float | None:
        xs = [m.peak_memory_gb for m in self.train_series]
        return max(xs) if xs else None

    def summary(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "hf_id": self.hf_id,
            "learning_rate": self.learning_rate,
            "iters": self.iters,
            "batch_size": self.batch_size,
            "num_layers": self.num_layers,
            "n_train_examples": self.n_train_examples,
            "trainable_pct": self.trainable_pct,
            "total_train_seconds": self.total_train_seconds,
            "n_train_reports": len(self.train_series),
            "n_val_reports": len(self.val_series),
            "final_train_loss": self.final_train_loss,
            "final_perplexity": self.final_perplexity,
            "best_val_loss": self.best_val_loss,
            "best_val_iter": self.best_val_iter,
            "best_val_perplexity": self.best_val_perplexity,
            "train_val_gap": self.train_val_gap,
            "epochs": self.epochs,
            "mean_tokens_per_sec": self.mean_tokens_per_sec,
            "peak_memory_gb": self.peak_memory_gb,
        }


def write(tm: TrainMetrics, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "train", **asdict(m)}) for m in tm.train_series]
    lines += [json.dumps({"kind": "val", **asdict(v)}) for v in tm.val_series]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    (run_dir / "summary.json").write_text(json.dumps(tm.summary(), indent=2))


def _opt_float(x: object) -> float | None:
    return float(x) if x is not None else None  # type: ignore[arg-type]


def load(run_dir: Path) -> TrainMetrics:
    summary = json.loads((run_dir / "summary.json").read_text())
    train_series: list[IterMetric] = []
    val_series: list[ValMetric] = []
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            kind = d.pop("kind")
            if kind == "train":
                train_series.append(IterMetric(**d))
            elif kind == "val":
                val_series.append(ValMetric(**d))
    return TrainMetrics(
        model_key=str(summary["model_key"]),
        hf_id=str(summary["hf_id"]),
        learning_rate=float(summary["learning_rate"]),
        iters=int(summary["iters"]),
        batch_size=int(summary["batch_size"]),
        num_layers=int(summary["num_layers"]),
        n_train_examples=int(summary["n_train_examples"]),
        train_series=train_series,
        val_series=val_series,
        trainable_pct=_opt_float(summary.get("trainable_pct")),
        total_train_seconds=_opt_float(summary.get("total_train_seconds")),
    )
