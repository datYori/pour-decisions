"""mlx-lm TrainingCallback adapters: record metrics, fan out, optional TensorBoard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pour_decisions.matrix.telemetry import IterMetric, ValMetric


class MetricsRecorder:
    """Collects mlx-lm train/val reports into in-memory series."""

    def __init__(self) -> None:
        self.train_series: list[IterMetric] = []
        self.val_series: list[ValMetric] = []

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        self.train_series.append(
            IterMetric(
                iteration=int(info["iteration"]),
                train_loss=float(info["train_loss"]),
                learning_rate=float(info["learning_rate"]),
                it_per_sec=float(info.get("iterations_per_second", 0.0)),
                tokens_per_sec=float(info.get("tokens_per_second", 0.0)),
                trained_tokens=int(info.get("trained_tokens", 0)),
                peak_memory_gb=float(info.get("peak_memory", 0.0)),
            )
        )

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        self.val_series.append(
            ValMetric(
                iteration=int(info["iteration"]),
                val_loss=float(info["val_loss"]),
                val_time_s=float(info.get("val_time", 0.0)),
            )
        )


class TensorBoardSink:
    """Writes scalars to TensorBoard event files. No-op if tensorboardX is absent."""

    def __init__(self, log_dir: Path) -> None:
        self._writer: Any = None
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print(
                "[telemetry] tensorboardX not installed; skipping TensorBoard "
                "(install the 'mlx' extra). Metrics + PNG still emitted.",
                file=sys.stderr,
            )
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(logdir=str(log_dir))

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        if self._writer is None:
            return
        step = int(info["iteration"])
        self._writer.add_scalar("loss/train", float(info["train_loss"]), step)
        self._writer.add_scalar("lr", float(info["learning_rate"]), step)
        self._writer.add_scalar(
            "throughput/tokens_per_sec", float(info.get("tokens_per_second", 0.0)), step
        )
        self._writer.add_scalar(
            "throughput/it_per_sec", float(info.get("iterations_per_second", 0.0)), step
        )
        self._writer.add_scalar("mem/peak_gb", float(info.get("peak_memory", 0.0)), step)

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("loss/val", float(info["val_loss"]), int(info["iteration"]))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


class CompositeCallback:
    """Fans out mlx-lm's single-callback interface to several sinks."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = sinks

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        for s in self._sinks:
            s.on_train_loss_report(info)

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        for s in self._sinks:
            s.on_val_loss_report(info)

    def close(self) -> None:
        for s in self._sinks:
            close = getattr(s, "close", None)
            if callable(close):
                close()
