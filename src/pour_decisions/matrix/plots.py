"""Render a training dashboard PNG from TrainMetrics. No-op if matplotlib is absent."""

from __future__ import annotations

import sys
from pathlib import Path

from pour_decisions.matrix.telemetry import TrainMetrics


def render_dashboard(tm: TrainMetrics, out_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[telemetry] matplotlib not installed; skipping dashboard PNG "
            "(install the 'mlx' extra).",
            file=sys.stderr,
        )
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"{tm.model_key} -- LoRA lr={tm.learning_rate:g}, {tm.iters} iters")

    ax = axes[0][0]
    if tm.train_series:
        ax.plot(
            [m.iteration for m in tm.train_series],
            [m.train_loss for m in tm.train_series],
            label="train",
        )
    if tm.val_series:
        ax.plot(
            [v.iteration for v in tm.val_series],
            [v.val_loss for v in tm.val_series],
            "o-",
            label="val",
        )
    ax.set_title("loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0][1]
    ax.plot(
        [m.iteration for m in tm.train_series],
        [m.learning_rate for m in tm.train_series],
    )
    ax.set_title("learning rate")
    ax.set_xlabel("iteration")

    ax = axes[1][0]
    ax.plot(
        [m.iteration for m in tm.train_series],
        [m.tokens_per_sec for m in tm.train_series],
    )
    ax.set_title("throughput (tokens/sec)")
    ax.set_xlabel("iteration")

    ax = axes[1][1]
    ax.axis("off")

    def _f(x: float | None) -> str:
        return f"{x:.3f}" if x is not None else "n/a"

    text = "\n".join(
        [
            f"final train loss : {_f(tm.final_train_loss)}",
            f"final perplexity : {_f(tm.final_perplexity)}",
            f"best val loss    : {_f(tm.best_val_loss)} @ {tm.best_val_iter}",
            f"train/val gap    : {_f(tm.train_val_gap)}",
            f"epochs           : {tm.epochs:.2f}",
            f"mean tokens/sec  : {_f(tm.mean_tokens_per_sec)}",
            f"peak memory (GB) : {_f(tm.peak_memory_gb)}",
            f"trainable params : {_f(tm.trainable_pct)} %",
        ]
    )
    ax.text(0.0, 1.0, text, va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
