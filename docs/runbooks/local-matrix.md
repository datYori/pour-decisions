# Runbook: local fine-tune + delta matrix (Apple Silicon)

Prereqs: Apple Silicon Mac, `uv`, `hf` CLI logged in for gated models.

1. Install deps: `just local-setup`            # uv sync --extra mlx
2. Build data:   `just data`                   # data/prepared/{train,val,test}.jsonl
3. Gated models: `hf auth login` then accept the license on the model's HF page
   (google/gemma-3-270m, google/gemma-3-1b-pt, meta-llama/Llama-3.2-1B).
4. One model:    `just local-one ministral-3-3b-base`
5. Full matrix:  `just local-matrix`           # MLX on this Mac; pass backend=peft only on a CUDA workstation
6. Read report:  `reports/local-matrix.md` (+ machine-readable `reports/local-matrix.json`)

Notes:
- `just` arguments are positional, not `name=value`. `just local-one ministral-3-3b-base` uses
  the defaults (backend=mlx, both eval modes). To override, pass in order:
  `just local-one qwen2.5-0.5b mlx unconstrained`. To bypass `just` entirely:
  `uv run python -m pour_decisions.matrix.run_matrix --backend mlx --key qwen2.5-0.5b --modes unconstrained,constrained`.
- Each row is that model's own untuned -> tuned delta on the 15 gold IBA test cocktails.
- Unconstrained mode shows the real JSON-validity lift; constrained mode (Outlines) is prod-faithful.
- runs/local/<key>/ holds the adapter + per-model logs (gitignored).

## Validation findings (2026-06-05, Apple M4 Pro 24GB, mlx-lm 0.31.3)

First end-to-end MLX runs on this Mac:

- **qwen2.5-0.5b (apache-2.0): clean headline demonstration.** Train ~58s.
  Unconstrained JSON validity 0.067 -> 1.000 (the base almost never emits parseable JSON; tuning
  fixes it). Constrained quantity 0.409 -> 1.000, unit 0.379 -> 0.985. A small base with large
  headroom gains a large, measurable delta. (lr 1e-4 default.)

## Root-cause investigation: the mistral3 "garbage" was the learning rate (2026-06-06)

An earlier version of this note blamed the `fix_mistral_regex` tokenizer warning for
ministral-3-3b-base producing garbage after tuning. **That was wrong.** A systematic investigation
(Apple M4 Pro 24GB, mlx-lm 0.31.3, transformers 5.10.2) found the real cause is the LoRA learning
rate, not the tokenizer and not the `mistral3` arch.

**Root cause: lr 1e-4 was 10x mlx-lm's 1e-5 default and mode-collapsed this saturated 3B base.**
Holding model + data + adapter config identical and changing only the LR:

- **lr 1e-4:** training *looks* fine (loss drifts down) but free-running generation is degenerate
  repetitive soup (`{"name":"0SS...iceiceice...`), JSON validity 0.0. It is already degenerate at
  the first saved checkpoint (iter 100: nearly pure `.`/`0`), and training only drags the collapse
  toward the target vocabulary (iter 200 loops `"ingredient"`, iter 300 mixes schema fragments).
  Classic "loss went down, model is broken" from an update magnitude that is too hot.
- **lr 1e-5:** training descends smoothly (val 1.74 -> 0.43), generation is clean schema-valid
  JSON.

**Why qwen2.5-0.5b survived the same 1e-4 recipe:** headroom. Its untuned unconstrained validity
was 0.067 (a strong, well-conditioned signal to learn). The 3B was near-saturated (untuned
validity ~0.93), so 1e-4 had almost nothing legitimate to learn and overshot into collapse. LR
must scale inversely with base competence; `registry.py` now sets >=1.5B bases to 1e-5 (validated
at 3B-Ministral; 1.5B/1.7B extrapolated, still unrun).

**Suspects ruled out, with evidence:**

1. **`fix_mistral_regex` warning: red herring.** 0.000% token divergence across 16,838 corpus
   tokens with vs without the flag; round-trip lossless; the canonical `'The'` case is already
   tokenized correctly by default. It is the known false-positive that fires when loading from a
   directory holding both model weights and tokenizer files (transformers issue #42591); Mistral
   itself rates the real effect at <1% of tokens. The *untuned* model uses the same tokenizer yet
   scores 0.93, so the tokenizer cannot be what corrupts only the tuned model.
2. **Adapter misrouting on `mistral3`: ruled out.** `self.layers` -> `language_model.model.layers`
   is used at both train and load (`mlx_lm/models/mistral3.py`), adapter keys match the module
   tree, and the tuned output starts on-format (`{"name":"`) -> the adapter is applied to the
   right modules.
3. **`mistral3` arch / forward bug: ruled out.** Untuned generation is flawless and gentle-LR
   (1e-5) tuned generation is flawless; the text-only forward path is correct.

**Result (lr 1e-5, re-run 2026-06-06):** ministral-3-3b-base now trains cleanly and produces valid
JSON -- the garbage is gone. As the headroom argument predicts, the delta is small:

- unconstrained: JSON validity 0.933 -> 1.000, quantity 0.984 -> 1.000 (unit/ingredient already 1.0);
- constrained: quantity 0.985 -> 1.000, unit 0.985 -> 1.000;
- tuned hallucination 0.000 in both modes.

The base was already near-perfect, so the fine-tune mostly removes the markdown fence and the last
few unit/quantity errors. The large-delta demonstration remains qwen2.5-0.5b (unconstrained
validity 0.067 -> 1.000). Both rows are in `reports/local-matrix.{md,json}`.

Note: constrained mode on a 3B+ via Outlines is memory-heavy; the both-mode 3B eval takes ~8 min.
