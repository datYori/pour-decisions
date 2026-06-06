# 03 — mistral-finetune Internals

## What mistral-finetune is

`mistral-finetune` is Mistral AI's official LoRA training library, purpose-built for
their own model family. It handles the full training loop (data loading, LoRA injection,
gradient accumulation, checkpointing) and two essential utilities: `reformat_data` (which
normalizes your JSONL into the exact tokenization format the trainer expects) and
`validate_data` (which checks for malformed records, excessively long sequences, and
correct message roles before training starts). Using the official tooling ensures your
adapter is guaranteed compatible with Mistral's consolidated checkpoint format.

## What each field in `finetune/7B.yaml` controls

### `data` block
```yaml
data:
  instruct_data: "data/prepared/train.jsonl"
  data: ""
  eval_instruct_data: "data/prepared/val.jsonl"
```
`instruct_data` is the fine-tuning JSONL in `{"messages": [...]}` format (one record per
line). `data` is for completion-style (non-instruct) data — we leave it empty. `eval_instruct_data`
is evaluated periodically according to `eval_freq` to track validation loss.

### `model_id_or_path`
Path to the Mistral consolidated checkpoint directory (containing `consolidated.safetensors`,
`tokenizer.model.v3`, `params.json`). Must be the Mistral native format — not HuggingFace.

### `lora.rank`
The rank of the low-rank decomposition. Higher rank = more capacity but more parameters
and more overfitting risk on small datasets. Rank 16 is a safe default for a specialized
extraction task trained on ~90 cocktails. See `01-lora-and-adapters.md` for the math.

### `seq_len`
Maximum token length per training example. mistral-finetune's default is 32768. Our
cocktail records are very short (the longest prompt + JSON target is well under 1000
tokens), so setting `seq_len: 8192` has no effect on data quality but significantly
reduces the memory allocated for activation buffers. If you hit OOM, halve this first.

### `batch_size`
Number of examples per gradient step per GPU. With a single GPU and short sequences,
`batch_size: 1` is stable. Increasing it increases throughput but also VRAM use.
Effective batch size is `batch_size * gradient_accumulation_steps` (not exposed in
this YAML — the tool handles accumulation internally).

### `max_steps`
Total training steps. With 62 training cocktails (IBA gold only) and batch_size=1,
`max_steps: 300` is roughly 4–5 passes over the data, which is sufficient for
convergence on a narrow extraction task. Watch the eval loss curve; stop early if it
flattens or rises.

### `optim` block
```yaml
optim:
  lr: 6.e-5
  weight_decay: 0.1
  pct_start: 0.05
```
Standard AdamW optimizer. `lr: 6e-5` is in the middle of the LoRA-safe range (1e-5 to
1e-4). `weight_decay: 0.1` provides mild L2 regularization on adapter weights. `pct_start: 0.05`
means 5% of `max_steps` (15 steps) are used for the linear learning-rate warm-up before
the cosine decay.

### Checkpoint and eval flags
- `ckpt_freq: 100` — save a checkpoint every 100 steps.
- `eval_freq: 50` — evaluate on `val.jsonl` every 50 steps.
- `save_adapters: True` — save only the LoRA adapter weights (not the full model), which
  is what we want; the base model is unchanged.
- `run_dir: "runs/cocktail-lora"` — output directory for checkpoints and logs.

## FSDP: single-GPU vs multi-GPU

mistral-finetune uses PyTorch's FSDP (Fully Sharded Data Parallel) for distributed
training. The entry point is always `torchrun`:

```bash
# Single GPU (our setup):
torchrun --nproc-per-node 1 --master_port $RANDOM -m train finetune/7B.yaml

# Multi-GPU (e.g., 4x L40S):
torchrun --nproc-per-node 4 --master_port $RANDOM -m train finetune/7B.yaml
```

With `--nproc-per-node 1`, FSDP degenerates to a no-op sharding pass (one process holds
everything). No changes to the YAML are needed — mistral-finetune auto-detects the world
size from torchrun. Increasing `--nproc-per-node` to match available GPUs will linearly
increase throughput; the effective batch size also scales.

## The two OOM knobs

If training crashes with an out-of-memory error, the two levers to pull are:

1. **`seq_len`** — halving seq_len roughly halves the activation memory. Go from 8192 →
   4096 → 2048. Check that no training record exceeds the new limit (validate_data will
   warn you).

2. **`batch_size`** — dropping to 1 is usually already the minimum; if you're already at
   1 and still OOM, reduce seq_len further before trying other approaches.

Do not reduce `lora.rank` as a first response to OOM — the memory saving is marginal and
the quality hit can be significant.

## Why output is Mistral-format `lora.safetensors` — and why that requires a merge for vLLM

mistral-finetune saves adapters in Mistral's own serialization format (not HuggingFace
PEFT `.bin`). The checkpoint structure is:

```
runs/cocktail-lora/checkpoints/checkpoint_000300/
└── consolidated/
    └── lora.safetensors   # LoRA A+B matrices only; base weights NOT included
```

vLLM's standard serving path loads a complete model checkpoint. It does support dynamic
LoRA loading (`--enable-lora`), but that endpoint is intended for trusted-admin use and
requires the adapter to be in a specific directory structure. For simplicity and
portability, Phase 1 uses `utils.merge_lora` (bundled with mistral-finetune) to fold the
adapter into the base, producing a standalone Mistral checkpoint that vLLM serves without
any special flags. See `docs/runbooks/serve.md` for the exact merge command.
