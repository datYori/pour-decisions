# Regenerate the merged model from the adapter + base weights

The merged model (`merged/consolidated.safetensors`, ~13.5GB) is **not stored** --
it is regenerable from two inputs:

1. **Base open weights:** `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0), Mistral
   consolidated format.
2. **LoRA adapter:** `your-hf-username/mistral-7b-cocktail-measures-lora` (private HF repo;
   final = `checkpoints/checkpoint_000300/consolidated/lora.safetensors`). Also kept
   locally under `runs/cocktail-lora/` (gitignored).

Merging is `base + scaling x (B*A)` per LoRA layer -- deterministic, so the
output reproduces the served "tuned" model exactly.

All commands run locally on your laptop.

## Requirements

- **~32 GB RAM** (the base loads to CPU before merge -- 16GB thrashes).
- GPU is **not** required for the merge.
- `mistral-finetune` installed:

**Local:**
```bash
just setup-mistral-finetune
```

- `hf` CLI logged in (the base model is gated; the adapter repo is private).

## 1. Fetch the two inputs

**Local:**
```bash
# base open weights (consolidated format: 3 files)
hf download mistralai/Mistral-7B-Instruct-v0.3 \
  consolidated.safetensors params.json tokenizer.model.v3 \
  --local-dir ~/m7b-base

# adapter (final checkpoint) -- from HF, or skip if you have it under runs/
hf download your-hf-username/mistral-7b-cocktail-measures-lora \
  checkpoints/checkpoint_000300/consolidated/lora.safetensors \
  --local-dir ~/cocktail-adapter
```

## 2. Merge

**Local:**
```bash
just merge-lora
```

Runs `utils.merge_lora` from the mistral-finetune repo (cwd requirement handled internally), copies `tokenizer.model.v3` and `params.json` into the merged dir, and prints the output path on success.

Default paths (`~/m7b-base`, `~/cocktail-adapter`, `~/merged`, `~/mistral-finetune`) match step 1 above. Override any via env:

**Local:**
```bash
MERGED_DIR=/data/merged ADAPTER_DIR=~/runs/cocktail-lora just merge-lora
```

## 3. Serve the merged model (optional)

**Local** (requires Docker with the nvidia runtime):
```bash
just serve-local
```

Pre-warms the page cache (cold gp3 mmap is IOPS-bound; see 07 §D2), then starts vLLM on `:8000` as `cocktail-tuned`. Uses `MERGED_DIR` (default `~/merged`).

## Notes

- **Determinism:** same base + same adapter + same `--scaling` -- byte-stable merge,
  so the regenerated model matches the one behind the committed eval numbers
  (`reports/eval-latest.md`).
- **Why not store the merged file?** It is 13.5GB and fully derived; the 80MB adapter
  + the public base weights are the minimal, durable inputs.
- **Format:** Mistral consolidated, not HF transformers -- serve via vLLM
  `--load_format mistral`, not `AutoModel.from_pretrained`.
