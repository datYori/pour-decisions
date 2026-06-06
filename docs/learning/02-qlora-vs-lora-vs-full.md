# 02 — QLoRA vs LoRA vs Full Fine-tune

## The VRAM ladder

Three techniques, in order of decreasing VRAM requirement at training time:

| Technique | Weights stored as | Optimizer states | Typical VRAM (7B model) | When to use |
|-----------|------------------|-----------------|------------------------|-------------|
| Full fine-tune | bf16 (14 GB) | fp32 Adam for all 7B params (~56 GB) | ~80 GB+ | Massive data, large domain shift |
| LoRA | bf16 (14 GB) | fp32 Adam for adapter params only (~17–25 MB) | ~18–24 GB | Moderate data, specialized task |
| QLoRA | int4 (≈3.5 GB) | fp32 Adam for adapter params only (~17–25 MB) | ~8–12 GB | Tight VRAM budget (consumer GPU) |

## Why optimizer states dominate

Adam stores two extra tensors per parameter: the first moment (mean of gradients) and
the second moment (uncentered variance). Both are kept in fp32 regardless of the weight
dtype. For full fine-tuning of 7B parameters that is `7B * 3 * 4 bytes ≈ 84 GB` just
for weights + gradients + optimizer states — before activations or batch. This is why full fine-tuning a 7B
model requires multiple high-end GPUs.

LoRA sidesteps this by only maintaining Adam states for the adapter parameters. With
2.1M trainable parameters the optimizer overhead is `2.1M * 3 * 4 bytes ≈ 25 MB` —
essentially free.

## Why a 48 GB card is comfortable for 7B LoRA

Working through the numbers for our setup:

- Frozen base weights in bf16: `7.24B * 2 bytes ≈ 14.5 GB`
- LoRA trainable weights (bf16): `2.1M * 2 bytes ≈ 4 MB`
- Adam states for adapters (fp32): `2.1M * 8 bytes ≈ 17 MB`
- Activations for seq_len=8192, batch_size=1: roughly 2–4 GB
- KV cache during training forward pass: small (short sequences)

Total: approximately 17–19 GB. The L40S has 48 GB, leaving ample headroom. Even a 24 GB
A10G would be sufficient for inference of the merged model (base weights only, no
optimizer). See `06-gpu-memory-math.md` for the full breakdown.

## What QLoRA adds — and why we didn't need it here

QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantization of the frozen base
weights using NF4 (Normal Float 4), a data type optimized for normally distributed
weights. This compresses the 14 GB base from bf16 to about 3.5 GB, enabling 7B LoRA on
a single 24 GB or even 16 GB GPU.

The cost: QLoRA introduces quantization noise in the frozen weights, which can slightly
lower final model quality compared to bf16 LoRA. It also requires bitsandbytes and is
not supported in mistral-finetune's official tooling (which uses bf16 LoRA).

For our task — a 48 GB L40S training on a dataset of ~90 cocktails — there is no memory
pressure. bf16 LoRA via mistral-finetune is both simpler and higher quality. QLoRA would
only be relevant if we needed to train on a consumer GPU (16 GB RTX 4080, for example).

## Summary decision tree

```
Available VRAM?
├── < 16 GB  → QLoRA (bitsandbytes + PEFT)
├── 16–24 GB → LoRA in bf16; tight but feasible for 7B
└── ≥ 24 GB  → LoRA in bf16 comfortably (our case: 48 GB L40S)
                └── >> 80 GB → consider full fine-tune if data justifies it
```
