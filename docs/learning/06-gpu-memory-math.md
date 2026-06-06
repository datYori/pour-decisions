# 06 — GPU Memory Math

## Why GPU memory planning matters

Running out of VRAM mid-training wastes hours on the GPU box and (on spot) costs real
money. This doc works through the numbers for a 7B model so you can size a GPU before
provisioning.

## Weights

A model's VRAM floor is its weight size:

```
weights_bytes = num_params * bytes_per_param
```

For Mistral-7B (7.24B parameters) in bf16 (2 bytes per parameter):

```
7,241,732,096 * 2 = 14,483,464,192 bytes ≈ 14.5 GB
```

At inference (serving only), this is essentially all you need to budget for weights.
fp32 would be 28 GB; int4 (QLoRA / GGUF) would be ~3.5 GB.

## Optimizer states (training only)

Adam maintains two momentum tensors per trainable parameter, both in fp32 (4 bytes each):

```
optimizer_bytes = num_trainable_params * 2 * 4 bytes
```

For **full fine-tuning** (all 7.24B params trainable):
```
7.24B * 2 * 4 = 57.9 GB
```
Plus the model weights (14.5 GB bf16) and gradients (~14.5 GB bf16): total ~87 GB. This
is why full fine-tuning a 7B model requires at minimum two 80 GB A100s.

For **LoRA** (only 2.1M adapter params are trainable):
```
2,097,152 * 2 * 4 = 16.8 MB
```
Effectively zero. The 14.5 GB base is loaded frozen; gradients flow only through the
tiny adapter. This is the fundamental efficiency win of LoRA.

## KV cache

During training, the KV cache (key and value tensors for every attention head across
the sequence) scales with batch size and sequence length:

```
kv_bytes ≈ 2 * num_layers * num_heads * head_dim * seq_len * batch_size * 2 bytes (bf16)
```

For Mistral-7B (32 layers, grouped-query attention with 8 KV heads, head_dim=128) at
seq_len=8192, batch_size=1:

```
2 * 32 * 8 * 128 * 8192 * 1 * 2 = 1,073,741,824 bytes = 1 GB
```

Using seq_len=32768 (the mistral-finetune default):
```
~4 GB per item in batch
```

Our choice of `seq_len: 8192` keeps this at ~1 GB. This is the main reason to set
seq_len to match actual data length rather than the model's maximum context window.

## Activations

Activations (intermediate tensors saved for the backward pass) are harder to compute
exactly — they depend on the attention kernel, mixed-precision choices, and whether
activation checkpointing is enabled. A rough rule is 1–3× the batch KV cache. For our
setup: ~1–3 GB.

## Budget summary for our setup (training)

| Component | Size |
|-----------|------|
| Frozen base weights (bf16) | ~14.5 GB |
| LoRA adapter weights (bf16) | < 5 MB |
| Adam optimizer states (adapter only) | < 20 MB |
| KV cache (seq_len=8192, batch=1) | ~1 GB |
| Activations | ~1–3 GB |
| **Total estimate** | **~17–19 GB** |

The L40S has **48 GB** — we are well under its limit. Even the A10G (24 GB) would be
sufficient for training.

## Training vs serving: why the GPU sizes differ

- **Training (g6e.xlarge, 1× L40S 48 GB):** needs the full 14.5 GB base + optimizer
  states + activations. We chose 48 GB for headroom and speed.

- **Serving (g5.xlarge, 1× A10G 24 GB):** only needs the merged model weights (~14.5 GB)
  plus KV cache for concurrent requests. With `max-model-len=8192` and moderate
  concurrency, 24 GB is sufficient and saves cost.

## How to size a GPU for an arbitrary model

1. **Serving:** `num_params * bytes_per_param` for the weight floor. Add 20–30% for KV
   cache at your expected concurrency. Example: 13B in bf16 = 26 GB → need 40 GB (A100).

2. **LoRA training:** same weight floor + ~1–3 GB activations per batch item. Optimizer
   overhead is negligible. A 24 GB card handles 7B comfortably; a 48 GB card handles
   13B.

3. **Full fine-tuning:** ~6× the bf16 weight floor (≈87 GB for 7B) — equivalently ~4×
   the fp32 floor; needs multi-GPU or heavy gradient checkpointing.
