# 01 — LoRA and Adapters

## The core idea: frozen weights + a tiny side-path

Full fine-tuning updates every parameter in the model. For a 7B-parameter model that is
expensive in compute, memory, and storage. LoRA (Low-Rank Adaptation) leaves the original
weight matrices completely frozen and instead trains a small pair of matrices that are
added to the frozen ones at inference time.

For a weight matrix **W** of shape `(d_out, d_in)`, LoRA injects:

```
W_effective = W + B · A
```

where **A** has shape `(rank, d_in)` and **B** has shape `(d_out, rank)`. The rank is a
small integer — 16 in our config. Because `rank << d_in` and `rank << d_out`, the number
of trainable parameters per matrix is `rank*(d_in + d_out)` instead of `d_in*d_out`.

At the start of training, **B** is initialized to all zeros so that `B·A = 0` and the
adapter introduces no change. As training progresses, only A and B are updated; W is never
touched.

## What "adapter" means

The term *adapter* refers generically to any small, trainable module added on top of a
frozen pre-trained model. LoRA is the most common adapter type today. The adapter weights
alone (A and B matrices for every targeted layer) are what gets saved to `lora.safetensors`.

This has two practical consequences:

1. **The base model is reusable.** You can share one copy of Mistral-7B-Instruct-v0.3 on
   disk and load different adapters for different tasks. No duplication of the 14 GB base.

2. **Adapters are small.** Our `lora.safetensors` is on the order of megabytes, not
   gigabytes, so it is easy to version, ship, and swap.

## The 0.03% number in our run

When training starts, mistral-finetune prints:

```
2,097,152 out of 7,241,732,096 parameters are finetuned (0.03%).
```

Where does 2,097,152 come from? **2,097,152 is the authoritative runtime-reported count**
(`2,097,152 / 7,241,732,096 ≈ 0.029%`). The exact value depends on two factors: (1) which
projection matrices mistral-finetune targets by default, and (2) Mistral-7B's use of
Grouped-Query Attention (GQA) — the K and V projections have only 8 KV heads, so they are
1024-dim rather than 4096-dim. Naively assuming all four projections are 4096×4096 would
produce a ~8× overcount; GQA and the default target subset together bring the real figure
to 2.1M.

The 7.24B total includes all weights — embedding table, all attention + feed-forward
layers, output head — of which we update only the LoRA injection points.

## "Adapter" vs "merged"

After training you have two options for serving:

- **Adapter mode:** load the base model + the small `lora.safetensors` at inference time.
  The adapter is added on the fly during the forward pass. Requires the serving stack to
  understand LoRA (e.g., vLLM's dynamic LoRA endpoint).

- **Merged mode:** use `merge_lora` (mistral-finetune's utility) to permanently fold
  `B·A` into `W`, producing a standard Mistral checkpoint with no LoRA structure. The
  merged model runs on any vLLM instance without special flags. This is what Phase 1 does.

The tradeoff: merged is simpler and slightly faster at inference; adapter mode lets you
hot-swap tasks without re-loading the base.
