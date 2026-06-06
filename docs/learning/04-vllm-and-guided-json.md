# 04 — vLLM and Guided JSON

## vLLM in one paragraph

vLLM is a high-throughput inference server for large language models built around two core ideas: **PagedAttention** and **continuous batching**. PagedAttention manages the KV cache in fixed-size pages (analogous to OS virtual memory), eliminating the fragmentation that wastes GPU memory when sequences have variable lengths — a 7B model that would otherwise fit one or two concurrent requests can handle many more. Continuous batching (also called iteration-level scheduling) goes further: rather than waiting for an entire batch to finish before starting new requests, vLLM slots new tokens into the engine at every decoding step, keeping GPU utilization near 100%. The result is that vLLM exposes an OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`) so any code written against the OpenAI SDK works unchanged against a local or self-hosted vLLM endpoint — exactly what `serve_client.py` exploits.

## What `response_format: json_schema` / `strict: True` does

When you pass `response_format: {"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": <JSON Schema>}}` to a vLLM endpoint that has constrained decoding enabled, vLLM compiles the JSON Schema into a grammar and masks logits during decoding (current default backend: `xgrammar`; `outlines`/`guidance` are selectable alternatives, version-dependent). At each decoding step, the logit mask from that grammar zeroes out every token that would produce output not conformable with the schema. The model never *chooses* an invalid token — it is structurally impossible. The consequence is:

- **JSON validity approaches 100%** regardless of model quality. The metric becomes a guardrail rather than signal.
- **The schema is the contract.** If the schema says `ingredients` is a required array of objects with `quantity: number | null`, the output will always satisfy that. Field *accuracy* (are the values semantically correct?) is the real headline metric.
- `strict: True` tells vLLM to apply the mask strictly rather than best-effort; without it some backends may fall back to unconstrained generation when the FSM is expensive to evaluate.

We derive the schema from `Cocktail.model_json_schema()` — the same Pydantic model that defines training targets — so training distribution and decoding constraint are always in sync.

## The exact serve command (and why those flags)

```bash
docker run --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /models/merged \
  --tokenizer_mode mistral \
  --load_format mistral \
  --config_format mistral \
  --served-model-name cocktail-tuned \
  --max-model-len 8192
```

**`--tokenizer_mode mistral`** — Mistral's tokenizer uses a custom `SentencePiece` variant (`tokenizer.model.v3`) rather than the HuggingFace `tokenizer.json` format. Without this flag vLLM falls back to the HF tokenizer path, which either fails to load or silently misaligns token IDs, producing garbled output.

**`--load_format mistral`** — The merged checkpoint produced by `mistral-finetune`'s `merge_lora` utility is saved in Mistral's own consolidated safetensors layout (`consolidated.safetensors` + `params.json`), not HuggingFace's sharded `model.safetensors.index.json` layout. `--load_format mistral` tells vLLM's weight loader to use the Mistral-native reader instead of the default HF reader. Without it the loader looks for HF shards, finds none, and aborts.

**`--config_format mistral`** — Companion to `--load_format mistral`; reads `params.json` (Mistral format) rather than `config.json` (HF format) to determine model architecture.

**`--max-model-len 8192`** — Our training sequences are short (cocktail records are ~200 tokens). Capping the context window at 8192 rather than the model's default 32768 reduces KV cache allocation and lets the A10G (24GB) serve the model without OOM.

## Dynamic LoRA and why Phase 1 serves merged

vLLM supports a `--enable-lora` mode where multiple LoRA adapters can be loaded and hot-swapped at request time via the `/v1/load_lora_adapter` endpoint. This is appealing for multi-tenant scenarios: one base model instance, many per-customer adapters loaded on demand.

**The security caveat:** the dynamic LoRA endpoint accepts an arbitrary `lora_path` from the HTTP request body and loads weights from that path on the server's filesystem (or a remote URL). This means any caller who can reach the endpoint can cause the server to load arbitrary weights — a path-traversal or supply-chain attack surface. vLLM's documentation explicitly flags this as a trusted-admin-only endpoint; it must never be exposed on a public or multi-tenant network without strong authentication.

**Why Phase 1 serves the merged checkpoint instead:** For a single fine-tuned model, there is no operational benefit to dynamic LoRA. We merge the adapter into the base weights once (using `mistral-finetune`'s `merge_lora` utility), ship the merged `consolidated.safetensors`, and serve it as a standalone model. This eliminates the dynamic-loading attack surface entirely, simplifies the serving stack (no adapter registry, no hot-swap logic), and is what the `serving/docker-compose.yml` reflects. Dynamic LoRA would only be worth the operational complexity if we needed to serve many adapters without the memory cost of holding multiple full models simultaneously.
