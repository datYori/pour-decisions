# pour-decisions

Fine-tune a Mistral open-weight to turn messy free-text cocktail measures
(`"1 1/2 oz Tequila"`, `"2 dashes Angostura"`, `"Juice of 1 lime"`) into
schema-valid JSON — trained with Mistral's own `mistral-finetune`, served on vLLM.

**The domain is a demo. The pipeline is the product:** schema-validated extraction,
constrained decoding, an eval gate, and IaC'd GPU serving. Point it at invoices,
dosages, or lab values and it's the same machine.

---

## Quickstart (local, GPU-free)

```bash
just install    # uv sync --extra dev
just data       # build data/prepared/{train,val,test}.jsonl from committed IBA gold
just test       # ruff + mypy + pytest (24 tests, no GPU needed)
```

Run `just` (no args) to list all available recipes, including GPU workflow commands.

For the GPU steps (train, merge, serve, eval) see `docs/runbooks/`:

- `docs/runbooks/finetune.md` — provision `g5.2xlarge`, run `mistral-finetune`, stream logs
- `docs/runbooks/serve.md` — enable serve instances, trigger vLLM, smoke test, run eval
- `docs/runbooks/regenerate-merged.md` — rebuild merged weights locally from adapter + base

---

## Architecture

```
OpenTofu  -->  g5.2xlarge (train, A10G 24GB)  +  g5.2xlarge x2 (serve, A10G 24GB each)  + S3

[1] data_prep  iba-cocktails (MIT gold) + CocktailDB (CC0 silver, train-only)
      |         raw ingredient line -> target JSON {quantity, unit, ingredient}
      v
[2] mistral-finetune  LoRA single GPU: torchrun --nproc-per-node 1 -m train finetune/7B.yaml
      |               output: consolidated/lora.safetensors  |  MLflow: loss + run id
      v
[3] utils.merge_lora  lora.safetensors + base  ->  merged 7B (bf16)
      v
[4] eval  base vs tuned on held-out gold IBA test
      |   metrics: field accuracy (qty/unit/ingredient), JSON validity, hallucination rate
      |   gate: CI fails if tuned < base
      v
[5] vLLM serve  --tokenizer_mode mistral --load_format mistral  (OpenAI-compatible :8000)
      |          guided_json enforces the schema -> 100% valid responses
      v
[6] serve_client  smoke + structured requests
```

---

## Eval metrics

Three signals, one headline:

| Metric | What it measures | Role |
|---|---|---|
| **Field accuracy** | Exact match on `quantity`, `unit`, normalized `ingredient` | Headline — the base-vs-tuned signal |
| **JSON validity** | % responses matching the Pydantic schema | Guardrail — expected ~100% under `guided_json` |
| **Hallucination rate** | % of `null` gold fields where the model invented a value | Safety floor |

See `docs/learning/05-eval-metrics.md` for the worked Negroni example.

---

## Repo layout

```
src/pour_decisions/    # schema, data prep, metrics, eval runner, serve client
tests/                 # 24 unit tests (all GPU-free)
finetune/7B.yaml       # mistral-finetune single-GPU config
serving/               # docker-compose.yml for vLLM
infra/                 # OpenTofu: g6e train + g5 serve + S3 artifacts
data/raw/              # committed datasets (IBA MIT gold + CocktailDB CC0 silver)
data/prepared/         # generated splits (just data)
docs/learning/         # plain-language explainers for every concept used
docs/runbooks/         # copy-paste GPU steps
reports/               # committed eval reports
.github/workflows/     # CI: lint + mypy + pytest + eval-gate
```

---

## Learning docs

Each concept introduced in the pipeline has a plain-language explainer:

- `01-lora-and-adapters.md` — frozen base + low-rank A·B; what `lora.safetensors` is
- `02-qlora-vs-lora-vs-full.md` — VRAM ladder; why optimizer states dominate
- `03-mistral-finetune-internals.md` — YAML config fields; FSDP single vs multi-GPU; OOM knobs
- `04-vllm-and-guided-json.md` — PagedAttention; constrained decoding; exact serve flags
- `05-eval-metrics.md` — field accuracy vs validity vs hallucination; worked example
- `06-gpu-memory-math.md` — weights vs KV cache vs optimizer; how to size a GPU

---

## Roadmap

**Phase 2 — Bartender agent (conversational assistant)**
A chat assistant that roleplays a bartender, learns the guest's taste/mood through conversation, recommends a cocktail, and renders its structured recipe. Base `Mistral-7B-Instruct` handles free-form chat + recommendation; it calls the Phase-1 fine-tuned extraction adapter as a structured "recipe tool" (`vLLM` `guided_json`) to emit always-valid recipe JSON. Showcases agent + tool-use and vLLM multi-LoRA (`--enable-lora`, hot-swapping base↔adapter per turn). Low infra, high demo value. Depends on Phase 1's trained adapter.

**Phase 3 — production serving platform**
Lift train + serve onto EKS GPU nodegroup; multi-GPU FSDP (`g5.12xlarge`); KEDA/HPA autoscaling on queue depth; Prometheus/Grafana (token throughput, GPU util, TTFT, p95 latency); load test; cost dashboard.

**Phase 4 — Production customer onboarding**
Wrap as a production-ready deliverable: architecture doc, onboarding runbook, deployment guide, eval + cost/latency summary.

**OSS track (parallel)**
The `mistral-finetune` single-GPU / vLLM-adapter-export friction → upstream docs/issue/PR. Cookbook examples for `mistral-common`.

---

## Data attribution

`data/raw/iba/` — IBA official cocktails compiled by Rasmus Bååth, licensed **MIT**.
Source: `rasmusab/iba-cocktails` (`iba-web/` folder only; `wikipedia/` excluded — CC-BY-SA 3.0).
See `data/raw/SOURCES.md` for full provenance.

`data/raw/cocktaildb/` — sourced from TheCocktailDB via Kaggle (`pxxthik/the-cocktail-db-recipe-collection`), licensed **CC0 / Public Domain**. Used for silver training augmentation only; eval stays 100% gold IBA.

Project code: **MIT** — see `LICENSE`.
