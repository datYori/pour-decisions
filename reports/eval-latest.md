# Eval: base vs tuned (gold IBA test)

| field | base | tuned | Δ |
|---|---|---|---|
| quantity | 0.924 | 1.000 | +0.076 |
| unit | 0.924 | 1.000 | +0.076 |
| ingredient | 1.000 | 1.000 | +0.000 |

Tuned hallucination rate: 0.000

**Gate: PASS**

<!--
First real run: 2026-05-31, eu-central-1.
LoRA via mistral-finetune (Mistral-7B-Instruct-v0.3, rank 16, seq_len 2048, 300 steps),
merged, served on vLLM (two g5.2xlarge boxes: tuned :8000 + base :8001), evaluated on
the 15 held-out gold IBA cocktails. Tuned reaches perfect quantity+unit normalization
vs base 92.4%, with zero hallucinated ingredients.
-->
