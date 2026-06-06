# 05 — Eval metrics

## Why three metrics

The tuning goal is to turn free-text ingredient lines into structured JSON that matches the IBA gold set field-for-field. Three metrics each measure a different failure mode:

**Field accuracy** is the headline number. For every ingredient pair (predicted vs. gold, matched by position), it counts how many of the three fields — `quantity`, `unit`, `ingredient` — are correct. `ingredient` matching is case-insensitive. This is the metric used in the eval gate: tuned must meet or beat base.

**Hallucination rate** measures a specific error the field accuracy number can hide. When the gold has `quantity: null` or `unit: null` (meaning the source literally has no quantity or no unit), did the model invent a value anyway? A model that guesses `"1 oz"` for every ingredient will score well on field accuracy when quantities happen to be present, but its hallucination rate will be high on the null-slot cases. Field accuracy and hallucination rate together tell the full story.

**JSON validity** is a guardrail, not a signal. Under vLLM's guided decoding (`response_format: json_schema`), the output is guaranteed by construction to satisfy the schema — so a validity rate below ~100% in production indicates the guided decoding was bypassed or the schema changed. In offline eval against recorded outputs it is a sanity check: if the model ever produced a string that fails schema validation, something is wrong with the eval pipeline.

## Worked example: Negroni

The IBA Negroni has three ingredients:

| position | gold quantity | gold unit | gold ingredient |
|----------|--------------|-----------|-----------------|
| 0        | 30.0         | ml        | Gin             |
| 1        | 30.0         | ml        | Campari         |
| 2        | 30.0         | ml        | Sweet Vermouth  |

Suppose a base model outputs:

| position | pred quantity | pred unit | pred ingredient  |
|----------|--------------|-----------|------------------|
| 0        | 45.0         | ml        | Gin              |
| 1        | 30.0         | ml        | Campari          |
| 2        | 30.0         | oz        | Sweet Vermouth   |

**Field accuracy computation:**

There are 3 ingredients × 3 fields = 9 slots total.

- `quantity`: position 0 wrong (45 ≠ 30), positions 1 and 2 correct → 2/3 = **0.667**
- `unit`: position 2 wrong (oz ≠ ml), positions 0 and 1 correct → 2/3 = **0.667**
- `ingredient`: all three match (case-insensitive) → 3/3 = **1.000**

**Hallucination rate computation:**

All gold quantities and units are non-null (30 ml, 30 ml, 30 ml). There are therefore zero null slots. The hallucination rate is 0/0, which by convention returns **0.0** (nothing to invent).

**JSON validity computation:**

If the model output is `{"name":"Negroni","ingredients":[{"quantity":45,"unit":"ml","ingredient":"Gin"},{"quantity":30,"unit":"ml","ingredient":"Campari"},{"quantity":30,"unit":"oz","ingredient":"Sweet Vermouth"}]}`, this parses and validates against `Cocktail.model_json_schema()` → **1.0** (valid).

## Why eval uses gold IBA only — never silver

The silver data (CocktailDB, parsed by `silver.py`) is machine-generated: a rule-based parser extracts quantities and units from raw strings. If we evaluated against silver labels, we would be measuring whether the model agrees with our own parser, not whether it is correct. When the parser makes a systematic error — say, mis-extracting a fraction or misidentifying a unit — that error would be invisible in the eval score because both gold and prediction come from the same flawed source.

The IBA data is human-curated and licensed under MIT. Every quantity, unit, and ingredient name was entered by a person. Grading against IBA measures genuine correctness. Grading against silver measures circularity.

Silver is train-only augmentation. It widens the model's exposure to ingredient variety. It never touches `val.jsonl` or `test.jsonl`.

One subtlety worth noting for completeness: silver augmentation (CocktailDB) can reintroduce a handful of cocktail *names* that also appear in the gold val/test splits — for example, "Negroni" or "Margarita" exist in both corpora. This is not a problem. The recipes differ (CocktailDB uses `oz`; IBA uses `ml`) and the training signal is on the raw-string → structured-JSON parse, not on memorising a particular cocktail's recipe. More importantly, **eval labels are always gold** — the model is graded against IBA-curated `{quantity, unit, ingredient}` triples, not against anything derived from the silver corpus. The train/test split guarantee that matters is on the gold set: no gold cocktail appears in both train and test. Silver name-overlap adds training noise at worst, not metric inflation or answer leakage.

## Why position-aligned matching — and the reorder risk

The prompt instructs the model to preserve the input order of ingredient lines. Position alignment is therefore the correct matching strategy: ingredient 0 in the prediction corresponds to ingredient 0 in the gold.

The alternative — set-based matching that finds the best pairing — would silently forgive reordering. That matters because downstream consumers (the dataset builder, the serve client) emit ingredient lines in the original order, and a model that shuffles them may produce semantically correct JSON that is structurally misaligned with the prompt.

If a model systematically reorders ingredients, position-aligned field accuracy will undercount its semantic correctness. That is intentional: we want the metric to penalise reordering, because the prompt contract says order must be preserved. A surprising drop in `ingredient` accuracy on an otherwise capable model is the diagnostic signal to look for reordering behaviour.

A related edge case: ingredients are zipped position-wise using `zip(strict=False)`, so a prediction with *fewer* ingredients than gold is only graded on the pairs that exist — a model could in principle inflate field accuracy by emitting fewer ingredients. This is acceptable for Phase 1 because base and tuned are evaluated identically on the same inputs, so the comparison remains fair. A length-penalized denominator (dividing by `max(len(pred), len(gold))`) is the natural hardening for future phases.
