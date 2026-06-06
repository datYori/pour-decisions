# Local fine-tune matrix: per-model untuned -> tuned delta

Eval on the 15 held-out gold IBA cocktails (n=15; deltas are noisy at this size).

| model | params | license | backend | mode | qty u->t (d) | unit u->t (d) | ingr u->t (d) | valid u->t | halluc | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.5B | apache-2.0 | mlx | unconstrained | 1.000->0.938 (-0.062) | 1.000->0.969 (-0.031) | 1.000->0.923 (-0.077) | 0.067->1.000 | 0.500 | 38 |
| qwen2.5-0.5b | 0.5B | apache-2.0 | mlx | constrained | 0.409->1.000 (+0.591) | 0.379->0.985 (+0.606) | 0.955->0.970 (+0.015) | 1.000->1.000 | 0.500 | 38 |
| ministral-3-3b-base | 3B | apache-2.0 | mlx | unconstrained | 0.984->1.000 (+0.016) | 1.000->1.000 (+0.000) | 1.000->1.000 (+0.000) | 0.933->1.000 | 0.000 | 204 |
| ministral-3-3b-base | 3B | apache-2.0 | mlx | constrained | 0.985->1.000 (+0.015) | 0.985->1.000 (+0.015) | 1.000->1.000 (+0.000) | 1.000->1.000 | 0.000 | 204 |

## Training

| model | final train loss | best val loss | dashboard |
|---|---|---|---|
| qwen2.5-0.5b | 0.086 | 0.448 | [png](runs/local/qwen2.5-0.5b/training_dashboard.png) |
| ministral-3-3b-base | 0.111 | 0.359 | [png](runs/local/ministral-3-3b-base/training_dashboard.png) |
