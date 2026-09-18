# Cost-estimation equation from swept run logs

`cost_model.py` repeatedly runs the solver with different parameters, logs how
many **input/output tokens each model consumes**, and fits a linear cost
equation per model:

```
cost_usd(model) ≈ a · input_tokens + b · output_tokens
```

The fitted coefficients are the model's $-per-token rates
(`a·1e6` = $/Mtok input, `b·1e6` = $/Mtok output). A whole-pipeline estimate is
the sum of the per-model equations:

```
total_cost ≈ Σ_model ( a_model · input_tokens_model + b_model · output_tokens_model )
```

## Why this is meaningful (even though cost is deterministic)

The Anthropic adapter computes `usd` as `(in·price_in + out·price_out)/1e6` per
model, so cost is *exactly* linear in tokens. The regression therefore:

1. **confirms linearity** — `R² = 1` with a near-zero intercept,
2. **recovers and documents the per-model $/Mtok rates** straight from logged
   usage (no need to trust the table by hand), and
3. **quantifies the error of a single blended rate**. The script also fits one
   mix-agnostic equation `total_cost ≈ A·total_in + B·total_out`; because the
   pipeline routes different roles to different-priced models, the blended fit
   has `R² < 1` (in our offline check ≈ 0.01–0.3) — empirical proof that you
   need the per-model decomposition, not one average rate, to estimate cost.

## How the sweep is designed

The **independent variables are the solver knobs** in `chess_study._key_params`
(`pipeline.K_executor`, `pipeline.max_split_depth`, `pipeline.max_atoms_per_split`,
`pipeline.max_node_calls`, `mcts.N`, …) — *not* the number of plies and *not*
per-model output-token caps. Varying those knobs is what spreads the
input/output token volume per model, which is what the regression needs.
Problem size (`--max-plies`) is held **fixed** as a cost bound, not swept.

`concord/config/cost_model.sweep.yaml` therefore, for each model in the set:

- routes `models.{execution,splitter,combiner}.model` through
  `claude-sonnet-4-6`, `claude-opus-4-8`, `claude-haiku-4-5` (model coverage —
  the per-model equation needs each model to accrue usage), and
- sweeps the knobs (`K_executor`, `max_split_depth`, `max_atoms_per_split`,
  `max_node_calls`) to vary token volume.

The fitter flags `identifiable=false` when a model's input/output volumes come
out near-collinear (then only a blended `a + b·ratio` is recoverable); widen
the knob sweep if you see it on real data.

## Run

Offline plumbing test (mock LLM — generation is mocked but the **real model
names** are preserved, so token usage is attributed per model; `usd` is then
imputed from the price table since the mock bills $0):

```bash
python -m concord.experiments.cost_model --mock \
    --sweep concord/config/cost_model.sweep.yaml
```

Real collection (needs `ANTHROPIC_API_KEY`):

```bash
export ANTHROPIC_API_KEY=sk-...
python -m concord.experiments.cost_model \
    --config concord/config/wide_and_deep.yaml \
    --sweep  concord/config/cost_model.sweep.yaml
```

Refit existing logs (from this script or any `*.jsonl` of
`{model,input_tokens,output_tokens,usd}` rows) without re-running:

```bash
python -m concord.experiments.cost_model \
    --fit-only results/cost_model/<ts>/cost_samples.jsonl
```

Useful flags: `--models claude-opus-4-8,claude-sonnet-4-6` (restrict the fit),
`--target {auto,usd,usd_table}` (auto fits logged `usd`, falling back to
table-imputed `usd_table` when all logged costs are 0), `--repeats N`.

## Outputs (`results/cost_model/<timestamp>/`)

| file | contents |
|---|---|
| `cost_samples.jsonl` / `.csv` | one row per (run, model): tokens, calls, logged `usd`, table `usd` |
| `cost_model.json` | per-model fits (coefficients, $/Mtok, R², intercept, identifiability), blended aggregate, combined equation |
| `cost_model.md` | readable equation tables (fitted vs reference $/Mtok) |

## What an offline run recovers

```
COST EQUATION  (target=usd_table)
  claude-sonnet-4-6   cost_usd ≈ 3.00e-06·input_tokens + 1.50e-05·output_tokens   R²=1.0000
  claude-opus-4-8     cost_usd ≈ 1.50e-05·input_tokens + 7.50e-05·output_tokens   R²=1.0000
  claude-haiku-4-5    cost_usd ≈ 8.00e-07·input_tokens + 4.00e-06·output_tokens   R²=1.0000
  blended single-rate R²≈0.01   <- one average rate cannot fit the model mix
```

i.e. fitted $/Mtok = {opus 15/75, sonnet 3/15, haiku 0.8/4}, matching
`anthropic_adapter._PRICE_PER_MTOK`. The script imports that table as the
reference column so any drift between the fitted rates and the configured
prices is visible at a glance.
