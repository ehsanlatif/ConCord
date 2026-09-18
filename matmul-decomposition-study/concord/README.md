# Concord

Adaptive Tree-search with Unanimous-Confidence Decomposition — research
prototype matching `SPEC.md` and built per `IMPLEMENTATION_PLAN.md`.

## What's here

MCTS-driven decomposition search with:

- **Semantic Density** confidence (not cosine-to-centroid; cosine is shipped
  as an ablation arm)
- **Hybrid clustering**: cosine pre-filter + entailment confirm; sub-quadratic
  in K (asserted in tests)
- **Sample-weight oracle** auto-detected per model (white-box logprobs ↔
  black-box Laplace-smoothed class frequency)
- **PUCT + progressive widening** (the old beam `M` is the new widening
  coefficient `C`)
- **Guarded-max backup** with `n_min` visit guard (test proves single-rollout
  inflation is blocked)
- **Incremental hard coherence gate** with **σ always recorded** — soft-relax
  fallback when no coherent terminal is found
- **Concrete depth governance**: `min(critical_path(G') + slack, B/L)` +
  marginal-gain stop rule

## Repo layout

```
concord/
  config/
    default.yaml             # all §9 defaults
    ablations/*.yaml         # one per §8.4 arm
  core/
    config.py types.py telemetry.py orchestrator.py expansion.py
    llm/            # client protocol + mock + Anthropic adapter
    confidence/     # embed, entailment, cluster, sample-weight, SD, verifier, score
    structure/      # graph, decompose, coherence
    search/         # mcts, widening, transposition, depth, fallback
  experiments/
    run.py          # one question, end-to-end
    compare.py      # v2 vs B0..B3 on a slice
    ablate.py       # one ablation arm
    sweep.py        # c_puct / n_min / K grid
    aggregate.py    # emits RESULTS.md
    baselines/      # b0_cot, b1_self_consistency, b2_concord_v1, b3_mcts_no_sd_no_gate
    metrics.py
  tests/
    unit/ integration/ edge_cases/    # 65 tests across M0..M6
```

## Running tests (no API calls)

```bash
.venv/bin/python -m pytest concord/tests/ -v
```

All 65 tests pass deterministically on the mock LLM.

## Running one question end-to-end

Mock LLM (no API, sanity check):
```bash
.venv/bin/python -m concord.experiments.run --mock --index 0 --N 8 --K 3
```

Real Anthropic API (single model for every role):
```bash
export ANTHROPIC_API_KEY=sk-...
.venv/bin/python -m concord.experiments.run \
    --dataset data/eval_set.json --index 0 \
    --model claude-sonnet-4-6 --N 16 --K 4
```

Trace is written to `results/concord/<run-id>.jsonl`.

## Using different models per role

Concord has four LLM-using roles. Each can be pinned to a different model
in a YAML config:

| role | what it does | when it fires |
|---|---|---|
| `execution` | the K-sample LLM calls inside MCTS expansion (the actual reasoning) | every node expansion |
| `decomposition` | proposes the next subproblem when no explicit graph is exposed | implicit decomposer only |
| `classification` | Phase-1 task type / domain lookup | once per problem if `domain` not supplied |
| `verification` | LLM-judge for response quality | every scored response (when distinct from `execution`) |

Example YAML (also shipped at [config/per_role.yaml](config/per_role.yaml)):

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6      # fallback for any role that doesn't override
  temperature: 1.0

models:
  execution:
    provider: anthropic
    model: claude-opus-4-7      # strongest model — does the heavy lifting
    temperature: 1.0
  decomposition:
    model: claude-sonnet-4-6
    temperature: 0.3
  classification:
    model: claude-haiku-4-5     # cheap one-shot category lookup
    temperature: 0.0
  verification:
    model: claude-haiku-4-5     # cheap LLM-judge
    temperature: 0.0
```

Run with the per-role config:
```bash
.venv/bin/python -m concord.experiments.run_all \
    --config concord/config/per_role.yaml --N 16 --K 4
```

### Shipped profiles

| YAML | profile |
|---|---|
| [config/default.yaml](config/default.yaml) | mock LLM everywhere; the testing baseline |
| [config/cheap.yaml](config/cheap.yaml) | haiku everywhere — cheap pipeline validation |
| [config/quality.yaml](config/quality.yaml) | opus everywhere, larger budget — highest expected accuracy |
| [config/per_role.yaml](config/per_role.yaml) | opus / sonnet / haiku / haiku per role — recommended default |
| [config/wide_and_deep.yaml](config/wide_and_deep.yaml) | wider + deeper search (N=128, K=8, C=4, slack=4) for hard problems |

## Widening + deepening the search

When the default settings don't solve a problem, the search is usually starved
on one of four axes. Here is the mental model:

```
Per node:       width  = floor(C * N(node)^β)         ← progressive widening
Per expansion:  K LLM calls                            ← samples-per-subproblem
Total budget:                  N LLM calls (hard ceiling)
Depth cap:    d_max = min(critical_path(G') + slack,  N // K)
Stop early:   ΔU_s < λ · K                            ← marginal-gain rule
```

### Knob reference

| Knob | YAML | What it controls | Default | Try if… |
|---|---|---|---|---|
| `N` | `mcts.N` | total LLM-call budget for the whole search | 16 | …the problem isn't solving — almost always the dominant lever |
| `K` | `sampling.K_blackbox` | samples per expansion (per node) | 4 | …meaning classes are collapsing too eagerly (raise to 8–16); but each ↑K eats your rollout budget faster |
| `C` | `mcts.C` | progressive widening coefficient | 2 | …the tree is staying narrow even at high N (raise to 4–8) |
| `β` | `mcts.beta` | widening exponent | 0.5 | …you want near-linear widening (0.7–1.0); rarely needed |
| `slack` | `depth.slack` | extra depth above `critical_path(G')` | 2 | …on chess/chemistry/logic problems where `extract_graph` returns a single-node graph, **this is the only depth knob** (try 6–12) |
| `λ` | `depth.lam` | marginal-gain stop threshold | 0.02 | …the search keeps stopping early; lower (0.005) or pass `--no-stop` |
| `c_puct` | `mcts.c_puct` | PUCT exploration constant | 3.0 | …search is fixating on one branch (raise to 4–6) |
| `n_min` | `mcts.n_min` | visits required before child counts for backup | 3 | …backed-up Q is too slow to propagate (rarely lower) |

### Two important consequences of the depth formula

1. **`d_max = min(critical_path + slack, N // K)`** — when `N // K` is the
   binding constraint, you are **budget-limited**, not structure-limited.
   With `N=16, K=4`, you get only `N//K = 4` expansions worth of depth.
   Raising slack does nothing in that regime — you need more N (or less K).

2. **For problems without `Problem node_K:` markers** (chess / chemistry /
   logic / cs in the eval set), `extract_graph` falls back to a single-node
   graph: `critical_path = 0`, so structural cap = `slack`. To go deep on
   those, you must raise `slack` (or pin `ablation.depth: fixed` with a high
   `budget_cap`).

### Two ways to widen and deepen

**Use the shipped preset** (recommended starting point):

```bash
.venv/bin/python -m concord.experiments.run_all \
    --config concord/config/wide_and_deep.yaml
```

The preset bumps `N: 16→128`, `K: 4→8`, `C: 2→4`, `slack: 2→4`, `c_puct: 3→4`,
`lam: 0.02→0.01`.

**Or override individual knobs from the CLI:**

```bash
.venv/bin/python -m concord.experiments.run_all \
    --N 256 --K 8 --C 6 --slack 8 --c-puct 4 --lam 0.005
```

Every knob in the table above has a matching `--<flag>`. `--no-stop` disables
the marginal-gain stop entirely (search runs the full `N` rollouts). Pass
`--limit 1` while you're tuning so you only burn budget on one question.

### Diagnosing whether widening actually engaged

Drop the resulting `*_tree.json` into the viewer or grep it:

```bash
jq '.nodes | map(.n_children) | max' results/concord/<run-id>_tree.json
jq '.nodes | map(.depth) | max'      results/concord/<run-id>_tree.json
```

If `max n_children == 1` even at high `C`, the LLM is producing near-identical
samples — raise temperature or K. If `max depth` is well below your target,
either `N // K` is binding (raise N or lower K) or `slack` is too small.

## Different models per role: CLI overrides

Any role can be pinned from the command line without writing YAML:

```bash
.venv/bin/python -m concord.experiments.run_all \
    --execution-model claude-opus-4-7 \
    --verification-model claude-haiku-4-5 \
    --N 16 --K 4
```

### Cost accounting

When roles point at distinct models, the result's `cost.per_role` dict
breaks down LLM spend by role:

```json
"cost": {
  "calls": 73,
  "input_tokens": 12480,
  "output_tokens": 4521,
  "usd": 0.0623,
  "per_role": {
    "execution":      {"calls": 64, "usd": 0.0598, ...},
    "decomposition":  {"calls": 0,  "usd": 0.0,    ...},
    "classification": {"calls": 1,  "usd": 0.0001, ...},
    "verification":   {"calls": 8,  "usd": 0.0024, ...}
  }
}
```

When multiple roles share the same model spec they share a client and
the same `cost` entry — that's the right behaviour, not double-counting.

## Visualization & per-question debugging

Every `solve()` writes three artifacts under `results/concord/`:

| File | Contents |
|---|---|
| `<run-id>.jsonl` | High-level telemetry (run_start, problem, classified, role_models, verifier, phase1, rollouts cost, run_end). |
| `<run-id>_tree.json` | Final MCTS tree snapshot: `meta` (config + role models), `summary` (answer, σ, chosen_node_id, per-role cost), `nodes[]` (every node with `id`/`parent_id`/`N`/`Q`/`prior`/`U_s`/`sigma`/`terminal`/`gated_fail`/`state.resolved`/`state.bindings`/`subproblem_text`/`answer_text`), `edges[]`. |
| `<run-id>_rollouts.jsonl` | One row per rollout: `selected_path` (root→leaf node ids), `expanded_parent_id` / `new_child_id`, `leaf` summary, `value_backed_up`, `backup_path` (post-rollout N+Q per ancestor), `expansion` (subproblem id, prompt excerpt, **K samples with text + extracted answer + class index + weight**, **clusters with mass/U_s/SD_mean/v_mean/members**, oracle mode, queried_pairs), and `gate` (sigma + violated/all_checked constraints). |

### Tree viewer (no install)

```bash
# Run a question first to produce artifacts:
.venv/bin/python -m concord.experiments.run --mock --index 0 --N 8 --K 3

# Then open the viewer:
open concord/viz/tree_viewer.html
# (or `python3 -m http.server 8765` from the repo root and visit /concord/viz/tree_viewer.html)
```

In the viewer:
- Drag-and-drop `<run-id>_tree.json` onto the page to render the tree.
- Optionally drop `<run-id>_rollouts.jsonl` to step through rollouts in the right panel.
- Each node is colored by Q (red→green) and sized by visit count.
- Chosen-path edges are highlighted in blue.
- Terminal nodes get a green border; `gated_fail` nodes get a red border.
- Click any node to inspect its `state.resolved`, the answer that produced it, and its bookkeeping.
- Click a rollout entry to see the selected path, the K LLM samples, the meaning-class assignments, the gate result, and the per-ancestor backup updates from that exact rollout.

### Manual inspection without the viewer

`tree.json` is plain JSON, `rollouts.jsonl` is JSONL. Either is straightforward to grep / `jq` / load into a notebook.

```bash
jq '.nodes[] | select(.gated_fail) | {id, depth, sigma, state}' \
    results/concord/<run-id>_tree.json

jq '. | select(.expansion) | .expansion.classes' \
    results/concord/<run-id>_rollouts.jsonl
```

## Producing the §10 results table

To produce the comparison report the plan calls for:

```bash
# 1) v2 vs baselines on a slice
.venv/bin/python -m concord.experiments.compare \
    --model claude-sonnet-4-6 --n 8 --N 16 --K 4

# 2) §8.4 ablations
for arm in concord/config/ablations/*.yaml; do
    .venv/bin/python -m concord.experiments.ablate \
        --ablation "$arm" --model claude-sonnet-4-6 --n 8 --N 16 --K 4
done

# 3) c_puct / n_min / K sweep
.venv/bin/python -m concord.experiments.sweep \
    --model claude-sonnet-4-6 --n 4 --N 16 \
    --c-puct 2 3 5 --n-min 2 3 --K 4 8

# 4) emit the markdown report
.venv/bin/python -m concord.experiments.aggregate --out RESULTS.md
```

## Milestones — what's done

| Milestone | Module | Plan §7 gate | Status |
|---|---|---|---|
| M0 | `core/{config,types,telemetry,orchestrator}.py`, `llm/mock.py` | end-to-end stub, cost tally, budget ceiling | ✓ 6 tests |
| M1 | `core/confidence/{embed,entailment,cluster,sample_weight,semantic_density,verifier,score}.py` | synonyms cluster, contradictions split, SD ranks dense > sparse, both oracles, sub-quadratic | ✓ 9 tests |
| M2 | `core/structure/{graph,decompose}.py` | cycles condense, critical path matches hand-calc, atomicity terminates | ✓ 13 tests |
| M3 | `core/search/{mcts,widening,transposition}.py` | widening sublinear, n_min blocks single-rollout inflation, backup reaches root, budget ceiling | ✓ 10 tests |
| M4 | `core/structure/coherence.py`, `core/search/{depth,fallback}.py` | spec §7 edge cases: empty class, bimodal, whole-level gated, soft-relax | ✓ 13 tests |
| M5 | `core/llm/anthropic_adapter.py`, `core/expansion.py`, orchestrator | coherent path on easy / flagged fallback on unsat | ✓ 3 integration tests |
| M6 | `experiments/{compare,ablate,baselines,metrics}.py` | comparison table across v2 + B0..B3 | ✓ 11 tests |
| M7 | `experiments/{sweep,aggregate}.py`, `RESULTS.md` | §10 acceptance — needs `ANTHROPIC_API_KEY` for live numbers | tooling done; numbers gated on API access |

## Design decisions matched against the plan

§2 of the plan freezes 8 decisions. All are implemented as defaults with
the alternates available as ablation arms:

| Decision | Default | Ablation arm |
|---|---|---|
| 1. SD confidence | `ablation.confidence: semantic_density` | `cosine_centroid` (in `cosine_centroid.yaml`) |
| 2. Sample-weight oracle | auto by `llm.supports_logprobs` | `weight_oracle: whitebox` / `blackbox` |
| 3. MCTS + progressive widening | `widening: progressive`, `C=2`, `β=0.5` | `widening: fixed_beam` |
| 4. Guarded-max backup | `backup: guarded_max`, `n_min=3` | `backup: average` (`average_backup.yaml`) |
| 5. Incremental hard gate | `gate: incremental_hard` | `terminal_only` (`terminal_gate.yaml`), `off` (`no_gate.yaml`) |
| 6. Soft-relax fallback | `soft_relax: true` (always on; σ always recorded) | — |
| 7. Concrete depth | `depth: adaptive`, `slack=2`, `λ=0.02` | `fixed` (`fixed_depth.yaml`) |
| 8. High c_puct + n_min | `c_puct=3.0`, `n_min=3` | sweep target |
