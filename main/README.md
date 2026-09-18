# Concord — Adaptive Tree-search with Unanimous-Confidence Decomposition

Concord is a research prototype for **decomposition-based reasoning under a fixed
LLM-call budget**. It reformulates "decompose a hard problem, solve the pieces,
and reassemble" as a **Monte-Carlo Tree Search** over a problem's dependency
structure, where each node's value comes from a **semantic-density confidence
signal** rather than a single sampled answer, and every assembled solution must
pass an **incremental coherence gate** before it is accepted.

The full algorithm is specified in [`SPEC.md`](SPEC.md) and the
build is tracked against [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
Deep usage docs (every search knob, the trace/viewer format, per-role model
configs) live in [`concord/README.md`](concord/README.md).

The full test suite (176 tests) passes deterministically on a mock LLM — **no
API key needed to explore the code**.

## The idea in one picture

```
Phase 1  —  characterize & bound
   ExtractGraph(P) → G → condense to DAG G' → critical path → depth cap d_max

Phase 2  —  MCTS with value backup
   repeat until budget N exhausted or marginal gain < λ·cost:
     SELECT   via PUCT (high c_puct)
     EXPAND   progressive widening; each meaning-class is a child
              K LLM samples → hybrid cluster (cosine pre-filter + entailment)
     EVALUATE U_s = α·SemanticDensity + (1−α)·verifier   (α = 0.25)
              incremental hard coherence gate → σ recorded, val=0 on violation
     BACKUP   guarded-max to root (n_min visit guard blocks single-rollout inflation)

Selection
   coherent terminal (σ=0)? → return best-Q coherent path
   else                     → return min-σ terminal, FLAGGED best-effort
```

## Key ideas

- **Semantic Density confidence.** A response is scored by how much probability
  mass agrees with it under an entailment kernel — not cosine-to-centroid (which
  ships only as an ablation arm).
- **Hybrid clustering.** Cosine pre-filter then entailment confirm, sub-quadratic
  in the sample count `K` (asserted in tests).
- **Sample-weight oracle, auto-detected per model.** White-box length-normalized
  log-probabilities when available; black-box Laplace-smoothed meaning-class
  frequency otherwise.
- **PUCT + progressive widening.** The old fixed beam width becomes a widening
  coefficient; under-visited nodes stay narrow, promising nodes widen.
- **Guarded-max backup.** An `n_min` visit guard prevents one lucky rollout from
  inflating a node's value.
- **Incremental hard coherence gate** with `σ` always recorded, and a soft-relax
  fallback when no fully coherent terminal is found within budget.
- **Concrete depth governance.** `d_max = min(critical_path(G') + slack, N // K)`
  with a marginal-gain early-stop rule.

Each of the eight frozen design decisions in the spec ships as the default with
its alternate available as an ablation arm (see
[`concord/config/ablations/`](concord/config/ablations)).

## Repo layout

```
Concord/
├── README.md                         ← you are here
├── SPEC.md                           ← the algorithm specification
├── IMPLEMENTATION_PLAN.md            ← milestone-by-milestone build plan
├── requirements.txt
├── concord/                          ← the Python package
│   ├── README.md                     ← deep usage: knobs, traces, viewer, per-role models
│   ├── core/                         ← core: confidence / structure / search / llm / pipeline
│   ├── config/                       ← default + profile + ablation YAMLs
│   ├── experiments/                  ← run / compare / ablate / sweep / aggregate + chess study
│   ├── tests/                        ← 176-test suite across milestones M0–M6
│   └── viz/                          ← drag-and-drop MCTS tree viewer (tree_viewer.html)
├── data/
│   └── eval_set.json              ← 24-question eval set (math/logic/cs/chess/chemistry)
├── verification/
│   └── uci_to_fen_easy_6_trace.json  ← python-chess ground truth for the chess study
├── notebooks/
│   └── chess_study_analysis.ipynb ← parameter-impact analysis (cost / tokens / success)
└── results/
    └── chess_study/               ← showcase runs (see "Results" below)
```

## Install

Python 3.10+ (3.11+ recommended; on 3.10 the optional `tomli` backport covers
`--config`).

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`anthropic` and `datasets` are the only hard runtime dependencies; `openai` and
`tomli` are optional (OpenAI model support and `--config` on Python < 3.11).

## Quickstart

**Run the test suite (no API calls, fully deterministic on the mock LLM):**

```bash
python -m pytest concord/tests/ -v
```

**Solve one question end-to-end on the mock LLM (sanity check):**

```bash
python -m concord.experiments.run --mock --index 0 --N 8 --K 3
```

**Solve a real question (needs an API key):**

```bash
export ANTHROPIC_API_KEY=sk-...
python -m concord.experiments.run \
    --dataset data/eval_set.json --index 0 \
    --model claude-sonnet-4-6 --N 16 --K 4
```

Each solve writes three artifacts under `results/concord/`: a high-level
telemetry `*.jsonl`, a full MCTS `*_tree.json` snapshot, and a per-rollout
`*_rollouts.jsonl`. Drag the tree (and optionally the rollouts) onto
[`concord/viz/tree_viewer.html`](concord/viz/tree_viewer.html) to inspect the
search. The widening/depth knob reference and the full trace schema are
documented in [`concord/README.md`](concord/README.md).

**Different models per role.** Concord has four LLM-using roles (execution,
decomposition, classification, verification); each can be pinned to a different
model from a YAML config or CLI flag. The recommended profile is
[`concord/config/per_role.yaml`](concord/config/per_role.yaml) (opus / sonnet /
haiku / haiku). See `concord/README.md` for the full table.

## Results — the `uci_to_fen` chess study

The showcased experiment runs the pipeline solver (Split → Solve → Combine →
Verify) on the chess problem `uci_to_fen_easy_6` — replaying a long UCI move
sequence to a final FEN — and **grades it at every decomposition level** against
a python-chess ground truth, recording the **cost/tokens** each parameter
combination spends. Methodology: [`concord/experiments/CHESS_STUDY.md`](concord/experiments/CHESS_STUDY.md).

Four runs are included under `results/chess_study/`:

| run | what it varies | headline |
|---|---|---|
| `20260623-145814` | `pipeline.max_split_depth` 1→6 (20-ply game) | full report + plots + regression analysis |
| `20260624-113438` | `max_plies` 20→650 (problem size) | how accuracy/cost scale with difficulty |
| `20260622-164441` | named multi-parameter presets (5-ply) | consumed by the analysis notebook |
| `20260622-165017` | `mcts.N` and `K_executor` sweeps (20-ply) | consumed by the analysis notebook |

### What the depth sweep shows (`20260623-145814`)

Splitting deeper is not monotonically better — there is a cost/accuracy sweet
spot, and gradeability degrades as splits get finer:

- Exact-FEN success appeared in 3 of 6 runs; the **cheapest correct run was
  depth 3** at ~136k tokens / $0.96.
- **Depth 5** had the strongest atom pass-rate (86%) but burned **4.3× the
  tokens** of the depth-3 solve.
- **Depth 4 is a clear regression** — it spent ~278k tokens / $1.60 and still
  returned a wrong FEN.
- The bottleneck at deep splits is **gradeability**: depth 5 left 94% of atoms
  ungradeable, depth 6 left 87% and was flagged `budget_truncated`.

Rendered tables, the cost/accuracy frontier, and per-role cost mix are in
[`results/chess_study/20260623-145814/study_report.md`](results/chess_study/20260623-145814/study_report.md),
[`study_plots_and_insights.md`](results/chess_study/20260623-145814/study_plots_and_insights.md),
[`study_regression_analysis.md`](results/chess_study/20260623-145814/study_regression_analysis.md),
and the SVGs under [`plots/`](results/chess_study/20260623-145814/plots).

### How problem size scales (`20260624-113438`)

Holding the solver fixed and growing the game from 20 to 650 plies shows the
approach solving the truncated 20-ply instance but degrading sharply on longer
games — quantifying where the current pipeline breaks. See
[`results/chess_study/20260624-113438/study_report.md`](results/chess_study/20260624-113438/study_report.md).

### Analysis notebook

[`notebooks/chess_study_analysis.ipynb`](notebooks/chess_study_analysis.ipynb)
loads the `20260622-164441` (collective presets) and `20260622-165017`
(individual sweeps) runs and works through the cost and solvability models —
parameters → calls → tokens → USD, and expected success per decomposition
level. It auto-detects the repo root, so it runs from a checkout with no path
edits.

### Reproduce

```bash
# offline plumbing check (no API key)
python -m concord.experiments.chess_study --mock --max-plies 8 \
    --sweep concord/config/chess_study.sweep.yaml

# real study (needs ANTHROPIC_API_KEY; start truncated to bound spend)
export ANTHROPIC_API_KEY=sk-...
python -m concord.experiments.chess_study \
    --config concord/config/wide_and_deep.yaml \
    --sweep  concord/config/chess_study.sweep.yaml \
    --max-plies 40
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0–M4 | skeleton, confidence stack, structure/graph, MCTS, coherence + fallback | ✓ |
| M5 | Anthropic adapter + end-to-end integration | ✓ |
| M6 | comparison harness across Concord + baselines B0–B3 | ✓ |
| M7 | sweep/aggregate tooling + results table | tooling done; live numbers gated on API access |

## License

See [`LICENSE`](LICENSE).
