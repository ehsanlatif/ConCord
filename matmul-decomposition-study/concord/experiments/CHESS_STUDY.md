# Chess `uci_to_fen` cost / token study

A standalone experiment harness that runs the **pipeline** solver
(Split → Solve → Combine → Verify) on the chess problem `uci_to_fen_easy_6`,
**verifies it at every decomposition level**, and records the **cost / tokens**
each parameter combination spends so you can see how accuracy trades off
against spend.

It is a *new* runner — it does not change `run_all.py`. Two files:

| file | role |
|---|---|
| `chess_grader.py` | grades a run against the python-chess ground-truth trace |
| `chess_study.py`  | runs a parameter sweep, grades each run, aggregates cost vs score |

The ground truth is `verification/uci_to_fen_easy_6_trace.json` — one verified
FEN per half-move (650 plies), produced with python-chess and confirmed to
reproduce the dataset's stated final FEN exactly.

---

## What gets graded (four levels)

| level | question | how |
|---|---|---|
| **L0 overall** | is the final answer FEN the gold FEN? | full-FEN exact match (the dataset criterion) + board-only match |
| **L1 per-subproblem** | did each atomic unit compute its own output FEN correctly? | parse the unit's input position + UCI moves, replay with python-chess, compare to the unit's answer → `PASS` / `FAIL` / `UNGRADEABLE` |
| **L2 decomposition fidelity** | did the splitter hand each unit the *correct* slice of the real game? | check the unit's moves equal the real game's contiguous moves at that ply |
| **L3 per-level** | at which depth does it break? | L1 pass-rate aggregated by split-tree depth |

`UNGRADEABLE` means the grader could not extract a self-contained
"(position, moves)" pair from the sub-question (e.g. a pure "combine the
sub-answers" step) — it is reported separately and never counted as wrong.

---

## Parameter study: solvability ratio & cost vs each parameter

The independent variables are the solver knobs in `_key_params`
(`solver`, `mcts.N`, `pipeline.K_executor`, `pipeline.K_combiner`,
`pipeline.max_split_depth`, `pipeline.max_atoms_per_split`,
`pipeline.max_node_calls`, `pipeline.verifier_accept`,
`pipeline.combiner_retries`, `confidence.alpha`, `llm.temperature`). `--ofat`
sweeps each one across **5 values** (see `PARAM_LEVELS`), holding the others at
the `wide_and_deep` baseline, and reports how the **solvability ratio** and
**cost** respond.

```bash
# all parameters, 5 values each, 3 repeats per value (needs ANTHROPIC_API_KEY):
python -m concord.experiments.chess_study --ofat --repeats 3 --max-plies 40

# a subset of parameters:
python -m concord.experiments.chess_study --ofat \
    --params mcts.N,pipeline.K_executor,pipeline.max_split_depth --repeats 3

# offline plumbing test:
python -m concord.experiments.chess_study --ofat --mock --max-plies 8 \
    --params mcts.N,pipeline.K_executor --repeats 2
```

`--repeats N` is what makes the **solvability ratio** meaningful: it is
`solved / N` at each parameter value (executor temperature is > 0). The
equivalent explicit, editable sweep is `concord/config/param_study.sweep.yaml`
(run it with `--sweep`). Output adds a `parameter_effects` block to
`study_summary.json` and a per-parameter table to `study_report.md`:

```
SOLVABILITY RATIO & COST vs PARAMETER (others held at baseline)
  pipeline.K_executor:
       value | solv.ratio | subprob pass | mean tokens |   mean $
           1 |       0.33 |        0.61  |       18420 |   0.07
           3 |       0.67 |        0.78  |       31200 |   0.12
           8 |       1.00 |        0.91  |       58900 |   0.24
```

### Run it

### Smoke test (no API key)

Uses the deterministic mock LLM and a truncated move list — exercises all the
plumbing (solve → provenance → grade → cost/token capture → summary) in
seconds. Answers will be wrong (mock isn't a chess engine) but every output
file is produced:

```bash
.venv/bin/python -m concord.experiments.chess_study --mock \
    --max-plies 8 --sweep concord/config/chess_study.sweep.yaml
```

### Real study (needs `ANTHROPIC_API_KEY`)

Start truncated to keep spend bounded, then scale up:

```bash
export ANTHROPIC_API_KEY=sk-...
.venv/bin/python -m concord.experiments.chess_study \
    --config concord/config/wide_and_deep.yaml \
    --sweep  concord/config/chess_study.sweep.yaml \
    --max-plies 40
```

- `--max-plies N` truncates the game to the first N half-moves; the gold FEN is
  taken from the trace at ply N. This is the main cost knob — raise it (or drop
  the flag for the full 650-ply game) to study how tokens grow with problem
  size.
- `--repeats K` re-runs each combination K times (executor temperature > 0, so
  scores vary).
- Without `--sweep`, a single `baseline` run uses `wide_and_deep.yaml` as-is.

### Grade an existing run offline

```bash
.venv/bin/python -m concord.experiments.chess_grader \
    --combine-tree results/.../combine_tree.json \
    --max-plies 40
```

---

## Outputs (under `results/chess_study/<timestamp>/`)

| file | contents |
|---|---|
| `runs.jsonl` | one line per run (params, models, cost, scores) — appended live |
| `runs/<label>_r<n>/record.json` | the run's full record |
| `runs/<label>_r<n>/grade.json`  | every atom's PASS/FAIL + move range + expected vs answered FEN |
| `study_summary.json` | all runs + `tokens_to_success_level` + `token_score_frontier` |
| `study_summary.csv`  | flat table (params × cost × scores) for plotting |
| `study_report.md`    | human-readable tables |

Per-run console report shows, for each combination:

```
RUN baseline_r1   overrides=(base)
  models: exec=anthropic/claude-sonnet-4-6  split=...  comb=...  verify=...
  params: N=64 K_exec=3 K_comb=3 split_depth=6 node_calls=100 accept=0.75
  COST:   calls=NN  tokens=NNNNN (in .. / out ..)  $0.NNNN  NN.Ns
  SCORE:  overall_correct=True  subproblems 7/9 pass (rate=0.778), 2 ungradeable  decomp_fidelity=1.000
  LEVELS: d0:1/1=1.00  d1:3/4=0.75  d2:3/4=0.75
```

The **`tokens_to_success_level`** table answers the headline question — for
each success bar (overall correct; subproblem pass-rate ≥ 1.0 / 0.9 / 0.75 /
0.5), it lists how many parameter combinations reached it and the **min /
median / max tokens** (and max `$`) required, plus the cheapest run that got
there. The **`token_score_frontier`** lists the best subproblem pass-rate
achievable at or below each token budget.

---

## Parameters the sweep varies

All knobs come from `wide_and_deep.yaml` and are overridden via dotted keys
(see `core/config.py`): `pipeline.max_split_depth`, `pipeline.K_executor`,
`pipeline.K_combiner`, `pipeline.max_node_calls`, `pipeline.max_atoms_per_split`,
`pipeline.verifier_accept`, `mcts.N`, and per-role models via
`models.<role>.model` (e.g. `models.execution.model: claude-opus-4-8`). Edit
`concord/config/chess_study.sweep.yaml` to add combinations.
