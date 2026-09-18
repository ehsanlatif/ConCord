# Matrix-chain decomposition study

The matmul analogue of the [`uci_to_fen` chess study](CHESS_STUDY.md). It runs
Concord's **pipeline** solver (Split → Solve → Combine → Verify → Synthesize)
on a long-horizon *running matrix product mod p*, grades it at every
decomposition level against a frozen golden dataset, and records the
**cost / tokens** each run spends.

## The task

Given an ordered sequence of `T` matrices `A_1..A_T` (each `d x d`, entries in
`0..p-1`), compute the running product modulo `p`:

    M_0 = I_d ,   M_t = (M_{t-1} · A_t) mod p ,   report M_T.

This is compounding and non-Markovian — one wrong entry corrupts every later
product — so it is a genuine long-horizon execution task, the same shape as the
chess move-replay. The whole chain is handed to the solver as **one problem**;
Concord must decompose it into sub-chains, solve each under a fixed per-node
call/token budget, and reassemble.

Two axes, mirroring the parent repo's monolithic matmul benchmark:

- **Horizon T** = chain length (`--max-turns`). The long-horizon axis.
- **Complexity n = d** = matrix dimension (`--dims`), ~`d³` scalar
  multiply-adds per step. `d=1` is the scalar running-product baseline. This is
  the `n = 1..8` sweep.

Ground truth is the parent repo's static dataset in `data/matmul/`
(`chains_d{d}.jsonl`: `matrices` + golden cumulative products `golden`, mod 97,
seed 7). Point `--data-dir` at it.

## What gets graded (four levels)

| level | question | how |
|---|---|---|
| **L0 overall** | is the final answer `M_T` the golden `M_T`? | exact matrix match |
| **L1 per-atom** | did each atomic unit compute its slice correctly? | match the matrices in the atom's question back to the known chain to recover its slice `[i..j]`, then compare its answer against the exact golden `M_j` (running-state framing) OR the standalone partial product `A_i..A_j` → `PASS`/`FAIL`/`UNGRADEABLE` |
| **L2 fidelity** | did the splitter hand each atom a correct *contiguous* slice, and do the atoms tile `1..T`? | `contiguity_rate` (fraction of matrix-bearing atoms that are gap-free slices) + `chain_coverage` (fraction of `1..T` covered); the reported `decomposition_fidelity` is their mean |
| **L3 per-level** | at which split depth does it break? | L1 pass-rate aggregated by split-tree depth |

`UNGRADEABLE` = the grader found no chain matrices in the atom (e.g. a pure
"combine the sub-answers" step) or no parseable answer — reported separately,
never counted wrong. Grading uses **exact** modular arithmetic against the
frozen labels, so an atom is scored against the true product for its slice,
not against its own framing.

Grading against a persisted run offline:

```bash
.venv/bin/python -m concord.experiments.matmul_grader \
    --combine-tree results/matmul_study/<ts>/runs/d3_r1/.../combine_tree.json \
    --dim 3 --max-turns 100 --data-dir ../data/matmul
```

## Models (per-role)

`concord/config/matmul_per_role.yaml` — chosen to match the parent repo's
monolithic baseline (executor fixed to Opus 4.8 so the decomposed-vs-monolithic
comparison is apples-to-apples):

| role | model |
|---|---|
| execution | `claude-opus-4-8` |
| splitter / combiner / synthesizer | `claude-sonnet-5` |
| classification / verification (block verifier) | `claude-haiku-4-5` |

The price table in `core/llm/anthropic_adapter.py` was corrected to current
Anthropic rates ($5/$25 Opus, $3/$15 Sonnet, $1/$5 Haiku) so the USD column is
accurate. `temperature` is omitted automatically for models that reject it.

## Run it

```bash
# offline plumbing test (no API key), short chain
.venv/bin/python -m concord.experiments.matmul_study --mock \
    --dims 1 2 --max-turns 8 --data-dir ../data/matmul

# real study — needs ANTHROPIC_API_KEY in the environment
#   (export it from the parent repo's .env: `set -a; . ../.env; set +a`)
.venv/bin/python -m concord.experiments.matmul_study \
    --config concord/config/matmul_per_role.yaml \
    --dims 1 2 3 4 5 6 7 8 --max-turns 100 --data-dir ../data/matmul

# stream to Weights & Biases as well (add --wandb; needs `pip install wandb`
# and `wandb login`, or WANDB_API_KEY set). WANDB_MODE=offline for a dry run.
.venv/bin/python -m concord.experiments.matmul_study \
    --config concord/config/matmul_per_role.yaml \
    --dims 1 2 3 4 5 6 7 8 --max-turns 100 --data-dir ../data/matmul \
    --wandb --wandb-project concord-matmul-study
```

W&B dashboards produced:
  * `run/*`   — one point per (n, repeat) as the sweep proceeds (overall
                correct, atom-pass, fidelity, tokens, $, calls, elapsed).
  * `by_n/*`  — solve rate / mean atom-pass / fidelity / tokens / $ vs n.
  * `levels/*`— L1 pass-rate per split depth, per n.
  * `summary/`— solve-rate-vs-n bar chart + a scannable table.

`--sample-id N` picks which golden chain (0..9); `--repeats K` re-runs each
dimension K times for a solve *rate*; `--max-turns` sets the horizon T.

## Outputs (`results/matmul_study/<ts>/`)

| file | contents |
|---|---|
| `runs.jsonl` | one line per run (dim, params, cost, scores) — appended live |
| `runs/d{d}_r{n}/record.json` | the run's full record |
| `runs/d{d}_r{n}/grade.json` | every atom's PASS/FAIL/UNGRADEABLE + slice + depth |
| `study_summary.json` | all runs + `by_dim` (solve rate, mean atom-pass, fidelity, tokens, $ per n) |
| `study_summary.csv` | flat table for plotting |
| `study_report.md` | human-readable tables |

## The hypothesis this tests

The monolithic baseline (`run_matmul_dataset.py`) resends the whole growing
history each turn — quadratic tokens, collapses at long horizon. Concord
decomposes so each sub-solve sees only its slice → **bounded tokens per call**.
The study measures whether decomposition-under-a-token-budget holds where the
single long context breaks, and at what cost — and, along the `n=d` axis, is
expected to degrade as the per-slice matrix payload grows (the splitter must
faithfully carry bulkier matrices into each sub-atom), mirroring the baseline's
`H_s`-vs-`d` collapse.
