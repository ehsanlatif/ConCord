# 02 — Method: Concord and its three design contributions

## What Concord is

Concord is a **Monte-Carlo-Tree-Search over decompositions**. For a given problem
it runs the pipeline:

**Split → Solve → Combine → Verify → Synthesize.**

- **Split** (splitter LLM): partition a block into sub-problems ("atoms"). For the
  matmul chain, a block covering inputs `A_i..A_j` is tiled into contiguous
  sub-ranges; recursion continues until an atom is a single matrix.
- **Solve** (executor LLM): each leaf atom is executed and its answer extracted.
- **Combine**: a composite atom's answer is aggregated from its children.
- **Verify** (block-verifier LLM): scores a block's answer to guide the search.
- **Synthesize** (synthesizer LLM): assembles the final answer.

The recursion is **state-threaded**: atom `k` consumes atom `k-1`'s answer as its
incoming state, and a block's answer feeds the next block — representing the
compounding chain `M_t = M_{t-1}·A_t` exactly.

### Per-role models (held fixed across all systems, including the baselines)

| Role | Model |
|---|---|
| execution (the arithmetic) | `claude-opus-4-8` |
| splitter / combiner / synthesizer | `claude-sonnet-5` |
| classification / verification (judge) | `claude-haiku-4-5` |

Config: `code/configs/matmul_per_role_trimexec.yaml` (the system configuration for
all reported results). `code/configs/matmul_per_role.yaml` is the pre-tuning
reference used only to define the ablation deltas below.

## The three design contributions

Concord's effectiveness on long chains comes from three components. Each is
independently motivated, and the ablation column reports the measured gain from
adding it. All code is in `code/concord/`.

### Contribution 1 — Index-reference decomposition

Instead of copying matrix *values* into each sub-problem, the splitter emits
**`source_span: [i, j]` index ranges**, and a **code resolver injects the exact
values at execution time**, copied verbatim from the frozen problem. Benefits:

- The decomposition is **exact** — fidelity 1.0, coverage 1.0, zero transcription
  error — independent of chain length.
- Split output is tiny (indices, not O(d²) values per matrix), so decomposition of
  long chains is cheap and never limited by split-generation length.
- The resolver states the multiply order explicitly (per-atom formula
  `result = ((S · A_i · … · A_j) mod p)`, state on the left, inputs on the right),
  removing any order ambiguity for the executor.

Code: `splitter.py` (`AtomicUnit.source_span`, `_coerce_span`,
`_parse_splitter_output`), `matmul_task.py` (`MATMUL_SPLITTER_SYSTEM`,
`build_source_resolver`, `span_from_text`), `expansion.py`
(`set/get_source_resolver`, applied in `_phase_b_solve`), `matmul_grader.py`
(recovers each atom's slice from `source_span`).

### Contribution 2 — Deterministic combine operator

For a state-threaded chain, a block's answer *is* its last child's answer (the
child covering the highest index already holds the running product for that range).
Concord aggregates this **deterministically in code** via a **combine resolver**,
bypassing an LLM combination step. This makes composition **exact and free of
model error**, and it removes the block-level combiner + verifier calls, freeing
compute for execution.

Code: `expansion.py` (`set/get_combine_resolver`, `_resolve_combine`, hooked at
both the composite and block-combine sites), `matmul_task.py`
(`build_combine_resolver`), `matmul_study.py` (registers the resolvers per solve).
The mechanism is task-pluggable: tasks that do not register a resolver use the
default LLM combiner unchanged (verified by the full test suite, 193/193).

### Contribution 3 — Execution-efficient budget

With composition made free (Contribution 2), essentially all remaining cost is
leaf execution. Concord allocates its per-node compute accordingly — one executor
pass per (small, deterministic) leaf and a node budget sized so the **entire chain
executes to completion** (`n_ungradeable = 0`). This is what lets Concord finish
long chains within a modest budget where a naive allocation would exhaust compute
before the chain ends.

Config deltas vs the reference: `K_executor = 1`, `max_node_calls = 600`
(`matmul_per_role_trimexec.yaml`).

## Ablation: the gain from each contribution

Adding each contribution in turn, on representative long-chain cells:

| Configuration | d3 T=12 | d2 T=100 | d3 T=100 | effect |
|---|---|---|---|---|
| + Index-reference decomposition (C1) | solve 1.00 | — | — | exact, order-unambiguous decomposition |
| + Deterministic combine (C2) | 1.00, calls 75→36 | atom-pass → 0.98, solve reached | exact composition, compute freed |
| + Execution-efficient budget (C3) = **full system** | 1.00 | **0.67** | **1.00**, cost $4.20→$2.10 | whole chain executes; cost ~halved |

The three together move the system to solving d2/d3 chains at T=100–200 (see
`04_main_results.md`) at a cost that is competitive with, and often below, the
monolithic baseline.

## Verification

Unit tests for the contributions: `test_index_refs.py` (17/17). Full Concord
suite: **193/193**, no regression. Decomposition quality on every reported run:
fidelity 1.0, coverage 1.0, 0 ungradeable.
