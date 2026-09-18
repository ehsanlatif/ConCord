# 03 — Experimental setup & assumptions

For the paper's **Experimental Setup** section.

## Systems

- **Concord** — the full system (all three contributions), config
  `code/configs/matmul_per_role_trimexec.yaml`.
- **Monolithic baselines** — Opus-4.8 and GPT-5.5, one call per turn on the whole
  chain.
- **SOTA decomposition baselines** — ARIES, Select-Then-Decompose, Graph-of-Thoughts
  (see `06_sota_comparison.md`).

## Controlled conditions (uniform across all systems)

1. **Identical problem prompt** — Concord's `build_problem_prompt`, imported by
   path so it cannot drift between systems.
2. **Identical per-role models** — executor `claude-opus-4-8`, planner
   `claude-sonnet-5`, judge `claude-haiku-4-5`. The executor is held fixed, so the
   comparison isolates the decomposition strategy rather than the base model.
3. **Intermediate scoring by an LLM verifier**, never a Python oracle — every
   decomposition system is judged on the same footing.
4. **Grading = exact whole-matrix match** of the final `M_T`. Modulus p = 97.
5. **Sampling** at the Anthropic API default (the Messages API no longer exposes a
   temperature parameter), uniform across systems.

## Cost metering

- **USD is the primary cost axis** — cache-aware and provider-fair. Cost is metered
  at the transport layer with one shared price table.
- Prices (USD per 1M tokens, input/output): opus-4-8 = 5 / 25; sonnet-5 = 3 / 15;
  haiku-4-5 = 1 / 5; gpt-5.5 = 1.25 / 10 (a documented estimate). Cache read 0.1×
  (Anthropic) / 0.5× (OpenAI) of input price; cache write 1.25× input.
- Monolithic baseline cost at any horizon T is read from per-turn cumulative traces
  (`data/baselines_monolithic/*.json`): cost@T = mean over samples of
  `per_sample[i][T-1].cum_usd`; solved@T = `task_accuracy_by_len[T-1] ≥ 0.5`.

## Dataset & horizons

- Static, golden-labeled matmul chains (`dataset/`), modulus 97, d = 1..8, T up to
  2000, 10 chains/dimension. Truncate to any T by slicing the first T inputs.
- Reported grid: **d ∈ {1, 2, 3}**, **T ∈ {12, 25, 50, 100, 200}**.

## Sample sizes (state these in every table caption)

- **Concord: n = 3 reps/cell.** Solve rates are therefore reported at 1/3-resolution.
- **Monolithic baselines:** Opus n = 6–8, GPT n = 5 (read from cumulative traces).
- **SOTA decomposition baselines:** n = 10 chains/cell (prior measured runs; not
  re-run). When Concord is tabled against them, both n's are stated in the caption.

## Configuration note

- Concord's per-node compute budget is `max_node_calls = 600` with one executor
  pass per leaf. Always report the node budget alongside any decomposition result,
  since decomposition cost and reach depend on it.
- Decomposition quality is exact on every reported run (fidelity 1.0, coverage 1.0,
  0 ungradeable atoms).

## Scope of the reported results

- Dimensions d = 1..3 (the dataset supports up to d = 8).
- Horizons up to T = 200 for Concord; monolithic baselines are read to T = 200 from
  their cumulative traces.
- The scalability claim is argued **theoretically** (`05_scalability_theory.md`);
  the experimental grid provides supporting evidence.
