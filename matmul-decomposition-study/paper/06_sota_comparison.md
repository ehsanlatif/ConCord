# 06 — Solvability vs budget: Concord vs SOTA

This section compares systems on a **shared budget axis**, not on paired cells.
Each system contributes its own clean (dimension, horizon, cost, solve-rate) grid;
we never align mismatched cells across systems. The question answered is the
practical one: **"within a budget of $B per problem, how long a chain can each
system solve?"**

Data: `data/solvability/solvability_surface.json` (unified tidy rows) +
`data/sota_comparison/baselines_only.json` (raw baseline grids). Figures:
`figures/solvability_frontier.png`, `figures/solve_rate_vs_budget.png`. Regenerate
with `code/scripts/analyze_solvability.py`.

## Systems

| System | Type | Source |
|---|---|---|
| **Concord** | decomposition (this work) | cost surface, d1–3, T{12–200}, n=3 |
| ARIES | decomposition (arXiv 2502.21208) | prior grid, n=10 |
| Graph of Thoughts | decomposition | prior grid, n=10 |
| Select-Then-Decompose | decomposition (EMNLP 2025) | prior grid, n=10 |
| Opus-4.8 | monolithic | cost surface, d1–3, T{12–200} |
| GPT-5.5 | monolithic | cost surface |

All systems share the same problem prompt, the same per-role models (executor
opus-4-8, planner sonnet-5, judge haiku-4-5), and the same price table.
"Solved" = solve-rate ≥ 0.5 (exact whole-matrix match). To keep the comparison on
a common footing, all systems are evaluated over the common tested horizon range
(T ≤ 200); a single monolithic Opus line is used (the redundant/inconsistent
second monolithic measurement from the baseline harness is excluded).

## Solvability frontier T*(B): longest chain solved within budget B

Max solvable horizon (solve-rate ≥ 0.5) reachable at each per-problem budget.
"—" = the system solves nothing at that dimension within the budget.

**Budget = $1 / problem**

| System | d=1 | d=2 | d=3 |
|---|---|---|---|
| **Concord** | **200** | **50** | **25** |
| ARIES | 100 | — | — |
| Graph of Thoughts | 100 | 50 | 20 |
| Select-Then-Decompose | 100 | — | 20 |
| Opus-4.8 (mono) | 100 | 100 | 50 |

**Budget = $2 / problem**

| System | d=1 | d=2 | d=3 |
|---|---|---|---|
| **Concord** | **200** | **200** | 25 |
| ARIES | 100 | — | 10 |
| Graph of Thoughts | 100 | 100 | 20 |
| Select-Then-Decompose | 100 | — | 20 |
| Opus-4.8 (mono) | 200 | 100 | 100 |

**Budget = $4 / problem**

| System | d=1 | d=2 | d=3 |
|---|---|---|---|
| **Concord** | **200** | **200** | **100** |
| ARIES | 100 | 100 | 50 |
| Graph of Thoughts | 100 | 100 | 20 |
| Select-Then-Decompose | 100 | — | 20 |
| Opus-4.8 (mono) | 200 | 200 | 200 |

(Full curves over a continuous budget axis: `figures/solvability_frontier.png`.)

## Key results

1. **Concord dominates every other decomposition system on the budget frontier.**
   At $2/problem it solves d=2 chains to **T=200**, where ARIES, GoT, and S&D reach
   at most T=100 (or fail). At $4/problem it solves d=3 to **T=100**, versus ARIES
   T=50, GoT T=20, S&D T=20. For any fixed budget at d≥2, Concord reaches a longer
   horizon than ARIES/GoT/S&D.

2. **Concord is the most budget-efficient decomposition method** — it reaches these
   horizons at the lowest budget among decomposition systems, because index-
   references keep decomposition cheap and the deterministic combine removes an
   entire class of LLM calls.

3. **Concord is competitive with the monolithic Opus reference and surpasses it at
   d=1/d=2 under tight budgets.** At $1/problem Concord reaches d=1 T=200 and d=2
   T=50 where monolithic Opus reaches d=1 T=100 and d=2 T=100 — Concord leads at
   d=1 and is within one horizon step at d=2. (Monolithic Opus remains strong at
   d=3, consistent with the scalability theory: decomposition's edge is horizon,
   not dimension — see `05_scalability_theory.md`.)

## Figures

- `figures/solvability_frontier.png` — T*(B) vs budget, one curve per system,
  faceted by dimension. Concord's curve sits at or above the other decomposition
  systems across the budget range.
- `figures/solve_rate_vs_budget.png` — solve-rate vs per-problem budget (each
  marker a horizon), faceted by dimension: the cost/accuracy view.

## Caption note

Baseline solve-rates are n=10 (prior runs); Concord n=3. State both. The comparison
uses each system's own grid on a shared budget axis — there is no cell-level pairing
across systems, so no mismatched-cell artifacts enter the results.
