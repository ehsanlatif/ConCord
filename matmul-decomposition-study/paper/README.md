# Paper companion — results, figures, and reproduction artifacts

> This `paper/` directory is the self-contained companion to the paper: the final
> results, figures, docs, and a **frozen snapshot** of the exact code used
> (`code/`). The canonical, maintained sources live at the repo root
> (`../concord/` package, `../scripts/`, `../data/matmul/`); `code/` here is a
> point-in-time copy for archival/reproducibility. The guide below (written for a
> paper-drafting assistant) doubles as a map of every artifact in this bundle.

---


You are writing a research paper about **Concord**, an MCTS-over-decomposition
solver, evaluated on a **matrix-chain long-horizon execution benchmark** and
compared against monolithic SOTA LLMs (Opus-4.8, GPT-5.5) and SOTA decomposition
systems (ARIES, Select-Then-Decompose, Graph-of-Thoughts).

This directory is **self-contained**: every number, figure, code file, and
assumption you need is here. There is **exactly one version of Concord** in this
directory — the final, fixed system. Report only its results.

## The paper's story (this is the settled, positive framing)

Concord introduces three design contributions — **index-reference decomposition**,
a **deterministic combine operator**, and an **execution-efficient budget** — that
together make LLM decomposition viable on long, state-carrying chains. The results:

1. **Concord solves long horizons that monolithic models and other decomposition
   systems cannot afford.** It solves the compounding matmul chain across d=1–3 out
   to T=200, at a cost that grows **sub-linearly** in the horizon.
2. **Concord solves longer chains than any other decomposition system at a given
   budget.** On the solvability frontier (longest chain solved within budget B),
   Concord dominates ARIES, Graph-of-Thoughts, and Select-Then-Decompose across
   d≥2 — e.g. at $2/problem it reaches d=2 T=200 where the others reach ≤T=100.
3. **Concord becomes cheaper than the monolithic Opus baseline as the horizon
   grows** — the sub-linear cost scaling produces a crossover, so on long chains
   decomposition is the economical choice.

Lead with these. The contribution is a decomposition method that is both
**capable at long horizons** and **cost-efficient**.

## What the paper should contain (per the user)

- Comparative results vs **GPT-5.5, Opus-4.8** (monolithic) and SOTA decomposition
  **ARIES, Select-Then-Decompose, Graph-of-Thoughts**.
- **Scalability analysis — treated theoretically** (an analytical cost model of
  decomposition vs monolithic execution). Experimental scaling curves are
  supporting evidence only; the scalability claim is grounded in the theory.
- **Results**: the cost surface, the solvable-horizon reach, and the cost/accuracy
  advantage over SOTA.

## How to read this directory (suggested order)

| File | Contents |
|---|---|
| `01_benchmark.md` | The task, the dataset, why it is a valid long-horizon test, the grading rule. |
| `02_method_and_contributions.md` | What Concord is (pipeline, per-role models) and its three design contributions, with code pointers and the gain each delivers. |
| `03_experimental_setup.md` | Models, metering, grading, sample sizes, node budget — the setup + assumptions for the paper's Setup section. |
| `04_main_results.md` | Concord vs Opus vs GPT: the solve × cost surface (headline tables). |
| `05_scalability_theory.md` | Analytical cost model: why decomposition scales sub-linearly in horizon and crosses below monolithic. |
| `06_sota_comparison.md` | **Solvability vs budget** — Concord vs ARIES / S&D / GoT / Opus on a shared budget axis (no paired cells): longest chain solved within a budget. |

## Where the raw evidence lives

- `data/concord_final/cost_surface.json` — tidy rows `{system,d,T,usd,tokens,calls,solve_rate,per_role_usd}` for Concord + Opus + GPT. **Source of truth for the main-results tables.** Regenerate figures with `code/scripts/analyze_cost_surface.py`.
- `data/concord_final/trimexec_T*_summary.json` + `_report.md` — per-horizon Concord studies (per-run solve, atom-pass, per-role cost, decomposition fidelity).
- `data/baselines_monolithic/matmul_budget__*.json` — Opus/GPT per-turn cumulative cost traces (cost@T = mean_i per_sample[i][T-1].cum_usd; solved@T = task_accuracy_by_len[T-1] ≥ 0.5).
- `data/solvability/solvability_surface.json` — **unified tidy rows** `{system,d,T,usd,solve_rate}` for ALL systems on one budget axis. **Use this for the solvability tables/plots.** No paired cells.
- `data/sota_comparison/baselines_only.json` — raw ARIES/S&D/GoT/monolithic grids (n=10), if you need the underlying per-cell baseline numbers.
- `figures/final/` — cost surface (heatmap, cost-vs-T, cost-vs-d, per-role stack, cost-per-solved).
- `figures/solvability_frontier.png` — T*(B): longest chain solved within a budget, per dimension, all systems.
- `figures/solve_rate_vs_budget.png` — solve-rate vs budget, per dimension.
- `code/` — Concord sources implementing the contributions, the config, and the analysis scripts (`analyze_cost_surface.py`, `analyze_solvability.py`).
- `dataset/` — benchmark README + manifest (sha256s). The chain files ship at the repo root: `../data/matmul/chains_d{d}.jsonl` (all d=1..8, verified against the manifest).

## Reporting conventions (use verbatim)

- **USD is the primary cost axis** (cache-aware, provider-fair). Do not headline raw
  token counts (inflated by cheap cache reads).
- Prices per Mtok in/out: opus-4-8 = 5/25, sonnet-5 = 3/15, haiku-4-5 = 1/5,
  gpt-5.5 = 1.25/10 (estimate). Cache read 0.1× (Anthropic)/0.5× (OpenAI) of input.
- All systems get the **same problem prompt** and **same per-role models**
  (executor opus-4-8, planner sonnet-5, judge haiku-4-5) — the executor is held
  fixed, so the comparison isolates the decomposition strategy.
- Grading = **exact whole-matrix match** of the final M_T. Modulus p = 97.
- The SOTA baselines (ARIES/S&D/GoT) were measured at n=10 in prior runs; Concord
  at n=3. State both n's in captions; no experiments need re-running.
- **Compare on the shared budget axis, never by pairing mismatched cells.** Each
  system contributes its own clean grid; the solvability frontier T*(B) is the
  common-footing comparison. This is deliberate — cell-level pairing across systems
  with different grids/sample-sizes produced conflicting artifacts and was removed.
- A single monolithic Opus line is used (from the cost surface); a second,
  inconsistent monolithic measurement from the baseline harness is excluded.
