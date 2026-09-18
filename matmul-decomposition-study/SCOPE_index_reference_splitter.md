# Scope: index-reference splitter for the matmul chain (Concord)

## Goal
Stop the splitter from **copying matrix values** into every sub-problem. Instead
have it reference source matrices by their existing labels/ranges (`A_i..A_j`),
and inject the exact values **programmatically at execution time** from the
frozen `GroundTruth`. This removes the transcription errors that drove the fixed
T=500 run to **4% atom-pass / 0% solve at $42.69**, and makes splits tiny/cheap.

## Root cause (confirmed, not hypothetical)
`_SPLITTER_SYSTEM` (`core/pipeline/splitter.py:260`) instructs: *"Inline every
known value... NEVER leave placeholders."* For a 500-matrix chain the splitter
(Sonnet) re-types ~15–18k tokens of matrices per split, recursively. Two
consequences: (1) the token blow-up we already patched (cap+timeout in
`matmul_per_role_hiT.yaml`); (2) **copy errors** — with the bug fixed and the
tree fully built (3057 calls), only **4% of atoms** were arithmetically gradeable-
correct, because the sub-problem matrices themselves were mis-copied.

## Current data flow (with the exact seams)
```
matmul_study.run_once
  gt = GroundTruth.load(...)                       # matmul_task.py:136  (authoritative values)
  prompt = build_problem_prompt(gt)                # matmul_task.py:187  (labels A_1..A_T with values)
  solve_multi(prompt, cfg)                         # core/multi_solve  (TEXT in / text out)
     └─ pipeline: split → expand/execute → combine
          splitter: _SPLITTER_SYSTEM               # splitter.py:260 (global; shared w/ chess)
          AtomicUnit{question, refs}               # splitter.py:78
          expansion._inline_atom_refs(atom,…)      # expansion.py:387 (inlines upstream ANSWERS)
          atom_q = …                               # expansion.py:330
          build_executor_prompt(atom_q)            # expansion.py:97
  grade: grade_atom matches matrices in the atom   # matmul_grader.py:105
         question back to the chain BY VALUE (find_matrices + value→pos idx)
```

## Design (index-reference contract)
Add a structured, machine-parseable source span to each atom rather than parsing
prose — the LLM emits *indices*, code supplies *values*.

1. **Atom schema** — extend `AtomicUnit` (`splitter.py:78`) with an optional
   `source_span: tuple[int,int] | None` (1-based inclusive `A_i..A_j`) and carry
   it through `to_dict`/`from_dict`, `SplitTree`, and the JSON trace. Keep `refs`
   (upstream-answer threading) untouched — it's a *different* namespace.

2. **Splitter prompt override (task-specific)** — `_SPLITTER_SYSTEM` is a GLOBAL
   constant used by chess too, so DO NOT edit it in place. Make it overridable:
   add `pipeline.splitter_system_override: str|null` to config (`core/config.py`)
   and use it at `splitter.py:447` when set. Provide a matmul override that says:
   *"reference the contiguous input range as `source_span:[i,j]`; NEVER copy the
   matrix values; still inline the upstream running-state answer via `refs`."*

3. **Source resolver (task-specific, deterministic, zero-LLM)** — new
   `build_source_resolver(gt)` in `matmul_task.py` returning
   `resolve(atom) -> str` that, given `source_span:[i,j]`, appends the EXACT
   `A_i..A_j` from `gt.matrices` (frozen source) to the atom's executor text.
   This is the crux: values are copied by code, not by the model.

4. **Pipeline hook** — thread an optional `source_resolver: Callable|None`
   from `matmul_study` → `solve_multi` → pipeline → `expansion`. In
   `expansion.py` call it right after `_inline_atom_refs` (≈`expansion.py:330`)
   so `build_executor_prompt` (`expansion.py:97`) sees concrete values.
   **Must default to `None` (no-op)** so chess/other tasks are unaffected.

5. **Grader** — `grade_atom` (`matmul_grader.py:105`) currently recovers the
   slice by matching matrix VALUES; with index-refs the question has no values →
   would be UNGRADEABLE. Change L1/L2 to read `source_span` directly (fallback to
   the old value-matching when `source_span` is absent → backward compatible).

6. **Config cleanup (payoff)** — once splits carry only ranges, revert the
   splitter in `matmul_per_role_hiT.yaml`: `max_output_tokens 32000→2048`,
   `request_timeout_s 900→default`. Splits become fast and ~free (this is where
   the $42 went).

## Files touched
| File | Change | Size |
|---|---|---|
| `core/pipeline/splitter.py` | `AtomicUnit.source_span`; honor prompt override at :447; parse `source_span` from split JSON | med |
| `core/config.py` | add `pipeline.splitter_system_override` | tiny |
| `core/multi_solve.py` + pipeline expansion | thread `source_resolver` (default None); call after `_inline_atom_refs` | med |
| `experiments/matmul_task.py` | `build_source_resolver(gt)` + matmul split-prompt string | med |
| `experiments/matmul_study.py` | pass resolver into `solve_multi` | tiny |
| `experiments/matmul_grader.py` | grade via `source_span`, keep value-matching fallback | med |
| `config/matmul_per_role_hiT.yaml` | revert splitter caps | tiny |

## Risks / open questions
- **Generic-API change** (threading `source_resolver` through `solve_multi`/
  pipeline): must default no-op; add a chess regression run to prove no behavior
  change. Biggest blast-radius item.
- **Splitter contract discipline**: the model must reliably emit `source_span`
  and NOT copy values. Structured field + a couple of few-shot examples in the
  override; validate on a mock split before spending on real runs.
- **This fixes the SPLITTER copy error, not executor arithmetic.** Even with
  perfect values injected, the executor (opus) still multiplies each leaf block.
  Only helps if leaves are small — with `max_split_depth=4, max_atoms=5` leaves
  are ~1–4 matrices, which opus handles (baseline is 100% single-step at d=3), so
  this should be fine, but it's the remaining performance ceiling to watch.
- **`refs` vs `source_span` conflation**: keep them separate namespaces; the
  split prompt must instruct both. Low risk if schema-separated.

## Effort
~1–2 focused days: schema+prompt-override (0.5d), resolver+pipeline hook (0.5–1d),
grader (0.25d), config + chess regression (0.25d).

## Validation plan (cheap)
1. Unit: mock a split emitting `source_span:[101,200]`; assert the resolver
   injects exactly `gt.matrices[100:200]` and grader recovers slice [101,200].
2. Chess regression: 1 `uci_to_fen` run before/after → identical (no-op default).
3. Re-run **d=3 T=100** (now cheap): expect atom-pass ≫ 4% and lower $ (splits
   tiny). Gate: only proceed if atom-pass jumps.
4. Re-run **d=3 T=500** through `run_matmul_budget` + `analyze_budget_frontier`;
   compare `T*(B)` vs Opus (T=347). This is the decisive re-test of the claim.

## Recommendation
Well-contained fix targeting the *confirmed* failure mode (splitter transcription
→ 4% atom-pass). Do steps 1–3 of validation first; if T=100 atom-pass doesn't
jump, stop — the ceiling is executor arithmetic, not copying, and the claim
stays refuted. If it does jump, the T=500 re-run is the fair, decisive test.
