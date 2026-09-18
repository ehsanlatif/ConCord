# Concord — Implementation Plan

**Audience:** the implementing agent.
**Goal:** a runnable, configurable, instrumented research prototype of Concord that
can be compared against baselines and ablated, on at least one decomposable+verifiable
task family. Algorithm semantics are fixed in `SPEC.md`; **this document is
about how to build it**, not why. Where this plan and the spec disagree, the spec wins —
flag the conflict, don't silently resolve it.

Definition of "done" is in §10. Read §2 (decisions already made — do not relitigate)
before writing any code.

---

## 1. Scope

Build the system in the spec: MCTS search over a decomposition tree, Semantic-Density
confidence behind a sample-weight oracle, hybrid entailment clustering, guarded-max value
backup, an incremental hard coherence gate with recorded `σ`, a soft-relax fallback, and
concrete depth governance. Plus a baseline/ablation harness and metrics.

Out of scope for v1 of the prototype: learned value/prior networks (AlphaZero-style),
distributed rollouts, UI. Leave clean seams for them (§5.8 note).

## 2. Decisions already made — do not relitigate

These were settled deliberately. Implement them; expose them as config flags so they can
be *ablated*, but the default configuration must match this list.

1. **Confidence = Semantic Density**, not cosine-to-centroid. SD also serves as the
   intrinsic confidence `c_i`; there is no separate self-confidence call.
2. **Sample-weight oracle is capability-detected per model.** White-box → length-normalized
   sequence probability. Black-box → Laplace-smoothed meaning-class frequency. Same SD
   formula consumes either.
3. **Search = MCTS** with PUCT selection, **progressive widening** (the old beam width `M`
   is now the widening coefficient `C`), and **value backup to the root**.
4. **Backup operator = guarded max** (Bellman/max, but a child is max-eligible only after
   `n_min` visits). Not average.
5. **Coherence = incremental hard gate** on `G'` constraints, evaluated at the depth each
   constraint first becomes checkable. Always record `σ`, even when gating hard.
6. **No-solution fallback = soft relax**: rank reached terminals by `σ`, return the best,
   flagged unverified.
7. **Depth** = `min(critical_path(G') + slack, budget_cap)`, governed online by a
   marginal-gain stopping rule. Not a guessed constant.
8. **High `c_puct`** and the `n_min` guard travel with decisions 3–4 (they compensate for
   sparse reward + max backup); they are tuning targets, not optional.

## 3. Stack and dependencies

- **Python 3.11+.**
- **LLM access:** thin provider-agnostic `LLMClient` protocol (§5.1). Concrete adapters for
  an OpenAI-compatible endpoint (exposes token logprobs → white-box) and Anthropic
  (no logprobs → black-box). Capability flag `supports_logprobs` drives the oracle.
- **Embeddings (cosine pre-filter):** `sentence-transformers` (start: `all-MiniLM-L6-v2`;
  swap to a stronger model via config).
- **Entailment kernel:** a cross-encoder NLI model (start: `microsoft/deberta-v3-base` MNLI
  head via `transformers`). Bidirectional entailment defines meaning-class equivalence.
- **Graph:** `networkx` (SCC condensation, DAG longest path).
- **Config:** `pydantic` v2 models loaded from YAML.
- **Experiment tracking:** structured JSONL always; optional `wandb`/`mlflow` behind a flag.
- **Tests:** `pytest`.
- Pin everything in `requirements.txt`; commit a `constraints.txt` lockfile.

## 4. Repository layout

```
core/
  config/
    default.yaml                # defaults from §9
    ablations/                  # one YAML per ablation arm
  core/
    __init__.py
    types.py                    # dataclasses: Node, MeaningClass, SolutionState, Constraint
    config.py                   # pydantic schema
    llm/
      client.py                 # LLMClient protocol
      openai_adapter.py         # white-box (logprobs)
      anthropic_adapter.py      # black-box
      mock.py                   # deterministic, for tests + cheap dev runs
    confidence/
      embed.py                  # embeddings + cosine pre-filter
      entailment.py             # NLI kernel k(r_i, r_j)
      cluster.py                # hybrid clustering -> meaning classes
      sample_weight.py          # the oracle (white/black-box)
      semantic_density.py       # SD(r)
      verifier.py               # v(r, q)
      score.py                  # U_s aggregation
    structure/
      graph.py                  # extract G, condense -> G', critical path
      decompose.py              # explicit/implicit decomposer, atomicity predicate
      coherence.py              # incremental gate, sigma
    search/
      mcts.py                   # Node ops, select/expand/evaluate/backup loop
      widening.py               # progressive-widening predicate
      transposition.py          # contextual transposition table
      depth.py                  # budget cap + marginal-gain stopping
      fallback.py               # soft relax
    orchestrator.py             # ties phases 1 + 2 together; public entry point
    telemetry.py                # JSONL logging, cost accounting, run metadata
  experiments/
    datasets/                   # loaders
    baselines/                  # B0..B3 (§8.2)
    metrics.py                  # §8.3
    run.py                      # single run
    sweep.py                    # config-matrix runner
    aggregate.py                # results table / plots
  tests/
    unit/  integration/  edge_cases/
```

## 5. Component contracts

Each subsection = one module: responsibility, signature, and the implementation notes that
encode the §2 decisions. Build in the order of §7, not the order below.

### 5.1 LLM client + sample-weight oracle
```python
@dataclass
class Generation:
    text: str
    token_logprobs: list[float] | None   # None when black-box
    finish_reason: str

class LLMClient(Protocol):
    supports_logprobs: bool
    def generate(self, prompt: str, temperature: float, n: int) -> list[Generation]: ...
    def cost(self) -> CostTally        # cumulative tokens + $ estimate
```
- `sample_weight.py` exposes `weight(gen, klass, K) -> float`:
  - white-box: `exp(sum(token_logprobs) / len(token_logprobs))` (length normalization is
    mandatory — raw sums bias toward short answers).
  - black-box: `(count(klass) + a) / (K + a*num_classes)` (Laplace `a`, default 1).
- The oracle, not the SD module, owns the white/black-box branch. SD stays agnostic.

### 5.2 Confidence stack
```python
# embed.py
def embed(texts: list[str]) -> np.ndarray
def cosine_groups(emb, tau_pre) -> list[list[int]]      # cheap pre-filter

# entailment.py
def entail_prob(a: str, b: str) -> float                 # P(a entails b)
def kernel(a: str, b: str) -> float                      # bidirectional: min(P(a|=b),P(b|=a))

# cluster.py
def hybrid_cluster(responses: list[str], cfg) -> list[MeaningClass]
# 1) cosine_groups to form candidate groups  -> O(K)
# 2) confirm/merge groups with entail kernel ONLY on borderline cross-group pairs -> O(K*k)
# Each MeaningClass: members[], mass (from oracle), representative

# semantic_density.py
def semantic_density(r_idx, responses, weights, kernel_matrix) -> float
# SD(r_i) = sum_j weights[j] * kernel_matrix[i][j]

# verifier.py
def verify(response: str, subproblem: str) -> float      # [0,1], pluggable per task

# score.py
def U_s(dominant_class, sd_scores, verifier_scores, alpha=0.25) -> float
# mean over dominant class of alpha*SD + (1-alpha)*v
```
Note: build `kernel_matrix` lazily and only within/between cosine candidate groups to keep
the entailment cost at `O(K·k)`, not `O(K²)` (this is the §6 complexity target).

### 5.3 Problem structure
```python
# graph.py
def extract_graph(problem) -> nx.DiGraph                 # explicit + implicit deps
def condense(G) -> nx.DiGraph                            # Tarjan SCC -> DAG (G')
def critical_path_len(Gp) -> int                         # longest path; structural depth

# decompose.py
def next_subproblem(state, Gp, decomposer, granularity) -> Subproblem | ATOMIC
# follows G' topological/critical-path order; granularity (coarse if parent U_s high) sets
# how large the next chunk is. Returns ATOMIC when directly solvable.
def is_atomic(state, Gp) -> bool
```

### 5.4 Coherence gate
```python
# coherence.py
def checkable_constraints(Gp, path) -> list[Constraint]  # those newly evaluable at this depth
def evaluate_gate(path, Gp) -> float                     # returns sigma in [0,1]; 0 == coherent
```
- Called at every node evaluation. If any newly-checkable constraint is violated, set
  `sigma > 0`, value `= 0`, mark node terminal-fail — but **always store sigma** (decision 6).
- Constraint evaluation is task-pluggable (math: numeric/identity consistency; code:
  compile+unit-test; planning: precondition/effect validity).

### 5.5 MCTS engine
```python
# types.py
@dataclass
class Node:
    state: SolutionState
    depth: int
    parent: "Node | None"
    children: dict[ClassKey, "Node"]
    untried: list[MeaningClass]      # populated once, on first expansion
    N: int = 0
    Q: float = 0.0
    prior: float = 0.0               # mass(class) * U_s(class)
    U_s: float = 0.0
    sigma: float = 0.0
    terminal: bool = False
    gated_fail: bool = False

# mcts.py
def select(root, cfg) -> Node            # PUCT descent over expanded children
def expand(node, ctx, cfg) -> Node       # progressive widening (§5.6); lazily decompose+sample
def evaluate(node, ctx, cfg) -> float    # U_s via confidence stack + incremental gate
def backup(node, value, cfg) -> None     # guarded max to root
def search(problem, cfg) -> SearchResult # the rollout loop
```
- **select:** `argmax Q(c) + c_puct * prior(c) * sqrt(sum N(siblings)) / (1 + N(c))`.
- **expand:** see §5.6. On a node's *first* expansion, call `next_subproblem`, draw `K`
  samples, `hybrid_cluster`, score each class' `U_s`, store classes as `untried` sorted by
  prior. Each subsequent widening pops the top `untried` class into a child. If `untried`
  empties and more width is permitted, draw `K` more samples (re-cluster, append new classes).
- **evaluate:** compute node `U_s`; run `evaluate_gate`; if gated, value 0 + terminal-fail.
  At an ATOMIC/`depth==d_max` node, assemble and do the final coherence check.
- **backup:** for each ancestor: `N += 1`; `Q = max(Q(c) for c in children if N(c) >= n_min)`;
  if no child qualifies, hold `Q = node.U_s` (or prior). This is the overestimation guard.

### 5.6 Progressive widening
```python
# widening.py
def may_widen(node, C, beta) -> bool:
    return floor(C * node.N**beta) > len(node.children)
```

### 5.7 Transposition table
```python
# transposition.py
def key(state, Gp) -> str
# hash of (remaining-subproblem signature + bindings of upstream resolutions that affect
# THIS subproblem's checkable constraints). Do NOT key on subproblem alone — a memoized
# solution coherent in one context can violate constraints in another.
```
Cache the `(classes, U_s)` of a decomposition; on hit, skip re-sampling. DAG-MCTS style node
sharing is allowed; `N`/`Q` stay per-path.

### 5.8 Depth governor + fallback + orchestrator
```python
# depth.py
def depth_cap(Gp, budget, per_expansion_cost) -> int     # min(critical_path+slack, B//L)
def should_stop(history, lam, cost_next) -> bool         # E[dU_s] < lam * cost_next

# fallback.py
def soft_relax(reached_terminals) -> FlaggedResult       # argmin sigma, flagged

# orchestrator.py
def solve(problem, cfg) -> Result        # Phase 1 (structure) then Phase 2 (search), then
                                         # §4 selection: best coherent path else soft_relax
```
Seam for future learned components: `prior` and `evaluate` are the two injection points for a
neural prior/value later — keep them behind interfaces so a learned version drops in.

## 6. Complexity targets (assert in tests)

Per expansion ≈ `K·(g+v) + K·k·s` (k = cosine neighbours ≪ K). Whole run worst case
`O(N·d·(K(g+v)+K·k·s))` — **bounded by the rollout budget `N`**; a test must assert total
LLM calls never exceed the configured budget. If the entailment cost shows `O(K²)` scaling in
profiling, the hybrid pre-filter (§5.2) is broken.

## 7. Build order (milestones with test gates)

Each milestone is independently testable against the **mock LLM** before any paid calls.

- **M0 — Skeleton.** Repo, config schema + `default.yaml`, `LLMClient` protocol, mock client,
  telemetry/cost tally. *Gate:* `solve()` stub runs on a toy problem end-to-end returning a
  dummy result; cost tally works.
- **M1 — Confidence stack.** embed → cosine pre-filter → entailment kernel → hybrid cluster →
  sample-weight oracle (both modes) → SD → verifier → `U_s`. *Gate:* unit tests with crafted
  responses: synonyms cluster together, contradictions don't; SD higher for dense clusters;
  white-box vs black-box weights both produce valid `U_s`.
- **M2 — Structure.** graph extract, SCC condense, critical path, decomposer + atomicity.
  *Gate:* cyclic `G` condenses correctly; critical path matches hand-computed cases; atomic
  problems return ATOMIC.
- **M3 — MCTS core.** Node, PUCT select, progressive widening, evaluate, guarded-max backup,
  transposition. Use mock LLM + a synthetic scalar value (skip real coherence). *Gate:*
  widening grows children sublinearly in `N`; `n_min` guard provably blocks single-rollout
  inflation (unit test); backup reaches the root; budget cap is never exceeded.
- **M4 — Coherence + governance.** incremental gate + `σ`, soft-relax fallback, depth cap +
  marginal-gain stop. *Gate:* edge-case suite (§7 table from spec) passes — empty class,
  bimodal, whole-level gated, no-coherent-solution → flagged fallback, etc.
- **M5 — Orchestrator.** wire Phase 1 + 2; walking skeleton on a real toy task with a real
  (small) model. *Gate:* produces a coherent path on an easy instance; produces a flagged
  fallback on a deliberately unsatisfiable instance.
- **M6 — Experiment harness.** dataset loaders, baselines B0–B3, metrics, ablation toggles,
  sweep runner, aggregation. *Gate:* one full comparison table (Concord vs B0–B3) on a small
  slice, with per-run cost logged.
- **M7 — Calibration + first results.** sweep `c_puct`, `n_min`, `K`; run ablations; write the
  results report. *Gate:* §10 acceptance criteria met.

## 8. Experiment harness

### 8.1 Datasets (pick ≥1 to start, structure them as decomposable + verifiable)
- **Math:** GSM8K / MATH — coherence = numeric/identity consistency; verifier from the gold
  answer (held out of the search).
- **Multi-hop QA:** HotpotQA — decomposition along supporting facts; coherence = supporting-
  fact consistency.
- **Code:** MBPP / HumanEval — coherence gate = compiles + hidden unit tests pass (a very
  natural hard gate).
- **Planning:** PlanBench / Blocksworld — coherence = precondition/effect validity.

Start with **one math set + one code set** — they give the cleanest, cheapest coherence
gates and ground-truth verifiers.

### 8.2 Baselines
- **B0** single-shot CoT (no tree, no sampling).
- **B1** self-consistency (K samples, majority vote) — isolates the value of search.
- **B2** Concord **v1** (cosine + hard τ threshold + chronological backtracking) — the thing
  v2 claims to beat; implement it from the v1 spec.
- **B3** vanilla MCTS/ToT without SD and without the coherence gate — isolates those two.

### 8.3 Metrics (log all per run, to JSONL)
- **Solve@budget** (task accuracy under fixed `N`).
- **Coherence rate** (fraction returned with `σ=0` vs soft-relax fallback).
- **Cost:** LLM calls, total tokens, `$` estimate, wall-clock.
- **Search efficiency:** rollouts to first coherent solution, max depth reached, expansions,
  gate-rejection rate, transposition hit rate.
- **`U_s` calibration:** ECE and AUROC of `U_s` against eventual correctness — does the
  confidence signal actually predict success? (Central to the SD claim.)

### 8.4 Ablations (one factor flipped vs the v2 default; each a YAML in `config/ablations/`)
SD ↔ cosine-centroid; entailment-cluster ↔ cosine-threshold; guarded-max ↔ average backup;
progressive widening ↔ fixed `M`-beam; incremental gate ↔ terminal-only ↔ no gate; adaptive
depth ↔ fixed depth; white-box weights ↔ frequency proxy (on a model exposing logprobs, so
both are measurable on the same instances).

### 8.5 Reproducibility
Fixed seeds; log temperature and every prompt/response (hashed) to JSONL; the mock LLM gives
fully deterministic CI runs; record git SHA + config hash in run metadata. Enforce the budget
`N` as a hard ceiling so a runaway sweep can't burn the account.

## 9. Configuration defaults (start here; §8.4 sweeps tune them)

| Param | Default | Rationale / tune via |
|---|---|---|
| `K` (samples/subproblem) | 8 white-box / 16 black-box | frequency proxy needs more samples to be a stable mass estimate |
| `alpha` (SD vs verifier) | 0.25 | verifier-weighted by design |
| `c_puct` | 3.0 | higher than AlphaZero (~1) to offset sparse gate + max backup; sweep 2–5 |
| `C` (widening coeff) | 2 | ≈2 children early; was the old beam `M` |
| `beta` (widening exp) | 0.5 | √-widening, standard |
| `n_min` (max-eligibility) | 3 | blocks single-rollout overestimation; sweep 2–5 |
| `N` (rollout budget) | 128 | cost cap; sweep 64–256 for cost/quality curve |
| `lambda` (stop) | small; stop when ΔU_s/level < ~0.02 per unit cost | the Pareto knob |
| entailment kernel threshold | bidirectional `P>0.5` → same class | calibrate on a labeled paraphrase set |
| `tau_pre` (cosine pre-filter) | loose (e.g. 0.6) | only a candidate-group filter; entailment confirms |
| `d_max` slack over critical path | +2 | headroom above the structural estimate |
| Laplace `a` | 1 | black-box smoothing |

## 10. Acceptance criteria (definition of done)

1. `solve()` runs end-to-end on a real task and returns either a coherent path (`σ=0`) or a
   flagged soft-relax result — never crashes on the §7 edge cases.
2. Every module in §5 has unit tests; the edge-case suite passes; total LLM calls per run are
   provably ≤ `N`.
3. The harness produces a results table comparing **v2 vs B0–B3** on ≥1 dataset, plus the
   §8.4 ablations, with cost and the calibration metrics logged.
4. CI runs the full suite on the mock LLM deterministically with no network/paid calls.
5. A short results report states, with numbers: does v2 beat B2 (v1) on solve@budget *at
   equal or lower cost*, and does SD improve `U_s` calibration (AUROC) over the cosine
   ablation? These two are the core hypotheses — report them whether or not they hold.

## 11. Known risks to surface early (not to fix silently)

- **Guarded-max + high `c_puct` is the fragile pairing.** If search either fixates (too low
  `c_puct`) or never converges (too high), report it from the M7 sweep rather than quietly
  switching to average backup — that would violate decision 4. Raise it for a design call.
- **The entailment kernel is itself a model.** Its errors propagate into both prior and value.
  Log kernel disagreement vs the verifier; if the NLI judge is the bottleneck, that's a
  finding, not a bug to paper over.
- **Logprob availability is per-provider.** Verify length normalization empirically (short
  answers must not win on weight alone).
- **Cost.** The default sweep can be expensive; gate paid runs behind an explicit budget flag
  and run the full matrix on the mock LLM first.
