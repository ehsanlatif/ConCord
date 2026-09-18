# Concord — Adaptive Tree-search with Unanimous-Confidence Decomposition

MCTS-based reformulation incorporating Semantic Density confidence, value backup
to the root, an incremental hard coherence gate, and concrete depth governance.

---

## 1. Parameters and oracles

| Symbol | Meaning |
|---|---|
| `P` | problem instance |
| `G` | extracted explicit/implicit dependency graph |
| `G'` | acyclic condensation of `G` (Tarjan SCC contraction) |
| `K` | samples per subproblem (per-model; larger for black-box) |
| `w(r)` | **sample-weight oracle** — white-box: length-normalized `p_model(r)`; black-box: Laplace-smoothed meaning-class frequency |
| `SD(r)` | Semantic Density of response `r` = `Σ_j w(r_j)·k(r, r_j)`, `k` = entailment kernel |
| `v(r,q)` | external verifier score in `[0,1]` |
| `U_s(q)` | unified score over the dominant meaning-class = `mean[ α·SD_i + (1−α)·v_i ]`, `α = 0.25` |
| `σ` | coherence severity of an assembly in `[0,1]` (0 = fully coherent) |
| `N` | rollout budget (cost cap) |
| `c_puct` | PUCT exploration constant (set high — see §5) |
| `C, β` | progressive-widening coefficient/exponent; `C` plays the role of the old beam width `M` |
| `n_min` | minimum visits before a child is eligible to set a parent's max |
| `B`, `L` | total cost budget, per-expansion cost estimate |

## 2. Phase 1 — characterization and concrete depth

```
 1: G        ← ExtractGraph(P)                  // explicit + implicit
 2: τ_type   ← CategorizeTask(P, G)
 3: D        ← SelectDecomposer(G, τ_type)      // explicit | implicit
 4: G'       ← Condense(G)                       // Tarjan SCC → DAG (handles cycles)
 5: d_init   ← LongestPath(G')                   // critical path = structural depth estimate
 6: d_budget ← floor(B / L)                       // affordable depth cap
 7: d_max    ← min(d_init_upper, d_budget)        // hard cap; stopping rule may halt earlier
```

## 3. Phase 2 — MCTS with value backup

```
 8: root ← Node(state = P, depth = 0)
 9: repeat                                        // until N rollouts or stopping rule (§ line 30)
10:     ── SELECT ───────────────────────────────────────────────
11:     s ← root
12:     while s is fully widened and not terminal do
13:         s ← argmax_a [ Q(s,a) + c_puct · P(s,a) · √(ΣN(s,·)) / (1 + N(s,a)) ]
14:
15:     ── EXPAND (progressive widening) ─────────────────────────
16:     if floor(C · N(s)^β) > childCount(s) and depth(s) < d_max then
17:         subproblem ← Decompose(s, D, granularity = f(U_s(parent)))   // coarser if U_s high
18:         R ← { LLM(subproblem, temp ~ U[0,1]) : i = 1..K }
19:         Classes ← HybridCluster(R)            // cosine pre-filter → entailment confirm
20:         for each unexpanded class κ ∈ Classes:
21:             P(s, κ) ← mass(κ) · U_s(κ)          // PUCT prior; each class is a child
22:         s ← Instantiate(highest-prior unexpanded κ)   // bimodal handled: classes coexist
23:
24:     ── EVALUATE + INCREMENTAL HARD GATE ──────────────────────
25:     val ← U_s(s)                               // SD oracle + verifier
26:     for each constraint c ∈ G' newly checkable at depth(s):
27:         if Violated(c, path(root→s)) then
28:             record σ(s); val ← 0; mark s terminal-fail; break    // gate fires early
29:
30:     ── BACKUP TO ROOT (guarded max / Bellman) ────────────────
31:     for each ancestor a on path(s → root):
32:         N(a) ← N(a) + 1
33:         Q(a) ← max over children c of a with N(c) ≥ n_min of Q(c)   // overestimation guard
34:     update best coherent path; check marginal-gain stopping rule
35: until budget N exhausted or  E[ΔU_s] < λ · cost(next level)
```

## 4. Solution selection and fallback

```
36: if ∃ coherent terminal (σ = 0) then
37:     return argmax_path Σ Q along coherent root→leaf path        // primary
38: else                                                            // last-resort soft relax
39:     return argmin_terminal σ , FLAGGED "unverified, best-effort"
```

## 5. Why the configuration choices travel together

- **Incremental hard gate** (lines 26–28) densifies the otherwise-sparse terminal
  reward so the value backup (line 30) is not silent until the first coherent leaf.
- **Guarded max backup** (line 33, `n_min`) prevents a single lucky rollout from
  inflating a node — the known maximization bias of pure-max under noisy reward.
- **High `c_puct`** compensates for greedy max backup under-exploring sibling
  subtrees that currently read as value 0 because unvisited.
- **Progressive widening** replaces the fixed M-beam: under-visited nodes stay
  narrow, promising nodes widen; memory-bounded and adaptive.

## 6. Complexity

Per expansion: `K·(g + v)` generation+verification + `K·k·s` hybrid clustering
(`k` neighbours ≪ K, vs `K²` for naïve all-pairs entailment).

| Case | Bound | Condition |
|---|---|---|
| Best | `O(K(g+v) + K·k·s)` | coherent high-`U_s` terminal found at shallow depth, ~constant in `N` |
| Worst | `O(N · d · (K(g+v) + K·k·s))` | full budget, gate rejects until exhaustion, then `O(N)` soft-relax pass — **bounded** (vs `O(Mᵈ)` for naïve backtracking) |
| Average | `O(N · E[d] · ρ · (K(g+v) + K·k·s))` | `ρ` = branch survival past gate; `E[d] < d_max` via marginal-gain stop |

The bounded worst case is the payoff of the rollout budget `N`: naïve chronological
backtracking could re-walk `O(Mᵈ)` retained states.

## 7. Edge cases and handling

| Edge case | Handling |
|---|---|
| Empty dominant class (`N_unan = 0`) | re-sample with higher K, or re-decompose finer; never divide by zero |
| Bimodal / tied clusters | each meaning-class is a separate PUCT child — explored, not arbitrarily collapsed |
| Whole level fails gate | becomes a low-value subtree; selection abandons it (no hard dead-end) |
| Non-decomposition (trivial split) | atomicity predicate halts recursion at directly-solvable subproblems |
| Cycles in `G` | SCC condensation to `G'` before depth/critical-path computation |
| Confident-but-incoherent assembly | incremental hard gate on `G'` constraints |
| No coherent solution in budget | soft-relax fallback, ranked by recorded `σ`, flagged |
| Max-backup overestimation | visit guard `n_min` on max eligibility |
| Transposition under contextual constraints | key the memo on (subproblem + constraint-relevant bindings), not subproblem alone |
| Black-box SD instability at small K | larger per-model K + Laplace smoothing on class counts |
