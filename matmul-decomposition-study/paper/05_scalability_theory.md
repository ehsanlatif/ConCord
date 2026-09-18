# 05 — Scalability: an analytical cost model

The scalability claim is argued **theoretically** here; the experimental grid
(`04_main_results.md`) provides corroborating measurements. The claim is about the
**horizon axis** `T` (chain length) — the axis a decomposition method is designed
to help.

## Setup

Let a problem have horizon `T` (number of chained matrices) and dimension `d`.
Cost is measured in USD (cache-aware). We model the dominant, order-determining
terms.

## Monolithic execution: linear in horizon

A monolithic solver processes the chain turn by turn: at turn `t` it holds the
running `d×d` state and folds in the next matrix. Even with prefix caching, each
turn contributes a roughly constant increment `c_mono` (it must emit a fresh `d×d`
state and consume one new matrix), and there is no reuse of *structure* across
turns. Hence

    C_mono(T) ≈ c_mono · T = Θ(T),      exponent b = 1.

Empirically the fitted exponent is **b ≈ 1.0** at every d (d1 1.04, d2 1.03,
d3 1.04) — exactly linear, as the model predicts.

## Concord decomposition: sub-linear in horizon

Concord tiles `[1, T]` with a balanced splitter (branching `b`, depth
`O(log_b T)`) into `≈ T/g` leaves of `g` factors each, then:

- **Index-reference decomposition (Contribution 1)** makes each atom's
  specification `O(1)` tokens (a range `[i, j]`), not `O(g·d²)` copied values. The
  values are injected by code at execution — so decomposition emits `O(T/g)`
  atoms at `O(1)` spec-cost each, and the split LLM cost is `O(T/g)` (internal
  nodes), *independent of d*.
- **Deterministic combine (Contribution 2)** performs composition in code:
  **zero** LLM calls for aggregation, at any depth.
- **Execution (Contribution 3)** is `T/g` leaf passes, each a heavily-cached
  executor call: the large shared prefix (task description, format, order rule) is
  cached at `0.1×` read cost across *all* leaves, so the marginal cost of the
  n-th leaf declines as more leaves reuse the cached prefix.

Collecting terms,

    C_concord(T) ≈ F  +  (κ_split + κ_exec) · (T/g)      (LLM calls)
    with an effective per-leaf cost that *decreases* in T because the cached
    shared prefix amortizes over a growing number of leaves.

Two forces push the **effective exponent below 1**: (i) a fixed decomposition
overhead `F` that amortizes as `T` grows, and (ii) prefix-cache amortization that
lowers the marginal per-leaf cost as the leaf count grows. Over practical horizons
this yields

    C_concord(T) ≈ Θ(T^b),   b < 1.

Empirically the fitted exponent is **b ≈ 0.85 for d≥2** (d2 0.84, d3 0.86) and
**0.51 for d=1** — clearly sub-linear, as the model predicts.

## The crossover

With `C_mono(T) = c_mono·T` (b=1) and `C_concord(T) = F + a·T^{b}` (b<1), the two
curves cross once, at a horizon `T*` beyond which decomposition is strictly
cheaper:

- For `T < T*`, the fixed decomposition overhead `F` dominates → monolithic is
  cheaper.
- For `T > T*`, the smaller sub-linear marginal wins → **Concord is cheaper, and
  the gap widens with `T`.**

Measured crossovers agree: Concord/Opus cost ratio passes below 1 around **T≈50
for d=1** and **T≈200 for d≥2**, reaching **0.42×** (d1), **0.89×** (d2), **0.79×**
(d3) at T=200. See the ratio table in `04_main_results.md`.

## Why the advantage is horizon, not dimension

Per-leaf execution must read `O(g·d²)` input entries and emit `O(d²)` output
entries, so Concord's cost grows with `d` at least as fast as `Θ(d²)` per leaf —
comparable to or steeper than monolithic per-turn cost. Empirically Concord's
cost-vs-`d` exponent is `a ≈ 1.1–1.7` vs monolithic `a ≈ 0.8–1.1`. Decomposition
adds per-atom overhead that scales with the dimension, so:

- **Long, narrow chains (large T, small d): decomposition wins** (its design target).
- **Short, wide chains (small T, large d): monolithic is preferable.**

This is the precise, defensible scalability statement: **Concord's contribution is
sub-linear cost scaling in the horizon**, which is what makes long-chain execution
economical.

## Reproduce the empirical exponents (corroboration)

```python
import json, numpy as np
d = json.load(open("data/concord_final/cost_surface.json"))
idx = {(r["system"], r["d"], r["T"]): r for r in d["rows"]}
for s in ["concord","opus-4.8"]:
    for dd in [1,2,3]:
        xy=[(np.log(T),np.log(idx[(s,dd,T)]["usd"])) for T in [12,25,50,100,200] if (s,dd,T) in idx]
        print(s, dd, round(np.polyfit(*zip(*xy),1)[0],2))
```
