# 01 — The benchmark: matrix-chain long-horizon execution

## Task definition

A model maintains a running **d × d integer matrix** `STATE`, starting at the
identity `I_d`. At each **turn** `t` it receives one d × d matrix `A_t` and must
update the state by **right-multiplication**, reducing every entry mod `p`:

```
STATE_0 = I_d
STATE_t = (STATE_{t-1} · A_t)  mod 97          (t = 1 … T)
```

The model reports the full matrix after each turn. The golden answer for turn
`t` is the cumulative product `M_t = (M_{t-1} · A_t) mod 97`. Read/report format
is `solution = [[...],[...],...]`. **Rows outer, columns inner.**

Right-multiplication order matters — matrix multiplication is non-commutative,
so `M_{t-1}·A_t ≠ A_t·M_{t-1}`. A decomposition method must therefore specify the
multiply order unambiguously when it hands a sub-range to a solver (see
`02_method_and_contributions.md`).

## Two independent difficulty axes

| Axis | Symbol | Controlled by | Meaning |
|---|---|---|---|
| **Horizon** | `T` | chain length (# turns) | length of the compounding, state-carrying chain; **one wrong entry corrupts every later product** |
| **Complexity** | `d` | matrix dimension | ≈ `d³` scalar multiply-adds per step; per-step arithmetic load, independent of `T` |

- `d = 1` is the scalar running-product baseline (1×1 matrices mod 97). It is a
  **cost control**: monolithic Opus solves it to very long horizons, so any
  decomposition overhead there is pure waste.
- Secondary knob: modulus `p = 97` (larger ⇒ harder individual multiplies).

## Why it is a valid long-horizon-execution test

- **State is model-maintained and non-Markovian**: turn `t` depends on the
  model's *own* turn `t-1` output, not on anything re-presented in the prompt.
- **Errors compound**: a single wrong entry at any turn propagates to every
  later product, so whole-chain success requires *every* step correct. This is
  what makes it a genuine test of long-horizon reliability rather than one-shot
  arithmetic.
- It is the matrix generalization of a scalar running-sum long-horizon task.

## The dataset (static, golden-labeled, frozen)

Full manifest with sha256 per file: `dataset/config.json`. Human description:
`dataset/README.md`.

- **Dimensions** d = 1..8; **horizon** T up to 2000; **10 independent chains per
  dimension**; modulus 97; base seed 7; per-chain seed `7*100000 + d*1000 +
  sample_id`.
- Each line of `chains_d{d}.jsonl` is one chain:
  `{id, dim, modulus, n_turns, seed, matrices:[A_1..A_T], golden:[M_1..M_T]}`.
- **Truncation to any shorter horizon** = slice the first `L` entries of
  `matrices`/`golden`. Concord experiments use sample_id 0 (n=3 reps); the SOTA
  baseline runs use samples 0–9 (n=10), truncated to the target `T`.
- Golden streams were verified at generation by recomputing the running product;
  regeneration is byte-for-byte reproducible (sha256 in the manifest).

**Note for the paper:** the chain files ship at the repo root
(`../data/matmul/chains_d{d}.jsonl`, all d=1..8, verified against the manifest);
this bundle carries the manifest + README that fully specify them.

## Grading

A cell counts as **solved only on an exact match of the whole final matrix
`M_T`** (every entry). This is the strict metric used in all tables. Auxiliary
diagnostics recorded per Concord run (in the `trimexec_T*_summary.json` files):

- `atom_pass_rate` — fraction of decomposition sub-answers (atoms) that match
  their golden slice. Because of cascading, a single wrong atom can drop the
  whole-chain result to failure even at high atom-pass.
- `decomposition_fidelity`, `chain_coverage`, `contiguity_rate` — whether the
  splitter tiled the chain into contiguous, complete, non-overlapping ranges
  (Concord: fidelity 1.0, coverage 1.0 — the decomposition is exact).
- `n_ungradeable` — atoms that never executed (0 for Concord = the whole chain
  runs to completion within budget).
