#!/usr/bin/env python3
"""
Materialize the STATIC matrix-chain long-horizon dataset (with golden answers).

Produces a frozen, reproducible dataset for the horizon x complexity benchmark
(see matmul_exec_common.py / run_matmul_dataset.py). Every chain and every
golden per-turn matrix is written to disk so experiments can be re-run
identically and compared across models without any regeneration.

Task recap
----------
The model maintains a running d x d integer matrix STATE (starting at I_d) and,
each turn, multiplies it on the right by a given matrix A_t and reduces mod p:

    M_0 = I_d ,   M_t = (M_{t-1} . A_t) mod p .

The golden response the model should emit at turn t is M_t (the cumulative
product). HORIZON = number of turns T; COMPLEXITY = matrix dimension d.

On-disk layout (under --out-dir, default data/matmul)
----------------------------------------------------
  config.json          — parameters + a manifest (per-file sha256 + sizes).
  README.md            — human-readable spec (this script writes it).
  chains_d{d}.jsonl    — one file per complexity level d in --dims. Each LINE is
                         one independent chain (sample), a JSON object:
      {
        "id":       int,                 # 0..num_samples-1, unique within the file
        "dim":      d,                   # matrix dimension (complexity)
        "modulus":  p,                   # entries live in [0, p-1]
        "n_turns":  T,                   # chain length (horizon)
        "seed":     int,                 # the RNG seed that produced this chain
        "matrices": [A_1, ..., A_T],     # inputs; each A_t is a d x d nested list
        "golden":   [M_1, ..., M_T]      # golden cumulative products, d x d each
      }

`matrices` is the input stream (analogous to the LHE dataset's `input`/`values`);
`golden` is the golden output stream (analogous to LHE's `output`). To evaluate
at a shorter horizon L, take the first L entries of both. `golden[t]` is exactly
what a perfect model replies as `STATE=` on turn t+1.

Determinism
-----------
Each chain's seed is `--seed * 100000 + d * 1000 + sample_id`, the SAME scheme
run_matmul_dataset.py uses for on-the-fly generation, so this static dataset is
identical to what the runner would produce live with the same --seed. Regenerate
byte-for-byte with the same flags.

Usage
-----
    python make_matmul_dataset.py                        # d=1..8, T=2000, n=10
    python make_matmul_dataset.py --dims 1 2 3 --max-turns 200 --num-samples 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from matmul_exec_common import identity, make_chain, matmul_mod

DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "matmul"


def chain_seed(base_seed: int, dim: int, sample_id: int) -> int:
    """Matches run_matmul_dataset.run_model_dim's ep_seed exactly."""
    return base_seed * 100000 + dim * 1000 + sample_id


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sample(rec: dict) -> None:
    """Re-derive the golden products from the matrices and assert they match."""
    d, p = rec["dim"], rec["modulus"]
    acc = identity(d)
    for t, (a, g) in enumerate(zip(rec["matrices"], rec["golden"]), start=1):
        acc = matmul_mod(acc, a, p)
        if acc != g:
            raise AssertionError(
                f"golden mismatch id={rec['id']} d={d} turn={t}")


def write_readme(out_dir: Path, args, manifest: list[dict]) -> None:
    dims = ", ".join(str(d) for d in args.dims)
    files_tbl = "\n".join(
        f"| `{m['file']}` | {m['dim']} | {m['num_samples']} | {m['n_turns']} | "
        f"{m['bytes'] / 1e6:.2f} MB | `{m['sha256'][:12]}…` |"
        for m in manifest
    )
    readme = f"""# Matrix-Chain Long-Horizon Execution Dataset

A **static, golden-labeled** dataset for the two-axis (horizon × complexity)
long-horizon *execution* benchmark. Frozen so experiments are reproducible and
comparable across models.

## The task

A model maintains a running **d × d integer matrix** `STATE`, starting at the
identity `I_d`. Each **turn** it is given one d × d matrix `A_t`, and must update:

```
STATE_0 = I_d
STATE_t = (STATE_{{t-1}} · A_t)  mod {args.modulus}      # ordinary matmul, then reduce each entry
```

and reply with the full matrix `STATE_t`. The **golden answer** for turn *t* is
the cumulative product `M_t` stored in `golden[t-1]`.

Two independent difficulty axes:

| Axis | Symbol | Controlled by | Meaning |
|---|---|---|---|
| **Horizon** | `T` | chain length (# turns) | length of the compounding, state-carrying chain; one wrong entry corrupts every later product |
| **Complexity** | `d` | matrix dimension | ≈ `d³` scalar multiply-adds per step; per-step arithmetic load, independent of `T` |

`d = 1` is the scalar running-product baseline (1×1 matrices mod {args.modulus}).
Secondary complexity knob: the modulus `p = {args.modulus}` (larger ⇒ harder
individual multiplies).

Why this is a valid long-horizon test: state is **model-maintained and
non-Markovian** (turn *t* depends on the model's own turn *t-1* matrix, not on
anything re-presented) and **errors compound**. This is the matrix generalization
of the scalar running-sum task in `horizon_exec_common.py`.

## Parameters

- **Dimensions (complexity) d:** {dims}
- **Horizon (max turns) T:** {args.max_turns}
- **Samples per dimension:** {args.num_samples} independent chains
- **Modulus p:** {args.modulus} (entries in `[0, {args.modulus - 1}]`)
- **Base seed:** {args.seed}
- **Per-chain seed:** `{args.seed} * 100000 + d * 1000 + sample_id`

## Files

| File | d | samples | T | size | sha256 |
|---|---|---|---|---|---|
{files_tbl}

`config.json` holds the full parameter set + this manifest. Each `chains_d{{d}}.jsonl`
has one JSON object per line:

```json
{{
  "id": 0, "dim": {args.dims[-1]}, "modulus": {args.modulus},
  "n_turns": {args.max_turns}, "seed": 700000,
  "matrices": [ [[...],...], ... ],   // T input matrices A_1..A_T (d×d each)
  "golden":   [ [[...],...], ... ]    // T golden cumulative products M_1..M_T
}}
```

## How to use

Truncate to any shorter horizon by slicing the first `L` entries of `matrices`
(and `golden`). To evaluate complexity `d` at horizon `L`, load `chains_d{{d}}.jsonl`
and use `matrices[:L]` as the turn inputs and `golden[:L]` as the answers.

The runner loads it automatically:

```bash
python run_matmul_dataset.py --models claude-opus-4-8 --dims 1 2 3 \\
    --max-turns 200 --num-samples 10        # reads data/matmul/ if present
```

## Integrity

Every golden stream was verified at generation time by recomputing the running
product from `matrices` and asserting equality with `golden`. Re-verify anytime:

```bash
python make_matmul_dataset.py --verify-only
```

Regenerating with identical flags reproduces every file **byte-for-byte** (the
sha256 values above are the check).

*Generated by `make_matmul_dataset.py`.*
"""
    (out_dir / "README.md").write_text(readme)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                   help="Complexity levels (matrix dimensions) to generate.")
    p.add_argument("--max-turns", type=int, default=2000,
                   help="Horizon: chain length T (turns) per sample.")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Independent chains per dimension.")
    p.add_argument("--modulus", type=int, default=97,
                   help="Entries live in [0, p-1]; larger p = harder multiplies.")
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--verify-only", action="store_true",
                   help="Re-verify golden products of an existing dataset; write "
                        "nothing.")
    args = p.parse_args()

    if args.verify_only:
        return verify_existing(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating matmul dataset -> {args.out_dir}")
    print(f"  dims={args.dims}  T={args.max_turns}  n={args.num_samples}  "
          f"mod={args.modulus}  seed={args.seed}\n")

    manifest: list[dict] = []
    for d in args.dims:
        fname = f"chains_d{d}.jsonl"
        fpath = args.out_dir / fname
        with open(fpath, "w") as f:
            for sid in range(args.num_samples):
                seed = chain_seed(args.seed, d, sid)
                ep = make_chain(d, args.max_turns, args.modulus, seed=seed)
                rec = {
                    "id": sid, "dim": d, "modulus": args.modulus,
                    "n_turns": args.max_turns, "seed": seed,
                    "matrices": [t.matrix for t in ep.turns],
                    "golden": [t.cumulative for t in ep.turns],
                }
                verify_sample(rec)           # golden self-check at write time
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        size = fpath.stat().st_size
        digest = sha256_of(fpath)
        manifest.append({
            "file": fname, "dim": d, "num_samples": args.num_samples,
            "n_turns": args.max_turns, "bytes": size, "sha256": digest,
        })
        print(f"  {fname:18s}  {args.num_samples} chains x {args.max_turns} turns "
              f"| {size/1e6:6.2f} MB | sha256 {digest[:12]}…")

    config = {
        "benchmark": "matmul_chain_horizon_execution",
        "description": ("Static golden-labeled dataset. Running matrix product "
                        "mod p; horizon T = #turns, complexity d = matrix dim."),
        "dims": args.dims, "max_turns": args.max_turns,
        "num_samples": args.num_samples, "modulus": args.modulus,
        "seed": args.seed,
        "per_chain_seed_formula": "seed*100000 + dim*1000 + sample_id",
        "files": manifest,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_readme(args.out_dir, args, manifest)

    total = sum(m["bytes"] for m in manifest)
    print(f"\nWrote config.json + README.md. Total dataset size: {total/1e6:.1f} MB")
    print(f"All golden products verified. Dataset ready at {args.out_dir}")
    return 0


def verify_existing(out_dir: Path) -> int:
    config = json.loads((out_dir / "config.json").read_text())
    print(f"Verifying dataset in {out_dir} ...")
    ok = True
    for m in config["files"]:
        fpath = out_dir / m["file"]
        digest = sha256_of(fpath)
        if digest != m["sha256"]:
            print(f"  {m['file']}: SHA256 MISMATCH (file changed since generation)")
            ok = False
            continue
        n = 0
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    verify_sample(json.loads(line))
                    n += 1
        print(f"  {m['file']}: sha256 OK, {n} chains golden-verified")
    print("All good." if ok else "PROBLEMS FOUND.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
