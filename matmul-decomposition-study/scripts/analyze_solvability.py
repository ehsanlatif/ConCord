#!/usr/bin/env python3
"""Solvability-vs-budget analysis for the matmul chain.

Avoids overlapping-cell head-to-heads. Each system contributes its OWN clean
(dimension, horizon, cost, solve-rate) grid; budget (USD) is the shared axis.

Inputs:
  --surface  cost_surface.json      (concord, opus-4.8, gpt-5.5; fields d,T,usd,solve_rate)
  --baselines baselines_only.json   (aries, graph_of_thoughts, select_then_decompose,
                                      monolithic_cot; fields dim,turns,mean_usd,solve_rate)

Outputs (into --out):
  solvability_surface.json   unified tidy rows {system,d,T,usd,solve_rate,n}
  solvability_frontier.png   T*(B) = max solvable horizon within budget B, facet by d
  solve_rate_vs_budget.png   solve-rate vs cost (per horizon), facet by d
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {"concord": "Concord", "opus-4.8": "Opus-4.8 (mono)",
          "gpt-5.5": "GPT-5.5 (mono)", "aries": "ARIES",
          "graph_of_thoughts": "Graph of Thoughts",
          "select_then_decompose": "Select-Then-Decompose",
          "monolithic_cot": "Monolithic CoT"}
ORDER = ["Concord", "ARIES", "Graph of Thoughts", "Select-Then-Decompose",
         "Opus-4.8 (mono)", "GPT-5.5 (mono)", "Monolithic CoT"]
COLOR = {"Concord": "#238b45", "ARIES": "#ef6548", "Graph of Thoughts": "#3690c0",
         "Select-Then-Decompose": "#8c6bb1", "Opus-4.8 (mono)": "#525252",
         "GPT-5.5 (mono)": "#bdbdbd", "Monolithic CoT": "#000000"}
MARK = {"Concord": "o", "ARIES": "s", "Graph of Thoughts": "^",
        "Select-Then-Decompose": "D", "Opus-4.8 (mono)": "x",
        "GPT-5.5 (mono)": "+", "Monolithic CoT": "."}
THRESH = 0.5


# The SOTA-harness "monolithic_cot" is dropped: it is the same system as the
# cost-surface monolithic Opus but measured on a different chain set (they
# disagree, e.g. d2 T50 0.50 vs 1.00). Keeping one monolithic line avoids a
# self-conflict. Decomposition baselines (aries/got/s&d) share the same prompt,
# per-role models, and price table as the cost-surface runs.
DROP_SYSTEMS = {"monolithic_cot"}


def load(surface, baselines, max_horizon):
    rows = []
    s = json.load(open(surface))
    for r in s["rows"]:
        if int(r["T"]) > max_horizon:
            continue
        rows.append({"system": PRETTY.get(r["system"], r["system"]),
                     "d": int(r["d"]), "T": int(r["T"]),
                     "usd": float(r["usd"]), "solve_rate": float(r["solve_rate"]),
                     "n": int(r.get("n", 0) or 0)})
    if baselines and Path(baselines).exists():
        for r in json.load(open(baselines)):
            if r["system"] in DROP_SYSTEMS or int(r["turns"]) > max_horizon:
                continue
            rows.append({"system": PRETTY.get(r["system"], r["system"]),
                         "d": int(r["dim"]), "T": int(r["turns"]),
                         "usd": float(r["mean_usd"]), "solve_rate": float(r["solve_rate"]),
                         "n": int(r.get("n", 0) or 0)})
    return rows


def systems_present(rows):
    present = {r["system"] for r in rows}
    return [s for s in ORDER if s in present]


def frontier(rows, sysn, d, budgets):
    """T*(B): max horizon with solve_rate>=THRESH and usd<=B, for each B."""
    pts = [(r["usd"], r["T"]) for r in rows
           if r["system"] == sysn and r["d"] == d and r["solve_rate"] >= THRESH]
    out = []
    for B in budgets:
        aff = [T for (u, T) in pts if u <= B]
        out.append(max(aff) if aff else np.nan)
    return out


def plot_frontier(rows, out, dims):
    systems = systems_present(rows)
    all_usd = [r["usd"] for r in rows if r["usd"] > 0]
    budgets = np.logspace(np.log10(min(all_usd) * 0.9),
                          np.log10(max(all_usd) * 1.1), 60)
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims), 4.2),
                             squeeze=False, sharey=False)
    for ax, d in zip(axes[0], dims):
        for sysn in systems:
            ys = frontier(rows, sysn, d, budgets)
            if np.all(np.isnan(ys)):
                continue
            ax.step(budgets, ys, where="post", color=COLOR.get(sysn),
                    label=sysn, lw=2 if sysn == "Concord" else 1.4,
                    alpha=0.95 if sysn == "Concord" else 0.8)
        ax.set_xscale("log")
        ax.set_xlabel("budget (USD / problem)")
        ax.set_ylabel("max solvable horizon  T*(B)")
        ax.set_title(f"d = {d}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Solvability frontier: longest chain solved within a budget "
                 f"(solve-rate ≥ {THRESH})")
    fig.tight_layout()
    fig.savefig(out / "solvability_frontier.png", dpi=140)
    plt.close(fig)


def plot_solve_vs_budget(rows, out, dims):
    systems = systems_present(rows)
    fig, axes = plt.subplots(1, len(dims), figsize=(5.0 * len(dims), 4.2),
                             squeeze=False, sharey=True)
    for ax, d in zip(axes[0], dims):
        for sysn in systems:
            pts = sorted([(r["usd"], r["solve_rate"], r["T"]) for r in rows
                          if r["system"] == sysn and r["d"] == d and r["usd"] > 0])
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, MARK.get(sysn, "o") + "-", color=COLOR.get(sysn),
                    label=sysn, lw=2 if sysn == "Concord" else 1.3, ms=6,
                    alpha=0.95 if sysn == "Concord" else 0.8)
        ax.axhline(THRESH, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("budget (USD / problem, log)")
        ax.set_ylabel("solve rate")
        ax.set_title(f"d = {d}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.suptitle("Solve rate vs budget (each marker = one horizon T)")
    fig.tight_layout()
    fig.savefig(out / "solve_rate_vs_budget.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", required=True)
    ap.add_argument("--baselines", default="")
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-horizon", type=int, default=200,
                    help="cap all systems at this horizon (common tested max)")
    ap.add_argument("--out", type=Path, default=Path("plots_solvability"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load(args.surface, args.baselines, args.max_horizon)
    (args.out / "solvability_surface.json").write_text(json.dumps(
        {"threshold": THRESH, "dims": args.dims,
         "systems": systems_present(rows), "rows": rows}, indent=2))
    plot_frontier(rows, args.out, args.dims)
    plot_solve_vs_budget(rows, args.out, args.dims)

    # console: T*(B) at a few budgets
    systems = systems_present(rows)
    for B in [0.5, 1.0, 2.0, 4.0]:
        print(f"\n=== max solvable horizon within ${B:.2f}/problem (solve>={THRESH}) ===")
        print(f"  {'system':24}" + "".join(f"  d={d}" for d in args.dims))
        for sysn in systems:
            cells = []
            for d in args.dims:
                v = frontier(rows, sysn, d, [B])[0]
                cells.append("  -- " if (v != v) else f"{int(v):>4}")
            print(f"  {sysn:24}" + "".join(f"  {c}" for c in cells))
    print(f"\nwrote {args.out}/solvability_frontier.png, solve_rate_vs_budget.png, "
          f"solvability_surface.json")


if __name__ == "__main__":
    main()
