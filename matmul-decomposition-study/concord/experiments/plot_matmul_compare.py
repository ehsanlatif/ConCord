#!/usr/bin/env python3
"""Before/after comparison of two matmul studies (e.g. broken-state vs fixed).

    python plot_matmul_compare.py <old_summary.json> <new_summary.json> <out.png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"gray": "#999999", "green": "#009E73", "blue": "#0072B2",
      "orange": "#E69F00", "verm": "#D55E00"}
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666",
})


def series(summary_path):
    s = json.loads(Path(summary_path).read_text())
    runs = {r["dim"]: r for r in s["runs"]}
    return runs


def grouped(ax, ns, old_vals, new_vals, ylabel, title, ylim=None):
    w = 0.38
    xs_o = [n - w / 2 for n in ns]
    xs_n = [n + w / 2 for n in ns]
    ov = [v if v is not None else 0 for v in old_vals]
    nv = [v if v is not None else 0 for v in new_vals]
    ax.bar(xs_o, ov, width=w, color=OI["gray"], zorder=3,
           label="before fix (severed state)")
    ax.bar(xs_n, nv, width=w, color=OI["green"], zorder=3,
           label="after fix (threaded state)")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("complexity  n = matrix dimension d")
    ax.set_xticks(ns)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")


def main():
    old_p, new_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    old, new = series(old_p), series(new_p)
    ns = sorted(set(old) | set(new))

    def get(runs, n, key, default=None):
        r = runs.get(n)
        return r.get(key, default) if r else default

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    grouped(axes[0], ns,
            [get(old, n, "chain_coverage") for n in ns],
            [get(new, n, "chain_coverage") for n in ns],
            "chain coverage (fraction of 1..T executed)",
            "A. Chain coverage  (state threading)", ylim=(0, 1.08))
    grouped(axes[1], ns,
            [get(old, n, "atom_pass_rate") for n in ns],
            [get(new, n, "atom_pass_rate") for n in ns],
            "atom pass-rate", "B. Per-atom pass-rate", ylim=(0, 1.08))
    grouped(axes[2], ns,
            [1 if get(old, n, "overall_correct") else 0 for n in ns],
            [1 if get(new, n, "overall_correct") else 0 for n in ns],
            "solved (final M_T exact)", "C. Whole-chain solved",
            ylim=(0, 1.15))
    axes[2].set_yticks([0, 1])
    fig.suptitle(
        "Concord matmul decomposition — before vs after the state-threading fix "
        "(d=1..8, T=100)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "n=1 coverage understated by a grader artifact (1x1 value "
             "collisions); its L0 solve is exact. High-d (6-8): splitter stops "
             "decomposing — a separate failure mode, not state threading.",
             ha="center", fontsize=8, color="#777777", style="italic")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
