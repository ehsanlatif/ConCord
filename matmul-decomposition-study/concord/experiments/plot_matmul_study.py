#!/usr/bin/env python3
"""Plot a matmul decomposition study (study_summary.json) as PNGs.

Single-axis panels only (no dual-axis), Okabe-Ito colorblind-safe hues,
direct value labels. Usage:

    python plot_matmul_study.py results/matmul_study/<run>/study_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Okabe-Ito colorblind-safe palette.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "gray": "#999999", "black": "#222222"}
DEPTH_COLORS = [OI["sky"], OI["blue"], OI["orange"], OI["verm"], OI["purple"]]

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666",
})


def load(path: Path):
    s = json.loads(path.read_text())
    runs = sorted(s["runs"], key=lambda r: r["dim"])
    return s, runs


def _label_bars(ax, bars, vals, fmt="{:.0f}", dy=0.01, color="#222222"):
    top = max((b.get_height() for b in bars), default=1) or 1
    for b, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * top,
                fmt.format(v), ha="center", va="bottom", fontsize=8.5,
                color=color)


def panel_solve(ax, runs):
    ns = [r["dim"] for r in runs]
    ok = [1 if r["overall_correct"] else 0 for r in runs]
    colors = [OI["green"] if v else OI["gray"] for v in ok]
    bars = ax.bar(ns, ok, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, ok):
        ax.text(b.get_x() + b.get_width() / 2, 0.5, "correct" if v else "wrong",
                ha="center", va="center", rotation=90, fontsize=8.5,
                color="white" if v else "#333333", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wrong", "correct"])
    ax.set_xlabel("complexity  n = matrix dimension d")
    ax.set_xticks(ns)
    ax.set_title("A. Final-answer correctness vs n  (L0, 1 chain/dim, T=100)",
                 fontsize=10.5, loc="left", fontweight="bold")


def panel_cost(ax, runs):
    ns = [r["dim"] for r in runs]
    toks = [r["cost"]["total_tokens"] / 1000 for r in runs]
    usd = [r["cost"]["usd"] for r in runs]
    bars = ax.bar(ns, toks, color=OI["blue"], width=0.62, zorder=3)
    _label_bars(ax, bars, usd, fmt="${:.2f}", dy=0.02, color=OI["verm"])
    ax.set_ylabel("total tokens (thousands)")
    ax.set_xlabel("complexity  n = matrix dimension d")
    ax.set_xticks(ns)
    ax.set_title("B. Spend vs n  (bar = tokens; label = USD)",
                 fontsize=10.5, loc="left", fontweight="bold")


def panel_calls(ax, runs, budget):
    ns = [r["dim"] for r in runs]
    calls = [r["cost"]["calls"] for r in runs]
    trunc = ["budget_truncated" in str(r["flagged"] or "") for r in runs]
    colors = [OI["orange"] if t else OI["blue"] for t in trunc]
    bars = ax.bar(ns, calls, color=colors, width=0.62, zorder=3)
    _label_bars(ax, bars, calls, fmt="{:.0f}")
    if budget:
        ax.axhline(budget, ls="--", lw=1.4, color=OI["verm"], zorder=2)
        ax.text(ns[-1], budget, f"  node-call budget = {budget}",
                va="bottom", ha="right", fontsize=8, color=OI["verm"])
    ax.set_ylabel("agent (LLM) calls")
    ax.set_xlabel("complexity  n = matrix dimension d")
    ax.set_xticks(ns)
    ax.legend(handles=[Patch(color=OI["orange"], label="hit budget (truncated)"),
                       Patch(color=OI["blue"], label="completed")],
              fontsize=8, loc="upper right", frameon=False)
    ax.set_title("C. Decomposition effort vs n  (calls; rise then collapse)",
                 fontsize=10.5, loc="left", fontweight="bold")


def panel_depth(ax, runs):
    # Gather per-depth pass rates per n.
    depths = set()
    data = {}  # n -> {depth: pass_rate}
    for r in runs:
        d = {}
        for ds, lv in (r.get("levels") or {}).items():
            if lv.get("pass_rate") is not None:
                d[int(ds)] = lv["pass_rate"]
                depths.add(int(ds))
        if d:
            data[r["dim"]] = d
    depths = sorted(depths)
    ns = sorted(data)
    if not ns:
        ax.text(0.5, 0.5, "no gradeable atoms", ha="center", va="center")
        ax.set_axis_off()
        return
    nd = len(depths)
    w = 0.8 / nd
    for i, dep in enumerate(depths):
        xs = [n + (i - (nd - 1) / 2) * w for n in ns]
        ys = [data[n].get(dep, 0) for n in ns]
        ax.bar(xs, ys, width=w, color=DEPTH_COLORS[dep % len(DEPTH_COLORS)],
               zorder=3, label=f"split depth {dep}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("atom pass-rate")
    ax.set_xlabel("complexity  n = matrix dimension d")
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=1)
    ax.set_title("D. Per-atom pass-rate by split depth  (L3; deeper = worse)",
                 fontsize=10.5, loc="left", fontweight="bold")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: plot_matmul_study.py <study_summary.json>")
    path = Path(sys.argv[1])
    s, runs = load(path)
    out_dir = path.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    budget = None
    try:
        budget = runs[0]["key_params"].get("pipeline.max_node_calls")
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    panel_solve(axes[0, 0], runs)
    panel_cost(axes[0, 1], runs)
    panel_calls(axes[1, 0], runs, budget)
    panel_depth(axes[1, 1], runs)
    T = s["meta"]["max_turns"]
    models = runs[0]["role_models"]
    fig.suptitle(
        f"Concord decomposition on the matrix-chain task  "
        f"(T={T}, exec={models['execution'].split('/')[-1]}, "
        f"split/comb={models['splitter'].split('/')[-1]})",
        fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "n=1 fidelity/coverage understated (1x1 value collisions in the "
             "grader's chain-matching); L0 correctness is exact.",
             ha="center", fontsize=8, color="#777777", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    combined = out_dir / "matmul_study_overview.png"
    fig.savefig(combined, bbox_inches="tight")
    print(f"wrote {combined}")

    # Also emit each panel standalone.
    for name, fn in [("A_solve_vs_n", lambda a: panel_solve(a, runs)),
                     ("B_cost_vs_n", lambda a: panel_cost(a, runs)),
                     ("C_calls_vs_n", lambda a: panel_calls(a, runs, budget)),
                     ("D_passrate_by_depth", lambda a: panel_depth(a, runs))]:
        f, a = plt.subplots(figsize=(6.6, 4.6))
        fn(a)
        f.tight_layout()
        p = out_dir / f"{name}.png"
        f.savefig(p, bbox_inches="tight")
        plt.close(f)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
