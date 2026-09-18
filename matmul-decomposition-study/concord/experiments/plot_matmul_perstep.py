#!/usr/bin/env python3
"""Per-step accuracy vs horizon for a Concord matmul study, in the same shape
as the monolithic baseline's report (task accuracy + per-step accuracy at every
prefix length L), one line per complexity n (= matrix dimension d).

Reconstructed from the per-atom grades already recorded (no new runs):
`rp[t]` = 1 iff turn t is covered by a running_state PASS atom (the running
product M_t is verified against the golden label), else 0. Then

  * task accuracy(L)  = 1 if rp[1..L] all correct  (compounds → cliff at the
                        first wrong/lost step) — the headline H_s curve.
  * per-step acc(L)   = mean(rp[1..L])              (fraction of M_1..M_L right)

    python plot_matmul_perstep.py <study_dir>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888",
})


def running_product_correct(grade_path: Path, T: int) -> list[int]:
    """rp[t] over t=1..T: 1 iff t lies inside a running_state PASS slice."""
    recs = json.loads(Path(grade_path).read_text())["atom_records"]
    passv = [False] * (T + 1)
    failv = [False] * (T + 1)
    for r in recs:
        if r.get("style") == "running_state" and r.get("slice"):
            i, j = r["slice"]
            for t in range(max(1, i), min(T, j) + 1):
                if r["status"] == "PASS":
                    passv[t] = True
                else:
                    failv[t] = True
    return [1 if (passv[t] and not failv[t]) else 0 for t in range(1, T + 1)]


def curves(rp: list[int]):
    T = len(rp)
    task, step = [], []
    still = True
    correct_so_far = 0
    for L in range(1, T + 1):
        if rp[L - 1] != 1:
            still = False
        task.append(1.0 if still else 0.0)
        correct_so_far += rp[L - 1]
        step.append(correct_so_far / L)
    return task, step


def main():
    study_dir = Path(sys.argv[1])
    s = json.loads((study_dir / "study_summary.json").read_text())
    T = s["meta"]["max_turns"]
    runs = {r["dim"]: r for r in s["runs"]}
    dims = sorted(runs)
    xs = list(range(1, T + 1))

    # sequential color by complexity (d is ordered).
    norm = Normalize(vmin=min(dims), vmax=max(dims))
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    for d in dims:
        gp = glob.glob(str(study_dir / f"runs/d{d}_r*/grade.json"))
        if not gp:
            continue
        rp = running_product_correct(Path(gp[0]), T)
        task, step = curves(rp)
        c = cmap(norm(d))
        ax1.plot(xs, task, color=c, lw=2, label=f"n={d}")
        ax2.plot(xs, step, color=c, lw=2, label=f"n={d}")

    ax1.set_title("Task accuracy vs horizon  (whole chain M_1..M_L correct)",
                  fontsize=11, loc="left", fontweight="bold")
    ax1.set_xlabel("horizon  L  (chain length)")
    ax1.set_ylabel("task accuracy")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlim(1, T)

    ax2.set_title("Per-step accuracy vs horizon  (fraction of M_1..M_L correct)",
                  fontsize=11, loc="left", fontweight="bold")
    ax2.set_xlabel("horizon  L  (chain length)")
    ax2.set_ylabel("per-step running-product accuracy")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlim(1, T)

    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.035, pad=0.02,
                        ticks=dims)
    cbar.set_label("complexity  n = matrix dimension d")

    fig.suptitle(
        f"Concord — per-step & task accuracy vs horizon, by complexity  "
        f"(T={T}, one chain/dim)", fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             "Single chain per dim (curves are 0/1 per step, not a 10-sample "
             "rate). n=1 sparse (1x1 grader-collision artifact); its final "
             "answer is exact. n=6-8: splitter did not decompose.",
             ha="center", fontsize=8, color="#777777", style="italic")
    out = study_dir / "plots" / "H_perstep_accuracy_vs_horizon.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
