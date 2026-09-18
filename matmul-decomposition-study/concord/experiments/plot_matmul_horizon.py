#!/usr/bin/env python3
"""Per-dimension horizon plots: how far along the chain Concord keeps the
running product verified-correct, reconstructed from a study's per-atom grades
(no new runs). One small-multiple panel per dimension, horizon t on the x-axis.

    python plot_matmul_horizon.py <study_dir>

`study_dir` is a results/matmul_study/<run> dir containing runs/d{d}_r*/grade.json
and study_summary.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"green": "#009E73", "gray": "#999999", "verm": "#D55E00",
      "blue": "#0072B2", "amber": "#E69F00"}
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888",
})


def reconstruct(grade_path: Path, T: int):
    """Return (verified_correct[t], H_rs) over t=1..T.

    verified_correct[t] = 1 if t is inside a running_state PASS atom's slice,
    0 if inside a running_state FAIL slice, None if no running_state coverage.
    H_rs = longest prefix 1..L fully covered by running_state PASS (the horizon
    up to which the running product is verified correct)."""
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
    verified = []
    for t in range(1, T + 1):
        if failv[t]:
            verified.append(0)
        elif passv[t]:
            verified.append(1)
        else:
            verified.append(None)
    # longest contiguous prefix of PASS (broken by a FAIL or a gap)
    H_rs = 0
    for t in range(1, T + 1):
        if passv[t] and not failv[t]:
            H_rs = t
        else:
            break
    return verified, H_rs


def main():
    study_dir = Path(sys.argv[1])
    s = json.loads((study_dir / "study_summary.json").read_text())
    T = s["meta"]["max_turns"]
    runs = {r["dim"]: r for r in s["runs"]}
    dims = sorted(runs)

    ncol = 4
    nrow = (len(dims) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.7 * nrow),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    hrs_by_dim = {}

    for k, d in enumerate(dims):
        ax = axes[k]
        gp = glob.glob(str(study_dir / f"runs/d{d}_r*/grade.json"))
        rec = runs[d]
        solved = rec["overall_correct"]
        if not gp:
            ax.text(0.5, 0.5, "no grade", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            verified, H_rs = reconstruct(Path(gp[0]), T)
            hrs_by_dim[d] = H_rs
            ts = list(range(1, T + 1))
            # verified-correct region (step fill up to H_rs)
            ax.fill_between(range(0, H_rs + 1), 0, 1, color=OI["green"],
                            alpha=0.20, step="post", zorder=1)
            # per-position markers
            gx = [t for t, v in zip(ts, verified) if v == 1]
            rx = [t for t, v in zip(ts, verified) if v == 0]
            ax.plot(gx, [1] * len(gx), "|", color=OI["green"], ms=7,
                    mew=1.2, zorder=3)
            if rx:
                ax.plot(rx, [0.5] * len(rx), "x", color=OI["verm"], ms=5,
                        zorder=3)
            # frontier line at H_rs
            if H_rs > 0:
                ax.axvline(H_rs, color=OI["green"], ls="--", lw=1.2, zorder=2)
            # beyond H_rs: state lost / unverified
            if H_rs < T:
                ax.axvspan(H_rs, T, color=OI["gray"], alpha=0.10, zorder=0)
            note = (f"solved (M_{T} exact)" if solved
                    else (f"verified to t≈{H_rs}" if H_rs > 0
                          else "no decomposition"))
            cov = rec.get("chain_coverage")
            ax.text(0.5, 0.5,
                    note + (f"\ncoverage {cov:.2f}" if cov is not None else ""),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8.5, color="#444444")
        ax.set_title(f"n = d = {d}" + ("   ✓ SOLVED" if solved else ""),
                     fontsize=10, loc="left",
                     fontweight="bold",
                     color=(OI["green"] if solved else "#333333"))
        ax.set_ylim(-0.08, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["wrong", "correct"])
        ax.set_xlim(0, T)

    for k in range(len(dims), len(axes)):
        axes[k].set_axis_off()
    for k in range(len(dims)):
        if k // ncol == nrow - 1 or k >= len(dims) - ncol:
            axes[k].set_xlabel("horizon  t  (turn / matrices multiplied)")

    fig.suptitle(
        "Concord running-product correctness vs horizon, per complexity n "
        f"(T={T}, one chain/dim)",
        fontsize=13, fontweight="bold", y=1.005)
    fig.text(0.5, -0.02,
             "Green ticks: M_t verified against golden as a running-state "
             "result. Dashed line / shaded tail: horizon where state threading "
             "is lost. n=1 sparse (1x1 grader-collision artifact); its L0 solve "
             "is exact.", ha="center", fontsize=8, color="#777777",
             style="italic")
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    out = study_dir / "plots" / "F_horizon_by_dim.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # Summary: verified horizon vs n.
    if hrs_by_dim:
        f2, a2 = plt.subplots(figsize=(6.8, 4.4))
        ns = sorted(hrs_by_dim)
        vals = [hrs_by_dim[n] for n in ns]
        colors = [OI["green"] if runs[n]["overall_correct"] else OI["blue"]
                  for n in ns]
        bars = a2.bar(ns, vals, color=colors, width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            a2.text(b.get_x() + b.get_width() / 2, v + T * 0.01, str(v),
                    ha="center", va="bottom", fontsize=9)
        a2.axhline(T, ls=":", color=OI["gray"], lw=1)
        a2.text(ns[-1], T, f" full horizon T={T}", va="bottom", ha="right",
                fontsize=8, color="#777777")
        a2.set_xlabel("complexity  n = matrix dimension d")
        a2.set_ylabel("verified horizon  H_rs  (turns kept correct)")
        a2.set_xticks(ns)
        a2.set_title("Verified running-product horizon vs complexity  "
                     "(green = whole chain solved)", fontsize=10.5,
                     loc="left", fontweight="bold")
        f2.tight_layout()
        out2 = study_dir / "plots" / "G_verified_horizon_vs_n.png"
        f2.savefig(out2, bbox_inches="tight")
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
