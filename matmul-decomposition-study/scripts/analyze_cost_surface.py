#!/usr/bin/env python3
"""Cost-vs-(dimension d, horizon T) surface for the matmul chain.

Ingests:
  * Concord decomposition studies  — matmul_study `study_summary.json` files
    (one per horizon T; each carries every dim it swept). Provides per-cell
    mean USD / tokens / calls / solve-rate, plus a per-role USD breakdown that
    it re-aggregates from the individual run records.
  * Monolithic metered baselines   — run_matmul_budget `matmul_budget__*.json`
    files. Their per-turn cumulative cost lets us read the cost of ANY horizon
    T <= max_turns for free: cost@T = mean_i per_sample[i][T-1].cum_usd, and
    solved@T = task_accuracy_by_len[T-1] >= threshold.

Emits (into --out):
  * cost_surface.json                 — tidy rows {system,d,T,usd,tokens,calls,
                                        solve_rate,per_role_usd}
  * cost_heatmap.png                  — USD over d x T, one panel per system
  * cost_vs_horizon.png               — USD vs T, line per d, facet per system
  * cost_vs_dim.png                   — USD vs d, line per T
  * concord_role_cost.png             — per-role USD stack vs T (Concord)
  * cost_per_solved.png               — USD / solve-rate, Concord vs baselines

USD is the primary axis (cache-aware / fair); tokens are recorded too.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- pretty names / ordering ------------------------------------------------
CONCORD = "concord"
ROLE_ORDER = ["execution", "verification", "splitter", "combiner",
              "synthesizer", "decomposition", "classification", "synth_verifier"]
ROLE_LABEL = {"execution": "executor", "verification": "verifier",
              "splitter": "splitter", "combiner": "combiner",
              "synthesizer": "synth", "decomposition": "decomp",
              "classification": "classify", "synth_verifier": "synth-verify"}


def _model_pretty(model_id: str) -> str:
    m = model_id.lower()
    if "opus" in m:
        return "opus-4.8"
    if "gpt" in m:
        return "gpt-5.5"
    if "sonnet" in m:
        return "sonnet-5"
    return model_id


# ---------------------------------------------------------------------------
# Concord studies
# ---------------------------------------------------------------------------
def load_concord(globs: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: dict[tuple, dict] = {}          # (d,T) -> row, newest study wins
    files = []
    for g in globs:
        files.extend(glob.glob(g))
    for f in sorted(files):
        try:
            s = json.load(open(f))
        except Exception as e:                                    # noqa: BLE001
            print(f"  ! skip {f}: {e}")
            continue
        T = int(s.get("meta", {}).get("max_turns") or 0)
        by_dim = s.get("by_dim", {})
        # per-role USD aggregated from the run records of this study
        role_by_dim: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        role_n: dict[str, int] = defaultdict(int)
        for r in s.get("runs", []):
            d = str(r.get("dim"))
            role_n[d] += 1
            for role, c in (r.get("per_role_cost") or {}).items():
                role_by_dim[d][role] += float(c.get("usd", 0.0))
        for d, b in by_dim.items():
            n = max(1, role_n.get(d, b.get("n_runs", 1)))
            per_role = {role: role_by_dim[d].get(role, 0.0) / n for role in ROLE_ORDER}
            row = {
                "system": CONCORD, "d": int(d), "T": T,
                "usd": float(b.get("mean_usd", 0.0)),
                "tokens": float(b.get("mean_total_tokens", 0.0)),
                "calls": float(b.get("mean_calls", 0.0)),
                "solve_rate": float(b.get("solve_rate", 0.0)),
                "atom_pass": float(b.get("mean_atom_pass_rate", 0.0)),
                "n": int(b.get("n_runs", 0)),
                "per_role_usd": per_role,
                "_src": Path(f).parent.name,
            }
            key = (int(d), T)
            # keep the study with more repeats (tie -> later file)
            if key not in seen or row["n"] >= seen[key]["n"]:
                seen[key] = row
    rows.extend(seen.values())
    return rows


# ---------------------------------------------------------------------------
# Monolithic baselines (per-turn cumulative)
# ---------------------------------------------------------------------------
def load_baselines(globs: list[str], horizons: list[int]) -> list[dict]:
    rows: list[dict] = []
    best: dict[tuple, dict] = {}          # (model,d,T) -> row, longest run wins
    files = []
    for g in globs:
        files.extend(glob.glob(g))
    for f in sorted(files):
        try:
            d = json.load(open(f))
        except Exception as e:                                    # noqa: BLE001
            print(f"  ! skip {f}: {e}")
            continue
        results = d.get("results", {})
        for model_id, dim_node in results.items():
            sys_name = _model_pretty(model_id)
            for dim_s, node in dim_node.items():
                dim = int(dim_s)
                summ = node.get("summary", {})
                per_sample = node.get("per_sample", [])
                acc_by_len = summ.get("task_accuracy_by_len", [])
                thr = float(summ.get("threshold", 0.5))
                # Use the LONGEST sample, not evaluated_length (= the shortest,
                # so one API-censored sample would blank whole columns). We read
                # cost from whatever samples reached T (coverage-conditioned).
                max_len = max((len(s) for s in per_sample), default=0)
                for T in horizons:
                    if T > max_len or T < 1:
                        continue
                    # mean cumulative USD/tokens at turn index T-1
                    usds, toks = [], []
                    for smp in per_sample:
                        if len(smp) >= T:
                            usds.append(smp[T - 1].get("cum_usd", math.nan))
                            toks.append(smp[T - 1].get("cum_tokens", math.nan))
                    if not usds:
                        continue
                    solve = (float(acc_by_len[T - 1]) if T - 1 < len(acc_by_len)
                             else math.nan)
                    row = {
                        "system": sys_name, "d": dim, "T": T,
                        "usd": float(np.nanmean(usds)),
                        "tokens": float(np.nanmean(toks)),
                        "calls": float(T),          # 1 API call / turn
                        "solve_rate": 1.0 if (not math.isnan(solve) and solve >= thr) else 0.0,
                        "task_acc": solve,
                        "n": len(usds),
                        "per_role_usd": {},
                        "_src": Path(f).name,
                    }
                    key = (sys_name, dim, T)
                    if key not in best or max_len > best[key]["_maxlen"]:
                        row["_maxlen"] = max_len
                        best[key] = row
    for r in best.values():
        r.pop("_maxlen", None)
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _grid(rows, systems, dims, horizons, field):
    """Return {system: 2D array [dim, T]} of `field` (nan when missing)."""
    out = {}
    idx = {(r["system"], r["d"], r["T"]): r for r in rows}
    for sysn in systems:
        M = np.full((len(dims), len(horizons)), np.nan)
        for i, d in enumerate(dims):
            for j, T in enumerate(horizons):
                r = idx.get((sysn, d, T))
                if r is not None:
                    M[i, j] = r.get(field, np.nan)
        out[sysn] = M
    return out


def plot_heatmap(rows, systems, dims, horizons, out):
    grids = _grid(rows, systems, dims, horizons, "usd")
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    vmax = np.nanmax([np.nanmax(g) if np.isfinite(g).any() else 0 for g in grids.values()]) or 1
    for ax, sysn in zip(axes[0], systems):
        M = grids[sysn]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(horizons))); ax.set_xticklabels(horizons)
        ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
        ax.set_xlabel("horizon T"); ax.set_ylabel("dimension d")
        ax.set_title(f"{sysn}  (USD)")
        for i in range(len(dims)):
            for j in range(len(horizons)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < vmax * 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cost surface: mean USD over dimension x horizon")
    fig.tight_layout()
    fig.savefig(out / "cost_heatmap.png", dpi=140)
    plt.close(fig)


def plot_cost_vs_horizon(rows, systems, dims, horizons, out):
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False, sharey=True)
    cmap = plt.get_cmap("plasma")
    for ax, sysn in zip(axes[0], systems):
        for i, d in enumerate(dims):
            xs, ys = [], []
            for T in horizons:
                r = next((r for r in rows if r["system"] == sysn and r["d"] == d and r["T"] == T), None)
                if r and r["usd"] > 0:
                    xs.append(T); ys.append(r["usd"])
            if xs:
                ax.plot(xs, ys, "o-", color=cmap(i / max(1, len(dims) - 1)),
                        label=f"d={d}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("horizon T"); ax.set_title(sysn)
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("mean USD (log)")
    axes[0][-1].legend(title="dim", fontsize=8)
    fig.suptitle("Cost vs horizon (log-log): scaling with chain length")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_horizon.png", dpi=140)
    plt.close(fig)


def plot_cost_vs_dim(rows, systems, dims, horizons, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cmap = plt.get_cmap("viridis")
    styles = {"concord": "o-", "opus-4.8": "s--", "gpt-5.5": "^:"}
    for sysn in systems:
        for k, T in enumerate(horizons):
            xs, ys = [], []
            for d in dims:
                r = next((r for r in rows if r["system"] == sysn and r["d"] == d and r["T"] == T), None)
                if r and r["usd"] > 0:
                    xs.append(d); ys.append(r["usd"])
            if xs:
                ax.plot(xs, ys, styles.get(sysn, "o-"),
                        color=cmap(k / max(1, len(horizons) - 1)),
                        label=f"{sysn} T={T}", alpha=0.85)
    ax.set_yscale("log"); ax.set_xlabel("dimension d")
    ax.set_ylabel("mean USD (log)"); ax.set_xticks(dims)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Cost vs dimension")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_dim.png", dpi=140)
    plt.close(fig)


def plot_role_cost(rows, dims, horizons, out):
    crows = [r for r in rows if r["system"] == CONCORD]
    if not crows:
        return
    fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.8),
                             squeeze=False, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, d in zip(axes[0], dims):
        Ts = [T for T in horizons
              if any(r["d"] == d and r["T"] == T for r in crows)]
        if not Ts:
            ax.set_visible(False); continue
        bottoms = np.zeros(len(Ts))
        for ri, role in enumerate(ROLE_ORDER):
            vals = []
            for T in Ts:
                r = next((r for r in crows if r["d"] == d and r["T"] == T), None)
                vals.append(r["per_role_usd"].get(role, 0.0) if r else 0.0)
            vals = np.array(vals)
            if vals.sum() <= 0:
                continue
            ax.bar([str(t) for t in Ts], vals, bottom=bottoms,
                   label=ROLE_LABEL[role], color=cmap(ri % 10))
            bottoms += vals
        ax.set_title(f"d={d}"); ax.set_xlabel("horizon T")
    axes[0][0].set_ylabel("mean USD")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Concord per-role cost breakdown vs horizon")
    fig.tight_layout()
    fig.savefig(out / "concord_role_cost.png", dpi=140)
    plt.close(fig)


def plot_cost_per_solved(rows, systems, dims, horizons, out):
    fig, axes = plt.subplots(1, len(dims), figsize=(4.4 * len(dims), 3.8),
                             squeeze=False, sharey=True)
    styles = {"concord": "o-", "opus-4.8": "s--", "gpt-5.5": "^:"}
    for ax, d in zip(axes[0], dims):
        for sysn in systems:
            xs, ys = [], []
            for T in horizons:
                r = next((r for r in rows if r["system"] == sysn and r["d"] == d and r["T"] == T), None)
                if not r or r["usd"] <= 0:
                    continue
                sr = r["solve_rate"]
                xs.append(T)
                ys.append(r["usd"] / sr if sr > 0 else np.nan)
            if xs:
                ax.plot(xs, ys, styles.get(sysn, "o-"), label=sysn)
                # mark unsolved (inf) as open markers at top
                for x, y, T in zip(xs, ys, [t for t in horizons if t in xs]):
                    if math.isnan(y):
                        ax.scatter([x], [ax.get_ylim()[1]], marker="x", color="red")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel("horizon T"); ax.set_title(f"d={d}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("USD / solve-rate (log)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Cost per solved problem (lower = better value; x = never solved)")
    fig.tight_layout()
    fig.savefig(out / "cost_per_solved.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concord", nargs="+", required=True,
                    help="globs of matmul_study study_summary.json (combine-fixed)")
    ap.add_argument("--baseline", nargs="+", default=[],
                    help="globs of run_matmul_budget matmul_budget__*.json")
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[12, 25, 50, 100, 200])
    ap.add_argument("--out", type=Path, default=Path("plots_cost_surface"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[concord]"); crows = load_concord(args.concord)
    print(f"  loaded {len(crows)} concord cells")
    print("[baseline]"); brows = load_baselines(args.baseline, args.horizons)
    print(f"  loaded {len(brows)} baseline cells")
    rows = crows + brows

    systems = [CONCORD] + sorted({r["system"] for r in brows})
    (args.out / "cost_surface.json").write_text(json.dumps(
        {"dims": args.dims, "horizons": args.horizons, "systems": systems,
         "rows": rows}, indent=2))

    plot_heatmap(rows, systems, args.dims, args.horizons, args.out)
    plot_cost_vs_horizon(rows, systems, args.dims, args.horizons, args.out)
    plot_cost_vs_dim(rows, systems, args.dims, args.horizons, args.out)
    plot_role_cost(rows, args.dims, args.horizons, args.out)
    plot_cost_per_solved(rows, systems, args.dims, args.horizons, args.out)

    # console table
    print("\n=== COST SURFACE (mean USD | solve-rate) ===")
    hdr = "sys/d\\T   " + "".join(f"{T:>12}" for T in args.horizons)
    for sysn in systems:
        print(f"\n[{sysn}]")
        print(hdr)
        for d in args.dims:
            cells = []
            for T in args.horizons:
                r = next((r for r in rows if r["system"] == sysn and r["d"] == d and r["T"] == T), None)
                cells.append(f"{r['usd']:6.2f}|{r['solve_rate']:.2f}" if r else f"{'--':>11}")
            print(f"  d={d}   " + "".join(f"{c:>12}" for c in cells))
    print(f"\nwrote plots + cost_surface.json to {args.out}/")


if __name__ == "__main__":
    main()
