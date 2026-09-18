"""Derive a cost-estimation equation from parameter-swept run logs.

Goal
----
Repeatedly run the solver with different parameters, log how many input/output
tokens each *model* consumes and what it cost, then fit a linear equation

        cost_usd(model)  ≈  a · input_tokens  +  b · output_tokens

for every model in a specified set. The fitted coefficients are the model's
$-per-token rates (a·1e6 = $/Mtok input, b·1e6 = $/Mtok output). A whole-run
estimate is the sum of the per-model equations:

        total_cost  ≈  Σ_model ( a_m · in_m  +  b_m · out_m )

The script also fits a single *blended* equation total_cost ≈ A·total_in +
B·total_out and reports its error, to show how badly a mix-agnostic rate
estimates a pipeline that routes different roles to different-priced models.

Why this works here: the Anthropic adapter computes usd deterministically as
(in·price_in + out·price_out)/1e6 per model, so cost is exactly linear in
tokens and the regression recovers the per-model rates (R²≈1). The empirical
fit therefore (a) confirms linearity, (b) recovers and documents the rates from
real logged usage, and (c) quantifies the blended-rate error.

Two phases, either can run alone:
  COLLECT  — vary parameters, run solve_multi, append per-model samples to
             cost_samples.jsonl.
  FIT      — read samples, fit per-model + blended equations, write
             cost_model.json / .md / .csv.

USAGE
-----
Offline plumbing test (mock LLM; usd is imputed from the price table so the
fit has a non-zero target — auto-enabled when all logged usd are 0):

    .venv/bin/python -m concord.experiments.cost_model --mock \\
        --sweep concord/config/cost_model.sweep.yaml

Real collection (needs ANTHROPIC_API_KEY):

    .venv/bin/python -m concord.experiments.cost_model \\
        --config concord/config/wide_and_deep.yaml \\
        --sweep  concord/config/cost_model.sweep.yaml

Refit existing logs without re-running:

    .venv/bin/python -m concord.experiments.cost_model \\
        --fit-only results/cost_model/<ts>/cost_samples.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Make `core`, sibling experiment modules importable however launched.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # concord/    -> import core
sys.path.insert(0, str(_HERE.parent))          # experiments/ -> import chess_study

import yaml                                      # noqa: E402

from core.config import Config                  # noqa: E402
from core.llm.factory import RoleClients        # noqa: E402
from core.multi_solve import solve_multi        # noqa: E402

# Reuse the chess_study helpers (config building w/ dotted overrides + mock
# enforcement, prompt truncation, question loading).
from chess_study import build_config, load_question, truncate_prompt  # noqa: E402

# The single source of truth for how usd is computed from tokens. Imported
# lazily-safe (the adapter only imports `anthropic` inside __init__).
try:
    from core.llm.anthropic_adapter import _PRICE_PER_MTOK as PRICE_PER_MTOK
except Exception:                                # noqa: BLE001
    # Fallback mirror — keep in sync with anthropic_adapter._PRICE_PER_MTOK.
    PRICE_PER_MTOK = {
        "claude-fable-5":    {"in": 10.0, "out": 50.0},
        "claude-opus-5":     {"in":  5.0, "out": 25.0},
        "claude-opus-4-8":   {"in":  5.0, "out": 25.0},
        "claude-opus-4-7":   {"in":  5.0, "out": 25.0},
        "claude-sonnet-5":   {"in":  3.0, "out": 15.0},
        "claude-sonnet-4-6": {"in":  3.0, "out": 15.0},
        "claude-haiku-4-5":  {"in":  1.0, "out":  5.0},
    }


def table_usd(model: str, in_tok: int, out_tok: int) -> float:
    """Reference cost from the price table (mirrors the adapter's _price)."""
    p = PRICE_PER_MTOK.get((model or "").split("[", 1)[0])
    if p is None:
        return 0.0
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000.0


# ===========================================================================
# Least squares (pure python; no numpy dependency)
# ===========================================================================

def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve A x = b (square A) by Gaussian elimination with partial pivoting.
    Returns None when A is singular / ill-conditioned to working precision."""
    n = len(A)
    # augment
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pivval
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def lstsq(rows: list[list[float]], y: list[float]) -> list[float] | None:
    """Ordinary least squares for X (m×k) and y (m): solve normal equations
    (XᵀX) c = Xᵀy. Returns the k coefficients, or None when XᵀX is singular
    (under-determined / collinear features)."""
    if not rows:
        return None
    k = len(rows[0])
    AtA = [[0.0] * k for _ in range(k)]
    Atb = [0.0] * k
    for xi, yi in zip(rows, y):
        for a in range(k):
            Atb[a] += xi[a] * yi
            for bb in range(k):
                AtA[a][bb] += xi[a] * xi[bb]
    return _solve_linear(AtA, Atb)


def r_squared(rows: list[list[float]], y: list[float],
              coeffs: list[float]) -> float:
    if not y:
        return float("nan")
    yhat = [sum(c * xij for c, xij in zip(coeffs, xi)) for xi in rows]
    ybar = sum(y) / len(y)
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    ss_tot = sum((yi - ybar) ** 2 for yi in y)
    if ss_tot <= 1e-30:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


def max_abs_resid(rows: list[list[float]], y: list[float],
                  coeffs: list[float]) -> float:
    if not y:
        return float("nan")
    return max(abs(yi - sum(c * xij for c, xij in zip(coeffs, xi)))
               for xi, yi in zip(rows, y))


# ===========================================================================
# Fitting one model's samples
# ===========================================================================

def fit_one(samples: list[dict], target: str = "usd") -> dict:
    """Fit cost = a·in + b·out (and a 3-param a·in+b·out+c variant) for one
    model's samples. `target` selects the cost column ('usd' logged, or
    'usd_table' imputed from the price table)."""
    ins = [float(s["input_tokens"]) for s in samples]
    outs = [float(s["output_tokens"]) for s in samples]
    y = [float(s.get(target, 0.0)) for s in samples]

    out: dict[str, Any] = {
        "n_samples": len(samples),
        "target": target,
        "input_tokens_range": [min(ins), max(ins)] if ins else None,
        "output_tokens_range": [min(outs), max(outs)] if outs else None,
        "mean_usd": (sum(y) / len(y)) if y else None,
    }

    # No-intercept model: cost = a·in + b·out
    rows2 = [[i, o] for i, o in zip(ins, outs)]
    c2 = lstsq(rows2, y) if len(samples) >= 2 else None
    if c2 is not None:
        a, b = c2
        out["fit"] = {
            "a_per_input_token": a,
            "b_per_output_token": b,
            "price_in_per_mtok": a * 1_000_000.0,
            "price_out_per_mtok": b * 1_000_000.0,
            "r2": r_squared(rows2, y, c2),
            "max_abs_residual_usd": max_abs_resid(rows2, y, c2),
            "equation": (f"cost_usd ≈ {a:.6e}·input_tokens "
                         f"+ {b:.6e}·output_tokens"),
            "identifiable": _is_identifiable(ins, outs),
        }
    else:
        out["fit"] = {"error": "under-determined (need ≥2 varied samples)"}

    # With-intercept model: cost = a·in + b·out + c  (c should be ≈ 0)
    rows3 = [[i, o, 1.0] for i, o in zip(ins, outs)]
    c3 = lstsq(rows3, y) if len(samples) >= 3 else None
    if c3 is not None:
        a, b, c = c3
        out["fit_with_intercept"] = {
            "a_per_input_token": a,
            "b_per_output_token": b,
            "intercept_usd": c,
            "r2": r_squared(rows3, y, c3),
        }

    # Reference rates from the price table (ground truth for this model).
    p = PRICE_PER_MTOK.get((samples[0].get("model") or "").split("[", 1)[0])
    if p is not None:
        out["reference_price_per_mtok"] = {"in": p["in"], "out": p["out"]}
    return out


def _is_identifiable(ins: list[float], outs: list[float]) -> bool:
    """Input/output token volumes must vary somewhat *independently* for a and
    b to be separately recoverable. Flag near-perfect collinearity (constant
    out/in ratio), where only a blended a+b·ratio is identifiable."""
    n = len(ins)
    if n < 3:
        return False
    mi, mo = sum(ins) / n, sum(outs) / n
    sii = sum((i - mi) ** 2 for i in ins)
    soo = sum((o - mo) ** 2 for o in outs)
    sio = sum((i - mi) * (o - mo) for i, o in zip(ins, outs))
    if sii <= 1e-9 or soo <= 1e-9:
        return False
    corr = sio / ((sii * soo) ** 0.5)
    return abs(corr) < 0.999


# ===========================================================================
# Collection: one run -> per-model samples
# ===========================================================================

def per_model_cost(clients: RoleClients) -> dict[str, dict]:
    """Sum (input_tokens, output_tokens, usd, calls) per MODEL across the run.

    RoleClients shares one client among roles with identical specs, so we
    dedup by client identity (counting each unique client once) and attribute
    it to its model. Two roles on the same model but different temperature are
    distinct clients and are summed — correct, since they bill separately."""
    seen: set[int] = set()
    per_model: dict[str, dict] = {}
    for role in RoleClients.ROLE_NAMES:
        c = getattr(clients, role, None)
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        model = clients.specs[role].model
        t = c.cost()
        d = per_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0,
                                         "usd": 0.0, "calls": 0})
        d["input_tokens"] += t.input_tokens
        d["output_tokens"] += t.output_tokens
        d["usd"] += t.usd
        d["calls"] += t.calls
    return per_model


def make_clients(cfg: Config, mock: bool) -> RoleClients:
    """Per-role clients. Real mode: RoleClients.from_config (Anthropic).

    Mock mode: deterministic MockLLM clients, but with the REAL per-role model
    names preserved on `specs` — so token usage is still attributed to
    sonnet/opus/haiku and the price table applies. (chess_study.build_config's
    own --mock path wipes model names to mock-v0, which would collapse every
    role to one unpriced model and defeat a per-model cost fit.) Distinct specs
    get distinct mock clients, mirroring from_config's dedup-by-spec so cost
    attribution per model is exact."""
    if not mock:
        return RoleClients.from_config(cfg)
    from core.llm.mock import MockLLM
    specs = {role: cfg.role_model(role) for role in RoleClients.ROLE_NAMES}
    by_key: dict[tuple, MockLLM] = {}

    def client_for(spec):
        key = tuple(sorted(spec.model_dump().items()))
        if key not in by_key:
            by_key[key] = MockLLM(responses={}, with_logprobs=False)
        return by_key[key]

    return RoleClients(
        execution=client_for(specs["execution"]),
        decomposition=client_for(specs["decomposition"]),
        classification=client_for(specs["classification"]),
        verification=client_for(specs["verification"]),
        splitter=client_for(specs["splitter"]),
        combiner=client_for(specs["combiner"]),
        synthesizer=client_for(specs["synthesizer"]),
        synth_verifier=client_for(specs["synth_verifier"]),
        specs=specs,
    )


def collect_run(*, label: str, repeat: int, overrides: dict, base_config: Path,
                mock: bool, solver: str | None, prompt: str, domain: str,
                log_dir: Path) -> list[dict]:
    """Run once; return one sample row per model used."""
    run_tag = f"{label}_r{repeat}"
    # Build the config WITHOUT mock enforcement so per-role model names (and
    # any model-swap overrides) survive; mock only swaps the client.
    cfg = build_config(base_config, mock=False, solver=solver,
                       overrides=overrides, log_dir=log_dir / run_tag)
    clients = make_clients(cfg, mock=mock)
    t0 = time.time()
    res = solve_multi(prompt, cfg=cfg, clients=clients, domain=domain,
                      tag=run_tag, progress=None)
    elapsed = time.time() - t0
    pm = per_model_cost(clients)

    rows: list[dict] = []
    for model, d in pm.items():
        if d["calls"] == 0 and d["input_tokens"] == 0 and d["output_tokens"] == 0:
            continue  # role never fired (e.g. synthesizer in a single-block solve)
        rows.append({
            "run_tag": run_tag,
            "label": label,
            "repeat": repeat,
            "overrides": overrides,
            "model": model,
            "input_tokens": d["input_tokens"],
            "output_tokens": d["output_tokens"],
            "calls": d["calls"],
            "usd": round(d["usd"], 8),
            "usd_table": round(table_usd(model, d["input_tokens"],
                                         d["output_tokens"]), 8),
            "elapsed_s": round(elapsed, 2),
        })
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    total_usd = sum(r["usd"] for r in rows)
    print(f"  [{run_tag}] models={[r['model'] for r in rows]} "
          f"tokens(in/out)={total_in}/{total_out} "
          f"${total_usd:.4f} {elapsed:.1f}s")
    return rows


# ===========================================================================
# Fitting all models + aggregate
# ===========================================================================

def fit_all(samples: list[dict], models: list[str] | None,
            target: str) -> dict:
    by_model: dict[str, list[dict]] = {}
    for s in samples:
        by_model.setdefault(s["model"], []).append(s)
    if models:
        by_model = {m: v for m, v in by_model.items() if m in models}

    per_model_fit = {m: fit_one(v, target=target) for m, v in by_model.items()}

    # Aggregate blended fit: total_cost ≈ A·total_in + B·total_out, per RUN.
    by_run: dict[str, dict] = {}
    for s in samples:
        r = by_run.setdefault(s["run_tag"],
                              {"in": 0.0, "out": 0.0, "usd": 0.0,
                               "usd_table": 0.0})
        r["in"] += s["input_tokens"]
        r["out"] += s["output_tokens"]
        r["usd"] += s["usd"]
        r["usd_table"] += s["usd_table"]
    runs = list(by_run.values())
    ycol = "usd_table" if target == "usd_table" else "usd"
    rows2 = [[r["in"], r["out"]] for r in runs]
    y = [r[ycol] for r in runs]
    cblend = lstsq(rows2, y) if len(runs) >= 2 else None
    blended = None
    if cblend is not None:
        A, B = cblend
        blended = {
            "A_per_input_token": A,
            "B_per_output_token": B,
            "blended_price_in_per_mtok": A * 1_000_000.0,
            "blended_price_out_per_mtok": B * 1_000_000.0,
            "r2": r_squared(rows2, y, cblend),
            "max_abs_residual_usd": max_abs_resid(rows2, y, cblend),
            "n_runs": len(runs),
            "note": ("A single blended rate is only valid for the model mix "
                     "that produced these runs; r2<1 means the mix shifted "
                     "across parameters and per-model coefficients are "
                     "required for accurate estimates."),
        }

    return {
        "target": target,
        "models": list(per_model_fit.keys()),
        "per_model": per_model_fit,
        "blended_aggregate": blended,
        "combined_equation": _combined_equation(per_model_fit),
    }


def _combined_equation(per_model_fit: dict) -> str:
    terms = []
    for m, f in per_model_fit.items():
        fit = f.get("fit", {})
        if "a_per_input_token" in fit:
            terms.append(
                f"[{m}] {fit['a_per_input_token']:.3e}·in_{_short(m)} "
                f"+ {fit['b_per_output_token']:.3e}·out_{_short(m)}")
    return "total_cost_usd ≈ " + "  +  ".join(terms) if terms else "n/a"


def _short(model: str) -> str:
    for tag in ("opus", "sonnet", "haiku"):
        if tag in model:
            return tag
    return model.replace("claude-", "").replace("-", "")[:6]


# ===========================================================================
# Reporting
# ===========================================================================

def write_outputs(samples: list[dict], fit: dict, out_dir: Path,
                  meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_model.json").write_text(
        json.dumps({"meta": meta, "fit": fit, "n_samples": len(samples)},
                   indent=2))

    # samples CSV
    with (out_dir / "cost_samples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_tag", "label", "model", "input_tokens",
                    "output_tokens", "calls", "usd", "usd_table"])
        for s in samples:
            w.writerow([s["run_tag"], s["label"], s["model"],
                        s["input_tokens"], s["output_tokens"], s["calls"],
                        s["usd"], s["usd_table"]])

    # markdown report
    lines = ["# Cost-estimation equation\n",
             f"- fit target: `{fit['target']}`"
             + ("  (imputed from price table — mock/offline)"
                if fit["target"] == "usd_table" else "  (logged usd)"),
             f"- samples: {len(samples)}   models: {', '.join(fit['models'])}\n",
             "## Per-model equations  (cost_usd ≈ a·input + b·output)\n",
             "| model | n | fitted $/Mtok in | fitted $/Mtok out | "
             "reference $/Mtok in | reference out | R² | identifiable |",
             "|---|--|--|--|--|--|--|--|"]
    for m, f in fit["per_model"].items():
        ft = f.get("fit", {})
        ref = f.get("reference_price_per_mtok", {})
        if "price_in_per_mtok" in ft:
            lines.append(
                f"| {m} | {f['n_samples']} | "
                f"{ft['price_in_per_mtok']:.4f} | {ft['price_out_per_mtok']:.4f} | "
                f"{ref.get('in', '—')} | {ref.get('out', '—')} | "
                f"{ft['r2']:.4f} | {ft.get('identifiable')} |")
        else:
            lines.append(f"| {m} | {f['n_samples']} | (under-determined) | | "
                         f"{ref.get('in', '—')} | {ref.get('out', '—')} | | |")
    lines.append("\n### Equations\n")
    for m, f in fit["per_model"].items():
        ft = f.get("fit", {})
        if "equation" in ft:
            lines.append(f"- **{m}**: `{ft['equation']}`  (R²={ft['r2']:.4f})")
    lines.append(f"\n**Combined:** `{fit['combined_equation']}`\n")

    bl = fit.get("blended_aggregate")
    if bl:
        lines += [
            "## Blended single-rate estimate (mix-agnostic)\n",
            f"`total_cost ≈ {bl['A_per_input_token']:.3e}·total_in + "
            f"{bl['B_per_output_token']:.3e}·total_out`",
            f"- blended $/Mtok: in={bl['blended_price_in_per_mtok']:.4f}, "
            f"out={bl['blended_price_out_per_mtok']:.4f}",
            f"- R²={bl['r2']:.4f} over {bl['n_runs']} runs, "
            f"max residual ${bl['max_abs_residual_usd']:.4f}",
            f"- {bl['note']}\n",
        ]
    (out_dir / "cost_model.md").write_text("\n".join(lines) + "\n")


def print_summary(fit: dict) -> None:
    print("\n" + "=" * 78)
    print(f"COST EQUATION  (target={fit['target']})")
    for m, f in fit["per_model"].items():
        ft = f.get("fit", {})
        if "equation" in ft:
            ident = "" if ft.get("identifiable", True) else "  [!collinear]"
            print(f"  {m:22s} {ft['equation']}  R²={ft['r2']:.4f}{ident}")
            ref = f.get("reference_price_per_mtok", {})
            print(f"  {'':22s}   fitted $/Mtok in={ft['price_in_per_mtok']:.3f} "
                  f"out={ft['price_out_per_mtok']:.3f}   "
                  f"(table in={ref.get('in')} out={ref.get('out')})")
        else:
            print(f"  {m:22s} under-determined (need ≥2 varied samples)")
    bl = fit.get("blended_aggregate")
    if bl:
        print(f"  blended single-rate R²={bl['r2']:.4f} "
              f"(in={bl['blended_price_in_per_mtok']:.3f}, "
              f"out={bl['blended_price_out_per_mtok']:.3f} $/Mtok)")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("concord/config/wide_and_deep.yaml"))
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--question-id", type=str, default="uci_to_fen_easy_6")
    p.add_argument("--trace", type=Path,
                   default=Path("verification/uci_to_fen_easy_6_trace.json"),
                   help="used only to truncate the move list (--max-plies)")
    p.add_argument("--sweep", type=Path, default=None,
                   help="YAML with a `runs:` list of {label, overrides, "
                        "max_plies?}. Without it, a small built-in grid runs.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=24,
                   help="global truncation (a run's own max_plies overrides it)")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated model set to fit (default: all models "
                        "seen in the logs)")
    p.add_argument("--solver", type=str, default=None, choices=("mcts", "pipeline"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--target", type=str, default="auto",
                   choices=("auto", "usd", "usd_table"),
                   help="cost column to fit. 'auto' uses logged usd unless all "
                        "are 0 (mock), then imputes from the price table.")
    p.add_argument("--fit-only", type=Path, default=None,
                   help="skip collection; fit an existing cost_samples.jsonl")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- gather samples (collect, or load for --fit-only) ----
    if args.fit_only is not None:
        samples = [json.loads(l) for l in args.fit_only.read_text().splitlines()
                   if l.strip()]
        out_dir = args.out or args.fit_only.parent
        meta = {"mode": "fit-only", "source": str(args.fit_only),
                "n_samples": len(samples)}
        print(f"[cost_model] fit-only: {len(samples)} samples from {args.fit_only}")
    else:
        trace = json.loads(args.trace.read_text())
        moves = trace.get("moves_uci") or [s["move_uci"] for s in trace["steps"]]
        q = load_question(args.dataset, args.question_id)

        if args.sweep is not None:
            spec = yaml.safe_load(args.sweep.read_text())
            runs = spec.get("runs") or [{"label": "baseline", "overrides": {}}]
            args.repeats = spec.get("repeats", args.repeats)
        else:
            runs = _builtin_grid()

        out_dir = args.out or Path("results/cost_model") / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "cost_samples.jsonl"
        print(f"[cost_model] {len(runs)} param combos x {args.repeats} repeats "
              f"-> {len(runs) * args.repeats} runs   mock={args.mock}")
        print(f"[cost_model] output dir: {out_dir}")

        samples = []
        for run in runs:
            label = run.get("label", "run")
            overrides = run.get("overrides", {}) or {}
            mp = run.get("max_plies", args.max_plies)
            prompt = (truncate_prompt(q["prompt"], moves, mp)
                      if mp is not None else q["prompt"])
            for rep in range(1, args.repeats + 1):
                try:
                    rows = collect_run(
                        label=label, repeat=rep, overrides=overrides,
                        base_config=args.config, mock=args.mock,
                        solver=args.solver, prompt=prompt, domain=q["domain"],
                        log_dir=out_dir / "runs")
                except Exception as e:                              # noqa: BLE001
                    import traceback
                    print(f"  [run {label}_r{rep} FAILED] {e}")
                    traceback.print_exc()
                    rows = []
                for r in rows:
                    r["max_plies"] = mp
                samples.extend(rows)
                with samples_path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
        meta = {"mode": "collect", "question_id": args.question_id,
                "base_config": str(args.config), "mock": args.mock,
                "repeats": args.repeats, "started_at": ts,
                "n_runs": len(runs) * args.repeats}

    if not samples:
        raise SystemExit("[cost_model] no samples collected — nothing to fit.")

    # ---- choose target ----
    target = args.target
    if target == "auto":
        any_nonzero = any(float(s.get("usd", 0.0)) > 0 for s in samples)
        target = "usd" if any_nonzero else "usd_table"
        if target == "usd_table":
            print("[cost_model] all logged usd are 0 (mock/offline) -> fitting "
                  "imputed 'usd_table' from the price table.")
    meta["fit_target"] = target

    # ---- fit + report ----
    fit = fit_all(samples, models, target)
    write_outputs(samples, fit, out_dir, meta)
    print_summary(fit)
    print("\n" + "#" * 78)
    print(f"[cost_model] equation : {out_dir}/cost_model.json")
    print(f"[cost_model] report   : {out_dir}/cost_model.md")
    print(f"[cost_model] samples  : {out_dir}/cost_samples.csv")


# Variations of the INDEPENDENT solver parameters (the key_params knobs). These
# — not the problem size or per-model output caps — are what we vary to spread
# token volume. Each entry is a set of dotted overrides applied on top of the
# baseline; `max_plies` is held FIXED (a cost bound, not an independent var).
_KNOB_VARIATIONS: list[dict] = [
    {},                                                         # baseline
    {"pipeline.K_executor": 1, "pipeline.K_combiner": 1},
    {"pipeline.K_executor": 5, "pipeline.K_combiner": 5,
     "pipeline.max_node_calls": 200},
    {"pipeline.max_split_depth": 2},
    {"pipeline.max_split_depth": 4, "pipeline.max_node_calls": 200},
    {"pipeline.max_atoms_per_split": 5, "pipeline.max_node_calls": 200},
    {"mcts.N": 128, "pipeline.max_node_calls": 200},
]


def _builtin_grid() -> list[dict]:
    """Default sweep: vary the INDEPENDENT solver parameters (key_params knobs)
    to spread token volume, and route the executor / splitter / combiner (the
    dominant token consumers) through each model in the set so every model
    accrues usage for its own cost equation. Problem size (max_plies) is fixed
    by the CLI; it is NOT an independent variable here."""
    grid: list[dict] = []
    for model in ("claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"):
        for i, kv in enumerate(_KNOB_VARIATIONS):
            ov = dict(kv)
            ov["models.execution.model"] = model
            ov["models.splitter.model"] = model
            ov["models.combiner.model"] = model
            grid.append({"label": f"{_short(model)}_v{i}", "overrides": ov})
    return grid


if __name__ == "__main__":
    main()
