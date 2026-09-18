"""Cost / token parameter study on the `uci_to_fen_easy_6` chess problem.

This is a NEW experiment runner (it does not modify run_all.py). Its job is to
answer two questions the strategy review posed:

  1. Does the Split→Solve→Combine→Verify pipeline actually solve the chess
     problem *at every decomposition level* — i.e. are the individual
     sub-problem (per move-chunk) answers correct, not just the final FEN?
  2. How does the score trade off against cost? For a grid of parameters it
     records the $ / tokens each run consumes, so we can see how accuracy
     varies with the knobs and what the MAXIMUM token spend is to reach a
     given level of success.

For each parameter combination it:
  * starts from `wide_and_deep.yaml` (the production pipeline preset),
  * applies the combination's dotted overrides (e.g. `pipeline.K_executor: 5`,
    `mcts.N: 128`, `models.execution.model: claude-opus-4-8`),
  * solves the problem with `solve_multi`,
  * grades the run with `chess_grader` against the python-chess ground-truth
    trace (overall FEN, per-subproblem pass/fail, per-level pass rate,
    decomposition fidelity),
  * records the resolved parameters + role models + cost (calls / input
    tokens / output tokens / $) + scores,
  * prints a one-line report, and
  * appends to a study JSONL.

At the end it writes a study summary (JSON + CSV + Markdown) including a
"tokens required to reach a success level" table.

USAGE
-----
Smoke test (no API key needed; mock LLM, truncated move list):

    .venv/bin/python -m concord.experiments.chess_study --mock \\
        --max-plies 8 --sweep concord/config/chess_study.sweep.yaml

Real study (needs ANTHROPIC_API_KEY; start truncated to control spend):

    .venv/bin/python -m concord.experiments.chess_study \\
        --config concord/config/wide_and_deep.yaml \\
        --sweep concord/config/chess_study.sweep.yaml \\
        --max-plies 40

Drop `--max-plies` for the full 650-ply problem (expensive).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

# Make `core`, `chess_grader` importable however this file is launched.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # concord/  -> `import core`
sys.path.insert(0, str(_HERE.parent))          # experiments/ -> `import chess_grader`

import yaml                                      # noqa: E402

import chess_grader as G                         # noqa: E402
from core.config import Config, LLMCfg          # noqa: E402
from core.llm.factory import RoleClients        # noqa: E402
from core.multi_solve import solve_multi        # noqa: E402


# Roles surfaced in every run record (resolved provider/model after overrides).
_REPORT_ROLES = ("execution", "splitter", "combiner", "verification",
                 "synthesizer", "synth_verifier", "decomposition",
                 "classification")

# "Key" config knobs pulled flat into the summary CSV so a sweep is scannable.
def _key_params(cfg: Config) -> dict[str, Any]:
    return {
        "solver": getattr(cfg, "solver", "mcts"),
        "mcts.N": cfg.mcts.N,
        "pipeline.K_executor": cfg.pipeline.K_executor,
        "pipeline.K_combiner": cfg.pipeline.K_combiner,
        "pipeline.max_split_depth": cfg.pipeline.max_split_depth,
        "pipeline.max_atoms_per_split": cfg.pipeline.max_atoms_per_split,
        "pipeline.max_node_calls": cfg.pipeline.max_node_calls,
        "pipeline.verifier_accept": cfg.pipeline.verifier_accept,
        "pipeline.combiner_retries": cfg.pipeline.combiner_retries,
        "confidence.alpha": cfg.confidence.alpha,
        "llm.temperature": cfg.llm.temperature,
    }


# ---------------------------------------------------------------------------
# Question loading + optional truncation
# ---------------------------------------------------------------------------

_SEQ_RE = re.compile(r"(\(UCI format\):\s*)([a-h1-8nbrqkNBRQK ]+?)(\n)")


def load_question(dataset: Path, question_id: str) -> dict:
    data = json.loads(dataset.read_text())
    for q in data["questions"]:
        if q["question_id"] == question_id:
            return q
    raise SystemExit(f"question_id {question_id!r} not found in {dataset}")


def truncate_prompt(prompt: str, moves: list[str], max_plies: int) -> str:
    """Rewrite the UCI move sequence in the prompt to its first `max_plies`
    half-moves so a study can scale problem size and watch tokens grow."""
    truncated = " ".join(moves[:max_plies])
    new, n = _SEQ_RE.subn(lambda m: f"{m.group(1)}{truncated}{m.group(3)}", prompt)
    if n == 0:
        raise SystemExit("could not locate the UCI sequence to truncate")
    return new


# ---------------------------------------------------------------------------
# Config construction (base + mock + dotted overrides)
# ---------------------------------------------------------------------------

def set_dotted(cfg: Config, key: str, value: Any) -> None:
    """Set a dotted attribute path on the Config, with special handling for
    `models.<role>.<field>` (instantiate the role block from the top-level
    `llm:` defaults when it is unset, mirroring run.py)."""
    parts = key.split(".")
    if parts[0] == "models" and len(parts) == 3:
        _, role, field = parts
        existing = getattr(cfg.models, role, None)
        if existing is None:
            existing = LLMCfg(**cfg.llm.model_dump())
            existing.provider = "anthropic"
        setattr(existing, field, value)
        setattr(cfg.models, role, existing)
        return
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def build_config(base_path: Path, *, mock: bool, solver: str | None,
                 overrides: dict[str, Any], log_dir: Path) -> Config:
    cfg = Config.from_yaml(base_path)
    if solver:
        cfg.solver = solver
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    if mock:
        # Force every role to the deterministic mock (no API), applied LAST so
        # that model-swap overrides (models.<role>.model) collapse to mock for
        # an offline smoke test while pipeline/mcts/etc. overrides still apply.
        # Clearing the per-role models makes RoleClients resolve all roles to
        # one shared mock client, exactly like run.py --mock.
        cfg.llm = LLMCfg(provider="mock", model="mock-v0",
                         temperature=cfg.llm.temperature)
        cfg.models = type(cfg.models)()
    cfg.telemetry.log_dir = str(log_dir)
    return cfg


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(*, label: str, repeat: int, overrides: dict[str, Any],
            base_config: Path, mock: bool, solver: str | None,
            prompt: str, domain: str, trace: dict, max_plies: int | None,
            out_dir: Path, vary: str | None = None,
            value: Any = None) -> dict:
    run_tag = f"{label}_r{repeat}"
    log_dir = out_dir / "runs" / run_tag
    cfg = build_config(base_config, mock=mock, solver=solver,
                       overrides=overrides, log_dir=log_dir)
    role_models = {r: f"{cfg.role_model(r).provider}/{cfg.role_model(r).model}"
                   for r in _REPORT_ROLES}

    clients = RoleClients.from_config(cfg)
    t0 = time.time()
    res = solve_multi(prompt, cfg=cfg, clients=clients, domain=domain,
                      tag=run_tag, progress=None)
    elapsed = time.time() - t0

    blocks = G.blocks_from_result_cost(res.cost)
    grade = G.grade_run(trace=trace, blocks=blocks,
                        final_answer=res.answer, max_plies=max_plies)

    cost = res.cost or {}
    in_tok = int(cost.get("input_tokens", 0))
    out_tok = int(cost.get("output_tokens", 0))
    record = {
        "label": label,
        "repeat": repeat,
        "run_tag": run_tag,
        "vary": vary,                       # which key parameter this run varies
        "value": value,                     # the value it was set to (OFAT)
        "max_plies": max_plies,             # problem size used for THIS run
        "overrides": overrides,
        "key_params": _key_params(cfg),
        "role_models": role_models,
        "cost": {
            "calls": int(cost.get("calls", 0)),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "usd": float(cost.get("usd", 0.0)),
        },
        "per_role_cost": cost.get("per_role"),
        "elapsed_s": round(elapsed, 2),
        "rollouts": res.rollouts,
        "flagged": res.flagged,
        "answer": res.answer,
        # grade summary (full per-atom detail saved to the per-run file)
        "overall_correct": grade["overall_correct"],
        "overall_board_match": grade["overall_board_match"],
        "atom_pass_rate": grade["atom_pass_rate"],
        "n_atoms": grade["n_atoms"],
        "n_gradeable": grade["n_gradeable"],
        "n_pass": grade["n_pass"],
        "n_fail": grade["n_fail"],
        "n_ungradeable": grade["n_ungradeable"],
        "decomposition_fidelity": grade["decomposition_fidelity"],
        "levels": grade["levels"],
        "trace_path": res.trace_path,
        "run_dir": str(log_dir),
    }

    # Per-run file holds the full grade (every atom) for deep inspection.
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "grade.json").write_text(json.dumps(grade, indent=2))
    (log_dir / "record.json").write_text(json.dumps(record, indent=2))

    _print_run_report(record)
    return record


def _print_run_report(rec: dict) -> None:
    c = rec["cost"]
    apr = rec["atom_pass_rate"]
    apr_s = f"{apr:.3f}" if apr is not None else "n/a"
    df = rec["decomposition_fidelity"]
    df_s = f"{df:.3f}" if df is not None else "n/a"
    print("=" * 78)
    print(f"RUN {rec['run_tag']}   overrides={rec['overrides'] or '(base)'}")
    print(f"  models: exec={rec['role_models']['execution']}  "
          f"split={rec['role_models']['splitter']}  "
          f"comb={rec['role_models']['combiner']}  "
          f"verify={rec['role_models']['verification']}")
    kp = rec["key_params"]
    print(f"  params: N={kp['mcts.N']} K_exec={kp['pipeline.K_executor']} "
          f"K_comb={kp['pipeline.K_combiner']} "
          f"split_depth={kp['pipeline.max_split_depth']} "
          f"node_calls={kp['pipeline.max_node_calls']} "
          f"accept={kp['pipeline.verifier_accept']}")
    print(f"  COST:   calls={c['calls']}  tokens={c['total_tokens']} "
          f"(in {c['input_tokens']} / out {c['output_tokens']})  "
          f"${c['usd']:.4f}  {rec['elapsed_s']}s")
    print(f"  SCORE:  overall_correct={rec['overall_correct']} "
          f"(board_match={rec['overall_board_match']})  "
          f"subproblems {rec['n_pass']}/{rec['n_gradeable']} pass "
          f"(rate={apr_s}), {rec['n_ungradeable']} ungradeable  "
          f"decomp_fidelity={df_s}")
    levels = rec["levels"]
    if levels:
        lv = "  ".join(
            f"d{d}:{b['pass']}/{b['gradeable']}"
            + (f"={b['pass_rate']:.2f}" if b["pass_rate"] is not None else "")
            for d, b in levels.items())
        print(f"  LEVELS: {lv}")
    if rec["flagged"]:
        print(f"  FLAG:   {rec['flagged']}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _tokens_to_success(records: list[dict]) -> dict:
    """For a set of success thresholds, report which runs reached them and the
    token spend required (min / median / max). Answers 'what is the maximum
    token budget needed to hit a given level of success?'."""
    def stats(rs: list[dict]) -> dict | None:
        if not rs:
            return None
        toks = sorted(r["cost"]["total_tokens"] for r in rs)
        usds = sorted(r["cost"]["usd"] for r in rs)
        return {
            "n_runs": len(rs),
            "min_tokens": toks[0],
            "median_tokens": int(median(toks)),
            "max_tokens": toks[-1],
            "min_usd": round(usds[0], 4),
            "max_usd": round(usds[-1], 4),
            "cheapest_run": min(rs, key=lambda r: r["cost"]["total_tokens"])["run_tag"],
        }

    levels = {
        "overall_correct": [r for r in records if r["overall_correct"]],
        "subproblems>=1.0": [r for r in records
                             if (r["atom_pass_rate"] or 0) >= 1.0],
        "subproblems>=0.9": [r for r in records
                             if (r["atom_pass_rate"] or 0) >= 0.9],
        "subproblems>=0.75": [r for r in records
                              if (r["atom_pass_rate"] or 0) >= 0.75],
        "subproblems>=0.5": [r for r in records
                             if (r["atom_pass_rate"] or 0) >= 0.5],
    }
    return {k: stats(v) for k, v in levels.items()}


def _token_score_frontier(records: list[dict]) -> list[dict]:
    """Sort by token spend ascending; track the best subproblem pass-rate
    achievable at or below each token budget (the cost/accuracy frontier)."""
    rs = sorted(records, key=lambda r: r["cost"]["total_tokens"])
    best = -1.0
    frontier: list[dict] = []
    for r in rs:
        apr = r["atom_pass_rate"] or 0.0
        if apr > best:
            best = apr
            frontier.append({
                "run_tag": r["run_tag"],
                "total_tokens": r["cost"]["total_tokens"],
                "usd": r["cost"]["usd"],
                "atom_pass_rate": r["atom_pass_rate"],
                "overall_correct": r["overall_correct"],
            })
    return frontier


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _parameter_effects(records: list[dict]) -> dict:
    """Group runs by the parameter they vary, then by value, and report how the
    SOLVABILITY RATIO and COST move with that parameter.

    solvability_ratio = (# runs with overall_correct) / (# repeats at that
    value) — that's why --repeats matters. Also reports the mean subproblem
    pass-rate (a finer-grained solvability signal), mean tokens and mean $."""
    by_param: dict[str, dict[Any, list[dict]]] = {}
    for r in records:
        param = r.get("vary")
        if param is None:
            continue
        by_param.setdefault(param, {}).setdefault(r.get("value"), []).append(r)

    out: dict[str, list[dict]] = {}
    for param, by_val in by_param.items():
        rows = []
        for val, rs in by_val.items():
            n = len(rs)
            solved = sum(1 for r in rs if r.get("overall_correct"))
            rows.append({
                "value": val,
                "n_repeats": n,
                "solvability_ratio": solved / n if n else None,
                "mean_subproblem_pass_rate":
                    _mean([r.get("atom_pass_rate") for r in rs]),
                "mean_total_tokens":
                    _mean([r["cost"]["total_tokens"] for r in rs]),
                "mean_usd": _mean([r["cost"]["usd"] for r in rs]),
                "mean_calls": _mean([r["cost"]["calls"] for r in rs]),
            })
        # sort by value numerically when possible
        try:
            rows.sort(key=lambda x: float(x["value"]))
        except (TypeError, ValueError):
            rows.sort(key=lambda x: str(x["value"]))
        out[param] = rows
    return out


def write_summary(records: list[dict], out_dir: Path, meta: dict) -> None:
    summary = {
        "meta": meta,
        "n_runs": len(records),
        "parameter_effects": _parameter_effects(records),
        "tokens_to_success_level": _tokens_to_success(records),
        "token_score_frontier": _token_score_frontier(records),
        "runs": records,
    }
    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))

    # CSV — one row per run, key params + cost + scores flat.
    csv_path = out_dir / "study_summary.csv"
    param_keys = list(records[0]["key_params"].keys()) if records else []
    fields = (["run_tag", "label", "repeat"] + param_keys
              + ["exec_model", "splitter_model", "combiner_model",
                 "verification_model", "calls", "input_tokens",
                 "output_tokens", "total_tokens", "usd", "elapsed_s",
                 "rollouts", "overall_correct", "overall_board_match",
                 "atom_pass_rate", "n_pass", "n_gradeable", "n_ungradeable",
                 "decomposition_fidelity", "flagged"])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {"run_tag": r["run_tag"], "label": r["label"],
                   "repeat": r["repeat"]}
            row.update(r["key_params"])
            row.update({
                "exec_model": r["role_models"]["execution"],
                "splitter_model": r["role_models"]["splitter"],
                "combiner_model": r["role_models"]["combiner"],
                "verification_model": r["role_models"]["verification"],
                "calls": r["cost"]["calls"],
                "input_tokens": r["cost"]["input_tokens"],
                "output_tokens": r["cost"]["output_tokens"],
                "total_tokens": r["cost"]["total_tokens"],
                "usd": r["cost"]["usd"],
                "elapsed_s": r["elapsed_s"],
                "rollouts": r["rollouts"],
                "overall_correct": r["overall_correct"],
                "overall_board_match": r["overall_board_match"],
                "atom_pass_rate": r["atom_pass_rate"],
                "n_pass": r["n_pass"],
                "n_gradeable": r["n_gradeable"],
                "n_ungradeable": r["n_ungradeable"],
                "decomposition_fidelity": r["decomposition_fidelity"],
                "flagged": r["flagged"],
            })
            w.writerow(row)

    # Markdown report.
    md = out_dir / "study_report.md"
    lines: list[str] = []
    lines.append(f"# Chess uci_to_fen parameter study\n")
    lines.append(f"- problem: `{meta['question_id']}`")
    lines.append(f"- base config: `{meta['base_config']}`")
    lines.append(f"- max_plies: {meta['max_plies']}  (gold FEN: `{meta['gold_fen']}`)")
    lines.append(f"- mock: {meta['mock']}   runs: {len(records)}\n")

    # Parameter-effect tables come first — this is the study's headline:
    # solvability ratio and cost vs each independent parameter.
    pe = _parameter_effects(records)
    if pe:
        lines.append("## Solvability ratio & cost vs each parameter\n")
        lines.append("Each parameter is swept alone (others at baseline); "
                     "solvability_ratio = solved / repeats.\n")
        for param, rows in pe.items():
            lines.append(f"### `{param}`\n")
            lines.append("| value | n | solvability ratio | mean subproblem "
                         "pass-rate | mean tokens | mean $ |")
            lines.append("|---|--|--|--|--|--|")
            for r in rows:
                sr = r["solvability_ratio"]
                mp = r["mean_subproblem_pass_rate"]
                mt = r["mean_total_tokens"]
                mu = r["mean_usd"]
                lines.append(
                    f"| {r['value']} | {r['n_repeats']} | "
                    f"{sr:.2f} | "
                    + (f"{mp:.3f}" if mp is not None else "n/a") + " | "
                    + (f"{mt:.0f}" if mt is not None else "n/a") + " | "
                    + (f"{mu:.4f}" if mu is not None else "n/a") + " |")
            lines.append("")

    lines.append("## Per-run results\n")
    lines.append("| run | exec model | N | K_exec | split_depth | tokens | $ | "
                 "overall | subproblem pass | decomp fidelity |")
    lines.append("|---|---|--|--|--|--|--|--|--|--|")
    for r in records:
        kp = r["key_params"]
        apr = r["atom_pass_rate"]
        apr_s = f"{r['n_pass']}/{r['n_gradeable']}" + (
            f" ({apr:.2f})" if apr is not None else "")
        df = r["decomposition_fidelity"]
        df_s = f"{df:.2f}" if df is not None else "n/a"
        lines.append(
            f"| {r['run_tag']} | {r['role_models']['execution']} | "
            f"{kp['mcts.N']} | {kp['pipeline.K_executor']} | "
            f"{kp['pipeline.max_split_depth']} | {r['cost']['total_tokens']} | "
            f"{r['cost']['usd']:.4f} | "
            f"{'✓' if r['overall_correct'] else '✗'} | {apr_s} | {df_s} |")
    lines.append("\n## Tokens required to reach a success level\n")
    lines.append("| success level | runs reaching it | min tokens | median | "
                 "max tokens | max $ | cheapest run |")
    lines.append("|---|--|--|--|--|--|---|")
    for lvl, s in _tokens_to_success(records).items():
        if s is None:
            lines.append(f"| {lvl} | 0 | — | — | — | — | — |")
        else:
            lines.append(
                f"| {lvl} | {s['n_runs']} | {s['min_tokens']} | "
                f"{s['median_tokens']} | {s['max_tokens']} | {s['max_usd']} | "
                f"{s['cheapest_run']} |")
    lines.append("\n## Cost / accuracy frontier (best pass-rate at or below a "
                 "token budget)\n")
    lines.append("| run | total tokens | $ | subproblem pass-rate | overall |")
    lines.append("|---|--|--|--|--|")
    for fr in _token_score_frontier(records):
        apr = fr["atom_pass_rate"]
        apr_s = f"{apr:.3f}" if apr is not None else "n/a"
        ok_s = "✓" if fr["overall_correct"] else "✗"
        lines.append(
            f"| {fr['run_tag']} | {fr['total_tokens']} | {fr['usd']:.4f} | "
            f"{apr_s} | {ok_s} |")
    md.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# One-factor-at-a-time (OFAT) sweep over the key_params knobs
# ---------------------------------------------------------------------------

# 5 values per independent parameter (the knobs in _key_params). Each is swept
# on its own, holding the others at the wide_and_deep baseline, so the effect
# of THAT parameter on solvability ratio and cost is isolated.
PARAM_LEVELS: dict[str, list[Any]] = {
    "mcts.N":                       [16, 32, 64, 128, 256],
    "pipeline.K_executor":          [1, 2, 3, 5, 8],
    "pipeline.K_combiner":          [1, 2, 3, 5, 8],
    "pipeline.max_split_depth":     [1, 2, 3, 4, 6],
    "pipeline.max_atoms_per_split": [2, 3, 4, 5, 6],
    "pipeline.max_node_calls":      [50, 100, 150, 200, 300],
    "pipeline.verifier_accept":     [0.50, 0.60, 0.75, 0.85, 0.95],
    "pipeline.combiner_retries":    [0, 1, 2, 3, 4],
    "confidence.alpha":             [0.0, 0.10, 0.25, 0.50, 0.75],
    "llm.temperature":              [0.0, 0.30, 0.50, 0.70, 1.0],
    # `solver` is categorical with only two meaningful values (the legacy MCTS
    # path produces no pipeline provenance, so per-subproblem metrics are n/a
    # for it; overall solvability is still graded).
    "solver":                       ["pipeline", "mcts"],
}


def ofat_runs(params: list[str] | None = None) -> list[dict]:
    """Build the OFAT run list: one run per (parameter, value)."""
    chosen = params or list(PARAM_LEVELS)
    runs: list[dict] = []
    for param in chosen:
        for value in PARAM_LEVELS[param]:
            runs.append({
                "label": f"{param}={value}",
                "vary": param,
                "value": value,
                "overrides": ({} if param == "solver" else {param: value}),
                **({"solver": value} if param == "solver" else {}),
            })
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("concord/config/wide_and_deep.yaml"),
                   help="base YAML config (parameters are read from here)")
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--question-id", type=str, default="uci_to_fen_easy_6")
    p.add_argument("--trace", type=Path,
                   default=Path("verification/uci_to_fen_easy_6_trace.json"))
    p.add_argument("--sweep", type=Path, default=None,
                   help="YAML/JSON sweep spec with a `runs:` list of "
                        "{label, overrides, vary?, value?}. If omitted (and no "
                        "--ofat), runs the base config once as 'baseline'.")
    p.add_argument("--ofat", action="store_true",
                   help="one-factor-at-a-time sweep over the key_params knobs: "
                        "5 values per parameter (see PARAM_LEVELS), each varied "
                        "alone. This is the parameter study — solvability ratio "
                        "and cost vs each independent parameter.")
    p.add_argument("--params", type=str, default=None,
                   help="comma-separated subset of key params to OFAT-sweep "
                        "(default: all). e.g. 'mcts.N,pipeline.K_executor'")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat each parameter VALUE N times. Needed for a "
                        "meaningful solvability RATIO (= solved / repeats) at "
                        "temperature>0.")
    p.add_argument("--max-plies", type=int, default=None,
                   help="truncate the move sequence to the first N half-moves "
                        "(gold FEN taken at ply N). Controls study cost.")
    p.add_argument("--solver", type=str, default=None,
                   choices=("mcts", "pipeline"),
                   help="override cfg.solver (default = the YAML's value)")
    p.add_argument("--mock", action="store_true",
                   help="force the mock LLM for all roles (no API; plumbing test)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default results/chess_study/<ts>)")
    args = p.parse_args()

    trace = json.loads(args.trace.read_text())
    gt = G.GroundTruth(trace)
    q = load_question(args.dataset, args.question_id)
    prompt = q["prompt"]
    if args.max_plies is not None:
        prompt = truncate_prompt(prompt, gt.moves, args.max_plies)
    gold_fen = gt.gold_fen(args.max_plies)

    # Build the run list: OFAT over key params, an explicit sweep, or a single
    # baseline run.
    if args.ofat:
        params = [s.strip() for s in args.params.split(",")] if args.params else None
        runs = ofat_runs(params)
    elif args.sweep is not None:
        spec = yaml.safe_load(args.sweep.read_text())
        runs = spec.get("runs") or [{"label": "baseline", "overrides": {}}]
        if "max_plies" in spec and args.max_plies is None:
            args.max_plies = spec["max_plies"]
            prompt = truncate_prompt(q["prompt"], gt.moves, args.max_plies)
            gold_fen = gt.gold_fen(args.max_plies)
        if "repeats" in spec:
            args.repeats = spec.get("repeats", args.repeats)
    else:
        runs = [{"label": "baseline", "overrides": {}}]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or Path("results/chess_study") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    study_jsonl = out_dir / "runs.jsonl"

    meta = {
        "question_id": args.question_id,
        "base_config": str(args.config),
        "solver_override": args.solver,
        "mock": args.mock,
        "max_plies": args.max_plies,
        "gold_fen": gold_fen,
        "repeats": args.repeats,
        "started_at": ts,
        "n_combinations": len(runs),
    }
    print(f"[chess_study] {len(runs)} param combinations x {args.repeats} "
          f"repeats -> {len(runs) * args.repeats} runs")
    print(f"[chess_study] problem={args.question_id} max_plies={args.max_plies} "
          f"mock={args.mock}")
    print(f"[chess_study] gold FEN: {gold_fen}")
    print(f"[chess_study] output dir: {out_dir}")

    records: list[dict] = []
    for run in runs:
        label = run.get("label", "run")
        overrides = run.get("overrides", {}) or {}
        vary = run.get("vary")
        value = run.get("value")
        # A run may pin its own solver (the OFAT `solver` factor); else use CLI.
        run_solver = run.get("solver", args.solver)
        # Per-run ply truncation. A PLIES sweep sets `vary: max_plies` /
        # `value: <n>` (or an explicit per-run `max_plies:` key) so a single
        # sweep can vary problem size and watch solvability + tokens move with
        # it; the prompt is re-truncated to <n> half-moves and the run is graded
        # against the gold FEN at ply <n>. Runs that don't ask for their own
        # plies fall back to the global CLI/spec truncation (`args.max_plies`).
        run_plies = run.get("max_plies")
        if run_plies is None and vary == "max_plies":
            run_plies = value
        if run_plies is None:
            run_plies = args.max_plies
        run_plies = int(run_plies) if run_plies is not None else None
        run_prompt = (truncate_prompt(q["prompt"], gt.moves, run_plies)
                      if run_plies is not None else q["prompt"])
        for rep in range(1, args.repeats + 1):
            try:
                rec = run_one(
                    label=label, repeat=rep, overrides=overrides,
                    base_config=args.config, mock=args.mock,
                    solver=run_solver, prompt=run_prompt,
                    domain=q["domain"], trace=trace,
                    max_plies=run_plies, out_dir=out_dir,
                    vary=vary, value=value)
            except Exception as e:                                  # noqa: BLE001
                import traceback
                print(f"  [run {label}_r{rep} FAILED] {e}")
                traceback.print_exc()
                rec = {"label": label, "repeat": rep,
                       "run_tag": f"{label}_r{rep}", "error": repr(e),
                       "vary": vary, "value": value, "max_plies": run_plies,
                       "overrides": overrides,
                       "cost": {"calls": 0, "input_tokens": 0,
                                "output_tokens": 0, "total_tokens": 0,
                                "usd": 0.0},
                       "overall_correct": False, "overall_board_match": False,
                       "atom_pass_rate": None, "n_atoms": 0, "n_gradeable": 0,
                       "n_pass": 0, "n_fail": 0, "n_ungradeable": 0,
                       "decomposition_fidelity": None, "levels": {},
                       "key_params": {}, "role_models": {}, "rollouts": 0,
                       "flagged": "error", "elapsed_s": 0.0}
            records.append(rec)
            with study_jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    write_summary([r for r in records if "error" not in r] or records,
                  out_dir, meta)

    # Parameter-effect tables to the console (the study headline).
    pe = _parameter_effects([r for r in records if "error" not in r])
    if pe:
        print("\n" + "=" * 78)
        print("SOLVABILITY RATIO & COST vs PARAMETER (others held at baseline)")
        for param, rows in pe.items():
            print(f"\n  {param}:")
            print(f"    {'value':>8} | {'solv.ratio':>10} | "
                  f"{'subprob pass':>12} | {'mean tokens':>11} | {'mean $':>8}")
            for r in rows:
                mp = r["mean_subproblem_pass_rate"]
                mt = r["mean_total_tokens"]
                print(f"    {str(r['value']):>8} | {r['solvability_ratio']:>10.2f} | "
                      + (f"{mp:>12.3f}" if mp is not None else f"{'n/a':>12}") + " | "
                      + (f"{mt:>11.0f}" if mt is not None else f"{'n/a':>11}") + " | "
                      + f"{r['mean_usd']:>8.4f}")

    # Final headline.
    n_ok = sum(1 for r in records if r.get("overall_correct"))
    print("\n" + "#" * 78)
    print(f"[chess_study] DONE. {len(records)} runs, {n_ok} solved the problem "
          f"(full-FEN correct).")
    print(f"[chess_study] summary : {out_dir}/study_summary.json")
    print(f"[chess_study] csv     : {out_dir}/study_summary.csv")
    print(f"[chess_study] report  : {out_dir}/study_report.md")


if __name__ == "__main__":
    main()
