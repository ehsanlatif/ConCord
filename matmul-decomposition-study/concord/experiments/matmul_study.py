"""Matrix-chain decomposition study — the matmul analogue of chess_study.py.

Runs Concord's Split -> Solve -> Combine -> Verify pipeline on the running
matrix-product task at a fixed long horizon T, sweeping the per-step
complexity axis n = matrix dimension d (1..8), and grades each run at every
decomposition level against the frozen golden dataset while recording the
cost / tokens each run spends.

The point of the study: show whether *intelligent decomposition under a fixed
per-node token/call budget* can solve a long-horizon compounding task that a
single monolithic long-context pass cannot — and at what cost.

USAGE
-----
Offline plumbing test (mock LLM, no API, short chain):

    .venv/bin/python -m concord.experiments.matmul_study --mock \\
        --dims 1 2 --max-turns 8

Real study (needs ANTHROPIC_API_KEY):

    .venv/bin/python -m concord.experiments.matmul_study \\
        --config concord/config/matmul_per_role.yaml \\
        --dims 1 2 3 4 5 6 7 8 --max-turns 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # concord/  -> import core
sys.path.insert(0, str(_HERE.parent))          # experiments/ -> local imports

import matmul_grader as G                        # noqa: E402
from matmul_task import (                         # noqa: E402
    GroundTruth, build_problem_prompt, MATMUL_SPLITTER_SYSTEM,
    build_source_resolver, build_combine_resolver,
)
from core.config import Config, LLMCfg           # noqa: E402
from core.llm.factory import RoleClients         # noqa: E402
from core.multi_solve import solve_multi         # noqa: E402
from core.pipeline.expansion import (  # noqa: E402
    set_source_resolver, set_combine_resolver,
)


_REPORT_ROLES = ("execution", "splitter", "combiner", "verification",
                 "synthesizer", "synth_verifier", "decomposition",
                 "classification")


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
    }


# --------------------------------------------------------------------------- #
# Config construction (base + mock + dotted overrides) — mirrors chess_study
# --------------------------------------------------------------------------- #

def set_dotted(cfg: Config, key: str, value: Any) -> None:
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
        cfg.llm = LLMCfg(provider="mock", model="mock-v0",
                         temperature=cfg.llm.temperature)
        cfg.models = type(cfg.models)()
    cfg.telemetry.log_dir = str(log_dir)
    return cfg


# --------------------------------------------------------------------------- #
# Weights & Biases logging (no-op unless --wandb)
# --------------------------------------------------------------------------- #

class WandbLogger:
    """Streams per-run scalars, then a complexity curve + summary table, all
    keyed by the complexity axis n (= matrix dimension d). Mirrors the shapes
    used by the parent repo's run_matmul_dataset.py so the two are comparable:
      * run/*      — one point per (dim, repeat) as the sweep proceeds.
      * by_n/*     — solve rate / atom-pass / fidelity / tokens / $ vs n.
      * levels/*   — L1 pass-rate per split depth, per n.
      * summary/   — a bar chart of solve rate vs n + a scannable table.
    """

    def __init__(self, enabled, project, entity, run_name, group, config):
        self.enabled = enabled
        self.wandb = None
        self._step = 0
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            raise SystemExit(
                "--wandb requested but the 'wandb' package is not installed.\n"
                "Install it into Concord's venv:  pip install wandb  "
                "(then `wandb login`).")
        self.wandb = wandb
        wandb.init(project=project, entity=entity, name=run_name, group=group,
                   config=config)
        wandb.define_metric("run/global_step")
        wandb.define_metric("run/*", step_metric="run/global_step")
        wandb.define_metric("by_n/n")
        wandb.define_metric("by_n/*", step_metric="by_n/n")
        wandb.define_metric("levels/n")
        wandb.define_metric("levels/*", step_metric="levels/n")

    def log_run(self, rec: dict) -> None:
        if not self.enabled:
            return
        self._step += 1
        c = rec["cost"]
        payload = {
            "run/global_step": self._step,
            "run/n": rec["dim"],
            "run/overall_correct": int(bool(rec["overall_correct"])),
            "run/total_tokens": c["total_tokens"],
            "run/input_tokens": c["input_tokens"],
            "run/output_tokens": c["output_tokens"],
            "run/usd": c["usd"],
            "run/calls": c["calls"],
            "run/elapsed_s": rec["elapsed_s"],
            "run/n_ungradeable": rec["n_ungradeable"],
        }
        for k in ("atom_pass_rate", "decomposition_fidelity",
                  "contiguity_rate", "chain_coverage"):
            if rec.get(k) is not None:
                payload[f"run/{k}"] = rec[k]
        self.wandb.log(payload)

    def log_summary(self, by_dim: dict, records: list) -> None:
        if not self.enabled:
            return
        # Complexity curves, one point per n.
        for d, s in sorted(by_dim.items()):
            row = {"by_n/n": d, "by_n/solve_rate": s["solve_rate"],
                   "by_n/mean_total_tokens": s["mean_total_tokens"] or 0,
                   "by_n/mean_usd": s["mean_usd"] or 0,
                   "by_n/mean_calls": s["mean_calls"] or 0}
            if s["mean_atom_pass_rate"] is not None:
                row["by_n/mean_atom_pass_rate"] = s["mean_atom_pass_rate"]
            if s["mean_fidelity"] is not None:
                row["by_n/mean_fidelity"] = s["mean_fidelity"]
            self.wandb.log(row)
        # Per-depth (L3) pass rate, per n — aggregated across a dim's runs.
        depth_acc: dict[int, dict[int, list[int]]] = {}
        for r in records:
            for depth_s, lv in (r.get("levels") or {}).items():
                depth = int(depth_s)
                if lv.get("pass_rate") is None:
                    continue
                depth_acc.setdefault(r["dim"], {}).setdefault(
                    depth, []).append(lv["pass_rate"])
        for d in sorted(depth_acc):
            row = {"levels/n": d}
            for depth, vals in sorted(depth_acc[d].items()):
                row[f"levels/depth_{depth}_pass_rate"] = sum(vals) / len(vals)
            self.wandb.log(row)
        # Scannable table + a bar chart of solve rate vs n.
        table = self.wandb.Table(
            columns=["n", "solve_rate", "atom_pass_rate", "fidelity",
                     "mean_tokens", "mean_usd", "mean_calls"])
        for d, s in sorted(by_dim.items()):
            table.add_data(
                d, s["solve_rate"], s["mean_atom_pass_rate"],
                s["mean_fidelity"], s["mean_total_tokens"], s["mean_usd"],
                s["mean_calls"])
            self.wandb.summary[f"solve_rate_n{d}"] = s["solve_rate"]
        self.wandb.log({
            "summary/solve_rate_by_n": self.wandb.plot.bar(
                table, "n", "solve_rate",
                title="Solve rate vs complexity n (=d)"),
            "summary/by_n_table": table,
        })

    def finish(self) -> None:
        if self.enabled and self.wandb is not None:
            self.wandb.finish()


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #

def run_one(*, dim: int, repeat: int, sample_id: int, overrides: dict[str, Any],
            base_config: Path, mock: bool, solver: str | None,
            data_dir: Path, max_turns: int, out_dir: Path,
            index_refs: bool = True) -> dict:
    run_tag = f"d{dim}_r{repeat}"
    log_dir = out_dir / "runs" / run_tag
    gt = GroundTruth.load(data_dir, dim, sample_id, max_turns)
    prompt = build_problem_prompt(gt)

    cfg = build_config(base_config, mock=mock, solver=solver,
                       overrides=overrides, log_dir=log_dir)
    # Index-reference mode: the splitter emits `source_span` ranges instead of
    # copying matrix values, and a code resolver injects the exact values at
    # execution. Removes splitter transcription errors + token blow-up.
    if index_refs:
        cfg.pipeline.splitter_system_override = MATMUL_SPLITTER_SYSTEM
        # The length-based short-circuit would mark our short range-questions
        # atomic WITHOUT a splitter call, stopping recursion at giant leaves.
        # Disable it so ranges keep splitting until they are single matrices
        # (the splitter returns a single atom when the range is one input).
        cfg.pipeline.atom_short_circuit_chars = 0
    role_models = {r: f"{cfg.role_model(r).provider}/{cfg.role_model(r).model}"
                   for r in _REPORT_ROLES}

    clients = RoleClients.from_config(cfg)
    if index_refs:
        set_source_resolver(build_source_resolver(gt))
        # Running-state combine: a block's answer is its last child's answer, so
        # aggregate deterministically in code instead of via the LLM combiner
        # (which re-multiplies and corrupts correct chained results).
        set_combine_resolver(build_combine_resolver(gt))
    t0 = time.time()
    try:
        res = solve_multi(prompt, cfg=cfg, clients=clients, domain="cs",
                          tag=run_tag, progress=None)
    finally:
        set_source_resolver(None)   # never leak the resolver across runs
        set_combine_resolver(None)
    elapsed = time.time() - t0

    blocks = G.blocks_from_result_cost(res.cost)
    grade = G.grade_run(gt, blocks, res.answer)

    cost = res.cost or {}
    in_tok = int(cost.get("input_tokens", 0))
    out_tok = int(cost.get("output_tokens", 0))
    record = {
        "dim": dim, "repeat": repeat, "run_tag": run_tag,
        "sample_id": sample_id, "max_turns": gt.T,
        "overrides": overrides, "key_params": _key_params(cfg),
        "role_models": role_models,
        "cost": {
            "calls": int(cost.get("calls", 0)),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "usd": float(cost.get("usd", 0.0)),
        },
        "per_role_cost": cost.get("per_role"),
        "elapsed_s": round(elapsed, 2),
        "rollouts": res.rollouts, "flagged": res.flagged,
        "answer": res.answer,
        "overall_correct": grade["overall_correct"],
        "atom_pass_rate": grade["atom_pass_rate"],
        "n_atoms": grade["n_atoms"], "n_gradeable": grade["n_gradeable"],
        "n_pass": grade["n_pass"], "n_fail": grade["n_fail"],
        "n_ungradeable": grade["n_ungradeable"],
        "decomposition_fidelity": grade["decomposition_fidelity"],
        "contiguity_rate": grade["contiguity_rate"],
        "chain_coverage": grade["chain_coverage"],
        "levels": grade["levels"],
        "trace_path": res.trace_path, "run_dir": str(log_dir),
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "grade.json").write_text(json.dumps(
        {k: v for k, v in grade.items()
         if k not in ("final_answer_matrix", "gold_final")}, indent=2))
    (log_dir / "record.json").write_text(json.dumps(record, indent=2))
    _print_run_report(record)
    return record


def _print_run_report(rec: dict) -> None:
    c = rec["cost"]
    apr = rec["atom_pass_rate"]
    apr_s = f"{apr:.3f}" if apr is not None else "n/a"
    fid = rec["decomposition_fidelity"]
    fid_s = f"{fid:.3f}" if fid is not None else "n/a"
    print("=" * 78)
    print(f"RUN {rec['run_tag']}  (n=d={rec['dim']}, T={rec['max_turns']})  "
          f"overrides={rec['overrides'] or '(base)'}")
    print(f"  models: exec={rec['role_models']['execution']}  "
          f"split={rec['role_models']['splitter']}  "
          f"comb={rec['role_models']['combiner']}  "
          f"synth={rec['role_models']['synthesizer']}")
    kp = rec["key_params"]
    print(f"  params: N={kp['mcts.N']} K_exec={kp['pipeline.K_executor']} "
          f"split_depth={kp['pipeline.max_split_depth']} "
          f"node_calls={kp['pipeline.max_node_calls']}")
    print(f"  COST:   calls={c['calls']}  tokens={c['total_tokens']} "
          f"(in {c['input_tokens']} / out {c['output_tokens']})  "
          f"${c['usd']:.4f}  {rec['elapsed_s']}s")
    print(f"  SCORE:  overall_correct={rec['overall_correct']}  "
          f"atoms {rec['n_pass']}/{rec['n_gradeable']} pass (rate={apr_s}), "
          f"{rec['n_ungradeable']} ungradeable  fidelity={fid_s}")
    levels = rec["levels"]
    if levels:
        lv = "  ".join(
            f"d{d}:{b['pass']}/{b['gradeable']}"
            + (f"={b['pass_rate']:.2f}" if b["pass_rate"] is not None else "")
            for d, b in levels.items())
        print(f"  LEVELS: {lv}")
    if rec["flagged"]:
        print(f"  FLAG:   {str(rec['flagged'])[:200]}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _by_dim(records: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    by: dict[int, list[dict]] = {}
    for r in records:
        by.setdefault(r["dim"], []).append(r)
    for d, rs in sorted(by.items()):
        n = len(rs)
        out[d] = {
            "n_runs": n,
            "solve_rate": sum(1 for r in rs if r["overall_correct"]) / n,
            "mean_atom_pass_rate": _mean([r["atom_pass_rate"] for r in rs]),
            "mean_fidelity": _mean([r["decomposition_fidelity"] for r in rs]),
            "mean_total_tokens": _mean([r["cost"]["total_tokens"] for r in rs]),
            "mean_usd": _mean([r["cost"]["usd"] for r in rs]),
            "mean_calls": _mean([r["cost"]["calls"] for r in rs]),
        }
    return out


def write_summary(records: list[dict], out_dir: Path, meta: dict) -> None:
    by_dim = _by_dim(records)
    summary = {"meta": meta, "n_runs": len(records),
               "by_dim": by_dim, "runs": records}
    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "study_summary.csv"
    param_keys = list(records[0]["key_params"].keys()) if records else []
    fields = (["run_tag", "dim", "repeat", "max_turns"] + param_keys
              + ["exec_model", "splitter_model", "combiner_model",
                 "calls", "input_tokens", "output_tokens", "total_tokens",
                 "usd", "elapsed_s", "rollouts", "overall_correct",
                 "atom_pass_rate", "n_pass", "n_gradeable", "n_ungradeable",
                 "decomposition_fidelity", "contiguity_rate", "chain_coverage",
                 "flagged"])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {"run_tag": r["run_tag"], "dim": r["dim"],
                   "repeat": r["repeat"], "max_turns": r["max_turns"]}
            row.update(r["key_params"])
            row.update({
                "exec_model": r["role_models"]["execution"],
                "splitter_model": r["role_models"]["splitter"],
                "combiner_model": r["role_models"]["combiner"],
                "calls": r["cost"]["calls"],
                "input_tokens": r["cost"]["input_tokens"],
                "output_tokens": r["cost"]["output_tokens"],
                "total_tokens": r["cost"]["total_tokens"],
                "usd": r["cost"]["usd"], "elapsed_s": r["elapsed_s"],
                "rollouts": r["rollouts"],
                "overall_correct": r["overall_correct"],
                "atom_pass_rate": r["atom_pass_rate"],
                "n_pass": r["n_pass"], "n_gradeable": r["n_gradeable"],
                "n_ungradeable": r["n_ungradeable"],
                "decomposition_fidelity": r["decomposition_fidelity"],
                "contiguity_rate": r["contiguity_rate"],
                "chain_coverage": r["chain_coverage"],
                "flagged": str(r["flagged"])[:120] if r["flagged"] else "",
            })
            w.writerow(row)

    md = out_dir / "study_report.md"
    lines = ["# Matrix-chain decomposition study\n",
             f"- horizon T (chain length): {meta['max_turns']}",
             f"- complexity axis n = matrix dimension d: {meta['dims']}",
             f"- base config: `{meta['base_config']}`",
             f"- mock: {meta['mock']}   runs: {len(records)}\n",
             "## Solve rate & cost vs complexity n (=d)\n",
             "| n (=d) | runs | solve rate | mean atom pass | mean fidelity "
             "| mean tokens | mean $ | mean calls |",
             "|--|--|--|--|--|--|--|--|"]
    for d, s in by_dim.items():
        def fmt(x, p=3):
            return f"{x:.{p}f}" if x is not None else "n/a"
        lines.append(
            f"| {d} | {s['n_runs']} | {s['solve_rate']:.2f} | "
            f"{fmt(s['mean_atom_pass_rate'])} | {fmt(s['mean_fidelity'])} | "
            f"{fmt(s['mean_total_tokens'],0)} | {fmt(s['mean_usd'],4)} | "
            f"{fmt(s['mean_calls'],0)} |")
    lines += ["\n## Per-run results\n",
              "| run | n | exec model | tokens | $ | overall | atom pass | "
              "fidelity | flag |", "|--|--|--|--|--|--|--|--|--|"]
    for r in records:
        apr = r["atom_pass_rate"]
        apr_s = f"{r['n_pass']}/{r['n_gradeable']}" + (
            f" ({apr:.2f})" if apr is not None else "")
        fid = r["decomposition_fidelity"]
        lines.append(
            f"| {r['run_tag']} | {r['dim']} | "
            f"{r['role_models']['execution']} | {r['cost']['total_tokens']} | "
            f"{r['cost']['usd']:.4f} | {'OK' if r['overall_correct'] else 'X'} "
            f"| {apr_s} | {fid:.2f} | "
            f"{'Y' if r['flagged'] else ''} |")
    md.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("concord/config/matmul_per_role.yaml"))
    p.add_argument("--data-dir", type=Path, default=Path("data/matmul"),
                   help="static golden dataset dir (parent repo's data/matmul)")
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3],
                   help="complexity axis n = matrix dimension d to sweep")
    p.add_argument("--max-turns", type=int, default=100,
                   help="horizon T = chain length (matrices per problem)")
    p.add_argument("--sample-id", type=int, default=0,
                   help="which golden chain (0..9) to use per dimension")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat each dim N times (executor temperature > 0 in "
                        "the mock; for a solve RATIO across stochastic runs)")
    p.add_argument("--solver", type=str, default=None,
                   choices=("mcts", "pipeline"))
    p.add_argument("--max-node-calls", type=int, default=None,
                   help="override pipeline.max_node_calls (the per-node "
                        "call/token budget). Raise it so a long chain can "
                        "finish executing all its atoms instead of truncating.")
    p.add_argument("--max-split-depth", type=int, default=None,
                   help="override pipeline.max_split_depth.")
    p.add_argument("--index-refs", dest="index_refs", action="store_true",
                   default=True, help="Splitter references input matrices by "
                   "range (source_span); code injects exact values (default).")
    p.add_argument("--no-index-refs", dest="index_refs", action="store_false",
                   help="Legacy: splitter inlines matrix values (transcription-"
                   "error-prone; for A/B comparison).")
    p.add_argument("--mock", action="store_true",
                   help="force the mock LLM (no API; plumbing test)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--wandb", action="store_true",
                   help="stream metrics to Weights & Biases (solve rate / "
                        "atom-pass / fidelity / tokens / $ vs n, per-depth "
                        "pass rates, and a summary table + bar chart).")
    p.add_argument("--wandb-project", default="concord-matmul-study")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None,
                   help="ties related runs together; default encodes T + dims.")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or Path("results/matmul_study") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    study_jsonl = out_dir / "runs.jsonl"

    meta = {"dims": args.dims, "max_turns": args.max_turns,
            "sample_id": args.sample_id, "base_config": str(args.config),
            "mock": args.mock, "repeats": args.repeats, "started_at": ts}
    print(f"[matmul_study] dims={args.dims} x {args.repeats} repeats, "
          f"T={args.max_turns}, sample={args.sample_id}, mock={args.mock}")
    print(f"[matmul_study] output dir: {out_dir}")

    dims_slug = "-".join(str(d) for d in args.dims)
    wb = WandbLogger(
        args.wandb, args.wandb_project, args.wandb_entity,
        args.wandb_run_name or f"matmul_T{args.max_turns}_n{dims_slug}",
        args.wandb_group or f"matmul_T{args.max_turns}_n{dims_slug}",
        {**meta, "solver": args.solver or "pipeline"})
    if args.wandb:
        print(f"[matmul_study] streaming to W&B project "
              f"'{args.wandb_project}'")

    overrides: dict[str, Any] = {}
    if args.max_node_calls is not None:
        overrides["pipeline.max_node_calls"] = args.max_node_calls
    if args.max_split_depth is not None:
        overrides["pipeline.max_split_depth"] = args.max_split_depth
    if overrides:
        print(f"[matmul_study] overrides: {overrides}")

    records: list[dict] = []
    for dim in args.dims:
        for rep in range(1, args.repeats + 1):
            try:
                rec = run_one(
                    dim=dim, repeat=rep, sample_id=args.sample_id,
                    overrides=overrides, base_config=args.config, mock=args.mock,
                    solver=args.solver, data_dir=args.data_dir,
                    max_turns=args.max_turns, out_dir=out_dir,
                    index_refs=args.index_refs)
            except Exception as e:                                  # noqa: BLE001
                import traceback
                print(f"  [run d{dim}_r{rep} FAILED] {e}")
                traceback.print_exc()
                continue
            records.append(rec)
            wb.log_run(rec)
            with study_jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    if records:
        write_summary(records, out_dir, meta)
        wb.log_summary(_by_dim(records), records)
    wb.finish()

    n_ok = sum(1 for r in records if r["overall_correct"])
    print("\n" + "#" * 78)
    print(f"[matmul_study] DONE. {len(records)} runs, {n_ok} solved the whole "
          f"chain (final M_T correct).")
    print(f"[matmul_study] summary : {out_dir}/study_summary.json")
    print(f"[matmul_study] report  : {out_dir}/study_report.md")


if __name__ == "__main__":
    main()
