"""Run Concord across every question in eval_set.json.

Usage:

    # 1) Sanity check on mock (free, fast, answers will be garbage):
    .venv/bin/python -m concord.experiments.run_all --mock

    # 2) Real run (needs ANTHROPIC_API_KEY):
    export ANTHROPIC_API_KEY=sk-...
    .venv/bin/python -m concord.experiments.run_all \\
        --model claude-sonnet-4-6 --N 16 --K 4

    # 3) Filter by domain / difficulty:
    .venv/bin/python -m concord.experiments.run_all --domain math

    # 4) Resume an interrupted run (skips question_ids already in the JSONL):
    .venv/bin/python -m concord.experiments.run_all --resume

Results are appended to `results/concord/run_all/<run-tag>.jsonl`; a final
summary lands at `<run-tag>_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from core.config import Config, LLMCfg
from core.llm.factory import RoleClients
from core.llm.mock import MockLLM
from core.multi_solve import solve_multi as v2_solve


def _load_config(path: Path | None, *, mock: bool,
                 model_override: str | None,
                 model_overrides: dict[str, str]) -> Config:
    """Load config from YAML; layer CLI overrides on top.

    `model_overrides` maps role name -> model string and pins that role's
    `models.<role>.model` field.
    """
    cfg_path = path or _HERE.parents[1] / "config" / "default.yaml"
    cfg = Config.from_yaml(cfg_path)

    if mock:
        # Force all roles to mock so we never hit the network.
        cfg.llm = LLMCfg(provider="mock", model="mock-v0",
                          temperature=cfg.llm.temperature)
        cfg.models = type(cfg.models)()
        return cfg

    if model_override:
        cfg.llm.provider = "anthropic"
        cfg.llm.model = model_override

    # Per-role CLI overrides (e.g. --execution-model claude-opus-4-7)
    for role, model in model_overrides.items():
        if model is None:
            continue
        # Build/replace the role spec, inheriting other knobs from cfg.llm
        existing = getattr(cfg.models, role) or LLMCfg(**cfg.llm.model_dump())
        existing.provider = "anthropic"
        existing.model = model
        setattr(cfg.models, role, existing)
    return cfg


def _make_clients_for_run(cfg: Config) -> RoleClients:
    return RoleClients.from_config(cfg)


def _grade(question: dict, predicted: str) -> bool | None:
    try:
        from longcot import verify
        from longcot._types import Question as LCQ
    except ImportError:
        return None
    try:
        q = LCQ(
            question_id=question["question_id"],
            domain=question["domain"],
            difficulty=question["difficulty"],
            prompt=question["prompt"],
            problem=question.get("problem"),
            answer=question.get("answer"),
        )
        body = predicted if "solution" in predicted.lower() else f"solution = {predicted}"
        return verify(q, body)
    except Exception as e:                                          # noqa: BLE001
        print(f"  [grade warning] {e}", file=sys.stderr)
        return None


def _load_done(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid and "error" not in obj:
                done.add(qid)
        except json.JSONDecodeError:
            continue
    return done


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Concord across all eval-set questions.")
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--domain", type=str, default=None,
                   help="filter by domain (math/chess/chemistry/logic/cs)")
    p.add_argument("--difficulty", type=str, default=None,
                   help="filter by difficulty (easy/medium/hard)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap to first N matching questions")
    p.add_argument("--N", type=int, default=None,
                   help="rollout budget cfg.mcts.N (overrides YAML)")
    p.add_argument("--K", type=int, default=None,
                   help="samples per subproblem (overrides YAML)")
    p.add_argument("--c-puct", type=float, default=None,
                   help="PUCT exploration constant (cfg.mcts.c_puct)")
    p.add_argument("--alpha", type=float, default=None,
                   help="SD vs verifier weight in U_s (cfg.confidence.alpha)")
    # --- widening + depth knobs ---
    p.add_argument("--C", type=float, default=None,
                   help="progressive widening coefficient (cfg.mcts.C). "
                        "Higher = more children per node. Default 2.")
    p.add_argument("--beta", type=float, default=None,
                   help="progressive widening exponent (cfg.mcts.beta). "
                        "Default 0.5 (sqrt). Raise toward 1.0 for near-linear "
                        "widening.")
    p.add_argument("--n-min", type=int, default=None,
                   help="guarded-max visit threshold (cfg.mcts.n_min)")
    p.add_argument("--slack", type=int, default=None,
                   help="extra depth above critical_path (cfg.depth.slack). "
                        "For non-explicit graphs THIS is the only depth knob.")
    p.add_argument("--lam", type=float, default=None,
                   help="marginal-gain stop threshold (cfg.depth.lam). "
                        "Lower = search keeps going longer.")
    p.add_argument("--no-stop", action="store_true",
                   help="disable the marginal-gain stop entirely "
                        "(equivalent to --lam 0).")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (e.g. config/per_role.yaml). "
                        "Per-role model assignments live under `models:`. "
                        "Defaults to config/default.yaml.")
    p.add_argument("--solver", type=str, default=None,
                   choices=("mcts", "pipeline"),
                   help="override cfg.solver. `mcts` = legacy expansion "
                        "(executor on whole block). `pipeline` = Split → "
                        "Solve → Combine → Verify with backtracking + "
                        "synthesizer. Default = whatever the YAML says "
                        "(wide_and_deep.yaml is pipeline; others are mcts).")
    p.add_argument("--mock", action="store_true",
                   help="force all roles to mock LLM (free, deterministic)")
    p.add_argument("--model", type=str, default=None,
                   help="override the top-level llm.model (applies to every "
                        "role that doesn't have its own pin)")
    # per-role overrides — let the user pin a single role without writing YAML
    p.add_argument("--execution-model", type=str, default=None,
                   help="override models.execution.model")
    p.add_argument("--decomposition-model", type=str, default=None,
                   help="override models.decomposition.model")
    p.add_argument("--classification-model", type=str, default=None,
                   help="override models.classification.model")
    p.add_argument("--verification-model", type=str, default=None,
                   help="override models.verification.model")
    p.add_argument("--resume", action="store_true",
                   help="skip question_ids already present in the output JSONL")
    p.add_argument("--tag", type=str, default=None,
                   help="custom run tag; default = timestamp")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/concord/run_all"))
    args = p.parse_args()

    role_overrides = {
        "execution": args.execution_model,
        "decomposition": args.decomposition_model,
        "classification": args.classification_model,
        "verification": args.verification_model,
    }

    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. Re-run with --mock for a "
                 "free sanity check, or `export ANTHROPIC_API_KEY=sk-...` first.")

    cfg = _load_config(args.config, mock=args.mock,
                        model_override=args.model,
                        model_overrides=role_overrides)

    # Loud guard: if we're about to use the mock provider but the user did
    # NOT pass --mock, abort. Silent fallback to mock is what made earlier
    # runs look like they "didn't make API calls" — the predicted answers
    # were `<mock:0>` strings.
    effective_provider = cfg.role_model("execution").provider
    if not args.mock and effective_provider == "mock":
        sys.exit(
            f"\nERROR: execution role resolved to provider=mock but --mock "
            f"was NOT passed.\n"
            f"  Loaded config: {args.config or '(default.yaml — which is mock!)'}\n"
            f"  To get real LLM calls, pass --config "
            f"concord/config/wide_and_deep.yaml (or another non-mock YAML),\n"
            f"  OR pass --mock explicitly if you intended a dry run.\n"
        )
    if args.N is not None:
        cfg.mcts.N = args.N
    if args.K is not None:
        cfg.sampling.K_blackbox = args.K
        cfg.sampling.K_whitebox = args.K
    if args.c_puct is not None:
        cfg.mcts.c_puct = args.c_puct
    if args.alpha is not None:
        cfg.confidence.alpha = args.alpha
    # widening + depth overrides
    if args.C is not None:      cfg.mcts.C = args.C
    if args.beta is not None:   cfg.mcts.beta = args.beta
    if args.n_min is not None:  cfg.mcts.n_min = args.n_min
    if args.slack is not None:  cfg.depth.slack = args.slack
    if args.lam is not None:    cfg.depth.lam = args.lam
    if args.no_stop:            cfg.depth.lam = 0.0
    # Solver override (after YAML load + other overrides). This is the
    # source of truth for which expansion policy runs.
    if args.solver is not None:
        cfg.solver = args.solver

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    qs = data["questions"]
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    if args.difficulty:
        qs = [q for q in qs if q["difficulty"] == args.difficulty]
    if args.limit:
        qs = qs[: args.limit]

    # Compose the run label from the *effective* execution model
    exec_spec = cfg.role_model("execution")
    run_label = "mock" if exec_spec.provider == "mock" else exec_spec.model.replace("/", "_")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: pick the most-recent file matching this model unless an
    # explicit --tag was supplied. New run: timestamp tag.
    if args.resume and not args.tag:
        candidates = sorted(out_dir.glob(f"*__{run_label}.jsonl"))
        if candidates:
            chosen = candidates[-1]
            tag = chosen.stem.split(f"__{run_label}")[0]
            print(f"Resume: continuing {chosen.name}")
        else:
            tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            print("Resume requested but no prior file found; starting fresh.")
    else:
        tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")

    out_path = out_dir / f"{tag}__{run_label}.jsonl"
    summary_path = out_dir / f"{tag}__{run_label}_summary.json"

    done = _load_done(out_path) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} question(s) already complete in {out_path}")

    # ----- Loud SOLVER banner -----
    # The single most-load-bearing config value. We surface it BEFORE the
    # role-model dump so the operator sees it as the first thing and
    # can ^C if the resolved solver isn't what they meant. Pipeline-only
    # roles are skipped from the role dump when running MCTS.
    solver_kind = getattr(cfg, "solver", "mcts")
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  SOLVER: {solver_kind.upper()}    "
          f"({'Split→Solve→Combine→Verify + synthesizer' if solver_kind == 'pipeline' else 'legacy MCTS expansion'})")
    if solver_kind == "mcts" and args.config is None:
        print("  NOTE: no --config was passed, so the default mcts solver is "
              "active. Pass --solver pipeline (or --config "
              "concord/config/wide_and_deep.yaml) for the new pipeline.")
    elif solver_kind == "mcts" and args.config is not None \
            and "wide_and_deep" in str(args.config):
        print("  WARNING: --config wide_and_deep.yaml was passed but solver "
              "resolved to MCTS. Did you override with --solver mcts?")
    print(bar + "\n")

    print(f"Will run {len([q for q in qs if q['question_id'] not in done])} "
          f"of {len(qs)} filtered question(s). "
          f"Config: N={cfg.mcts.N} K={cfg.sampling.K_blackbox} "
          f"c_puct={cfg.mcts.c_puct}")
    # Show per-role assignments so the user can spot misconfiguration.
    print("Role models:")
    _roles = ("execution", "decomposition", "classification", "verification")
    if solver_kind == "pipeline":
        _roles = _roles + ("splitter", "combiner", "synthesizer", "synth_verifier")
    for role in _roles:
        spec = cfg.role_model(role)
        print(f"  - {role:14s} {spec.provider}/{spec.model} (T={spec.temperature})")
    print(f"Output: {out_path}\n")

    n_correct = 0
    n_graded = 0
    n_coherent = 0
    n_runs = 0
    # Token / cost accounting across all problems in this run.
    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    total_usd = 0.0
    t_start = time.time()
    by_domain: dict[str, dict[str, int]] = {}

    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            if q["question_id"] in done:
                print(f"[{i}/{len(qs)}] skip {q['question_id']} (resumed)")
                continue

            print(f"[{i}/{len(qs)}] {q['question_id']} "
                  f"({q['domain']}/{q['difficulty']}/"
                  f"{(q.get('problem') or {}).get('template')})")

            clients = _make_clients_for_run(cfg)
            t0 = time.time()
            error = None
            res = None
            try:
                res = v2_solve(q["prompt"], cfg=cfg, clients=clients,
                                domain=q["domain"], tag=f"run_all_{tag}")
            except Exception as e:                                  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                print(f"  ERROR: {error}", file=sys.stderr)
            elapsed = time.time() - t0

            row: dict
            if res is None:
                row = {"question_id": q["question_id"], "error": error,
                       "elapsed_s": round(elapsed, 1)}
            else:
                correct = _grade(q, res.answer)
                # Per-problem token consumption (input + output). For
                # multi-block problems this is the accurate, delta-measured
                # total across every block + the graph-builder + synthesizer.
                p_in = int(res.cost.get("input_tokens", 0) or 0)
                p_out = int(res.cost.get("output_tokens", 0) or 0)
                p_calls = int(res.cost.get("calls", 0) or 0)
                p_usd = float(res.cost.get("usd", 0.0) or 0.0)
                row = {
                    "question_id": q["question_id"],
                    "domain": q["domain"],
                    "difficulty": q["difficulty"],
                    "template": (q.get("problem") or {}).get("template"),
                    "gold_answer": q.get("answer"),
                    "predicted_answer": res.answer,
                    "coherent": res.coherent,
                    "sigma": res.sigma,
                    "rollouts": res.rollouts,
                    "flagged": res.flagged,
                    "cost": res.cost,
                    # Token consumption surfaced at the top level so it's
                    # easy to scan / aggregate without digging into `cost`.
                    "input_tokens": p_in,
                    "output_tokens": p_out,
                    "total_tokens": p_in + p_out,
                    "calls": p_calls,
                    "usd": round(p_usd, 6),
                    "elapsed_s": round(elapsed, 1),
                    "correct": correct,
                    "trace_path": res.trace_path,
                    "tree_path": res.cost.get("tree_path"),
                    "rollouts_path": res.cost.get("rollouts_path"),
                }
                n_runs += 1
                total_input_tokens += p_in
                total_output_tokens += p_out
                total_calls += p_calls
                total_usd += p_usd
                if res.coherent:
                    n_coherent += 1
                if correct is not None:
                    n_graded += 1
                    if correct:
                        n_correct += 1
                bucket = by_domain.setdefault(q["domain"],
                                               {"total": 0, "correct": 0,
                                                "coherent": 0})
                bucket["total"] += 1
                if res.coherent:
                    bucket["coherent"] += 1
                if correct:
                    bucket["correct"] += 1

                status = ("CORRECT" if correct else
                          "incorrect" if correct is False else
                          "ungraded")
                # Multi-block runs expose `per_block` on the cost dict; show
                # block count so the operator can see when the per-block
                # fanout kicked in (vs. a single solve).
                nb = res.cost.get("n_blocks")
                block_note = f"  blocks={nb}" if nb else ""
                print(f"  -> {status}  predicted={res.answer!r}  "
                      f"gold={q.get('answer')!r}  "
                      f"calls={p_calls} ${p_usd:.4f}  "
                      f"tokens={p_in + p_out} (in {p_in} / out {p_out})  "
                      f"{elapsed:.1f}s{block_note}")

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    total_elapsed = time.time() - t_start
    acc = (n_correct / n_graded) if n_graded else None
    coh = (n_coherent / n_runs) if n_runs else None

    summary = {
        "run_tag": tag,
        "model": run_label,
        "config": {"N": cfg.mcts.N, "K": cfg.sampling.K_blackbox,
                   "c_puct": cfg.mcts.c_puct, "alpha": cfg.confidence.alpha},
        "n_questions": len(qs),
        "n_runs": n_runs,
        "n_graded": n_graded,
        "n_correct": n_correct,
        "n_coherent": n_coherent,
        "accuracy": acc,
        "coherence_rate": coh,
        "by_domain": by_domain,
        # Token / cost totals across every solved problem in this run.
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_calls": total_calls,
        "total_usd": round(total_usd, 6),
        "avg_tokens_per_problem": (
            round((total_input_tokens + total_output_tokens) / n_runs, 1)
            if n_runs else None),
        "total_elapsed_s": round(total_elapsed, 1),
        "results_path": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Total: {n_runs} runs, {n_graded} graded, "
          f"{n_correct} correct ({acc if acc is None else f'{acc:.1%}'})")
    print(f"Coherence rate: {coh if coh is None else f'{coh:.1%}'}")
    total_tokens = total_input_tokens + total_output_tokens
    avg_tok = (total_tokens / n_runs) if n_runs else 0
    print(f"Tokens: {total_tokens:,} total "
          f"(in {total_input_tokens:,} / out {total_output_tokens:,})"
          f"{f'  ~{avg_tok:,.0f}/problem' if n_runs else ''}")
    print(f"Cost:   ${total_usd:.4f} over {total_calls:,} calls")
    print(f"Elapsed: {total_elapsed:.1f}s")
    if by_domain:
        print("\nBy domain:")
        for d, stats in sorted(by_domain.items()):
            domain_acc = stats["correct"] / stats["total"] if stats["total"] else 0
            print(f"  {d:10s} {stats['correct']}/{stats['total']} "
                  f"({domain_acc:.0%})  coherent={stats['coherent']}")
    print(f"\nResults:  {out_path}")
    print(f"Summary:  {summary_path}")
    print(f"\nPer-question viz artifacts: results/concord/<run-id>_tree.json + _rollouts.jsonl")
    print(f"  Open concord/viz/tree_viewer.html in a browser and drag-and-drop them.")


if __name__ == "__main__":
    main()
