"""Run Concord alongside B0..B3 on a slice of the eval set and emit the
comparison table the §10 acceptance criteria call for.

Usage:

    .venv/bin/python -m concord.experiments.compare --mock --n 4
    .venv/bin/python -m concord.experiments.compare --model claude-sonnet-4-6 --n 4 --N 16 --K 4

Writes per-arm JSONL summaries to `results/concord/compare/<arm>.jsonl`
and a human-readable table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from core.config import Config
from core.llm.mock import MockLLM
from core.orchestrator import solve as v2_solve

from experiments.baselines import BASELINES
from experiments.metrics import RunRecord, summarize


# ---------------------------------------------------------------------------

def _make_llm(args: argparse.Namespace):
    if args.mock:
        return MockLLM(responses={})   # echo-mode; correctness will be 0 but
                                       # the harness still exercises end-to-end
    from core.llm.anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        request_timeout_s=args.request_timeout_s,
    )


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
        print(f"  [grade] {e}", file=sys.stderr)
        return None


def _run_arm(name: str, solve_fn, questions: list[dict], *,
             cfg: Config, mk_llm, out_dir: Path) -> list[RunRecord]:
    out_path = out_dir / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            llm = mk_llm()
            t0 = time.time()
            try:
                res = solve_fn(q["prompt"], cfg=cfg, llm=llm,
                               domain=q["domain"], tag=f"compare_{name}")
                error = None
            except Exception as e:                                  # noqa: BLE001
                print(f"  [{name}/{q['question_id']}] error: {e}",
                      file=sys.stderr)
                res = None
                error = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            if res is None:
                rec = RunRecord(question_id=q["question_id"], correct=None,
                                coherent=False, sigma=1.0, rollouts=0,
                                calls=0, tokens_in=0, tokens_out=0, usd=0,
                                elapsed_s=elapsed)
                row = {"question_id": q["question_id"], "error": error}
            else:
                correct = _grade(q, res.answer)
                rec = RunRecord(
                    question_id=q["question_id"], correct=correct,
                    coherent=res.coherent, sigma=res.sigma,
                    rollouts=res.rollouts,
                    calls=res.cost.get("calls", 0),
                    tokens_in=res.cost.get("input_tokens", 0),
                    tokens_out=res.cost.get("output_tokens", 0),
                    usd=res.cost.get("usd", 0.0),
                    elapsed_s=elapsed,
                    u_s_sample=res.cost.get("u_s_sample"),
                )
                row = {
                    "arm": name,
                    "question_id": q["question_id"],
                    "domain": q["domain"],
                    "difficulty": q["difficulty"],
                    "predicted": res.answer,
                    "gold": q.get("answer"),
                    "correct": correct,
                    "coherent": res.coherent,
                    "sigma": res.sigma,
                    "rollouts": res.rollouts,
                    "cost": res.cost,
                    "flagged": res.flagged,
                    "elapsed_s": round(elapsed, 1),
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            status = ("OK" if rec.correct else
                      "ERR" if error else
                      "incorrect")
            print(f"  [{name}/{q['question_id']}] {status}  "
                  f"(calls={rec.calls}, {round(elapsed,1)}s)")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Concord vs B0..B3.")
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--n", type=int, default=4,
                   help="number of questions (from the start of the dataset)")
    p.add_argument("--domain", type=str, default=None,
                   help="filter by domain")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--arms", type=str,
                   default="concord,b0_cot,b1_self_consistency,b2_concord_v1,b3_mcts_no_sd_no_gate",
                   help="comma-separated subset of arms to run")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/concord/compare"))
    args = p.parse_args()

    cfg_path = args.config or _HERE.parents[1] / "config" / "default.yaml"
    cfg = Config.from_yaml(cfg_path)
    cfg.mcts.N = args.N
    cfg.sampling.K_blackbox = args.K
    cfg.sampling.K_whitebox = args.K

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    qs = data["questions"]
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    qs = qs[: args.n]
    print(f"Selected {len(qs)} questions from {args.dataset}")

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    solvers = {"concord": v2_solve, **BASELINES}
    for a in arms_to_run:
        if a not in solvers:
            raise SystemExit(f"unknown arm {a!r}; choices: {list(solvers)}")

    def mk_llm():
        return _make_llm(args)

    all_summaries: dict[str, dict] = {}
    for arm in arms_to_run:
        print(f"\n=== arm: {arm} ===")
        recs = _run_arm(arm, solvers[arm], qs, cfg=cfg, mk_llm=mk_llm,
                        out_dir=args.out_dir)
        all_summaries[arm] = summarize(recs)

    print("\n" + "=" * 60)
    print(f"{'arm':<28} {'acc':>6} {'coh':>6} {'roll':>5} {'calls':>6} {'usd':>8}")
    print("-" * 60)
    for arm, s in all_summaries.items():
        acc = s["accuracy"]
        coh = s["coherence_rate"]
        roll = s["rollouts_mean"]
        cost = s["cost"]
        acc_s = "nan" if (isinstance(acc, float) and acc != acc) else f"{acc:.1%}"
        coh_s = f"{coh:.1%}"
        print(f"{arm:<28} {acc_s:>6} {coh_s:>6} {roll:>5.1f} "
              f"{cost['calls']:>6d} ${cost['usd']:>7.4f}")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2),
                             encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
