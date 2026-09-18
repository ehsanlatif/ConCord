"""Calibration sweep over c_puct, n_min, K.

Plan §7 M7: 'sweep c_puct, n_min, K; run ablations'. Each grid point is one
run of `solve()` per question; results are appended to the same JSONL the
compare runner uses so downstream aggregation can pool them.

Usage:

    .venv/bin/python -m concord.experiments.sweep --mock --n 2 \\
        --c-puct 2 3 5 --n-min 2 3 --K 4 8

Live runs need `ANTHROPIC_API_KEY`. Even on the mock the sweep is useful for
exercising the runner — accuracy will be 0 but the cost/rollout metrics
exercise the harness.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from core.config import Config
from core.llm.mock import MockLLM
from core.orchestrator import solve as v2_solve

from experiments.metrics import RunRecord, summarize


def _make_llm(args):
    if args.mock:
        return MockLLM(responses={})
    from core.llm.anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(model=args.model,
                            max_output_tokens=args.max_output_tokens,
                            request_timeout_s=args.request_timeout_s)


def _grade(question: dict, predicted: str):
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
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep c_puct, n_min, K.")
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--N", type=int, default=16, help="rollout budget per run")
    p.add_argument("--c-puct", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    p.add_argument("--n-min", type=int,   nargs="+", default=[2, 3])
    p.add_argument("--K",     type=int,   nargs="+", default=[4, 8])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/concord/sweep"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_default = Config.from_yaml(_HERE.parents[1] / "config" / "default.yaml")

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    qs = data["questions"]
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    qs = qs[: args.n]

    grid = list(itertools.product(args.c_puct, args.n_min, args.K))
    print(f"Grid: {len(grid)} points x {len(qs)} questions = "
          f"{len(grid) * len(qs)} runs total")

    summary_path = args.out_dir / "sweep_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as sf:
        for (c_puct, n_min, K) in grid:
            cfg = cfg_default.model_copy(deep=True)
            cfg.mcts.c_puct = c_puct
            cfg.mcts.n_min = n_min
            cfg.mcts.N = args.N
            cfg.sampling.K_blackbox = K
            cfg.sampling.K_whitebox = K

            tag = f"c{c_puct}_n{n_min}_K{K}"
            print(f"\n=== sweep point {tag} ===")

            records: list[RunRecord] = []
            for q in qs:
                llm = _make_llm(args)
                t0 = time.time()
                try:
                    res = v2_solve(q["prompt"], cfg=cfg, llm=llm,
                                    domain=q["domain"], tag=tag)
                except Exception as e:                              # noqa: BLE001
                    print(f"  [{q['question_id']}] error: {e}")
                    continue
                elapsed = time.time() - t0
                correct = _grade(q, res.answer)
                records.append(RunRecord(
                    question_id=q["question_id"], correct=correct,
                    coherent=res.coherent, sigma=res.sigma,
                    rollouts=res.rollouts,
                    calls=res.cost.get("calls", 0),
                    tokens_in=res.cost.get("input_tokens", 0),
                    tokens_out=res.cost.get("output_tokens", 0),
                    usd=res.cost.get("usd", 0.0), elapsed_s=elapsed,
                ))
                print(f"  [{q['question_id']}] correct={correct} "
                      f"({res.cost.get('calls')} calls, {round(elapsed,1)}s)")
            s = summarize(records)
            s["c_puct"] = c_puct
            s["n_min"] = n_min
            s["K"] = K
            sf.write(json.dumps(s, ensure_ascii=False) + "\n")
            sf.flush()
            print(f"  -> accuracy={s['accuracy']} coh={s['coherence_rate']} "
                  f"cost.calls={s['cost']['calls']}")

    print(f"\nSweep summary: {summary_path}")


if __name__ == "__main__":
    main()
