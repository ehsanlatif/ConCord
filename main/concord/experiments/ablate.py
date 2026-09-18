"""Run a single ablation arm on a slice of the dataset.

Pairs with `config/ablations/*.yaml`. Each ablation YAML *overrides* a
subset of the default config; this runner applies that override and feeds
it into the same `solve()` used by the main pipeline. Output mirrors the
compare runner so results can be aggregated together later.

Usage:

    .venv/bin/python -m concord.experiments.ablate \\
        --ablation concord/config/ablations/no_gate.yaml \\
        --mock --n 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

import yaml

from core.config import Config
from core.llm.mock import MockLLM
from core.orchestrator import solve as v2_solve

from experiments.metrics import RunRecord, summarize


def _deep_update(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _apply_ablation(default_yaml: Path, ablation_yaml: Path) -> Config:
    base = yaml.safe_load(default_yaml.read_text()) or {}
    overlay = yaml.safe_load(ablation_yaml.read_text()) or {}
    merged = _deep_update(base, overlay)
    return Config.model_validate(merged)


def _make_llm(args: argparse.Namespace):
    if args.mock:
        return MockLLM(responses={})
    from core.llm.anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(model=args.model,
                             max_output_tokens=args.max_output_tokens,
                             request_timeout_s=args.request_timeout_s)


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
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Run one ablation arm.")
    p.add_argument("--ablation", type=Path, required=True,
                   help="path to config/ablations/<arm>.yaml")
    p.add_argument("--dataset", type=Path, default=Path("data/eval_set.json"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/concord/ablation"))
    args = p.parse_args()

    default_yaml = _HERE.parents[1] / "config" / "default.yaml"
    cfg = _apply_ablation(default_yaml, args.ablation)
    cfg.mcts.N = args.N
    cfg.sampling.K_blackbox = args.K
    cfg.sampling.K_whitebox = args.K

    arm_name = args.ablation.stem
    out_path = args.out_dir / f"{arm_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    qs = data["questions"]
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    qs = qs[: args.n]

    records: list[RunRecord] = []
    print(f"Ablation: {arm_name}  on {len(qs)} questions")
    with out_path.open("w", encoding="utf-8") as f:
        for q in qs:
            llm = _make_llm(args)
            t0 = time.time()
            try:
                res = v2_solve(q["prompt"], cfg=cfg, llm=llm,
                                domain=q["domain"], tag=f"abl_{arm_name}")
                error = None
            except Exception as e:                                  # noqa: BLE001
                res = None
                error = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0
            if res is None:
                row = {"question_id": q["question_id"], "arm": arm_name,
                       "error": error}
                rec = RunRecord(question_id=q["question_id"], correct=None,
                                coherent=False, sigma=1.0,
                                rollouts=0, calls=0, tokens_in=0, tokens_out=0,
                                usd=0.0, elapsed_s=elapsed)
            else:
                correct = _grade(q, res.answer)
                row = {
                    "arm": arm_name, "question_id": q["question_id"],
                    "domain": q["domain"], "predicted": res.answer,
                    "gold": q.get("answer"), "correct": correct,
                    "coherent": res.coherent, "sigma": res.sigma,
                    "rollouts": res.rollouts, "cost": res.cost,
                    "elapsed_s": round(elapsed, 1),
                    "flagged": res.flagged,
                }
                rec = RunRecord(question_id=q["question_id"], correct=correct,
                                coherent=res.coherent, sigma=res.sigma,
                                rollouts=res.rollouts,
                                calls=res.cost.get("calls", 0),
                                tokens_in=res.cost.get("input_tokens", 0),
                                tokens_out=res.cost.get("output_tokens", 0),
                                usd=res.cost.get("usd", 0.0),
                                elapsed_s=elapsed)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            print(f"  [{q['question_id']}] {rec.correct} ({rec.calls} calls)")

    summary = summarize(records)
    summary["arm"] = arm_name
    summary["config_overlay"] = yaml.safe_load(args.ablation.read_text())
    (args.out_dir / f"{arm_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
