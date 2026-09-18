"""Run Concord on one eval-set question.

Usage:

    .venv/bin/python -m concord.experiments.run \\
        --dataset data/eval_set.json \\
        --index 0 \\
        --model claude-sonnet-4-6 \\
        --N 16 --K 4

Or with the deterministic mock LLM (no API calls):

    .venv/bin/python -m concord.experiments.run --dataset data/eval_set.json \\
        --index 0 --mock

The script picks ONE question by `--index`, runs `solve()`, and writes the
result to `results/concord/<run_id>.json` next to the JSONL trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `core` importable when this file is executed as a script.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from core.config import Config, LLMCfg
from core.confidence import (
    LexicalEmbedder,
    LexicalKernel,
    default_verifier_for,
)
from core.llm.factory import RoleClients
from core.llm.mock import MockLLM
from core.multi_solve import solve_multi as solve


def _load_question(dataset_path: Path, *, index: int | None,
                    question_id: str | None) -> dict:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if question_id is not None:
        for q in questions:
            if q["question_id"] == question_id:
                return q
        raise SystemExit(f"question_id {question_id!r} not in dataset")
    if index is None:
        index = 0
    if index < 0 or index >= len(questions):
        raise SystemExit(f"index {index} out of range (0..{len(questions) - 1})")
    return questions[index]


def _grade(question: dict, predicted: str) -> bool | None:
    """Optional terminal grader using longcot.verify when available."""
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
        # Grader expects the predicted response embedded in a "solution = ..."
        # marker. Wrap if it isn't already.
        body = predicted if "solution" in predicted.lower() else f"solution = {predicted}"
        return verify(q, body)
    except Exception as e:                                          # noqa: BLE001
        print(f"  [grade warning] {e}", file=sys.stderr)
        return None


def _make_llm(args: argparse.Namespace):
    """Legacy helper for the one-LLM `solve(llm=...)` path.

    Preferred path is now `RoleClients.from_config(cfg)` — see main().
    """
    if args.mock:
        return MockLLM()
    from core.llm.anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        request_timeout_s=args.request_timeout_s,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run Concord on one eval-set question.")
    p.add_argument("--dataset", type=Path,
                   default=Path("data/eval_set.json"),
                   help="Path to eval_set.json")
    p.add_argument("--index", type=int, default=None,
                   help="0-based question index")
    p.add_argument("--question-id", type=str, default=None,
                   help="Pick by exact question_id")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (e.g. config/per_role.yaml). Per-role "
                        "model assignments live under `models:`. Defaults to "
                        "config/default.yaml.")
    p.add_argument("--solver", type=str, default=None,
                   choices=("mcts", "pipeline"),
                   help="override cfg.solver. `pipeline` = Split → Solve → "
                        "Combine → Verify with backtracking + synthesizer; "
                        "`mcts` = legacy expansion. Default = YAML's value "
                        "(wide_and_deep.yaml is pipeline; others are mcts).")
    p.add_argument("--N", type=int, default=16, help="rollout budget (cfg.mcts.N)")
    p.add_argument("--K", type=int, default=4, help="samples per subproblem (black-box)")
    p.add_argument("--c-puct", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="confidence: SD vs verifier weight (cfg.confidence.alpha)")
    p.add_argument("--tag", type=str, default=None,
                   help="appended to the run id in the trace path")
    p.add_argument("--mock", action="store_true",
                   help="force all roles to mock LLM (no API)")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6",
                   help="override top-level llm.model (legacy single-LLM path)")
    p.add_argument("--execution-model", type=str, default=None)
    p.add_argument("--decomposition-model", type=str, default=None)
    p.add_argument("--classification-model", type=str, default=None)
    p.add_argument("--verification-model", type=str, default=None)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--out", type=Path,
                   default=Path("results/concord/runs.jsonl"),
                   help="append the result summary to this JSONL")
    args = p.parse_args()

    cfg_path = args.config or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = Config.from_yaml(cfg_path)
    cfg.mcts.N = args.N
    cfg.sampling.K_blackbox = args.K
    cfg.sampling.K_whitebox = args.K
    if args.c_puct is not None:
        cfg.mcts.c_puct = args.c_puct
    if args.alpha is not None:
        cfg.confidence.alpha = args.alpha

    if args.mock:
        cfg.llm = LLMCfg(provider="mock", model="mock-v0",
                          temperature=cfg.llm.temperature)
        cfg.models = type(cfg.models)()    # clear per-role overrides
    else:
        # Top-level model override (kept for backward compat)
        if args.model and args.model != "claude-sonnet-4-6":
            cfg.llm.provider = "anthropic"
            cfg.llm.model = args.model
        elif cfg.llm.provider == "mock":
            # Default to sonnet when no explicit override AND base config is mock.
            cfg.llm.provider = "anthropic"
            cfg.llm.model = args.model
        # Per-role overrides
        for role, model in (("execution", args.execution_model),
                            ("decomposition", args.decomposition_model),
                            ("classification", args.classification_model),
                            ("verification", args.verification_model)):
            if model is None:
                continue
            existing = getattr(cfg.models, role) or LLMCfg(**cfg.llm.model_dump())
            existing.provider = "anthropic"
            existing.model = model
            setattr(cfg.models, role, existing)

    # Apply --solver override (after YAML load + role-model fix-ups).
    if args.solver is not None:
        cfg.solver = args.solver

    q = _load_question(args.dataset, index=args.index, question_id=args.question_id)

    # Loud SOLVER banner — single most-load-bearing knob; print before role
    # models so the operator sees it first.
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
    print(bar)

    print(f"Question {q['question_id']} ({q['domain']}/{q['difficulty']})")
    print(f"  config: N={cfg.mcts.N} K={cfg.sampling.K_blackbox} "
          f"c_puct={cfg.mcts.c_puct} alpha={cfg.confidence.alpha}")
    print("  role models:")
    _roles = ("execution", "decomposition", "classification", "verification")
    if solver_kind == "pipeline":
        _roles = _roles + ("splitter", "combiner", "synthesizer", "synth_verifier")
    for role in _roles:
        spec = cfg.role_model(role)
        print(f"    - {role:14s} {spec.provider}/{spec.model}")

    clients = RoleClients.from_config(cfg)
    t0 = time.time()
    res = solve(q["prompt"], cfg=cfg, clients=clients,
                 domain=q["domain"], tag=args.tag)
    elapsed = time.time() - t0

    correct = _grade(q, res.answer)

    summary = {
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
        "trace_path": res.trace_path,
        "cost": res.cost,
        "elapsed_s": round(elapsed, 2),
        "correct": correct,
        "config": {
            "N": cfg.mcts.N, "K": cfg.sampling.K_blackbox,
            "c_puct": cfg.mcts.c_puct, "alpha": cfg.confidence.alpha,
            "ablation": cfg.ablation.model_dump(),
        },
        "model": "mock" if args.mock else cfg.role_model("execution").model,
        "role_models": {role: f"{cfg.role_model(role).provider}/{cfg.role_model(role).model}"
                        for role in ("execution", "decomposition",
                                      "classification", "verification")},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Answer:    {res.answer!r}")
    print(f"Coherent:  {res.coherent}  sigma={res.sigma}")
    print(f"Flagged:   {res.flagged}")
    print(f"Correct:   {correct}")
    print(f"Rollouts:  {res.rollouts}  calls={res.cost.get('calls')}  ${res.cost.get('usd'):.4f}")
    print(f"Elapsed:   {elapsed:.1f}s")
    print(f"Trace:     {res.trace_path}")
    print(f"Tree:      {res.cost.get('tree_path')}")
    print(f"Rollouts:  {res.cost.get('rollouts_path')}")
    print(f"Summary:   {args.out}")
    print(f"\n  Visualize: open concord/viz/tree_viewer.html in a browser and")
    print(f"             drag-and-drop the tree.json + rollouts.jsonl files.")


if __name__ == "__main__":
    main()
