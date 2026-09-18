"""M5 gate — walking skeleton.

Plan §7 gate:
- produces a coherent path on an easy instance
- produces a flagged fallback on a deliberately unsatisfiable instance

The test drives the FULL orchestrator end-to-end against a scripted MockLLM,
so we exercise the real ExpansionPolicy + confidence stack + gate + soft-
relax wiring without paying for API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.expansion import build_subproblem_prompt
from core.llm.mock import MockLLM
from core.orchestrator import solve


# ---------------------------------------------------------------------------
# Easy instance: 3-node math chain — every subproblem has a clear integer
# majority answer in the scripted bank, so SD will pick the correct class.
# ---------------------------------------------------------------------------

EASY_PROBLEM = (
    "Problem node_0: Compute the value 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
    "Problem node_2: Use the answer from problem node_1 and multiply by 3.\n"
)


def _easy_responses() -> dict[str, list[str]]:
    """Compose the scripted responses by reconstructing the actual prompts the
    expansion policy will build for each subproblem. The bindings used in the
    prompt depend on which answer wins at each node — but we make sure the
    *majority* (and SD-dense cluster) at each level is the correct integer.
    """
    # Node 0 prompt: simple compute 5+5.
    sub0 = "Compute the value 5 + 5."
    prompt0 = build_subproblem_prompt(sub0, {})

    # Node 1 prompt is built with binding {node_0: "10"} (after extraction).
    sub1 = "Use the answer from problem node_0 and add 2."
    prompt1 = build_subproblem_prompt(sub1, {"node_0": "10"})

    # Node 2 prompt with binding {node_1: "12"}.
    sub2 = "Use the answer from problem node_1 and multiply by 3."
    prompt2 = build_subproblem_prompt(sub2, {"node_1": "12"})

    return {
        prompt0: [
            "Adding gives 10.\nsolution = 10",
            "Adding gives 10.\nsolution = 10",
            "5+5 = 10. solution = 10",
            "noise 7.\nsolution = 7",          # decoy
        ],
        prompt1: [
            "10 + 2 = 12.\nsolution = 12",
            "Adding 2 yields 12.\nsolution = 12",
            "= 12.\nsolution = 12",
            "garbage.\nsolution = 99",          # decoy
        ],
        prompt2: [
            "12 * 3 = 36.\nsolution = 36",
            "Product is 36.\nsolution = 36",
            "= 36.\nsolution = 36",
            "wrong.\nsolution = 100",          # decoy
        ],
    }


def test_solve_easy_instance_returns_coherent_answer(tmp_path: Path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    # With per-sample budget accounting (mock matches anthropic), each
    # expansion of K=4 samples costs 4 calls. The 3-node chain needs at
    # least 3 expansions to reach a terminal; allow generous widening too.
    cfg.mcts.N = 80
    cfg.sampling.K_blackbox = 4

    llm = MockLLM(responses=_easy_responses())
    res = solve(EASY_PROBLEM, cfg=cfg, llm=llm, domain="math",
                tag="m5_easy")

    # The dense correct cluster should win at every level; final answer = 36.
    assert res.answer == "36", f"got {res.answer!r}"
    assert res.coherent is True, (
        f"expected coherent path; got flagged={res.flagged!r} sigma={res.sigma}"
    )
    assert res.sigma == 0.0
    assert res.flagged is None
    # The trace was written to JSONL.
    assert Path(res.trace_path).exists()


# ---------------------------------------------------------------------------
# Deliberately unsatisfiable instance: every response is non-numeric, so the
# integer gate fires at every depth -> no coherent terminal -> soft-relax.
# ---------------------------------------------------------------------------

UNSAT_PROBLEM = EASY_PROBLEM    # same structure, but scripted answers are bad


def _unsat_responses() -> dict[str, list[str]]:
    sub0 = "Compute the value 5 + 5."
    sub1 = "Use the answer from problem node_0 and add 2."
    sub2 = "Use the answer from problem node_1 and multiply by 3."

    p0 = build_subproblem_prompt(sub0, {})
    p1a = build_subproblem_prompt(sub1, {"node_0": "garbage_x"})
    p1b = build_subproblem_prompt(sub1, {"node_0": "wibble"})
    p2 = build_subproblem_prompt(sub2, {"node_1": "wibble"})

    # Every response's "solution =" tail is non-numeric -> sigma=1 at gate.
    return {
        p0: [
            "I dunno.\nsolution = garbage_x",
            "Eh.\nsolution = wibble",
            "?\nsolution = nope",
        ],
        p1a: [
            "??\nsolution = blah",
            "..\nsolution = blah",
        ],
        p1b: [
            "??\nsolution = blah",
            "..\nsolution = blah",
        ],
        p2: [
            "no.\nsolution = wat",
            "no.\nsolution = nada",
        ],
    }


def test_solve_unsatisfiable_instance_returns_flagged_soft_relax(tmp_path: Path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 8
    cfg.sampling.K_blackbox = 4

    llm = MockLLM(responses=_unsat_responses())
    res = solve(UNSAT_PROBLEM, cfg=cfg, llm=llm, domain="math",
                tag="m5_unsat")

    # Math integer gate must fire on at least one depth -> no coherent path.
    assert res.coherent is False, f"unexpectedly coherent: {res}"
    assert res.flagged is not None
    assert res.sigma > 0.0 or "no terminal" in (res.flagged or "")


# ---------------------------------------------------------------------------
# Budget ceiling holds end-to-end with the real orchestrator
# ---------------------------------------------------------------------------

def test_solve_respects_budget_ceiling(tmp_path: Path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 6
    cfg.sampling.K_blackbox = 3

    llm = MockLLM(responses=_easy_responses())
    res = solve(EASY_PROBLEM, cfg=cfg, llm=llm, domain="math")
    # Calls bounded by rollouts * K + a little headroom; we just assert NEVER
    # exceeds N (the rollout budget enforced by telemetry.check_budget would
    # have raised BudgetExceeded otherwise — the call ceiling is per-rollout-loop).
    assert llm.cost().calls <= cfg.mcts.N
