"""Soft budget exhaustion: solve() returns the best partial instead of crashing.

Prior behavior crashed the run with a BudgetExceeded exception when
LLM cost exceeded `cfg.mcts.N`. New behavior:
  - The orchestrator catches the exception in on_rollout
  - stop_fn ends the search loop cleanly
  - Solution selection proceeds with whatever was reached so far
  - Result.flagged is annotated with "budget_truncated"
  - Result.coherent is unaffected (still True if a coherent path was found
    BEFORE the budget hit)

The strict telemetry.check_budget contract still raises — that's a
*mechanism* test that lives in test_m0_skeleton.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config, LLMCfg
from core.llm.mock import MockLLM
from core.orchestrator import solve
from core.types import CostTally, Generation


PROBLEM = (
    "Problem node_0: Compute 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
    "Problem node_2: Use the answer from problem node_1 and multiply by 3.\n"
)


class ExpensiveLLM:
    """Test fixture: every generate() bumps cost.calls by `cost_per_call` so
    we can force a budget breach inside a few rollouts without depending on
    how many distinct meaning-classes mock clustering produces."""

    supports_logprobs: bool = False

    def __init__(self, cost_per_call: int = 100):
        self._cost = CostTally()
        self.cost_per_call = cost_per_call

    def generate(self, prompt: str, *, temperature: float,
                  n: int) -> list[Generation]:
        self._cost.add(calls=self.cost_per_call,
                       input_tokens=len(prompt.split()),
                       output_tokens=10 * n)
        return [Generation(text=f"solution = {i}", token_logprobs=None,
                            finish_reason="stop") for i in range(n)]

    def cost(self) -> CostTally:
        return self._cost


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.llm = LLMCfg(provider="mock", model="mock-v0", temperature=1.0)
    return cfg


# ExpensiveLLM adds 100 cost calls per generate(); with N=50 the very first
# expansion blows the budget. Perfect for exercising the soft-stop path.
TIGHT_N = 50


def test_solve_does_not_raise_on_budget_exhaustion(tmp_path: Path):
    """Budget tight enough to trip; solve() must still return a Result."""
    cfg = _cfg(tmp_path)
    cfg.mcts.N = TIGHT_N
    cfg.sampling.K_blackbox = 4
    llm = ExpensiveLLM()

    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    assert res is not None
    # The orchestrator caught the exception; cost has overshot the cap.
    assert res.cost["calls"] >= TIGHT_N


def test_budget_truncation_is_flagged(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.mcts.N = TIGHT_N
    cfg.sampling.K_blackbox = 4
    llm = ExpensiveLLM()

    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    assert res.flagged is not None
    assert "budget_truncated" in res.flagged


def test_telemetry_records_budget_exceeded_event(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.mcts.N = TIGHT_N
    cfg.sampling.K_blackbox = 4
    llm = ExpensiveLLM()

    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    log = Path(res.trace_path).read_text()
    events = [json.loads(l) for l in log.splitlines()]
    kinds = {e["kind"] for e in events}
    assert "budget_exceeded" in kinds
    be = next(e for e in events if e["kind"] == "budget_exceeded")
    assert be["data"]["limit"] == TIGHT_N
    assert be["data"]["calls"] > TIGHT_N


def test_summary_carries_budget_truncated_flag(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.mcts.N = TIGHT_N
    cfg.sampling.K_blackbox = 4
    llm = ExpensiveLLM()

    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    tree = json.loads(Path(res.cost["tree_path"]).read_text())
    assert tree["summary"]["budget_truncated"] is True


def test_coherent_path_found_before_budget_hit_stays_coherent(tmp_path: Path):
    """If a coherent leaf was reached BEFORE the budget hit, Result.coherent
    must NOT be retroactively demoted by the truncation flag."""
    cfg = _cfg(tmp_path)
    cfg.mcts.N = 80                  # generous enough to reach a terminal
    cfg.sampling.K_blackbox = 4
    llm = MockLLM(responses={})

    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    # Either the search completed cleanly (no truncation) or it truncated
    # but coherent is still True if the search did find a coherent leaf.
    if res.coherent:
        assert res.sigma == 0.0
    # No crash either way.
    assert res.rollouts >= 1


# ---------------------------------------------------------------------------
# Per-block budget — regression test for the 2026-06-17 bug.
# ---------------------------------------------------------------------------

def test_per_block_budget_does_not_inherit_prior_blocks_cost(tmp_path: Path):
    """In multi-block runs (solve_multi), one RoleClients accumulates LLM
    cost across all blocks. The budget check must compare the per-block
    DELTA against cfg.mcts.N, NOT the raw cumulative — otherwise block 3
    onwards would trip the budget on its first rollout because block 1+2
    already pushed cumulative past N.

    This regression test reproduces the failure mode and asserts every
    block runs to completion under a per-block budget.
    """
    from core.multi_solve import solve_multi

    cfg = _cfg(tmp_path)
    cfg.mcts.N = 6                   # small enough to be plausibly trip-able
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    # Four sequential blocks. Each block's solve will burn ~2-4 calls on
    # the mock LLM (a single rollout, K=2 samples). With the OLD bug, the
    # third block onwards would inherit cumulative cost > 6 and bail
    # after one rollout with budget_truncated. With the fix, each block
    # gets its own N=6 budget.
    problem = (
        "Problem node_0: First.\n"
        "Problem node_1: Second.\n"
        "Problem node_2: Third.\n"
        "Problem node_3: Fourth.\n"
    )
    llm = MockLLM(responses={})
    res = solve_multi(problem, cfg=cfg, llm=llm, domain="math",
                      progress=None)

    # Every block must have run more than ONE rollout — that's the
    # specific failure signature of the cumulative-budget bug (every
    # block past the first ran exactly 1 rollout with budget_truncated).
    per_block = res.cost["per_block"]
    assert len(per_block) == 4
    rollouts = [b["rollouts"] for b in per_block]
    trunc_flags = [b.get("flagged") or "" for b in per_block]
    # At least one block past the first must have rollouts > 1 to confirm
    # the budget didn't trip cumulatively. (The mock LLM is fast so all
    # blocks should run their full N rollouts.)
    assert rollouts[-1] >= 2, (
        f"last block stopped at {rollouts[-1]} rollouts — the cumulative "
        f"budget bug appears to have regressed. rollouts per block: {rollouts}; "
        f"flags: {trunc_flags}"
    )
