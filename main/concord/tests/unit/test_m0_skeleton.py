"""M0 gate: skeleton runs end-to-end with the mock LLM, cost tally works,
budget ceiling is enforced.

Per the plan §7 M0 gate: `solve()` stub runs on a toy problem end-to-end
returning a dummy result; cost tally works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the package importable when running pytest from the project root.
PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.llm.mock import MockLLM
from core.orchestrator import solve
from core.telemetry import BudgetExceeded, Telemetry
from core.types import CostTally, Generation


def test_default_config_loads(tmp_path: Path) -> None:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    assert cfg.mcts.c_puct == 3.0
    assert cfg.mcts.n_min == 3
    assert cfg.confidence.alpha == 0.25
    assert cfg.ablation.confidence == "semantic_density"


def test_mock_llm_returns_n_completions() -> None:
    llm = MockLLM(responses={"Q": ["A1", "A2", "A3"]}, with_logprobs=True)
    gens = llm.generate("Q", temperature=1.0, n=5)
    assert len(gens) == 5
    # cycles deterministically when bank shorter than n
    assert gens[3].text == "A1"
    # white-box mode populates logprobs
    assert gens[0].token_logprobs is not None
    assert all(lp < 0 for lp in gens[0].token_logprobs)


def test_mock_llm_cost_accumulates() -> None:
    """Cost is counted PER SAMPLE (not per generate() invocation) so that the
    mock budget matches the Anthropic adapter's per-call accounting."""
    llm = MockLLM()
    assert llm.cost().calls == 0
    llm.generate("hello world", temperature=1.0, n=2)
    llm.generate("hello world", temperature=1.0, n=3)
    c = llm.cost()
    assert c.calls == 5    # 2 + 3 samples
    assert c.input_tokens > 0
    assert c.output_tokens > 0


def test_solve_end_to_end_returns_result(tmp_path: Path) -> None:
    """End-to-end smoke test: solve() runs without crashing on a trivial
    problem and returns a Result with telemetry written. Behaviour beyond
    "doesn't crash, writes a trace" is the M5 walking-skeleton test's job.
    """
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 4
    cfg.sampling.K_blackbox = 2

    llm = MockLLM(responses={"any": ["solution = 4"]})
    res = solve("What is 2+2?", cfg=cfg, llm=llm, domain="math")

    # Returns a Result with a trace path
    log_path = Path(res.trace_path)
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    kinds = [json.loads(l)["kind"] for l in lines]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    assert "phase1" in kinds


def test_budget_check_raises_when_exceeded(tmp_path: Path) -> None:
    """Direct unit test of telemetry.check_budget contract.

    The orchestrator integration is more variable (caching, stop_fn) so
    we test the mechanism here and rely on the M5 integration test to
    verify it engages inside a real solve.
    """
    from core.telemetry import Telemetry
    from core.types import CostTally

    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2

    t = Telemetry.from_config(cfg)
    t.check_budget(CostTally(calls=2))   # at ceiling, ok
    with pytest.raises(BudgetExceeded):
        t.check_budget(CostTally(calls=3))   # over ceiling, fail


def test_telemetry_run_metadata_includes_config_hash(tmp_path: Path) -> None:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    llm = MockLLM()
    res = solve("hi", cfg=cfg, llm=llm)
    head = json.loads(Path(res.trace_path).read_text().splitlines()[0])
    assert head["kind"] == "run_start"
    assert head["data"]["config_hash"]
    assert head["data"]["config"]["mcts"]["c_puct"] == 3.0
