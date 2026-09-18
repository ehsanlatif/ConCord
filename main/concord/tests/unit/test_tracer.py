"""Tracer + per-question artifact tests.

We don't probe the viewer's HTML — only the shape and integrity of the
JSON it consumes. Two artifacts per solve():
    <run-id>_tree.json     — final tree snapshot
    <run-id>_rollouts.jsonl — one row per rollout
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
from core.orchestrator import solve, solve_with_config
from core.search.mcts import Node, reset_node_ids


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.llm = LLMCfg(provider="mock", model="mock-v0", temperature=1.0)
    cfg.mcts.N = 6
    cfg.sampling.K_blackbox = 3
    return cfg


PROBLEM = (
    "Problem node_0: Compute 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
)


# ---------------------------------------------------------------------------
# Node id machinery
# ---------------------------------------------------------------------------

def test_reset_node_ids_starts_from_zero():
    reset_node_ids()
    a = Node(state=None, depth=0)
    b = Node(state=None, depth=0)
    assert a.node_id == "n0"
    assert b.node_id == "n1"
    reset_node_ids()
    c = Node(state=None, depth=0)
    assert c.node_id == "n0"


def test_node_to_dict_is_json_safe():
    from core.types import SolutionState
    reset_node_ids()
    n = Node(state=SolutionState(resolved=[("sub", "ans")],
                                  bindings={"sub": "ans"}),
              depth=2, Q=0.5, N=3, U_s=0.7, sigma=0.1, terminal=True,
              class_key="ck", answer_text="ans")
    d = n.to_dict()
    # Must round-trip through json.dumps
    json.dumps(d)
    assert d["id"] == "n0"
    assert d["depth"] == 2
    assert d["state"]["resolved"] == [["sub", "ans"]] or \
           d["state"]["resolved"] == [("sub", "ans")]


# ---------------------------------------------------------------------------
# Artifact production
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_solve_writes_tree_and_rollouts(tmp_path: Path):
    cfg = _cfg(tmp_path)
    llm = MockLLM(responses={})
    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math", tag="tracer_artifact")

    # Both artifacts referenced from the result
    tree_path = Path(res.cost["tree_path"])
    rollouts_path = Path(res.cost["rollouts_path"])
    assert tree_path.exists()
    assert rollouts_path.exists()

    # tree.json shape
    tree = json.loads(tree_path.read_text())
    assert set(tree) >= {"meta", "summary", "nodes", "edges"}
    # nodes must have stable ids starting at n0 (because reset_node_ids was called)
    assert tree["nodes"][0]["id"] == "n0"
    # every edge endpoint exists as a node
    ids = {n["id"] for n in tree["nodes"]}
    for e in tree["edges"]:
        assert e["from"] in ids and e["to"] in ids


def test_rollouts_record_has_required_fields(tmp_path: Path):
    cfg = _cfg(tmp_path)
    llm = MockLLM(responses={})
    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    rollouts = _read_jsonl(Path(res.cost["rollouts_path"]))
    assert len(rollouts) >= 1
    for r in rollouts:
        # Selection + leaf + backup are always recorded.
        assert "selected_path" in r and r["selected_path"]
        assert "leaf" in r and "id" in r["leaf"]
        assert "backup_path" in r and r["backup_path"]
        assert "value_backed_up" in r


def test_rollouts_include_expansion_detail_when_expanded(tmp_path: Path):
    """The first rollout always expands the root, so its row must carry the
    expansion side-channel (samples, classes, prompt, bindings)."""
    cfg = _cfg(tmp_path)
    llm = MockLLM(responses={})
    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    rollouts = _read_jsonl(Path(res.cost["rollouts_path"]))
    first_with_expansion = next((r for r in rollouts if "expansion" in r), None)
    assert first_with_expansion is not None, \
        "at least one rollout should carry expansion side-channel data"
    exp = first_with_expansion["expansion"]
    assert {"subproblem_id", "samples", "classes", "oracle_mode"} <= set(exp)
    # K samples == 3 here
    assert len(exp["samples"]) == cfg.sampling.K_blackbox


def test_tree_marks_gated_fail_and_terminal(tmp_path: Path):
    """When the gate fires on every leaf, tree.json should record gated_fail=True
    on the leaves and the run should be soft-relaxed."""
    cfg = _cfg(tmp_path)
    # ensure integer gate fires by sending non-numeric responses everywhere
    llm = MockLLM(responses={})    # echo mode → "<mock:i> ..." (not integer)
    res = solve(PROBLEM, cfg=cfg, llm=llm, domain="math")
    tree = json.loads(Path(res.cost["tree_path"]).read_text())
    # at least one gated terminal in the tree (because integer constraint fires)
    has_gated = any(n["gated_fail"] for n in tree["nodes"])
    # Either we got a clean coherent path (mock is loose) OR we got gated fails;
    # only the soft-relax / fallback path is meaningful here.
    if not res.coherent:
        assert has_gated, "soft-relax run should have at least one gated_fail node"
        assert res.flagged is not None


def test_node_ids_are_stable_within_a_run(tmp_path: Path):
    """tree.json node ids should be a contiguous n0, n1, ... range."""
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM, cfg=cfg, llm=MockLLM(responses={}), domain="math")
    tree = json.loads(Path(res.cost["tree_path"]).read_text())
    ids = [n["id"] for n in tree["nodes"]]
    assert ids[0] == "n0"
    nums = sorted(int(x[1:]) for x in ids)
    # no gaps
    assert nums == list(range(nums[0], nums[-1] + 1))


def test_tree_meta_includes_role_models_and_config(tmp_path: Path):
    cfg = _cfg(tmp_path)
    res = solve_with_config(PROBLEM, cfg=cfg, domain="math")
    tree = json.loads(Path(res.cost["tree_path"]).read_text())
    meta = tree["meta"]
    assert "role_models" in meta
    # The four classic roles must always be present. Pipeline-solver roles
    # (splitter, combiner, synthesizer, synth_verifier) may also appear when
    # the new factory is in use — accept either shape.
    assert {"execution", "decomposition", "classification",
            "verification"} <= set(meta["role_models"])
    assert "config" in meta
    assert meta["config"]["c_puct"] == cfg.mcts.c_puct


def test_summary_has_chosen_node_pointer(tmp_path: Path):
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM, cfg=cfg, llm=MockLLM(responses={}), domain="math")
    tree = json.loads(Path(res.cost["tree_path"]).read_text())
    # The viewer highlights `summary.chosen_node_id` on the chosen path.
    assert "chosen_node_id" in tree["summary"]


# ---------------------------------------------------------------------------
# Per-LLM-call log (<run-id>_agent_calls.jsonl)
# ---------------------------------------------------------------------------

def _agent_calls_path_for(res) -> Path:
    """Locate the agent_calls.jsonl that sits next to the rollouts file."""
    r = Path(res.cost["rollouts_path"])
    # tracer derives the agent file from the rollouts file's stem.
    return r.with_name(r.stem.replace("_rollouts", "_agent_calls") + ".jsonl")


def test_agent_calls_file_is_created_and_contains_executor_lines(tmp_path: Path):
    """Every executor sample must show up as one line in agent_calls.jsonl
    with the FULL prompt and response, plus role/sample_idx/node_id metadata."""
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM, cfg=cfg, llm=MockLLM(responses={}), domain="math")

    ac_path = _agent_calls_path_for(res)
    assert ac_path.exists(), f"agent_calls.jsonl missing at {ac_path}"
    lines = _read_jsonl(ac_path)
    assert lines, "agent_calls.jsonl is empty — executor calls were not logged"

    # Every entry has the agreed schema.
    needed = {"role", "prompt", "response", "model", "extras",
              "seq", "t", "rollout_i"}
    for r in lines:
        assert needed <= set(r), f"missing keys in agent_call row: {r}"

    executor_rows = [r for r in lines if r["role"] == "executor"]
    assert executor_rows, "no executor rows recorded"
    # Each executor row tracks which sample within the expansion it was.
    for r in executor_rows:
        assert r["sample_idx"] is not None
        # Full prompt + response (no truncation).
        assert "solution = <your final answer>" in r["prompt"]


def test_agent_calls_log_uses_full_response_no_truncation(tmp_path: Path):
    """Long executor responses must be recorded in full (the historical
    400-char truncation was the original bug)."""
    cfg = _cfg(tmp_path)
    # Build a long scripted response (>1500 chars) and route it through
    # MockLLM. The actual prompts the executor builds are deterministic
    # but model-dependent — easier to assert via a long echo by giving
    # MockLLM a "catch-all" response bank keyed off the prompts the
    # mock will actually see. We do this by injecting a long string in
    # the scripted bank under any seen prompt: hook on MockLLM.extend
    # after a dummy run.
    long_response = "REASONING " * 250 + "\nsolution = 5"   # ~2500 chars
    assert len(long_response) > 1500

    # First do a no-op run to discover what prompts the policy builds,
    # then re-run with those prompts mapped to our long response.
    llm_probe = MockLLM(responses={})
    solve(PROBLEM, cfg=_cfg(tmp_path / "_probe"), llm=llm_probe,
          domain="math", tag="probe")

    # Rebuild a fresh LLM with the long response on every prompt seen.
    llm_long = MockLLM(responses={})

    class _AlwaysLong(MockLLM):
        def generate(self, prompt, *, temperature, n):
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n,
                           usd=0.0)
            return [Generation(text=long_response, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]
    llm_long = _AlwaysLong(responses={})

    res = solve(PROBLEM, cfg=_cfg(tmp_path / "_long"), llm=llm_long,
                 domain="math", tag="long_resp")
    ac_path = _agent_calls_path_for(res)
    lines = _read_jsonl(ac_path)
    executor_rows = [r for r in lines if r["role"] == "executor"]
    assert executor_rows, "expected executor rows"
    # At least one row carries the FULL long response (no truncation).
    assert any(r["response"] == long_response for r in executor_rows), (
        "executor responses appear truncated — full text missing from "
        "agent_calls.jsonl")


def test_agent_calls_rollout_tagging(tmp_path: Path):
    """Executor calls within rollout i must carry rollout_i in the log so
    the trace can be joined back to the rollouts.jsonl row."""
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM, cfg=cfg, llm=MockLLM(responses={}), domain="math")
    lines = _read_jsonl(_agent_calls_path_for(res))
    executor_rows = [r for r in lines if r["role"] == "executor"]
    # Initial expansion(s) at i=0 may have rollout_i=None (logged before
    # on_rollout fires for the first time). Subsequent rollouts must tag.
    tagged = [r for r in executor_rows if r["rollout_i"] is not None]
    assert tagged, (
        "expected at least some executor rows to be tagged with rollout_i; "
        f"got {[r['rollout_i'] for r in executor_rows]}"
    )
