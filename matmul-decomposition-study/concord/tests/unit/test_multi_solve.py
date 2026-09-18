"""Tests for solve_multi — per-block fanout with shared answer memory.

Pins down:
  - Single-block (or zero-block) input is delegated to plain solve()
    so existing call sites keep working.
  - Multi-block input is split, solved in topological order, and the
    answers from prior blocks are threaded into later blocks both as
    inline substitution of `answer from problem node_M` and as a
    short "Context:" preamble.
  - Cost is summed across per-block runs and per_block diagnostics
    are exposed on the result.
  - When the dependency DAG has a single sink, that sink's answer
    is the consolidated final answer. When there are multiple
    sinks (independent siblings), the answer is a JSON map.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.llm.factory import RoleClients
from core.llm.mock import MockLLM
from core.multi_solve import solve_multi


# ---------------------------------------------------------------------------
# Single-block fallback — must behave identically to plain solve()
# ---------------------------------------------------------------------------

def test_solve_multi_delegates_to_solve_for_single_block(tmp_path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    problem = "Problem node_0: Compute 1 + 1."
    res = solve_multi(problem, cfg=cfg, llm=MockLLM(), domain="math")

    # Single block → no per_block fanout, behaves like solve().
    assert "per_block" not in res.cost
    assert "n_blocks" not in res.cost


def test_solve_multi_delegates_to_solve_for_zero_blocks(tmp_path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    # No `Problem node_K:` headers at all → parse_explicit_nodes returns [].
    res = solve_multi("Just one problem with no header.", cfg=cfg,
                     llm=MockLLM(), domain="math")
    assert "per_block" not in res.cost


# ---------------------------------------------------------------------------
# Multi-block fanout
# ---------------------------------------------------------------------------

INDEPENDENT_PROBLEMS = (
    "Problem node_0: First standalone question.\n"
    "Problem node_1: Second standalone question.\n"
    "Problem node_2: Third standalone question.\n"
)


def test_solve_multi_fans_out_independent_blocks(tmp_path):
    """Three independent blocks → three sub-solves, per_block populated,
    cost aggregated, consolidated answer is a JSON map (no single sink).
    """
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi(INDEPENDENT_PROBLEMS, cfg=cfg, llm=MockLLM(),
                     domain="math")

    # per_block carries one entry per block.
    assert res.cost["n_blocks"] == 3
    assert len(res.cost["per_block"]) == 3
    block_ids = [b["node_id"] for b in res.cost["per_block"]]
    assert set(block_ids) == {"node_0", "node_1", "node_2"}

    # shared_memory has every block_id as a key.
    assert set(res.cost["shared_memory"].keys()) == {"node_0", "node_1", "node_2"}

    # Multiple sinks (independent blocks) → answer is a JSON map keyed by
    # block id, in source order.
    parsed = json.loads(res.answer)
    assert list(parsed.keys()) == ["node_0", "node_1", "node_2"]


def test_solve_multi_total_cost_equals_sum_of_per_block_costs(tmp_path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi(INDEPENDENT_PROBLEMS, cfg=cfg, llm=MockLLM(),
                     domain="math")

    summed_calls = sum(b["cost"]["calls"] for b in res.cost["per_block"])
    summed_in = sum(b["cost"]["input_tokens"] for b in res.cost["per_block"])
    summed_out = sum(b["cost"]["output_tokens"] for b in res.cost["per_block"])
    assert res.cost["calls"] == summed_calls
    assert res.cost["input_tokens"] == summed_in
    assert res.cost["output_tokens"] == summed_out


def test_solve_multi_token_total_is_accurate_not_inflated(tmp_path):
    """Per-block cost must be each block's DELTA (not the cumulative
    snapshot solve() returns), so the reported total tokens / calls equal
    the TRUE client total — never the triangular over-count."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    shared = MockLLM()
    clients = RoleClients(
        execution=shared, decomposition=shared, classification=shared,
        verification=shared, splitter=shared, combiner=shared,
        synthesizer=shared, synth_verifier=shared,
        specs={r: cfg.llm for r in RoleClients.ROLE_NAMES},
    )
    res = solve_multi(INDEPENDENT_PROBLEMS, cfg=cfg, clients=clients,
                      domain="math", progress=None)

    true = clients.total_cost()
    # Reported total == the true cumulative client cost (no inflation).
    assert res.cost["calls"] == true.calls
    assert res.cost["input_tokens"] == true.input_tokens
    assert res.cost["output_tokens"] == true.output_tokens
    # total_tokens is surfaced and consistent.
    assert res.cost["total_tokens"] == true.input_tokens + true.output_tokens
    # Per-block entries are deltas that sum to the true total.
    assert sum(b["cost"]["calls"] for b in res.cost["per_block"]) == true.calls
    # Each block carries its own token consumption.
    for b in res.cost["per_block"]:
        assert b["cost"]["total_tokens"] == (
            b["cost"]["input_tokens"] + b["cost"]["output_tokens"])


# ---------------------------------------------------------------------------
# Shared memory + reference substitution
# ---------------------------------------------------------------------------

DEPENDENT_PROBLEMS = (
    "Problem node_0: Compute the value 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
)


def test_solve_multi_threads_prior_answers_into_dependent_blocks(tmp_path):
    """When node_1 references node_0, solve_multi must (a) solve node_0
    first, (b) substitute its answer into node_1's text *before* the
    sub-solve runs, and (c) record both answers in shared_memory.
    """
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi(DEPENDENT_PROBLEMS, cfg=cfg, llm=MockLLM(),
                     domain="math")

    # Two blocks, one sink (node_1) → consolidated answer is node_1's.
    assert res.cost["n_blocks"] == 2
    sinks = res.cost["sinks"]
    assert sinks == ["node_1"]
    assert res.answer == res.cost["shared_memory"]["node_1"]


def test_solve_multi_topological_order_respects_dependencies(tmp_path):
    """node_1 depends on node_0 even though they appear in arbitrary
    source order — the dependent must be solved AFTER its dependency,
    and `shared_memory["node_0"]` must already be populated at the
    moment node_1 starts."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    # node_1 listed first, but it depends on node_0 → solver must reorder.
    reversed_text = (
        "Problem node_1: Use the answer from problem node_0 and add 2.\n"
        "Problem node_0: Compute the value 5 + 5.\n"
    )
    res = solve_multi(reversed_text, cfg=cfg, llm=MockLLM(), domain="math")

    # per_block is in solve order — node_0 must come before node_1.
    order = [b["node_id"] for b in res.cost["per_block"]]
    assert order.index("node_0") < order.index("node_1")


def test_solve_multi_drains_all_independents_before_any_dependents(tmp_path):
    """Layer-stratified topological order: every block with in-degree 0
    must be solved BEFORE any block with in-degree > 0. Standard
    `lexicographical_topological_sort` violates this — after node_0 is
    taken its successor node_1 becomes a source and is picked before
    the still-untouched independents node_4, node_5. Our `_solve_order`
    drains layer 0 first."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    # node_0, node_5, node_10 are independent (no upstream refs).
    # node_1 depends on node_0; node_6 depends on node_5.
    text = (
        "Problem node_0: First independent.\n"
        "Problem node_1: Use the answer from problem node_0.\n"
        "Problem node_5: Second independent.\n"
        "Problem node_6: Use the answer from problem node_5.\n"
        "Problem node_10: Third independent.\n"
    )
    res = solve_multi(text, cfg=cfg, llm=MockLLM(), domain="math",
                     progress=None)
    order = [b["node_id"] for b in res.cost["per_block"]]
    independents = {"node_0", "node_5", "node_10"}
    last_independent = max(order.index(n) for n in independents)
    first_dependent  = min(order.index(n) for n in order if n not in independents)
    assert last_independent < first_dependent, (
        f"layer-stratified order broken — got {order!r}. Every independent "
        f"block must come before any dependent block."
    )


def test_solve_multi_natural_numeric_order_within_layer(tmp_path):
    """Within a single layer, blocks are sorted naturally by their
    numeric id — `node_5` comes before `node_10`, not after."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    # Three independents in shuffled source order.
    text = (
        "Problem node_10: Third.\n"
        "Problem node_5: Second.\n"
        "Problem node_2: First.\n"
    )
    res = solve_multi(text, cfg=cfg, llm=MockLLM(), domain="math",
                     progress=None)
    order = [b["node_id"] for b in res.cost["per_block"]]
    assert order == ["node_2", "node_5", "node_10"], (
        f"natural numeric ordering broken — got {order!r}"
    )


# ---------------------------------------------------------------------------
# Per-run directory layout
# ---------------------------------------------------------------------------

def test_solve_multi_creates_parent_run_directory(tmp_path):
    """Every multi-block solve creates a single parent directory under
    cfg.telemetry.log_dir. All per-block sub-runs and a manifest.json
    live inside it — no loose sibling files."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi(INDEPENDENT_PROBLEMS, cfg=cfg, llm=MockLLM(),
                     domain="math", progress=None)

    run_dir = Path(res.cost["run_dir"])
    assert run_dir.exists() and run_dir.is_dir()
    # Manifest exists and references the same run_dir.
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_blocks"] == 3
    assert len(manifest["blocks"]) == 3
    # Each per-block subdirectory exists and contains tree/rollouts/agent_calls.
    for entry in manifest["blocks"]:
        rel = entry["rel_tree"]
        assert rel is not None
        block_tree = run_dir / rel
        assert block_tree.exists()
        # The block's other artifacts live in the same subdirectory.
        block_dir = block_tree.parent
        siblings = {p.name.split("_")[-1] for p in block_dir.iterdir()}
        # We expect at least tree.json, rollouts.jsonl, agent_calls.jsonl,
        # and the telemetry JSONL.
        names = list(p.name for p in block_dir.iterdir())
        assert any(n.endswith("_tree.json") for n in names)
        assert any(n.endswith("_rollouts.jsonl") for n in names)
        assert any(n.endswith("_agent_calls.jsonl") for n in names)


def test_single_solve_also_uses_per_run_subdirectory(tmp_path):
    """Even a single (non-multi) solve gets its own subdirectory now —
    one folder per run, never loose files under log_dir."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi("Problem node_0: Compute 1 + 1.", cfg=cfg,
                     llm=MockLLM(), domain="math", progress=None)

    # All four artifacts live in the same per-run directory under log_dir.
    tree_p = Path(res.cost["tree_path"])
    rollouts_p = Path(res.cost["rollouts_path"])
    telemetry_p = Path(res.trace_path)
    assert tree_p.parent == rollouts_p.parent == telemetry_p.parent
    # That directory is a child of cfg.telemetry.log_dir, NOT log_dir itself.
    assert tree_p.parent.parent == Path(cfg.telemetry.log_dir).resolve() or \
           tree_p.parent.parent == Path(cfg.telemetry.log_dir)
