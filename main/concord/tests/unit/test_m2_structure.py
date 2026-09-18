"""M2 gate — structure (graph extract, SCC condense, critical path, decomposer).

Plan §7 gate:
- cyclic G condenses correctly
- critical path matches hand-computed cases
- atomic problems return ATOMIC
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.structure import (
    ATOMIC,
    ExplicitDecomposer,
    ImplicitDecomposerStub,
    condense,
    critical_path_len,
    extract_graph,
    granularity_for,
    is_atomic,
    parse_explicit_nodes,
    topo_order,
)
from core.types import SolutionState


# ---------------------------------------------------------------------------
# Explicit graph extraction from eval-set-style prompts
# ---------------------------------------------------------------------------

CHAIN_PROMPT = """Solve this problem step by step.

Problem node_0: Compute X = 42 + 1.
Problem node_1: Using the answer from problem node_0, compute Y = X * 2.
Problem node_2: Using the answer from problem node_1, compute Z = Y - 5.
Problem node_3: Using the answer from problem node_2, return Z mod 7.
"""

DAG_PROMPT = """Two-source problem.

Problem node_0: Compute A.
Problem node_1: Compute B.
Problem node_2: Using the answer from problem node_0 and the answer from problem node_1, combine them.
"""


def test_parse_explicit_nodes_chain():
    nodes = parse_explicit_nodes(CHAIN_PROMPT)
    assert [n.node_id for n in nodes] == ["node_0", "node_1", "node_2", "node_3"]
    assert nodes[0].refs == []
    assert nodes[1].refs == ["node_0"]
    assert nodes[2].refs == ["node_1"]
    assert nodes[3].refs == ["node_2"]


def test_parse_explicit_nodes_dag():
    nodes = parse_explicit_nodes(DAG_PROMPT)
    assert nodes[2].refs == ["node_0", "node_1"]


def test_parse_picks_up_richly_phrased_cross_block_references():
    """The dataset uses many phrasings ending in `from problem node_K`:
    `the answer from`, `the fraction from`, `the solution set from`,
    `the total area from`, `the radical term from`, etc. Every variant
    must be detected as a dependency — the earlier `answer\\s+from\\s+
    problem\\s+node_(\\d+)` regex silently dropped most of them and
    made downstream blocks look independent (the user observed this
    on `conditional_easy_4`: node_6 referenced node_5 via "the
    denominator of the reduced form of the fraction from problem
    node_5" and got mis-ordered as a layer-0 source)."""
    text = (
        "Problem node_0: First.\n"
        "Problem node_5: A fraction problem.\n"
        "Problem node_6: Use the denominator of the reduced form of the "
        "fraction from problem node_5 and add 14.\n"
        "Problem node_10: A solution-set problem.\n"
        "Problem node_11: Use the x-coordinate of the second ordered "
        "pair in the solution set from problem node_10 and add 10.\n"
        "Problem node_12: Use the total area from problem node_11 "
        "and subtract 80.\n"
        "Problem node_13: Use the radical term from problem node_12.\n"
    )
    nodes = parse_explicit_nodes(text)
    by_id = {n.node_id: n for n in nodes}
    assert by_id["node_6"].refs == ["node_5"], (
        "node_6's `fraction from problem node_5` was not detected"
    )
    assert by_id["node_11"].refs == ["node_10"], (
        "node_11's `solution set from problem node_10` was not detected"
    )
    assert by_id["node_12"].refs == ["node_11"]
    assert by_id["node_13"].refs == ["node_12"]


def test_parse_does_not_match_unrelated_problem_node_mentions():
    """The broader regex requires the word `from` immediately before
    `problem node_K` — bare mentions like `Problem node_3:` (a header)
    or generic "problem node_3 is hard" must NOT be picked up as
    cross-block references."""
    text = (
        "Problem node_0: An unrelated mention of problem node_2 here.\n"
        "Problem node_2: Use the answer from problem node_0.\n"
    )
    nodes = parse_explicit_nodes(text)
    by_id = {n.node_id: n for n in nodes}
    assert by_id["node_0"].refs == [], (
        "bare mention of `problem node_2` (no `from`) must NOT be a ref"
    )
    assert by_id["node_2"].refs == ["node_0"]


def test_extract_graph_falls_back_to_root_when_no_markers():
    text = "Find the best chess move from this FEN: 8/8/8/8/..."
    G = extract_graph(text)
    assert list(G.nodes) == ["root"]
    assert G.nodes["root"]["source"] == "implicit"


def test_critical_path_chain_is_n_minus_1():
    G = extract_graph(CHAIN_PROMPT)
    Gp = condense(G)
    assert critical_path_len(Gp) == 3   # 4 nodes in a line -> 3 edges


def test_critical_path_dag_split():
    G = extract_graph(DAG_PROMPT)
    Gp = condense(G)
    assert critical_path_len(Gp) == 1   # root -> dependent (depth 1)


def test_critical_path_singleton_is_zero():
    G = extract_graph("a problem with no markers")
    Gp = condense(G)
    assert critical_path_len(Gp) == 0


# ---------------------------------------------------------------------------
# SCC condensation handles cycles
# ---------------------------------------------------------------------------

def test_cyclic_graph_condenses_correctly():
    G: nx.DiGraph = nx.DiGraph()
    # Cycle a -> b -> c -> a; plus an outsider d that depends on c.
    for n in ["a", "b", "c", "d"]:
        G.add_node(n, text=n)
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])

    Gp = condense(G)
    # 2 SCCs: {a, b, c} and {d}
    assert Gp.number_of_nodes() == 2
    sizes = sorted(len(Gp.nodes[n]["members"]) for n in Gp.nodes)
    assert sizes == [1, 3]
    # The result is now a DAG
    assert nx.is_directed_acyclic_graph(Gp)
    # critical path = 1 edge (the SCC -> d)
    assert critical_path_len(Gp) == 1


# ---------------------------------------------------------------------------
# Topological order is stable
# ---------------------------------------------------------------------------

def test_topo_order_stable():
    G = extract_graph(CHAIN_PROMPT)
    Gp = condense(G)
    order1 = topo_order(Gp)
    order2 = topo_order(Gp)
    assert order1 == order2


# ---------------------------------------------------------------------------
# Explicit decomposer walks G' correctly
# ---------------------------------------------------------------------------

def test_explicit_decomposer_walks_chain():
    G = extract_graph(CHAIN_PROMPT)
    Gp = condense(G)
    dec = ExplicitDecomposer()

    state = SolutionState()
    visited: list[str] = []
    for _ in range(10):
        sp = dec.next_subproblem(state, Gp)
        if sp == ATOMIC:
            break
        visited.append(sp.node_id)
        state = state.extend(sp.node_id, f"answer_for_{sp.node_id}")

    # Expect SCC nodes in topological order
    assert len(visited) == Gp.number_of_nodes()
    # And the loop terminates with ATOMIC once everything is resolved
    assert is_atomic(state, Gp)


def test_explicit_decomposer_respects_predecessor_completeness():
    """When a node has unresolved predecessors, it must NOT be returned next."""
    G = extract_graph(DAG_PROMPT)
    Gp = condense(G)
    dec = ExplicitDecomposer()

    # Resolve node_0 only; node_2 needs both 0 and 1 -> only 1 should be next.
    state = SolutionState().extend("scc_0", "A_value")
    # The SCC ids depend on how networkx numbers them; use the SCC that
    # contains 'node_0' as the resolved one. Re-look it up.
    sccs = {n: Gp.nodes[n]["members"] for n in Gp.nodes}

    def find_scc(orig: str) -> str:
        return next(s for s, m in sccs.items() if orig in m)

    state = SolutionState().extend(find_scc("node_0"), "A_value")
    sp = dec.next_subproblem(state, Gp)
    assert sp != ATOMIC
    # Next ready node must be node_1 (not node_2 which still has unresolved deps).
    assert "node_1" in Gp.nodes[sp.node_id]["members"]


def test_atomic_when_everything_resolved():
    G = extract_graph(CHAIN_PROMPT)
    Gp = condense(G)
    state = SolutionState()
    for n in Gp.nodes:
        state = state.extend(n, "x")
    assert is_atomic(state, Gp)


def test_implicit_stub_runs():
    G = extract_graph("a non-decomposed problem")
    Gp = condense(G)
    dec = ImplicitDecomposerStub()
    sp = dec.next_subproblem(SolutionState(), Gp)
    assert sp != ATOMIC
    assert sp.node_id == "scc_0"  # SCC wrapping the single 'root' node


def test_granularity_helper():
    assert granularity_for(0.9) >= granularity_for(0.5)
    assert granularity_for(0.3) == 1
