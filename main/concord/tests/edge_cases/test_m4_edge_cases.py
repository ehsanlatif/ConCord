"""M4 gate — coherence, governance, soft-relax, edge cases from spec §7.

Plan §7 M4 gate (edge-case suite from spec table):
- empty class
- bimodal / tied clusters
- whole level fails gate
- no coherent solution -> flagged fallback
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.confidence import (
    LexicalEmbedder,
    NullVerifier,
    ScriptedKernel,
    score_subproblem,
)
from core.search import (
    ExpandedClass,
    ExpandedNodeInfo,
    Node,
    depth_cap,
    evaluate,
    search,
    should_stop,
    soft_relax,
)
from core.search.depth import StopState
from core.structure import (
    condense,
    evaluate_gate,
    extract_graph,
    integer_answer_constraint,
    numeric_bound_constraint,
    predecessor_consistency_constraint,
)
from core.types import Generation, SolutionState


def _cfg():
    return Config.from_yaml(PKG_ROOT / "config" / "default.yaml")


# ===========================================================================
# Edge case 1 — empty dominant class
# ===========================================================================

def test_empty_class_does_not_divide_by_zero():
    """K=0 generations should NOT crash — orchestrator handles by re-sampling
    with higher K. We just verify the score_subproblem stays well-defined.
    """
    cfg = _cfg()
    res = score_subproblem(
        [],   # zero generations
        embedder=LexicalEmbedder(),
        kernel=ScriptedKernel(),
        verifier=NullVerifier(1.0),
        subproblem="?",
        cfg=cfg,
        supports_logprobs=False,
    )
    assert res.classes == []
    assert res.dominant is None
    assert math.isfinite(res.queried_pairs)


# ===========================================================================
# Edge case 2 — bimodal / tied clusters become separate PUCT children
# ===========================================================================

def test_bimodal_clusters_kept_as_separate_classes():
    """Two equal-mass meaning classes must coexist after clustering — they
    become two PUCT children, NOT collapsed into one.
    """
    cfg = _cfg()
    responses = ["yes", "yes", "no", "no"]
    pairs = {
        ("yes", "yes"): 1.0,
        ("no", "no"): 1.0,
        ("yes", "no"): 0.02,
    }
    gens = [Generation(text=r, token_logprobs=None, finish_reason="stop")
            for r in responses]
    res = score_subproblem(
        gens, embedder=LexicalEmbedder(),
        kernel=ScriptedKernel(pairs=pairs),
        verifier=NullVerifier(0.5), subproblem="?",
        cfg=cfg, supports_logprobs=False,
    )
    assert len(res.classes) == 2
    sizes = sorted(len(c.klass.members) for c in res.classes)
    assert sizes == [2, 2]


# ===========================================================================
# Edge case 3 — whole level fails gate becomes a low-value subtree
# ===========================================================================

def test_whole_level_failing_gate_yields_low_value():
    """If every expansion at depth d violates a constraint, the gate sets
    val=0 for all of them. Backup to root should converge to low Q.
    """
    cfg = _cfg()
    cfg.mcts.N = 16

    class AlwaysGoodPolicy:
        def expand(self, state, depth, *, parent_u_s, K):
            return ExpandedNodeInfo(
                classes=[ExpandedClass(class_key="bad", answer_text="not_a_number",
                                       mass=1.0, u_s=0.9)],
                subproblem_text="sub@" + str(depth), bindings={},
            )

    class DepthCap:
        def is_terminal(self, state, depth): return depth >= 2

    # An always-fires gate: every depth ≥ 1 the integer constraint fires
    # because answers are "not_a_number".
    constraints = [integer_answer_constraint()]
    def gate(state, depth):
        return evaluate_gate(state, depth, constraints).sigma

    terminals: list[Node] = []
    res = search(
        SolutionState(), cfg=cfg, policy=AlwaysGoodPolicy(),
        terminal_check=DepthCap(), gate_fn=gate, K=4,
        reached_terminals=terminals,
    )

    # Every reached terminal must be a gated_fail (sigma > 0) and contribute 0.
    assert terminals, "should have reached at least one terminal"
    assert all(t.gated_fail for t in terminals)
    assert all(t.sigma > 0 for t in terminals)
    # Root Q should be 0 — no coherent path exists.
    assert res.best_terminal is None


# ===========================================================================
# Edge case 4 — no coherent solution -> soft-relax fallback
# ===========================================================================

def test_soft_relax_returns_argmin_sigma():
    """Among reached terminals with σ > 0, soft_relax picks the one with the
    lowest sigma. Ties broken by higher Q. Since every terminal here is
    incoherent (σ > 0), the flag is the "all rejected" flag — not the
    softer "best-effort" wording — and `all_rejected` is True.
    """
    from core.search.fallback import ALL_REJECTED_FLAG

    t1 = Node(state=SolutionState(), depth=2, sigma=0.7, Q=0.2,
              terminal=True, gated_fail=True)
    t2 = Node(state=SolutionState(), depth=2, sigma=0.4, Q=0.3,
              terminal=True, gated_fail=True)
    t3 = Node(state=SolutionState(), depth=2, sigma=0.4, Q=0.8,
              terminal=True, gated_fail=True)

    fr = soft_relax([t1, t2, t3])
    assert fr is not None
    assert fr.node is t3   # min sigma, max Q breaks tie
    assert fr.sigma == 0.4
    assert fr.all_rejected is True
    assert fr.flag == ALL_REJECTED_FLAG


def test_soft_relax_keeps_best_effort_flag_when_any_coherent():
    """When at least one reached terminal has σ = 0, soft_relax still
    returns it (sigma 0, ranked first) with the softer "best-effort"
    flag — `all_rejected` must be False so the orchestrator does NOT
    suppress the answer.
    """
    bad = Node(state=SolutionState(), depth=2, sigma=0.5, Q=0.9,
               terminal=True, gated_fail=True)
    good = Node(state=SolutionState(), depth=2, sigma=0.0, Q=0.2,
                terminal=True)

    fr = soft_relax([bad, good])
    assert fr is not None
    assert fr.node is good
    assert fr.sigma == 0.0
    assert fr.all_rejected is False
    assert "soft-relax" in fr.flag


def test_soft_relax_returns_none_when_no_terminals():
    assert soft_relax([]) is None


# ===========================================================================
# Sigma always recorded (decision 6)
# ===========================================================================

def test_sigma_recorded_even_when_gate_fires():
    """Spec line 28: 'record σ(s); val ← 0; mark s terminal-fail'.

    The σ must be stored on the node, not just thrown away — soft-relax
    depends on this.
    """
    node = Node(state=SolutionState(), depth=1, U_s=0.9)

    def fire_gate(state, depth):
        return 0.6

    val = evaluate(node, terminal_check=None, gate_fn=fire_gate)
    assert val == 0.0
    assert node.gated_fail is True
    assert node.terminal is True
    assert node.sigma == 0.6


# ===========================================================================
# Constraints: applicable_at gates which fire at which depth
# ===========================================================================

def test_constraints_only_fire_when_applicable():
    constraints = [
        numeric_bound_constraint(lo=0, hi=100, subproblem_id="node_0"),
    ]
    # not yet resolved
    res = evaluate_gate(SolutionState(), depth=0, constraints=constraints)
    assert res.sigma == 0.0
    assert res.all_checked == []   # not applicable yet

    # resolved with valid answer
    s = SolutionState().extend("node_0", "42")
    res = evaluate_gate(s, depth=1, constraints=constraints)
    assert res.sigma == 0.0

    # resolved with out-of-bound answer
    s = SolutionState().extend("node_0", "9999")
    res = evaluate_gate(s, depth=1, constraints=constraints)
    assert res.sigma == 1.0
    assert res.violated


def test_predecessor_order_constraint_catches_out_of_order():
    G = extract_graph(
        "Problem node_0: foo\n"
        "Problem node_1: use the answer from problem node_0 to compute bar.\n"
    )
    Gp = condense(G)
    c = predecessor_consistency_constraint(Gp)

    # in-order: scc containing node_0 first, then node_1
    sccs = {n: Gp.nodes[n]["members"] for n in Gp.nodes}
    n0 = next(s for s, m in sccs.items() if "node_0" in m)
    n1 = next(s for s, m in sccs.items() if "node_1" in m)
    good = SolutionState().extend(n0, "x").extend(n1, "y")
    bad = SolutionState().extend(n1, "y").extend(n0, "x")

    assert evaluate_gate(good, depth=2, constraints=[c]).sigma == 0.0
    assert evaluate_gate(bad,  depth=2, constraints=[c]).sigma == 1.0


# ===========================================================================
# Depth governance — min(critical+slack, B//L) and marginal-gain stop
# ===========================================================================

def test_depth_cap_takes_min_of_structural_and_budget():
    Gp = condense(extract_graph(
        "Problem node_0: a\n"
        "Problem node_1: use the answer from problem node_0\n"
        "Problem node_2: use the answer from problem node_1\n"
        "Problem node_3: use the answer from problem node_2\n"
    ))
    # critical path = 3 edges, slack=2 -> structural=5
    structural = 5
    # budget_calls = 12, per_expansion = 8 -> budget_d = 1 (tight)
    cap_budget_tight = depth_cap(Gp, budget_calls=12, per_expansion_cost=8,
                                  slack=2)
    assert cap_budget_tight == 1

    # generous budget -> structural wins
    cap_budget_loose = depth_cap(Gp, budget_calls=10_000,
                                  per_expansion_cost=8, slack=2)
    assert cap_budget_loose == structural

    # fixed override (ablation arm)
    cap_fixed = depth_cap(Gp, budget_calls=10, per_expansion_cost=8,
                           slack=2, fixed=4)
    assert cap_fixed == 4


def test_marginal_gain_stop_fires_when_gain_low():
    s = StopState(best_at_depth={})
    s.update(1, 0.50)
    s.update(2, 0.51)     # gain = 0.01 only

    # cost_next = 1.0, lam = 0.02 -> threshold 0.02; gain 0.01 < threshold -> stop
    assert should_stop(s, current_depth=2, lam=0.02, cost_next=1.0) is True


def test_marginal_gain_stop_holds_when_gain_high():
    s = StopState(best_at_depth={})
    s.update(1, 0.30)
    s.update(2, 0.80)     # gain = 0.50, way above

    assert should_stop(s, current_depth=2, lam=0.02, cost_next=1.0) is False


def test_marginal_gain_holds_until_two_depths():
    s = StopState(best_at_depth={})
    s.update(0, 0.0)
    # only one depth seen for "gain" computation
    assert should_stop(s, current_depth=1, lam=0.02, cost_next=1.0) is False


# ===========================================================================
# Integration: gate + soft-relax in a real search
# ===========================================================================

def test_search_returns_no_coherent_then_soft_relax_chooses_min_sigma():
    cfg = _cfg()
    cfg.mcts.N = 12

    class MixedPolicy:
        """Every expansion produces two children: one with severity-1 violation
        (answer 'X') and one with severity-0.5 (answer 'Y'). Both fail the
        integer check, but with different sigma."""
        def expand(self, state, depth, *, parent_u_s, K):
            return ExpandedNodeInfo(
                classes=[
                    ExpandedClass(class_key="bad", answer_text="X",
                                  mass=0.5, u_s=0.5),
                    ExpandedClass(class_key="worse", answer_text="Y",
                                  mass=0.5, u_s=0.5),
                ],
                subproblem_text="sub@" + str(depth), bindings={},
            )

    class DepthCap:
        def is_terminal(self, state, depth): return depth >= 1

    # constraint: any non-integer is sigma 1.0
    def gate(state, depth):
        return 1.0 if any(not ans.lstrip("-").isdigit()
                          for _, ans in state.resolved) else 0.0

    terminals: list[Node] = []
    res = search(
        SolutionState(), cfg=cfg,
        policy=MixedPolicy(),
        terminal_check=DepthCap(),
        gate_fn=gate, K=4,
        reached_terminals=terminals,
    )
    assert res.best_terminal is None, (
        "no coherent terminal should be found — every leaf fails the gate"
    )
    assert terminals
    fr = soft_relax(terminals)
    assert fr is not None
    assert fr.sigma == 1.0    # both options have sigma 1 in this scenario
    # Every terminal here failed the gate → the stronger "all rejected"
    # flag fires, NOT the softer "best-effort" wording.
    from core.search.fallback import ALL_REJECTED_FLAG
    assert fr.all_rejected is True
    assert fr.flag == ALL_REJECTED_FLAG
