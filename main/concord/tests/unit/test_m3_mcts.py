"""M3 gate — MCTS core.

Plan §7 gate:
- widening grows children sublinearly in N
- n_min guard provably blocks single-rollout inflation
- backup reaches the root
- budget cap is never exceeded
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.search import (
    ExpandedClass,
    ExpandedNodeInfo,
    Node,
    backup,
    expand,
    may_widen,
    search,
    select,
    widening_cap,
)
from core.search.transposition import TTKey, TranspositionTable
from core.types import SolutionState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    for k, v in overrides.items():
        # nested setter for top-level groups
        if "." in k:
            grp, fld = k.split(".", 1)
            setattr(getattr(cfg, grp), fld, v)
        else:
            setattr(cfg, k, v)
    return cfg


class SyntheticPolicy:
    """Returns a fixed bouquet of K class summaries per call.

    Used as the M3 ExpansionPolicy fixture: the engine's MCTS mechanics can
    be tested without any LLM. `value_ladder` gives each successive class a
    slightly different u_s so the prior ordering is non-trivial.
    """

    def __init__(self, n_classes: int = 4, terminal_depth: int = 3,
                 base_u: float = 0.5):
        self.n_classes = n_classes
        self.terminal_depth = terminal_depth
        self.base_u = base_u
        self.calls = 0

    def expand(self, state: SolutionState, depth: int, *,
               parent_u_s: float, K: int) -> ExpandedNodeInfo:
        self.calls += 1
        classes = []
        for i in range(self.n_classes):
            classes.append(ExpandedClass(
                class_key=f"d{depth}_c{i}",
                answer_text=f"answer_{depth}_{i}",
                mass=(self.n_classes - i) / self.n_classes,
                u_s=self.base_u + 0.1 * (self.n_classes - i) / self.n_classes,
                terminal_hint=(depth + 1 >= self.terminal_depth),
            ))
        return ExpandedNodeInfo(classes=classes,
                                subproblem_text=f"sub@d{depth}",
                                bindings={})


class DepthTerminal:
    def __init__(self, depth: int):
        self.cap = depth
    def is_terminal(self, state: SolutionState, depth: int) -> bool:
        return depth >= self.cap


# ---------------------------------------------------------------------------
# Widening: sublinear in N
# ---------------------------------------------------------------------------

def test_widening_cap_grows_sublinearly():
    """C=2, beta=0.5 -> cap = floor(2 * sqrt(N)). Children << visits at large N."""
    for N in [1, 4, 9, 16, 25, 100]:
        cap = widening_cap(N, C=2.0, beta=0.5)
        assert cap == int(math.floor(2.0 * math.sqrt(N)))
        assert cap <= N + 1   # sub-linear


def test_widening_predicate_matches_cap():
    assert may_widen(N=1, n_children=0, C=2.0, beta=0.5) is True
    assert may_widen(N=1, n_children=2, C=2.0, beta=0.5) is False
    assert may_widen(N=4, n_children=4, C=2.0, beta=0.5) is False
    assert may_widen(N=9, n_children=4, C=2.0, beta=0.5) is True


def test_widening_grows_sublinearly_under_real_loop():
    """End-to-end check inside the rollout loop. After N rollouts the root
    has at most O(sqrt(N)) children (the widening cap)."""
    cfg = _cfg(**{"mcts.N": 100, "mcts.C": 2.0, "mcts.beta": 0.5})
    res = search(
        SolutionState(),
        cfg=cfg,
        policy=SyntheticPolicy(n_classes=10, terminal_depth=99),
        terminal_check=DepthTerminal(99),
        K=8,
    )
    root = res.root
    cap = widening_cap(root.N, cfg.mcts.C, cfg.mcts.beta)
    assert len(root.children) <= cap
    # And sublinear in N (proven by the cap itself, but assert numerically)
    assert len(root.children) <= int(2 * math.sqrt(cfg.mcts.N)) + 1


# ---------------------------------------------------------------------------
# n_min guard prevents single-rollout inflation
# ---------------------------------------------------------------------------

def test_n_min_blocks_single_rollout_inflation():
    """If a child has visited only once with a very high value, the parent's
    Q must NOT jump to that value while N(child) < n_min.
    """
    cfg = _cfg(**{"mcts.n_min": 3})

    parent = Node(state=SolutionState(), depth=0, U_s=0.5, N=0)
    parent.expanded = True
    # Two children. We will visit child A once with value 1.0, twice with 0.4.
    parent.children["A"] = Node(state=SolutionState(), depth=1, parent=parent,
                                U_s=0.4, N=0, Q=0.0)
    parent.children["B"] = Node(state=SolutionState(), depth=1, parent=parent,
                                U_s=0.6, N=0, Q=0.0)

    # First backup: child A, value 1.0. With n_min=3 nobody qualifies yet,
    # so parent.Q must equal parent.U_s (the guard).
    backup(parent.children["A"], 1.0, cfg)
    assert parent.children["A"].N == 1
    assert parent.N == 1
    assert parent.Q == pytest.approx(parent.U_s), (
        "single rollout must NOT push parent Q to 1.0 while n_min unmet"
    )

    # Run more backups so A reaches n_min.
    backup(parent.children["A"], 0.4, cfg)
    backup(parent.children["A"], 0.4, cfg)
    # Now A has N=3, Q=0.4 (last assigned). Parent Q should equal max(0.4) = 0.4.
    assert parent.children["A"].N == 3
    assert parent.Q == pytest.approx(0.4)

    # If we then send a single high-value rollout via B, parent Q should still
    # cap at A's qualified Q because B has N=1 < n_min.
    backup(parent.children["B"], 1.0, cfg)
    assert parent.children["B"].N == 1
    assert parent.Q == pytest.approx(0.4)   # guard still holds


def test_average_backup_ablation_does_not_guard():
    cfg = _cfg(**{"mcts.n_min": 999, "ablation.backup": "average"})
    parent = Node(state=SolutionState(), depth=0, U_s=0.5, N=0)
    parent.children["A"] = Node(state=SolutionState(), depth=1, parent=parent,
                                U_s=0.0, N=0, Q=0.0)
    backup(parent.children["A"], 1.0, cfg)
    # average backup uses the child's Q regardless of n_min — that's the
    # behavior we are warning about in §11 of the plan.
    assert parent.Q == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Backup reaches the root
# ---------------------------------------------------------------------------

def test_backup_reaches_root_under_full_loop():
    cfg = _cfg(**{"mcts.N": 32})
    res = search(
        SolutionState(),
        cfg=cfg,
        policy=SyntheticPolicy(n_classes=3, terminal_depth=2),
        terminal_check=DepthTerminal(2),
        K=4,
    )
    assert res.root.N == cfg.mcts.N
    # Root must have been updated by every rollout.
    assert res.root.N > 0


# ---------------------------------------------------------------------------
# Budget cap never exceeded
# ---------------------------------------------------------------------------

def test_rollout_budget_is_a_hard_ceiling():
    cfg = _cfg(**{"mcts.N": 7})
    res = search(
        SolutionState(),
        cfg=cfg,
        policy=SyntheticPolicy(n_classes=4, terminal_depth=2),
        terminal_check=DepthTerminal(2),
        K=4,
    )
    assert res.rollouts == 7
    # Sum of visits at root equals N (every rollout backs up through root).
    assert res.root.N == 7


# ---------------------------------------------------------------------------
# Transposition table caches expensive expansion calls
# ---------------------------------------------------------------------------

def test_transposition_table_records_hits_and_misses():
    tt = TranspositionTable()
    k = TTKey.make("sub", {"node_0": "42"})
    assert tt.get(k) is None
    assert tt.misses == 1
    tt.put(k, "value")
    assert tt.get(k) == "value"
    assert tt.hits == 1


def test_expand_reuses_transposition_for_same_state():
    """Two nodes with the same state + subproblem signature should hit the TT."""
    cfg = _cfg()
    policy = SyntheticPolicy(n_classes=3, terminal_depth=5)
    tt = TranspositionTable()

    s = SolutionState()
    a = Node(state=s, depth=0)
    b = Node(state=s, depth=0)

    expand(a, policy, K=4, cfg=cfg, tt=tt)
    calls_after_first = policy.calls
    expand(b, policy, K=4, cfg=cfg, tt=tt)
    # Same state -> TT hit -> policy NOT re-invoked.
    assert policy.calls == calls_after_first
    assert tt.hits == 1


# ---------------------------------------------------------------------------
# PUCT actually prefers high-prior children when N is small
# ---------------------------------------------------------------------------

def test_puct_select_descends_to_high_prior_child_when_unvisited():
    cfg = _cfg()
    root = Node(state=SolutionState(), depth=0, U_s=0.5, N=1)
    root.expanded = True
    # Two children, same Q=0 but different priors.
    low = Node(state=SolutionState(), depth=1, parent=root, prior=0.1, U_s=0.0)
    high = Node(state=SolutionState(), depth=1, parent=root, prior=0.9, U_s=0.0)
    root.children = {"low": low, "high": high}
    # mark root fully widened so select doesn't try to grow it
    # widening_cap(1, 2.0, 0.5) = 2; root already has 2 children -> done.
    chosen = select(root, cfg)
    assert chosen is high
