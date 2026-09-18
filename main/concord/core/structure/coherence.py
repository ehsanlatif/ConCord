"""Incremental hard coherence gate (§5.4 + spec §3 lines 26-28).

Two key invariants from the decisions in §2:

  - **Hard** when violated: value goes to 0 and the node is marked
    `terminal-fail`. The selection rule will abandon that subtree.
  - **Always record sigma** (even when gating hard). The recorded sigmas feed
    the soft-relax fallback when no fully-coherent terminal is found.

`Constraint` is the public unit of pluggable task knowledge: math constraints
check numeric/identity consistency; code constraints compile + run unit tests;
planning constraints check precondition/effect validity. M4 ships the
*protocol* + a generic numeric-equality constraint usable for the eval set's
math chain (answers are integers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import networkx as nx

from ..types import SolutionState


@dataclass
class Constraint:
    """One pluggable check.

    `applicable_at` decides whether enough of the path is resolved to even
    evaluate this constraint. `check` returns sigma_contribution in [0, 1];
    0 means satisfied, anything > 0 is a violation. The orchestrator sums /
    maxes them in `evaluate_gate`.
    """

    name: str
    applicable_at: Callable[[SolutionState, int], bool]
    check: Callable[[SolutionState], float]
    # constraint-relevant binding names, for transposition keying (§5.7)
    relevant_bindings: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class GateResult:
    sigma: float           # max severity across newly-checkable constraints
    violated: list[str]    # names of constraints that fired (>0)
    all_checked: list[str]


def checkable_constraints(state: SolutionState, depth: int,
                          constraints: list[Constraint]) -> list[Constraint]:
    """Filter to constraints whose `applicable_at` predicate is satisfied."""
    return [c for c in constraints if c.applicable_at(state, depth)]


def evaluate_gate(state: SolutionState, depth: int,
                  constraints: list[Constraint]) -> GateResult:
    """Run every newly-checkable constraint. Sigma is the MAX severity.

    Why max rather than sum: severity caps at 1.0 per the spec, and "any one
    violation -> hard fail" is the desired semantic. We record the worst,
    and let soft-relax rank by it later.
    """
    checked = checkable_constraints(state, depth, constraints)
    if not checked:
        return GateResult(sigma=0.0, violated=[], all_checked=[])
    sigma = 0.0
    fired: list[str] = []
    for c in checked:
        s = float(c.check(state))
        s = max(0.0, min(1.0, s))
        if s > 0.0:
            fired.append(c.name)
        if s > sigma:
            sigma = s
    return GateResult(sigma=sigma, violated=fired,
                      all_checked=[c.name for c in checked])


# ---------------------------------------------------------------------------
# Ready-made constraints
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"-?\d+")


def _extract_int(s: str) -> int | None:
    m = _INT_RE.search(s)
    return int(m.group(0)) if m else None


def integer_answer_constraint(subproblem_predicate: Callable[[str], bool]
                              = lambda _t: True) -> Constraint:
    """Require that every resolved subproblem whose text matches `predicate`
    yields an integer-shaped answer. the eval set's math templates produce
    integer answers, so this catches "I returned '24!' (with !)" or "I said
    'the answer is roughly 24'" — non-numeric noise.
    """
    def applicable(state: SolutionState, depth: int) -> bool:
        return bool(state.resolved)
    def check(state: SolutionState) -> float:
        for sub, ans in state.resolved:
            if not subproblem_predicate(sub):
                continue
            if _extract_int(ans) is None:
                return 1.0
        return 0.0
    return Constraint(name="integer_answer",
                      applicable_at=applicable, check=check)


def numeric_bound_constraint(lo: int, hi: int,
                             subproblem_id: str) -> Constraint:
    """`subproblem_id` answer must lie in [lo, hi]. Used when a node spec says
    e.g. 'the answer is at most 100'."""
    def applicable(state: SolutionState, depth: int) -> bool:
        return any(sub == subproblem_id for sub, _ in state.resolved)
    def check(state: SolutionState) -> float:
        for sub, ans in state.resolved:
            if sub != subproblem_id:
                continue
            v = _extract_int(ans)
            if v is None:
                return 1.0
            if v < lo or v > hi:
                return 1.0
        return 0.0
    return Constraint(
        name=f"bound[{subproblem_id}]<{lo},{hi}>",
        applicable_at=applicable, check=check,
        relevant_bindings=(subproblem_id,),
    )


def predecessor_consistency_constraint(Gp: nx.DiGraph) -> Constraint:
    """Generic: a node may only be 'resolved' after ALL its G' predecessors are.

    Catches out-of-order assembly that slipped past the decomposer.
    """
    def applicable(state: SolutionState, depth: int) -> bool:
        return depth > 0
    def check(state: SolutionState) -> float:
        resolved = [sub for sub, _ in state.resolved]
        for i, sub in enumerate(resolved):
            preds = list(Gp.predecessors(sub)) if sub in Gp.nodes else []
            for p in preds:
                if p not in resolved[:i]:
                    return 1.0
        return 0.0
    return Constraint(name="predecessor_order",
                      applicable_at=applicable, check=check)
