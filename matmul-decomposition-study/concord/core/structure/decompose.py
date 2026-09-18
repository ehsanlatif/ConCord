"""Decomposer + atomicity predicate (§5.3).

Drives Phase-2 expansion: given a SolutionState and the condensed graph G',
return the *next* subproblem to attack (in topological/critical-path order),
or the ATOMIC sentinel if the current state is directly solvable by the LLM.

`granularity` lets the orchestrator coarsen the chunk size when the parent's
U_s is already high — fewer, larger subproblems when we're confident, finer
when we're not. M2 ships the contract; M3/M5 calls it from the MCTS expand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import networkx as nx

from ..types import SolutionState


ATOMIC = "<ATOMIC>"


@dataclass
class Subproblem:
    node_id: str
    text: str
    depth: int                 # 0-based depth from root in G'
    bindings: dict[str, str]   # answers to predecessors, ready to substitute


class Decomposer(Protocol):
    def next_subproblem(self, state: SolutionState, Gp: nx.DiGraph,
                        granularity: int) -> Subproblem | str:
        """Return the next Subproblem or the ATOMIC sentinel."""
        ...


# ---------------------------------------------------------------------------
# Explicit decomposer — uses G' topological order directly.
# ---------------------------------------------------------------------------

class ExplicitDecomposer:
    """For problems where G' was parsed from explicit structure.

    Picks the lowest-index unresolved node whose predecessors are all
    resolved. `granularity` is accepted for interface compatibility but
    ignored here — the structure already pins chunk size.
    """

    def __init__(self):
        pass

    def next_subproblem(self, state: SolutionState, Gp: nx.DiGraph,
                        granularity: int = 1) -> Subproblem | str:
        resolved = {sub for sub, _ in state.resolved}
        # Topological order over G' (lex-stable).
        for n in nx.lexicographical_topological_sort(Gp):
            if n in resolved:
                continue
            preds = list(Gp.predecessors(n))
            if all(p in resolved for p in preds):
                # Bindings cover BOTH the SCC id and each of its original
                # members — the prompt substituter looks them up by the
                # original `node_K` ids that appear in the problem text.
                bindings: dict[str, str] = {}
                resolved_map = dict(state.resolved)
                for p in preds:
                    answer = resolved_map[p]
                    bindings[p] = answer
                    for orig in Gp.nodes[p].get("members", []):
                        bindings[orig] = answer
                depth = _depth_of(Gp, n)
                text = Gp.nodes[n].get("text", "")
                return Subproblem(node_id=n, text=text, depth=depth,
                                  bindings=bindings)
        return ATOMIC


def _depth_of(Gp: nx.DiGraph, node: str) -> int:
    """Longest path from any source to `node`, in edges."""
    # build a quick longest-path-to-node memo via topo order
    order = list(nx.lexicographical_topological_sort(Gp))
    dist: dict[str, int] = {n: 0 for n in order}
    for u in order:
        for v in Gp.successors(u):
            dist[v] = max(dist[v], dist[u] + 1)
    return dist.get(node, 0)


# ---------------------------------------------------------------------------
# Implicit decomposer — LLM-proposed subproblems (stub for M5)
# ---------------------------------------------------------------------------

class ImplicitDecomposerStub:
    """Placeholder: for problems with no explicit dependency graph (chess,
    chemistry single-query, etc.) the real implementation will ask the LLM
    for the next subproblem. M2 ships the protocol-conforming stub so the
    MCTS engine (M3) can compile against it.
    """

    def __init__(self, prompt_template: str = "Decompose: {state}"):
        self.prompt_template = prompt_template

    def next_subproblem(self, state: SolutionState, Gp: nx.DiGraph,
                        granularity: int = 1) -> Subproblem | str:
        # M2 stub: walk G' in topological order, returning each unresolved
        # node as a one-shot subproblem with the parent answers in bindings.
        # The real LLM-driven implementation lands in M5.
        resolved = {sub for sub, _ in state.resolved}
        for n in nx.lexicographical_topological_sort(Gp):
            if n in resolved:
                continue
            preds = list(Gp.predecessors(n))
            if all(p in resolved for p in preds):
                bindings = {p: dict(state.resolved)[p] for p in preds}
                return Subproblem(
                    node_id=n, text=Gp.nodes[n].get("text", ""),
                    depth=0, bindings=bindings,
                )
        return ATOMIC


# ---------------------------------------------------------------------------
# Atomicity predicate
# ---------------------------------------------------------------------------

def is_atomic(state: SolutionState, Gp: nx.DiGraph) -> bool:
    """All G' nodes resolved → atomic terminal."""
    resolved = {sub for sub, _ in state.resolved}
    return all(n in resolved for n in Gp.nodes)


# ---------------------------------------------------------------------------
# Granularity helper
# ---------------------------------------------------------------------------

def granularity_for(parent_u_s: float) -> int:
    """Coarser chunk when parent U_s is high (we're confident), finer otherwise.

    Returned int is consumed by the implicit decomposer (number of source-
    line / sentence units to bundle into one subproblem). The explicit
    decomposer ignores it.
    """
    if parent_u_s >= 0.85:
        return 3
    if parent_u_s >= 0.6:
        return 2
    return 1
