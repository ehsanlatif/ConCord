"""Concrete depth governance (§5.8, decision 7).

Two knobs:

  - **Hard cap**: `d_max = min(critical_path(G') + slack, B // L)`. The
    structural depth estimate from G' plus headroom, OR the affordable depth
    given the total budget B and per-expansion cost L — whichever is smaller.
  - **Online stop**: marginal-gain rule. Stop expanding deeper when the
    expected improvement in U_s per level drops below `λ * cost(next level)`.

These travel together with the rest of the v2 defaults — fixed depth is the
ablation arm (§8.4).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import networkx as nx

from ..structure.graph import critical_path_len


def depth_cap(Gp: nx.DiGraph, *, budget_calls: int,
              per_expansion_cost: int, slack: int = 2,
              fixed: int | None = None) -> int:
    """`d_max` per spec line 7.

    `budget_calls` is the LLM-call budget for this run (`cfg.mcts.N`).
    `per_expansion_cost` is the expected number of LLM calls per expansion
    (≈ K when no caching). When `fixed` is set we ignore the structural
    estimate (this is the `ablation.depth == "fixed"` path).
    """
    if fixed is not None:
        return max(1, int(fixed))
    structural = critical_path_len(Gp) + slack
    budget_d = max(1, budget_calls // max(1, per_expansion_cost))
    return max(1, min(structural, budget_d))


@dataclass
class StopState:
    """Rolling per-depth max U_s, used for the marginal-gain rule."""

    best_at_depth: dict[int, float]

    def update(self, depth: int, u_s: float) -> None:
        cur = self.best_at_depth.get(depth, -math.inf)
        if u_s > cur:
            self.best_at_depth[depth] = u_s

    def gain(self, depth: int) -> float | None:
        """U_s at this depth minus U_s at depth-1. None if depth-1 unseen."""
        if depth - 1 not in self.best_at_depth:
            return None
        if depth not in self.best_at_depth:
            return None
        return self.best_at_depth[depth] - self.best_at_depth[depth - 1]


def should_stop(stop_state: StopState, *, current_depth: int,
                lam: float, cost_next: float) -> bool:
    """Stop when expected ΔU_s < λ * cost_next.

    Uses the most recent depth-over-depth gain as the empirical estimate.
    Falls back to "don't stop" when we don't yet have two depths of data.
    """
    g = stop_state.gain(current_depth)
    if g is None:
        return False
    return g < lam * cost_next
