"""Progressive widening predicate (§5.6, plan §5.6).

A node may grow another child whenever
    floor(C * N(node)**beta) > childCount(node)

With default C=2, beta=0.5 (√-widening): ~2 children after 1 visit, ~3 after
4, ~4 after 9, etc. Sub-linear in N — that's the whole point.
"""

from __future__ import annotations

import math


def may_widen(N: int, n_children: int, C: float, beta: float) -> bool:
    """Return True iff the widening rule permits adding another child now."""
    cap = int(math.floor(C * (max(N, 1) ** beta)))
    return cap > n_children


def widening_cap(N: int, C: float, beta: float) -> int:
    """How many children the rule permits at visit-count N."""
    return int(math.floor(C * (max(N, 1) ** beta)))
