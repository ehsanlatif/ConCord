"""Soft-relax fallback (decision 6).

When the search exhausts its rollout budget without finding a coherent
terminal (σ = 0), return the BEST INCOHERENT terminal — ranked by `argmin σ`
— flagged as unverified. This is the spec's lines 38-39.

If EVERY reached terminal failed the gate (min σ > 0), the returned
FlaggedResult is marked `all_rejected = True` so the orchestrator can
suppress the misleading answer text rather than pretend the chosen
terminal is "the answer."
"""

from __future__ import annotations

from dataclasses import dataclass

from .mcts import Node


@dataclass
class FlaggedResult:
    node: Node
    sigma: float
    flag: str = "unverified, best-effort (soft-relax)"
    # True iff *every* reached terminal had σ > 0 (i.e. there is no
    # least-incoherent winner — they're all equally bad, and the chosen
    # node won by tiebreak alone). Callers should treat the chosen
    # node's answer as a diagnostic, NOT as a real prediction.
    all_rejected: bool = False


# Flag text used when every reached terminal failed the gate. Stable so
# callers (orchestrator, run_all, metrics) can match on it.
ALL_REJECTED_FLAG = "no coherent solution; all terminals rejected"


def soft_relax(reached_terminals: list[Node]) -> FlaggedResult | None:
    """`argmin sigma` over the supplied terminals; ties broken by `-Q` (higher
    Q wins). Returns None when no terminals were ever reached.

    NB: We do NOT filter by σ > 0 — if a coherent terminal exists in the
    list, soft_relax will still return it (with sigma 0, no flag suppressed
    yet). Callers should usually check for a coherent terminal first and
    only fall back here.

    When every reached terminal has σ > 0, the result is marked
    `all_rejected=True` and assigned the `ALL_REJECTED_FLAG` so callers
    know the "winner" is just whatever node won the tiebreak, not a real
    answer.
    """
    if not reached_terminals:
        return None
    best = min(reached_terminals, key=lambda n: (n.sigma, -n.Q))
    min_sigma = best.sigma
    all_rejected = min_sigma > 0.0
    flag = ALL_REJECTED_FLAG if all_rejected else "unverified, best-effort (soft-relax)"
    return FlaggedResult(node=best, sigma=min_sigma, flag=flag,
                         all_rejected=all_rejected)
