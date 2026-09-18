"""Core dataclasses shared across modules.

Kept deliberately small in M0 — fleshed out as later milestones land. The
fields here are the minimum the orchestrator + telemetry need to round-trip a
dummy result end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostTally:
    """Cumulative LLM cost accounting. Updated by LLMClient adapters."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def add(self, *, calls: int = 1, input_tokens: int = 0, output_tokens: int = 0,
            usd: float = 0.0) -> None:
        self.calls += calls
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
        }


@dataclass
class Generation:
    """One LLM completion. token_logprobs is None for black-box providers."""

    text: str
    token_logprobs: list[float] | None
    finish_reason: str


@dataclass
class SolutionState:
    """The state attached to an MCTS node — what has been resolved so far.

    Held flat in M0 (just an ordered list of (subproblem, answer) pairs and a
    free-form bindings dict). M2+ will replace `subproblem` with structured
    references into the dependency graph G'.
    """

    resolved: list[tuple[str, str]] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def extend(self, subproblem: str, answer: str, **bindings: Any) -> "SolutionState":
        new = SolutionState(
            resolved=self.resolved + [(subproblem, answer)],
            bindings={**self.bindings, **bindings},
        )
        return new


@dataclass
class Result:
    """Top-level return value from `solve()`."""

    answer: str
    coherent: bool
    sigma: float
    rollouts: int
    cost: dict[str, Any]
    flagged: str | None = None
    trace_path: str | None = None
