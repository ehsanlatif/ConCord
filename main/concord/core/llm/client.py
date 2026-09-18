"""Provider-agnostic LLM protocol.

`supports_logprobs` drives the sample-weight oracle (§5.1). White-box providers
expose token_logprobs and use length-normalized sequence probability; black-box
providers leave them None and fall back to Laplace-smoothed class frequency.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import CostTally, Generation


@runtime_checkable
class LLMClient(Protocol):
    """Minimum surface every adapter must expose."""

    supports_logprobs: bool

    def generate(self, prompt: str, *, temperature: float, n: int) -> list[Generation]:
        """Return `n` completions for `prompt` at the given temperature."""
        ...

    def cost(self) -> CostTally:
        """Cumulative cost since this client was constructed."""
        ...
