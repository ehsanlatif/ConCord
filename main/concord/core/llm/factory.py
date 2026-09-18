"""LLMClient factory.

Constructs an LLMClient from a `LLMCfg`. Used by the orchestrator to
materialize per-role clients defined under `cfg.models.{role}`.

A factory rather than direct adapter instantiation keeps:
  - `mock` available without importing anthropic
  - room to drop in OpenAI / other adapters later
  - one cost-aggregation point per role
"""

from __future__ import annotations

from typing import Iterable

from ..config import Config, LLMCfg
from ..types import CostTally
from .client import LLMClient
from .mock import MockLLM


def make_client(spec: LLMCfg) -> LLMClient:
    """Build one LLMClient from a single LLMCfg block."""
    if spec.provider == "mock":
        return MockLLM(responses={}, with_logprobs=False)
    if spec.provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(
            model=spec.model,
            max_output_tokens=spec.max_output_tokens,
            request_timeout_s=spec.request_timeout_s,
        )
    if spec.provider == "openai":
        # OpenAI adapter ships in a later milestone; route to mock with a
        # warning so an unfinished config doesn't crash a run.
        import warnings
        warnings.warn("OpenAI provider not implemented yet; falling back to MockLLM",
                       RuntimeWarning, stacklevel=2)
        return MockLLM(responses={}, with_logprobs=True)
    raise ValueError(f"unknown provider: {spec.provider!r}")


class RoleClients:
    """Container for the per-role LLMClient set used by `solve()`.

    Holds one client per role. Roles can SHARE the same client when their
    config blocks resolve to identical LLMCfg — saves on initialization
    cost and gives a single CostTally for the shared usage.

    Use `RoleClients.from_config(cfg)` to build from a Config; passes
    cost aggregation through `total_cost()` for telemetry.
    """

    # All known role names. Stable order matters for cost aggregation
    # (dedup keys are positional). New pipeline-solver roles are appended
    # at the end so existing code continues to see the original four.
    ROLE_NAMES = (
        "execution", "decomposition", "classification", "verification",
        "splitter", "combiner", "synthesizer", "synth_verifier",
    )

    def __init__(self, execution: LLMClient, decomposition: LLMClient,
                 classification: LLMClient, verification: LLMClient,
                 specs: dict[str, LLMCfg],
                 splitter: LLMClient | None = None,
                 combiner: LLMClient | None = None,
                 synthesizer: LLMClient | None = None,
                 synth_verifier: LLMClient | None = None):
        self.execution = execution
        self.decomposition = decomposition
        self.classification = classification
        self.verification = verification
        # Pipeline-solver roles. Default to the execution / verification
        # client when the caller didn't supply a binding — preserves
        # backward compatibility with construct-by-positional-args paths.
        self.splitter = splitter if splitter is not None else execution
        self.combiner = combiner if combiner is not None else execution
        self.synthesizer = synthesizer if synthesizer is not None else execution
        self.synth_verifier = (synth_verifier if synth_verifier is not None
                               else verification)
        self.specs = specs
        # Deduplicate by identity so total_cost() does not double-count
        # when multiple roles point at the same client instance.
        self._all_unique: list[LLMClient] = []
        seen: set[int] = set()
        for c in (execution, decomposition, classification, verification,
                  self.splitter, self.combiner, self.synthesizer,
                  self.synth_verifier):
            if id(c) not in seen:
                seen.add(id(c))
                self._all_unique.append(c)

    @classmethod
    def from_config(cls, cfg: Config) -> "RoleClients":
        # Each call to `role_model` returns a *merged* LLMCfg with role
        # overrides applied on top of the top-level `llm:` block.
        specs = {role: cfg.role_model(role) for role in cls.ROLE_NAMES}
        # If two roles resolve to the same spec, share the same client.
        clients_by_spec: dict[tuple, LLMClient] = {}

        def _client_for(spec: LLMCfg) -> LLMClient:
            key = tuple(sorted(spec.model_dump().items()))
            if key not in clients_by_spec:
                clients_by_spec[key] = make_client(spec)
            return clients_by_spec[key]

        return cls(
            execution=_client_for(specs["execution"]),
            decomposition=_client_for(specs["decomposition"]),
            classification=_client_for(specs["classification"]),
            verification=_client_for(specs["verification"]),
            splitter=_client_for(specs["splitter"]),
            combiner=_client_for(specs["combiner"]),
            synthesizer=_client_for(specs["synthesizer"]),
            synth_verifier=_client_for(specs["synth_verifier"]),
            specs=specs,
        )

    def total_cost(self) -> CostTally:
        """Sum CostTallies across unique underlying clients."""
        out = CostTally()
        for c in self._all_unique:
            ct = c.cost()
            out.add(calls=ct.calls, input_tokens=ct.input_tokens,
                    output_tokens=ct.output_tokens, usd=ct.usd)
        return out

    def per_role_cost(self) -> dict[str, dict]:
        """Diagnostic: per-role cost breakdown. Roles sharing a client will
        report the SAME tally — caller can interpret that as shared cost.
        """
        return {
            "execution": self.execution.cost().to_dict(),
            "decomposition": self.decomposition.cost().to_dict(),
            "classification": self.classification.cost().to_dict(),
            "verification": self.verification.cost().to_dict(),
            "splitter": self.splitter.cost().to_dict(),
            "combiner": self.combiner.cost().to_dict(),
            "synthesizer": self.synthesizer.cost().to_dict(),
            "synth_verifier": self.synth_verifier.cost().to_dict(),
        }
