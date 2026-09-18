"""Deterministic mock LLM. Used by unit tests, CI, and cheap dev runs.

Two modes:

- **scripted**: caller supplies a `responses` dict mapping prompt -> list of
  completion strings. `generate(prompt, n=K)` returns the first K, cycling if
  short. Token-logprobs are filled in deterministically when `with_logprobs`
  is true so white-box oracles can be exercised.
- **echo**: when no scripted answer matches, returns `f"<mock:{i}> {prompt[:60]}"`
  for each of the n slots. Keeps integration tests runnable without authoring
  a full script for every prompt.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

from ..types import CostTally, Generation


class MockLLM:
    supports_logprobs: bool

    def __init__(self, responses: dict[str, list[str]] | None = None,
                 with_logprobs: bool = False, seed: int = 0):
        self._responses = responses or {}
        self.supports_logprobs = with_logprobs
        self._cost = CostTally()
        self._seed = seed

    def _scripted(self, prompt: str, n: int) -> list[str] | None:
        if prompt not in self._responses:
            return None
        bank = self._responses[prompt]
        if not bank:
            return None
        # cycle deterministically
        return [bank[i % len(bank)] for i in range(n)]

    @staticmethod
    def _fake_logprobs(text: str, seed: int) -> list[float]:
        # Deterministic per-token logprob in (-3, 0); short answers stay
        # distinguishable from long ones so length normalization is testable.
        rng = random.Random(seed)
        toks = max(1, len(text.split()))
        return [-rng.uniform(0.1, 3.0) for _ in range(toks)]

    def generate(self, prompt: str, *, temperature: float, n: int) -> list[Generation]:
        scripted = self._scripted(prompt, n)
        if scripted is not None:
            texts = scripted
        else:
            texts = [f"<mock:{i}> {prompt[:60]}" for i in range(n)]

        out: list[Generation] = []
        for i, t in enumerate(texts):
            seed = int(hashlib.sha1(f"{self._seed}|{prompt}|{i}".encode()).hexdigest()[:8], 16)
            lp = self._fake_logprobs(t, seed) if self.supports_logprobs else None
            out.append(Generation(text=t, token_logprobs=lp, finish_reason="stop"))

        # Cost accounting: count one "call" per SAMPLE emitted, matching the
        # AnthropicAdapter (which loops K times internally because the
        # Messages API has no native n=K parameter). This keeps the budget
        # ceiling — telemetry.check_budget compares cost.calls > cfg.mcts.N —
        # consistent across providers; otherwise mock would silently mask
        # budget overruns that fire on the real API.
        in_tok = max(1, len(prompt.split())) * n
        out_tok = sum(max(1, len(g.text.split())) for g in out)
        self._cost.add(calls=n, input_tokens=in_tok, output_tokens=out_tok, usd=0.0)
        return out

    def cost(self) -> CostTally:
        return self._cost

    # Convenience for tests:
    def extend(self, responses: dict[str, list[str]]) -> None:
        for k, v in responses.items():
            self._responses.setdefault(k, []).extend(v)
