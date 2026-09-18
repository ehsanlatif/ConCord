"""Anthropic LLMClient adapter — black-box mode (no token logprobs).

Anthropic's Messages API exposes no token logprobs, so `supports_logprobs` is
False and the SD pipeline uses the Laplace-smoothed class-frequency oracle
(decision 2). We DO NOT enable extended thinking here — we want diversity
across samples for the SD calculation, not a single high-quality answer.

`generate(prompt, n=K)` makes K independent calls (Anthropic has no native
`n` parameter). That's the actual cost a v2 expansion pays. Calls are made
sequentially in M5; M5+ can parallelize via a worker pool.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..types import CostTally, Generation


_log = logging.getLogger(__name__)


# Rough public-pricing baseline — used only for the cost-estimate column in
# the run log. Update via env if Anthropic prices move; precision is not
# load-bearing.
_PRICE_PER_MTOK = {
    "claude-opus-4-7":      {"in":  15.0, "out": 75.0},
    "claude-opus-4-8":      {"in":  15.0, "out": 75.0},
    "claude-sonnet-4-6":    {"in":   3.0, "out": 15.0},
    "claude-haiku-4-5":     {"in":   0.8, "out":  4.0},
}


class AnthropicAdapter:
    supports_logprobs: bool = False

    def __init__(self, model: str = "claude-sonnet-4-6",
                 max_output_tokens: int = 1024,
                 request_timeout_s: float = 120.0,
                 max_retries: int = 4,
                 initial_retry_delay: float = 4.0):
        import anthropic  # lazy
        self._anthropic_mod = anthropic
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self._client = anthropic.Anthropic()
        self._cost = CostTally()
        # Some newer models (e.g. claude-opus-4-8) reject the `temperature`
        # argument with a 400 "temperature is deprecated for this model".
        # We discover this at runtime on the first call that fails and flip
        # this flag so subsequent calls omit the argument entirely.
        self._skip_temperature: bool = False

    def _price(self, in_tok: int, out_tok: int) -> float:
        p = _PRICE_PER_MTOK.get(self.model.split("[", 1)[0])  # strip [1m] etc.
        if p is None:
            return 0.0
        return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000.0

    def _build_kwargs(self, prompt: str, *, temperature: float) -> dict:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_s,
        )
        # Some newer models reject `temperature` outright (see
        # `_skip_temperature`). Omit the argument completely in that case;
        # do NOT pass `temperature=None` — the SDK serialises it.
        if not self._skip_temperature:
            kwargs["temperature"] = temperature
        return kwargs

    def _one_call(self, prompt: str, *, temperature: float) -> Generation:
        anthropic = self._anthropic_mod
        delay = self.initial_retry_delay
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    **self._build_kwargs(prompt, temperature=temperature),
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
                self._cost.add(calls=1, input_tokens=in_tok,
                               output_tokens=out_tok,
                               usd=self._price(in_tok, out_tok))
                return Generation(text=text, token_logprobs=None,
                                  finish_reason=resp.stop_reason or "stop")
            except anthropic.BadRequestError as e:
                # Special case: some models (claude-opus-4-8 onwards) have
                # deprecated the `temperature` argument and return 400 the
                # moment we pass it. Detect that ONE specific message,
                # flip the flag, and retry immediately — no sleep, no
                # increment toward the retry ceiling, because the prior
                # attempt was structurally invalid, not transient.
                msg = str(e).lower()
                if ("temperature" in msg and "deprecated" in msg
                        and not self._skip_temperature):
                    self._skip_temperature = True
                    _log.warning(
                        "Model %s rejects `temperature` (deprecated); "
                        "retrying call without it.",
                        self.model,
                    )
                    continue
                raise
            except (anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError,
                    anthropic.NotFoundError) as e:
                # non-retryable
                raise
            except Exception as e:                                  # noqa: BLE001
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        assert last_err is not None
        raise last_err

    def generate(self, prompt: str, *, temperature: float,
                 n: int) -> list[Generation]:
        return [self._one_call(prompt, temperature=temperature) for _ in range(n)]

    def cost(self) -> CostTally:
        return self._cost
