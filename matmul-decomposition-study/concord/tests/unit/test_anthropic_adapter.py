"""Tests for AnthropicAdapter retry / fallback paths.

We don't hit the real API here — instead we monkey-patch the adapter's
`_anthropic_mod` (which the adapter uses to look up exception classes)
and its `_client.messages.create` callable. That gives us full control
over what the simulated SDK raises on each call.

Pinned behavior:
  - Some models (claude-opus-4-8 onward) deprecate the `temperature`
    argument. The adapter must detect the 400 BadRequestError carrying
    "temperature is deprecated", flip `_skip_temperature`, and retry
    the SAME call without the offending kwarg. Subsequent calls must
    not pass `temperature` either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.llm.anthropic_adapter import AnthropicAdapter


# ---------------------------------------------------------------------------
# Fake SDK surface
# ---------------------------------------------------------------------------

class _FakeBadRequest(Exception):
    """Stand-in for anthropic.BadRequestError."""


class _FakeAuth(Exception):
    pass


class _FakePerm(Exception):
    pass


class _FakeNotFound(Exception):
    pass


def _make_fake_anthropic_mod() -> SimpleNamespace:
    return SimpleNamespace(
        BadRequestError=_FakeBadRequest,
        AuthenticationError=_FakeAuth,
        PermissionDeniedError=_FakePerm,
        NotFoundError=_FakeNotFound,
    )


def _ok_response(text: str = "ok", finish: str = "stop",
                 in_tok: int = 7, out_tok: int = 5):
    """Shape the adapter expects from messages.create()."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
        stop_reason=finish,
    )


class _FakeMessages:
    def __init__(self, plan):
        # plan: list of callables that take **kwargs and return / raise
        self.plan = list(plan)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.plan:
            raise RuntimeError("no plan step left")
        step = self.plan.pop(0)
        return step(**kwargs)


# ---------------------------------------------------------------------------
# Adapter constructed without touching the real SDK
# ---------------------------------------------------------------------------

def _make_adapter(plan) -> AnthropicAdapter:
    # Build a skeleton adapter without invoking the real anthropic SDK,
    # then attach our fakes.
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter._anthropic_mod = _make_fake_anthropic_mod()
    adapter.model = "claude-opus-4-8"
    adapter.max_output_tokens = 128
    adapter.timeout_s = 10.0
    adapter.max_retries = 4
    adapter.initial_retry_delay = 0.0   # no real sleeps in tests
    from core.types import CostTally
    adapter._cost = CostTally()
    adapter._skip_temperature = False
    adapter._client = SimpleNamespace(messages=_FakeMessages(plan))
    return adapter


# ---------------------------------------------------------------------------
# Test 1 — happy path
# ---------------------------------------------------------------------------

def test_one_call_passes_temperature_when_not_skipped():
    adapter = _make_adapter([lambda **kw: _ok_response("hi")])
    g = adapter._one_call("hello", temperature=0.7)
    assert g.text == "hi"
    # The single call carried temperature.
    assert adapter._client.messages.calls[0]["temperature"] == 0.7
    assert adapter._skip_temperature is False


# ---------------------------------------------------------------------------
# Test 2 — temperature deprecation triggers retry without temperature
# ---------------------------------------------------------------------------

def _raise_temp_deprecated(**kw):
    raise _FakeBadRequest(
        "Error code: 400 - {'type': 'error', 'error': "
        "{'type': 'invalid_request_error', 'message': "
        "'`temperature` is deprecated for this model.'}}")


def test_one_call_retries_without_temperature_on_deprecation_error():
    plan = [
        _raise_temp_deprecated,                            # 1st: 400 deprecated
        lambda **kw: _ok_response("ok now"),               # 2nd: succeeds
    ]
    adapter = _make_adapter(plan)

    g = adapter._one_call("hello", temperature=0.7)

    assert g.text == "ok now"
    assert adapter._skip_temperature is True

    # First call carried temperature; second call did NOT.
    calls = adapter._client.messages.calls
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


# ---------------------------------------------------------------------------
# Test 3 — once flipped, subsequent calls don't pass temperature at all
# ---------------------------------------------------------------------------

def test_skip_temperature_persists_for_subsequent_calls():
    plan = [
        _raise_temp_deprecated,
        lambda **kw: _ok_response("first ok"),
        lambda **kw: _ok_response("second ok"),
    ]
    adapter = _make_adapter(plan)

    adapter._one_call("p1", temperature=0.7)   # triggers flip
    adapter._one_call("p2", temperature=0.0)   # uses flipped flag immediately

    calls = adapter._client.messages.calls
    assert len(calls) == 3
    assert "temperature" in calls[0]      # the initial failing call
    assert "temperature" not in calls[1]  # the retry
    assert "temperature" not in calls[2]  # subsequent call honors the flag


# ---------------------------------------------------------------------------
# Test 4 — other BadRequestError messages STILL raise (no silent swallow)
# ---------------------------------------------------------------------------

def test_other_bad_request_errors_still_raise():
    def _raise_other(**kw):
        raise _FakeBadRequest("malformed prompt or whatever")
    adapter = _make_adapter([_raise_other])
    with pytest.raises(_FakeBadRequest):
        adapter._one_call("hello", temperature=0.7)
    # Flag must NOT have been flipped by an unrelated 400.
    assert adapter._skip_temperature is False
