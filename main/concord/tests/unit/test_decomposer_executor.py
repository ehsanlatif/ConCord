"""Tests for the proposer/answerer split.

What this nails down:
  - extract_answer is robust against verbose / prompt-echo responses.
  - The executor prompt contains the clean question and NOTHING from the
    original problem context or the `Problem node_K:` meta-syntax.
  - When the decomposition role has a distinct model, the LLM decomposer
    fires and produces the clean question; when not, raw substitution is
    used. (Answerers receive only the question text.)
  - template_needs_decomposing correctly classifies real templates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config, LLMCfg
from core.decomposer import (
    LLMDecomposer,
    template_needs_decomposing,
)
from core.expansion import build_executor_prompt, extract_answer
from core.llm.mock import MockLLM
from core.orchestrator import solve, solve_with_config
from core.types import Generation


# ---------------------------------------------------------------------------
# extract_answer robustness
# ---------------------------------------------------------------------------

def test_extract_answer_prefers_last_solution_marker():
    text = "Let me think.\nFirst, I compute X.\nsolution = 42\n"
    assert extract_answer(text) == "42"


def test_extract_answer_takes_last_when_multiple_solutions():
    text = "Initial guess: solution = 99\nReconsidering...\nsolution = 42"
    assert extract_answer(text) == "42"


def test_extract_answer_falls_back_to_boxed():
    text = "Reasoning step 1.\nReasoning step 2.\nThe answer is \\boxed{36}."
    assert extract_answer(text) == "36"


def test_extract_answer_falls_back_to_final_answer():
    text = "Long reasoning.\nfinal answer: 7"
    assert extract_answer(text) == "7"


def test_extract_answer_skips_prompt_echo():
    """The mock LLM (and sometimes weak models) echoes the prompt back.
    extract_answer must NOT return that as the answer — empty string is
    a better signal of extraction failure than 'Solve the following...'."""
    text = "<mock:0> Solve the following subproblem and conclude with..."
    assert extract_answer(text) == ""


def test_extract_answer_empty_on_blank_input():
    assert extract_answer("") == ""
    assert extract_answer("\n\n") == ""


def test_extract_answer_returns_empty_on_max_tokens_when_no_structured_marker():
    """A truncated response with no `solution = X` / boxed / final-answer line
    must NOT fall through to the "last non-prompt line" clause — that line
    is whatever mid-derivation fragment the model happened to be writing
    when the token cap fired, not the answer.
    """
    truncated_text = (
        "<think>\nLet me set up coordinates. KITE is a kite with IE "
        "being perpendicular bisector of KT.\n\n"
        "Let R be intersection of IE and KT.\n"
        "Set up: KT along x-axis, IE along y-axis.\n"
        "E→M: (0)(-e/2) - (k/2)(-e) = 0 + ke/2 ="
    )
    # finish_reason absent → legacy behavior: returns the last line
    assert extract_answer(truncated_text) != ""
    # finish_reason == "max_tokens" → returns empty (extraction failure)
    assert extract_answer(truncated_text, finish_reason="max_tokens") == ""


def test_extract_answer_still_returns_solution_on_max_tokens_when_present():
    """Even when finish_reason=max_tokens, a well-formed `solution = X`
    line earlier in the text is still honored — only the last-line
    fallback is suppressed.
    """
    text = "Reasoning A.\nsolution = 42\nNow let me double-check by considering"
    assert extract_answer(text, finish_reason="max_tokens") == "42"
    assert extract_answer(text, finish_reason="end_turn") == "42"


def test_extract_answer_still_returns_boxed_on_max_tokens():
    text = "We get \\boxed{36}. To verify, note that 36 ="
    assert extract_answer(text, finish_reason="max_tokens") == "36"


# ---------------------------------------------------------------------------
# build_executor_prompt is minimal — no scaffolding beyond one instruction
# ---------------------------------------------------------------------------

def test_executor_prompt_contains_only_the_clean_question_and_one_line():
    p = build_executor_prompt("What is 5 + 5?")
    # The clean question is present.
    assert "What is 5 + 5?" in p
    # The "solution = ..." instruction is the only extra scaffolding.
    assert "solution = <your final answer>" in p
    # No meta-syntax leaks in (no `Subproblem:`, no `Problem node_`, etc.)
    assert "Problem node_" not in p
    assert "Subproblem:" not in p
    assert "[For this value" not in p


# ---------------------------------------------------------------------------
# template_needs_decomposing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template,expected", [
    ("Compute 5 + 5.", False),
    ("Use the answer from problem node_0 and add 2.", True),
    ("Given that [MAKE] = [For this value use the answer from problem node_0 and add 15], find...", True),
    ("Solve the equation x^2 = 16.", False),
    ("Reference: node_3 of the above.", True),
])
def test_template_needs_decomposing(template, expected):
    assert template_needs_decomposing(template) is expected


# ---------------------------------------------------------------------------
# LLMDecomposer behavior
# ---------------------------------------------------------------------------

def test_decomposer_skips_call_for_clean_first_subtask():
    """First subtask with no parent references — no LLM call should fire."""
    llm = MockLLM(responses={})
    dec = LLMDecomposer(llm=llm)
    q = dec.make_subquestion(
        template_text="Compute 5 + 5.",
        resolved=[],
        original_problem="ignored",
    )
    assert q == "Compute 5 + 5."
    assert llm.cost().calls == 0


def test_decomposer_fires_when_template_has_node_refs():
    """When the template references a parent subtask, the decomposer LLM
    must be invoked."""
    # Script a response that contains a 'subproblem:' marker.
    script = {}    # any prompt → echo mode; we just verify the call fires
    llm = MockLLM(responses=script)
    dec = LLMDecomposer(llm=llm)
    dec.make_subquestion(
        template_text="Use the answer from problem node_0 and add 2.",
        resolved=[("scc_2", "10")],
        original_problem="ignored",
    )
    assert llm.cost().calls > 0


def test_decomposer_falls_back_to_template_when_llm_response_lacks_marker():
    """If the LLM doesn't include `subproblem:` in its response, the
    decomposer should still return *something* usable rather than crashing.
    """
    llm = MockLLM(responses={})    # echo mode → no `subproblem:` marker
    dec = LLMDecomposer(llm=llm)
    q = dec.make_subquestion(
        template_text="Use the answer from problem node_0 and add 2.",
        resolved=[("scc_2", "10")],
        original_problem="x",
    )
    assert q   # non-empty
    # And specifically NOT the raw template (because the LLM was called):
    assert "node_0" not in q or q != "Use the answer from problem node_0 and add 2."


def test_decomposer_extracts_text_after_subproblem_marker():
    """When the LLM includes the marker, extraction picks up everything
    after it."""
    class StubLLM:
        supports_logprobs = False
        def __init__(self):
            from core.types import CostTally
            self._cost = CostTally()
        def generate(self, prompt, *, temperature, n):
            from core.types import Generation, CostTally
            self._cost.add(calls=1)
            return [Generation(
                text="Some reasoning blah.\nsubproblem: Compute 12 + 3.",
                token_logprobs=None, finish_reason="stop")]
        def cost(self):
            return self._cost

    dec = LLMDecomposer(llm=StubLLM())
    q = dec.make_subquestion(
        template_text="Use the answer from problem node_0 and add 3.",
        resolved=[("scc_2", "12")],
        original_problem="x",
    )
    assert q == "Compute 12 + 3."


# ---------------------------------------------------------------------------
# Orchestrator wiring: decomposer fires only when distinct model
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path) -> Config:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.llm = LLMCfg(provider="mock", model="mock-v0", temperature=1.0)
    cfg.mcts.N = 12
    cfg.sampling.K_blackbox = 2
    return cfg


PROBLEM_WITH_REFS = (
    "Problem node_0: Compute 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
)


def test_orchestrator_skips_llm_decomposer_in_single_model_mode(tmp_path: Path):
    """Legacy single-LLM mode: decomposer events report raw_substitution."""
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM_WITH_REFS, cfg=cfg, llm=MockLLM(responses={}),
                 domain="math")
    log = Path(res.trace_path).read_text()
    import json
    events = [json.loads(l) for l in log.splitlines()]
    dec_events = [e for e in events if e["kind"] == "decomposer"]
    assert dec_events
    assert dec_events[0]["data"]["kind"] == "raw_substitution"


def test_orchestrator_uses_llm_decomposer_when_distinct_model(tmp_path: Path):
    """Per-role config: decomposer events report kind=llm."""
    cfg = _cfg(tmp_path)
    cfg.llm = LLMCfg(provider="mock", model="base", temperature=1.0)
    cfg.models.decomposition = LLMCfg(provider="mock", model="decomposer",
                                        temperature=0.3)
    res = solve_with_config(PROBLEM_WITH_REFS, cfg=cfg, domain="math")
    log = Path(res.trace_path).read_text()
    import json
    events = [json.loads(l) for l in log.splitlines()]
    dec_events = [e for e in events if e["kind"] == "decomposer"]
    assert dec_events
    assert dec_events[0]["data"]["kind"] == "llm"
    assert dec_events[0]["data"]["model"] == "decomposer"


# ---------------------------------------------------------------------------
# Executor must never see meta-syntax from the original template
# ---------------------------------------------------------------------------

def test_executor_prompts_strip_node_meta_syntax(tmp_path: Path):
    """Inspect the tracer's recorded executor prompts and verify NONE of
    them contain `Problem node_K`, `[For this value...]`, or other
    decomposition meta-syntax that the executor shouldn't have to parse."""
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM_WITH_REFS, cfg=cfg, llm=MockLLM(responses={}),
                 domain="math")
    import json
    rollouts_path = Path(res.cost["rollouts_path"])
    for line in rollouts_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        exp = row.get("expansion")
        if not exp:
            continue
        prompt = exp.get("prompt_excerpt", "")
        # The executor must NOT see the original `Problem node_K:` headers.
        assert "Problem node_" not in prompt, (
            f"executor prompt leaked node meta-syntax: {prompt[:200]!r}"
        )


# ---------------------------------------------------------------------------
# The user-facing invariant: final answer is a SOLUTION, not a prompt
# ---------------------------------------------------------------------------

def test_final_answer_never_contains_prompt_scaffolding(tmp_path: Path):
    """End-to-end check: Result.answer must never be a string that looks
    like a decomposition / executor prompt.

    Even with the mock LLM (which echoes prompt content), extract_answer
    rejects prompt-echo and returns "" — `state.resolved[-1][1]` may still
    be the raw echo from earlier steps, so the orchestrator runs it through
    extract_answer one more time before returning. That second pass is the
    invariant guarded here.
    """
    cfg = _cfg(tmp_path)
    res = solve(PROBLEM_WITH_REFS, cfg=cfg, llm=MockLLM(responses={}),
                 domain="math")
    forbidden = (
        "Solve the following subproblem",
        "Subproblem:",
        "Problem node_",
        "For this value use",
        "Provide your final answer",
    )
    for needle in forbidden:
        assert needle not in res.answer, (
            f"Final answer contains prompt scaffolding {needle!r}: {res.answer!r}"
        )
