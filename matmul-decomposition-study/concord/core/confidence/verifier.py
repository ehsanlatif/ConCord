"""External verifier v(r, q) -> [0, 1].

Per the spec, v is "pluggable per task". Implementations shipped:

- `NullVerifier`: always 1.0. Used when no task-specific verifier is
  available — the orchestrator should then set alpha = 1.0 so U_s falls
  back to pure SD.
- `LexicalVerifier`: lexical-kernel score against a gold answer if known.
- `MathIntermediateVerifier`: positive integer extraction + range check.
  Catches non-numeric noise on eval-set math intermediate nodes.
- `ChessIntermediateVerifier`: SAN/UCI-shape parse + length sanity.
- `CompositeVerifier`: dispatch by domain.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from .entailment import LexicalKernel


_log = logging.getLogger(__name__)


class Verifier(Protocol):
    def verify(self, response: str, subproblem: str) -> float:
        ...


class NullVerifier:
    """Constant verifier — for use when no signal is available."""

    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def verify(self, response: str, subproblem: str) -> float:
        return self.value


class LexicalVerifier:
    """Lexical proxy: kernel(response, gold) if a gold map is supplied.

    `golds` maps subproblem text -> the canonical gold answer. Unknown
    subproblems return 1.0 (treated as unverifiable, not failing).
    """

    def __init__(self, golds: dict[str, str] | None = None):
        self.golds = golds or {}
        self._k = LexicalKernel()

    def verify(self, response: str, subproblem: str) -> float:
        gold = self.golds.get(subproblem)
        if gold is None:
            return 1.0
        return self._k.kernel(response, gold)


# ---------------------------------------------------------------------------
# Domain verifiers (math + chess)
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"-?\d+")
_SOLUTION_LINE_RE = re.compile(r"solution\s*=\s*(.+)", re.IGNORECASE)


def _final_token(text: str) -> str:
    m = _SOLUTION_LINE_RE.findall(text)
    if m:
        return m[-1].strip().rstrip(". ")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


class MathIntermediateVerifier:
    """Cheap math-shape check for intermediate nodes.

    Returns:
      1.0 if the extracted final token is a single integer (the expected
          shape for every eval-set math template);
      0.5 if it is some other number-like token (decimal, fraction);
      0.0 if no number is found.

    This is a SHAPE verifier, not a CORRECTNESS verifier — the gold answer
    is hidden from the search and only consulted at the terminal grader.
    """

    def verify(self, response: str, subproblem: str) -> float:
        tail = _final_token(response)
        if not tail:
            return 0.0
        m = _INT_RE.fullmatch(tail.strip())
        if m:
            return 1.0
        if any(ch.isdigit() for ch in tail):
            return 0.5
        return 0.0


class ChessIntermediateVerifier:
    """Loose shape check for chess subproblem responses.

    Looks for a SAN move token (e.g. Nf3, e4, O-O, Qxe5) or a UCI move
    token (e.g. e2e4, g1f3, e7e8q). 1.0 on hit, 0.5 if numeric/coord-like
    fragments present, 0.0 otherwise.
    """

    _SAN_RE = re.compile(r"\b(O-O(-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?)\b")
    _UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def verify(self, response: str, subproblem: str) -> float:
        text = response.lower() + "\n" + response   # cheap both-casings
        if self._SAN_RE.search(response) or self._UCI_RE.search(text):
            return 1.0
        if re.search(r"[a-h][1-8]", text):
            return 0.5
        return 0.0


class CompositeVerifier:
    """Dispatch by domain string. Falls back to NullVerifier(1.0)."""

    def __init__(self, by_domain: dict[str, Verifier], default: Verifier | None = None):
        self.by_domain = by_domain
        self.default = default or NullVerifier(1.0)
        self._domain: str | None = None

    def set_domain(self, domain: str | None) -> None:
        self._domain = domain

    def verify(self, response: str, subproblem: str) -> float:
        v = self.by_domain.get(self._domain or "", self.default)
        return v.verify(response, subproblem)


def default_verifier_for(domain: str | None) -> Verifier:
    """One-shot factory: pick the right domain verifier."""
    if domain == "math":
        return MathIntermediateVerifier()
    if domain == "chess":
        return ChessIntermediateVerifier()
    return NullVerifier(1.0)


# ---------------------------------------------------------------------------
# LLM-judge verifier (uses the `verification` role model)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are a careful judge scoring a candidate answer to a subproblem.

Subproblem:
{subproblem}

Candidate answer:
{response}

Score this candidate on a 0-1 scale answering: "Is this a plausible, well-formed
answer to the subproblem?" Use:

  - 1.0 if the answer is well-formed AND consistent with the subproblem's question.
  - 0.5 if the shape is right but you are unsure of correctness.
  - 0.0 if the answer is malformed, refuses, or is clearly off-topic.

Reply with a single number on the last line, prefixed exactly with `score = `.
For example: `score = 0.7`
"""


class LLMJudgeVerifier:
    """Cross-checks each response with a separate (usually cheaper) model.

    Score parsing is tolerant of a final `score = <float>` line; falls back
    to a regex over the whole text. On any failure -> 0.5 (uncertain).
    Caches per (subproblem, response) so repeated lookups during a search
    do not re-call the model.
    """

    _SCORE_RE = re.compile(r"score\s*=\s*([-+]?\d*\.?\d+)", re.IGNORECASE)

    def __init__(self, llm, *, temperature: float = 0.0,
                 tracer: "object | None" = None,
                 model_name: str | None = None):
        # `llm` is an LLMClient — we keep this loose-typed to avoid an
        # import cycle (confidence -> llm.client).
        self.llm = llm
        self.temperature = temperature
        self.tracer = tracer
        self.model_name = model_name
        self._cache: dict[tuple[str, str], float] = {}

    def verify(self, response: str, subproblem: str) -> float:
        key = (subproblem[:200], response[:200])
        if key in self._cache:
            return self._cache[key]
        prompt = _JUDGE_PROMPT.format(subproblem=subproblem, response=response)
        text = ""
        finish: str | None = None
        error: str | None = None
        try:
            gens = self.llm.generate(prompt, temperature=self.temperature, n=1)
            if gens:
                text = gens[0].text or ""
                finish = gens[0].finish_reason
        except Exception as e:                                      # noqa: BLE001
            # Surface the real reason — earlier silent failures here made it
            # look like the verifier was running when it wasn't (e.g., bad
            # model name, auth error, or wrong max_tokens). The 0.5 fallback
            # still happens so the search keeps moving.
            _log.exception("LLMJudgeVerifier call failed; returning 0.5")
            error = repr(e)

        m = self._SCORE_RE.search(text) if text else None
        if not m:
            v = 0.5
        else:
            try:
                v = float(m.group(1))
            except ValueError:
                v = 0.5
        v = max(0.0, min(1.0, v))
        self._cache[key] = v

        # Log the verifier call (prompt sent to the judge + judge's full
        # response + parsed score). This lands in agent_calls.jsonl.
        if self.tracer is not None:
            try:
                self.tracer.log_agent_call(
                    role="verifier",
                    prompt=prompt,
                    response=text,
                    model=self.model_name,
                    finish_reason=finish,
                    extras={
                        "score": v,
                        "candidate_response": response,
                        "subproblem": subproblem,
                        "error": error,
                    },
                )
            except Exception:                                       # noqa: BLE001
                pass
        return v
