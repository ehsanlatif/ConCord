"""LLMBlockVerifier — score + issues + (optional) corrected answer.

The verifier sees the original block question, all atomic sub-answers
that fed into the candidate, and the candidate block answer. It returns
a verdict carrying:

  - `score` ∈ [0, 1]
  - `is_consistent`  : True iff the atomic answers cohere
  - `issues`         : short list of one-line issues (for combiner hint)
  - `corrected_answer`: optional verifier-proposed fix (only if obvious)

Decision policy (driven by the orchestrator using cfg.pipeline thresholds):

  score ≥ verifier_accept (default 0.75)     → commit
  verifier_retry ≤ score < verifier_accept   → retry combiner with hint
  score < verifier_backtrack (default 0.25)  → MCTS backtrack (revise atomic)

The verifier ITSELF does not make those decisions — it just emits the
verdict. The thresholds live in the policy.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import Config
from ..llm.client import LLMClient


_log = logging.getLogger(__name__)


_VERIFIER_SYSTEM = (
    "You are a careful answer auditor. Given a problem, the atomic "
    "sub-answers that decompose it, and a proposed final answer, judge "
    "whether the final answer is (a) consistent with the atomic answers, "
    "(b) well-formed, and (c) likely correct.\n\n"
    "Reply in strict JSON, no prose, no markdown:\n"
    "{\n"
    "  \"score\": <float 0..1>,\n"
    "  \"is_consistent\": <true|false>,\n"
    "  \"issues\": [\"short issue 1\", \"short issue 2\", ...],\n"
    "  \"corrected_answer\": \"<optional; only fill if you can fix the "
    "answer trivially, e.g. an arithmetic typo. Otherwise omit or leave "
    "empty.>\"\n"
    "}\n\n"
    "Scoring rubric:\n"
    "  1.0  the answer is well-formed AND consistent AND likely correct.\n"
    "  0.75 the answer is well-formed AND consistent; correctness uncertain.\n"
    "  0.5  the answer is well-formed but conflicts with at least one atomic.\n"
    "  0.25 the answer is malformed or contradicts most atomics.\n"
    "  0.0  the answer is garbage / off-topic / a refusal.\n"
)


def _format_atoms(atomic_answers: list[tuple[str, str]]) -> str:
    if not atomic_answers:
        return "(no atomic sub-answers)"
    return "\n".join(f"[atom {i}] Q: {q}\n          A: {a}"
                     for i, (q, a) in enumerate(atomic_answers))


def _build_verifier_prompt(*, original_question: str,
                            atomic_answers: list[tuple[str, str]],
                            candidate_answer: str) -> str:
    return (
        _VERIFIER_SYSTEM
        + "\n\nOriginal problem:\n"
        + original_question.strip()
        + "\n\nAtomic sub-answers:\n"
        + _format_atoms(atomic_answers)
        + "\n\nCandidate final answer:\n"
        + candidate_answer.strip()
        + "\n\nReturn ONLY the JSON object."
    )


@dataclass
class VerifierVerdict:
    """Structured verdict returned by LLMBlockVerifier.verify()."""

    score: float = 0.5
    is_consistent: bool = False
    issues: list[str] = field(default_factory=list)
    corrected_answer: str | None = None
    # Provenance.
    raw_response: str = ""
    parse_failed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "is_consistent": self.is_consistent,
            "issues": list(self.issues),
            "corrected_answer": self.corrected_answer,
            "parse_failed": self.parse_failed,
            "error": self.error,
        }

    def hint_text(self) -> str:
        """Format issues for injection into a combiner retry prompt."""
        if not self.issues:
            return "(verifier flagged the answer but listed no specific issues)"
        return "\n".join(f"- {iss}" for iss in self.issues)


@dataclass
class LLMBlockVerifier:
    """One call per (atomics, candidate) pair. Caches by candidate text."""

    llm: LLMClient
    cfg: Config
    temperature: float = 0.0
    tracer: "object | None" = None
    model_name: str | None = None

    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __post_init__(self) -> None:
        self._cache: dict[tuple[str, str], VerifierVerdict] = {}

    def verify(self, *, original_question: str,
               atomic_answers: list[tuple[str, str]],
               candidate_answer: str,
               node_id: str | None = None) -> VerifierVerdict:
        key = (original_question[:200], candidate_answer[:200])
        if key in self._cache:
            return self._cache[key]
        prompt = _build_verifier_prompt(
            original_question=original_question,
            atomic_answers=atomic_answers,
            candidate_answer=candidate_answer,
        )
        text = ""
        finish: str | None = None
        error: str | None = None
        try:
            gens = self.llm.generate(prompt, temperature=self.temperature, n=1)
            if gens:
                text = gens[0].text or ""
                finish = gens[0].finish_reason
        except Exception as e:                                      # noqa: BLE001
            _log.exception("LLMBlockVerifier call failed; defaulting to "
                            "neutral verdict")
            error = repr(e)

        verdict = self._parse(text)
        verdict.raw_response = text
        verdict.error = error
        self._cache[key] = verdict

        if self.tracer is not None:
            try:
                self.tracer.log_agent_call(
                    role="block_verifier",
                    prompt=prompt, response=text,
                    model=self.model_name, finish_reason=finish,
                    node_id=node_id,
                    extras={
                        "score": verdict.score,
                        "is_consistent": verdict.is_consistent,
                        "issues": list(verdict.issues),
                        "corrected_answer": verdict.corrected_answer,
                        "parse_failed": verdict.parse_failed,
                        "candidate_answer": candidate_answer,
                        "error": error,
                    },
                )
            except Exception:                                       # noqa: BLE001
                pass
        return verdict

    # ----- parsing -------------------------------------------------------

    def _parse(self, text: str) -> VerifierVerdict:
        if not text:
            return VerifierVerdict(score=0.5, parse_failed=True)
        m = self._JSON_RE.search(text)
        if not m:
            return VerifierVerdict(score=0.5, parse_failed=True)
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return VerifierVerdict(score=0.5, parse_failed=True)
        if not isinstance(obj, dict):
            return VerifierVerdict(score=0.5, parse_failed=True)
        try:
            score = float(obj.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))
        consistent = bool(obj.get("is_consistent", False))
        raw_issues = obj.get("issues") or []
        issues = [str(x).strip() for x in raw_issues
                  if isinstance(x, (str, int, float)) and str(x).strip()]
        corrected = obj.get("corrected_answer")
        if corrected is not None:
            corrected = str(corrected).strip() or None
        return VerifierVerdict(
            score=score, is_consistent=consistent,
            issues=issues, corrected_answer=corrected,
            parse_failed=False,
        )
