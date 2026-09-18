"""LLM-driven sub-question synthesis (the "PROPOSER" / decomposer role).

Separation of concerns:

  - The DECOMPOSER produces a clean, self-contained sub-question. It inlines
    every value from already-resolved subtasks, resolves any embedded
    arithmetic on those values (e.g. "use the answer and add 15"), and
    strips all `node_K` / `[For this value...]` meta-syntax.

  - The EXECUTOR receives ONLY the sub-question text — no original problem
    statement, no parent facts, no scaffolding beyond a single instruction.
    Self-containment is the decomposer's responsibility: the executor
    receives only the question text — no problem
    statement, no parent facts.

That separation is what makes each subproblem "independently solvable,
isolated, and have an answer not a new question."

The decomposer uses the `cfg.models.decomposition` LLM. If it's the SAME
client as execution (single-model run), we still call it — the clean
substitution and arithmetic resolution are too valuable to skip just to
save the call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm.client import LLMClient


# Patterns that indicate the raw template needs cleanup. If NONE of these
# match, the template is already self-contained and we can skip the
# decomposer LLM call entirely.
_NODE_REF_PATTERNS = [
    re.compile(r"\bproblem\s+node_\d+\b", re.IGNORECASE),
    re.compile(r"\bnode_\d+\b", re.IGNORECASE),
    re.compile(r"\[\s*for\s+this\s+value", re.IGNORECASE),
    re.compile(r"\banswer\s+from\b", re.IGNORECASE),
]


def template_needs_decomposing(template_text: str) -> bool:
    """Cheap check: does the template still reference parent subtasks?"""
    return any(p.search(template_text) for p in _NODE_REF_PATTERNS)


_DECOMPOSER_SYSTEM = (
    "You rewrite a multi-part problem-template into a single self-contained "
    "sub-question that an independent answerer can solve in isolation.\n\n"
    "The answerer will NOT see the original problem or any other context — "
    "they will see ONLY your output. Their job is to compute an answer, not "
    "to decompose further. Your output MUST therefore:\n\n"
    "  1. Inline every value from the resolved subtasks (substitute the "
    "actual numbers/strings, never use 'node_K' or '[For this value...]').\n"
    "  2. Resolve any embedded arithmetic on those values (e.g. if the "
    "template says 'use the answer from node_0 and add 15' and node_0 was "
    "42, write '57' directly, not '42 + 15').\n"
    "  3. Be a single specific question with a single deterministic answer.\n"
    "  4. Preserve the mathematical / logical content of the template — do "
    "NOT solve the question itself, just clean it.\n"
    "  5. Be self-contained — the answerer needs no other context.\n\n"
    "Output format (and nothing else):\n"
    "subproblem:\n"
    "<clean self-contained question on one or more lines>"
)


def _format_resolved(resolved: list[tuple[str, str]]) -> str:
    if not resolved:
        return "(none — this is the first subtask)"
    parts: list[str] = []
    for label, ans in resolved:
        parts.append(f"- {label}: {ans}")
    return "\n".join(parts)


def _build_decomposer_user(template_text: str,
                            resolved: list[tuple[str, str]],
                            problem_excerpt: str) -> str:
    return (
        f"Original problem (for reference; DO NOT echo this to the answerer):\n"
        f"{problem_excerpt}\n\n"
        f"Already-resolved subtasks (subtask label -> answer):\n"
        f"{_format_resolved(resolved)}\n\n"
        f"Next subtask template (clean this up):\n"
        f"{template_text}\n\n"
        f"Now produce the clean sub-question."
    )


_SUBPROBLEM_RE = re.compile(r"^\s*subproblem\s*:\s*(.*)\Z",
                              re.IGNORECASE | re.DOTALL | re.MULTILINE)


def _extract_subproblem(text: str) -> str:
    """Parse the decomposer's response — strip everything before the
    `subproblem:` marker. If the marker isn't found, take the whole text
    (the LLM probably just produced the bare question)."""
    m = _SUBPROBLEM_RE.search(text)
    if m:
        return m.group(1).strip()
    # No marker: take the whole text minus any leading reasoning prefix.
    return text.strip()


@dataclass
class LLMDecomposer:
    """Synthesizes a clean self-contained sub-question via the decomposition
    role LLM. Falls back to the raw template (no cleanup) when the LLM call
    fails — caller can still attempt to solve it, just with a messier prompt.
    """

    llm: LLMClient
    temperature: float = 0.3
    max_problem_excerpt: int = 2000
    # Optional reference to the per-question Tracer so each decomposer LLM
    # call is recorded in the agent_calls.jsonl stream. Set by the
    # orchestrator after construction (we don't accept it via __init__ to
    # avoid an import cycle).
    tracer: "object | None" = None
    model_name: str | None = None

    def make_subquestion(self, *, template_text: str,
                          resolved: list[tuple[str, str]],
                          original_problem: str,
                          node_id: str | None = None) -> str:
        if not template_needs_decomposing(template_text) and not resolved:
            # First subtask + no references — template IS the question.
            return template_text.strip()
        excerpt = original_problem[: self.max_problem_excerpt]
        prompt = (
            _DECOMPOSER_SYSTEM
            + "\n\n"
            + _build_decomposer_user(template_text, resolved, excerpt)
        )
        try:
            gens = self.llm.generate(prompt, temperature=self.temperature, n=1)
            if not gens:
                return template_text.strip()
            out = _extract_subproblem(gens[0].text)
            self._log_call(prompt=prompt, response=gens[0].text or "",
                           finish_reason=gens[0].finish_reason,
                           node_id=node_id,
                           template_text=template_text,
                           clean_question=out)
            return out
        except Exception as e:                                      # noqa: BLE001
            self._log_call(prompt=prompt, response="",
                           finish_reason=None, node_id=node_id,
                           template_text=template_text,
                           clean_question=template_text.strip(),
                           error=repr(e))
            return template_text.strip()

    def _log_call(self, *, prompt: str, response: str,
                  finish_reason: str | None,
                  node_id: str | None,
                  template_text: str,
                  clean_question: str,
                  error: str | None = None) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.log_agent_call(
                role="decomposer",
                prompt=prompt,
                response=response,
                model=self.model_name,
                finish_reason=finish_reason,
                node_id=node_id,
                extras={
                    "raw_template": template_text,
                    "clean_question": clean_question,
                    "error": error,
                },
            )
        except Exception:                                           # noqa: BLE001
            pass
