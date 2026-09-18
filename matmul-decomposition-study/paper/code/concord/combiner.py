"""LLMCombiner — atomic answers → block answer.

Reads the original block question + every atomic-unit answer collected
during phase B, and produces a single candidate block answer. Sampled
K times (cfg.pipeline.K_combiner) so the existing semantic-density
clustering can pick the most coherent candidate.

When the block verifier rejects with `verifier_retry ≤ score < accept`,
the combiner is invoked AGAIN with the verifier's `issues` injected as
a hint. Up to `cfg.pipeline.combiner_retries` retries are allowed
before MCTS backtracks to the atomic layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Config
from ..llm.client import LLMClient
from ..types import Generation


_log = logging.getLogger(__name__)


_COMBINER_SYSTEM = (
    "You are a problem solver. You are given:\n"
    "  (1) the ORIGINAL problem,\n"
    "  (2) a list of atomic sub-question answers that decompose it.\n\n"
    "Your job is to COMBINE the atomic answers into a single, final, "
    "self-contained answer to the original problem.\n\n"
    "RULES:\n"
    "  - Be concise. Show one short line of reasoning if useful, then the "
    "answer.\n"
    "  - The very LAST line of your response MUST be exactly:\n"
    "        solution = <your final answer>\n"
    "  - If the atomic answers conflict, prefer the most internally "
    "consistent interpretation and note the conflict briefly.\n"
    "  - Do NOT re-derive atomic answers — trust them unless they are "
    "obviously inconsistent.\n"
)


def _format_atoms(atomic_answers: list[tuple[str, str]]) -> str:
    if not atomic_answers:
        return "(no atomic sub-answers; you must solve the original problem directly)"
    lines = []
    for i, (q, a) in enumerate(atomic_answers):
        lines.append(f"[atom {i}] Q: {q}\n          A: {a}")
    return "\n".join(lines)


def _build_combiner_prompt(*, original_question: str,
                            atomic_answers: list[tuple[str, str]],
                            verifier_hint: str | None = None) -> str:
    hint_block = ""
    if verifier_hint:
        hint_block = (
            "\n\nThe previous attempt was rejected. Issues to address:\n"
            f"{verifier_hint}\n"
        )
    return (
        _COMBINER_SYSTEM
        + "\n\nOriginal problem:\n"
        + original_question.strip()
        + "\n\nAtomic sub-answers (in order):\n"
        + _format_atoms(atomic_answers)
        + hint_block
        + "\n\nNow produce the combined final answer."
    )


@dataclass
class LLMCombiner:
    """Stateless combiner. K samples per call, deterministic seeding."""

    llm: LLMClient
    cfg: Config
    temperature: float = 0.5
    tracer: "object | None" = None
    model_name: str | None = None

    def combine(self, *, original_question: str,
                atomic_answers: list[tuple[str, str]],
                verifier_hint: str | None = None,
                node_id: str | None = None) -> list[Generation]:
        """Run K_combiner samples. Returns the raw generations; callers
        feed them through the existing cluster+score pipeline."""
        prompt = _build_combiner_prompt(
            original_question=original_question,
            atomic_answers=atomic_answers,
            verifier_hint=verifier_hint,
        )
        K = self.cfg.pipeline.K_combiner
        try:
            gens = self.llm.generate(prompt, temperature=self.temperature, n=K)
        except Exception as e:                                      # noqa: BLE001
            _log.exception("LLMCombiner call failed; returning empty list")
            self._log_calls(prompt=prompt, gens=[], node_id=node_id,
                            error=repr(e),
                            verifier_hint=verifier_hint)
            return []
        self._log_calls(prompt=prompt, gens=gens, node_id=node_id,
                        error=None, verifier_hint=verifier_hint)
        return gens

    def _log_calls(self, *, prompt: str, gens: list[Generation],
                   node_id: str | None, error: str | None,
                   verifier_hint: str | None) -> None:
        if self.tracer is None:
            return
        if not gens:
            try:
                self.tracer.log_agent_call(
                    role="combiner", prompt=prompt, response="",
                    model=self.model_name, finish_reason=None,
                    node_id=node_id,
                    extras={"error": error,
                             "verifier_hint_used": bool(verifier_hint)},
                )
            except Exception:                                       # noqa: BLE001
                pass
            return
        for i, g in enumerate(gens):
            try:
                self.tracer.log_agent_call(
                    role="combiner", prompt=prompt,
                    response=g.text or "", model=self.model_name,
                    finish_reason=g.finish_reason, node_id=node_id,
                    sample_idx=i,
                    extras={"verifier_hint_used": bool(verifier_hint)},
                )
            except Exception:                                       # noqa: BLE001
                pass
