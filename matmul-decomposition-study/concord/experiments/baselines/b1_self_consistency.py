"""B1: self-consistency (K samples, majority vote).

Plan §8.2 baseline 1. K independent CoT samples; the most common
*extracted answer* wins. Isolates "the value of search vs. just sampling
more". Uses the same prompt as B0.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import Config
from core.expansion import extract_answer
from core.llm.client import LLMClient
from core.types import Result

from .b0_cot import _PROMPT_TMPL


def solve(problem: str, *, cfg: Config, llm: LLMClient, domain: str | None = None,
          tag: str | None = None) -> Result:
    prompt = _PROMPT_TMPL.format(problem=problem)
    K = cfg.sampling.K_blackbox if not llm.supports_logprobs else cfg.sampling.K_whitebox
    gens = llm.generate(prompt, temperature=cfg.llm.temperature, n=K)

    answers = [extract_answer(g.text) for g in gens]
    counts = Counter(a for a in answers if a)
    if not counts:
        return Result(answer="", coherent=False, sigma=1.0,
                      rollouts=1, cost=llm.cost().to_dict(),
                      flagged="baseline_b1_no_answer_extracted")
    answer, votes = counts.most_common(1)[0]
    # "confidence" of majority vote = vote share
    confidence = votes / max(1, len(answers))
    cost = llm.cost()
    return Result(
        answer=answer,
        coherent=(confidence == 1.0),
        sigma=1.0 - confidence,
        rollouts=1, cost=cost.to_dict(),
        flagged=("baseline_b1_self_consistency"
                 if confidence < 1.0 else None),
    )
