"""B0: single-shot chain-of-thought. No tree, no sampling.

Plan §8.2 baseline 0. The simplest possible reference point — one LLM call
per problem. Isolates "is the search adding any value at all over a single
strong prompt?".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import Config
from core.expansion import extract_answer
from core.llm.client import LLMClient
from core.types import Result


_PROMPT_TMPL = (
    "Solve the following problem step by step. End your response with a line "
    "of the form\n    solution = <your final answer>\n\n"
    "Problem:\n{problem}\n"
)


def solve(problem: str, *, cfg: Config, llm: LLMClient, domain: str | None = None,
          tag: str | None = None) -> Result:
    prompt = _PROMPT_TMPL.format(problem=problem)
    gens = llm.generate(prompt, temperature=cfg.llm.temperature, n=1)
    text = gens[0].text
    answer = extract_answer(text)
    cost = llm.cost()
    return Result(
        answer=answer, coherent=False, sigma=0.0,
        rollouts=1, cost=cost.to_dict(),
        flagged="baseline_b0_cot",
    )
