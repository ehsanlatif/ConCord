"""Phase-1 task classifier (spec line 2: `CategorizeTask(P, G)`).

Currently a one-shot LLM call that picks a domain label from a fixed set.
Used when the caller does NOT supply `domain=...` to `solve()`. Returns
None on failure — the orchestrator then falls back to no-verifier scoring.

Cheap by design: one call per run, short output. Configure via
`cfg.models.classification`.
"""

from __future__ import annotations

import re

from .llm.client import LLMClient


_PROMPT = """You categorize a problem into a one-word domain label. Choose ONE of:

  - math
  - chess
  - chemistry
  - logic
  - cs
  - other

Reply with a single line of the form `domain = <label>`. Nothing else.

Problem (excerpt):
{problem}
"""

_DOMAIN_RE = re.compile(r"domain\s*=\s*([A-Za-z]+)", re.IGNORECASE)
_KNOWN = {"math", "chess", "chemistry", "logic", "cs", "other"}


def classify(problem: str, *, llm: LLMClient, temperature: float = 0.0,
             max_chars: int = 4000,
             tracer: "object | None" = None,
             model_name: str | None = None) -> str | None:
    """Return one of {math, chess, chemistry, logic, cs, other} or None.

    If `tracer` is supplied, the classifier LLM call (prompt + response +
    parsed domain) is logged to the tracer's `agent_calls.jsonl` stream.
    """
    excerpt = problem[:max_chars]
    prompt = _PROMPT.format(problem=excerpt)
    text = ""
    finish: str | None = None
    error: str | None = None
    try:
        gens = llm.generate(prompt, temperature=temperature, n=1)
        if gens:
            text = gens[0].text or ""
            finish = gens[0].finish_reason
    except Exception as e:                                          # noqa: BLE001
        error = repr(e)

    m = _DOMAIN_RE.search(text) if text else None
    label: str | None = None
    if m:
        cand = m.group(1).lower().strip()
        if cand in _KNOWN:
            label = cand

    if tracer is not None:
        try:
            tracer.log_agent_call(
                role="classifier",
                prompt=prompt,
                response=text,
                model=model_name,
                finish_reason=finish,
                extras={"extracted_domain": label, "error": error},
            )
        except Exception:                                           # noqa: BLE001
            pass

    return label
