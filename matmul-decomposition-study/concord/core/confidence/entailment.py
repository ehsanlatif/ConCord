"""Entailment kernel k(a, b) = min(P(a |= b), P(b |= a)).

Bidirectional entailment defines meaning-class equivalence (§5.2). A kernel
value near 1 means the two responses entail each other (paraphrases /
equivalent answers); a value near 0 means at least one direction fails
(unrelated or contradictory).

Backends:

- `LexicalKernel`: cheap normalized-overlap proxy. Bidirectional by
  construction. Sufficient for unit tests that don't need real semantic
  understanding (we use crafted strings).
- `ScriptedKernel`: test-supplied dict[(a, b) -> float]. Lets unit tests
  set exact pairwise scores so clustering / SD math can be verified
  independently of any real model.
- `NLIKernel`: HuggingFace cross-encoder MNLI. Lazy-loaded; default model
  is `microsoft/deberta-v3-base-mnli`. Caches per-pair calls.
"""

from __future__ import annotations

import re
from typing import Protocol


class EntailmentKernel(Protocol):
    def kernel(self, a: str, b: str) -> float:
        """Bidirectional entailment probability in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# Lexical proxy
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\s]")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s) if t.strip()}


class LexicalKernel:
    """Min of directional containment scores.

    P(a|=b) ~= |a ∩ b| / |b|  (does b contain everything a says?)
    P(b|=a) ~= |a ∩ b| / |a|
    kernel = min(...) — so a strict subset of words doesn't claim bidirectional
    entailment. Coarse but bidirectional and deterministic.
    """

    def kernel(self, a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        if inter == 0:
            return 0.0
        p_ab = inter / len(tb)
        p_ba = inter / len(ta)
        return float(min(p_ab, p_ba))


# ---------------------------------------------------------------------------
# Scripted (for tests)
# ---------------------------------------------------------------------------

class ScriptedKernel:
    """Test fixture. Pairs are looked up canonically (sorted) so caller does
    not have to specify both directions. Missing pairs default to 0.0.
    """

    def __init__(self, pairs: dict[tuple[str, str], float] | None = None,
                 default: float = 0.0):
        self._table: dict[tuple[str, str], float] = {}
        self.default = default
        for (a, b), v in (pairs or {}).items():
            self._table[self._key(a, b)] = float(v)

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def kernel(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        return self._table.get(self._key(a, b), self.default)


# ---------------------------------------------------------------------------
# Real NLI cross-encoder (lazy)
# ---------------------------------------------------------------------------

class NLIKernel:
    """HuggingFace MNLI cross-encoder. Lazy-loads on first call.

    Computes P(entailment) in each direction and returns their min. Results
    are cached per (a, b, model_name) so repeated lookups during clustering
    don't re-run the model.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli"):
        self.model_name = model_name
        self._pipe = None
        self._cache: dict[tuple[str, str], float] = {}

    def _ensure(self) -> None:
        if self._pipe is None:
            # Use transformers + AutoTokenizer/AutoModel directly so we keep
            # control over the entailment label index. Done lazily to avoid
            # the heavy import at package load.
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch  # noqa: F401  (verify availability)

            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            mdl.eval()
            self._tok = tok
            self._mdl = mdl
            # MNLI label order: [contradiction, neutral, entailment] for most
            # checkpoints; we'll discover it from id2label.
            id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
            self._entail_idx = next(
                i for i, lbl in id2label.items() if "entail" in lbl
            )
            self._pipe = True  # sentinel so we don't reload

    def _p_entail(self, premise: str, hypothesis: str) -> float:
        import torch
        self._ensure()
        inputs = self._tok(premise, hypothesis, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._mdl(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return float(probs[self._entail_idx].item())

    def kernel(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        key = (a, b) if a <= b else (b, a)
        if key in self._cache:
            return self._cache[key]
        p_ab = self._p_entail(a, b)
        p_ba = self._p_entail(b, a)
        v = float(min(p_ab, p_ba))
        self._cache[key] = v
        return v
