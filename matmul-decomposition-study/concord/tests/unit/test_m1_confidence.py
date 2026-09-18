"""M1 gate — confidence stack.

Plan §7 gate:
- synonyms cluster together, contradictions don't
- SD higher for dense clusters
- white-box vs black-box weights both produce valid U_s
- (impl note) hybrid filter keeps entailment cost sub-quadratic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.confidence import (
    LexicalEmbedder,
    LexicalKernel,
    LexicalVerifier,
    NullVerifier,
    ScriptedKernel,
    hybrid_cluster,
    score_subproblem,
)
from core.types import Generation

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _cfg() -> Config:
    return Config.from_yaml(PKG_ROOT / "config" / "default.yaml")


def _gen(text: str, lp: list[float] | None = None) -> Generation:
    return Generation(text=text, token_logprobs=lp, finish_reason="stop")


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def test_synonyms_cluster_contradictions_do_not():
    """With scripted kernel scores, paraphrases should land in one class and a
    contradictory answer should land in its own.
    """
    responses = [
        "the answer is 42",
        "answer: 42",
        "forty-two is the answer",
        "the answer is 7",
    ]
    # scripted: high entailment among the first three, low between any of them
    # and the fourth.
    pairs = {
        (responses[0], responses[1]): 0.95,
        (responses[0], responses[2]): 0.9,
        (responses[1], responses[2]): 0.92,
        (responses[0], responses[3]): 0.05,
        (responses[1], responses[3]): 0.05,
        (responses[2], responses[3]): 0.05,
    }
    kernel = ScriptedKernel(pairs=pairs)
    cfg = _cfg().confidence
    emb = LexicalEmbedder().encode(responses)
    cr = hybrid_cluster(responses, emb, kernel, cfg)

    classes = [set(c.members) for c in cr.classes]
    # the three paraphrases should be in one class
    assert {0, 1, 2} in classes
    # the contradiction should be in its own class
    assert {3} in classes


def test_cluster_handles_empty_and_singleton():
    cfg = _cfg().confidence
    emb = LexicalEmbedder().encode([])
    cr = hybrid_cluster([], emb, ScriptedKernel(), cfg)
    assert cr.classes == []

    emb1 = LexicalEmbedder().encode(["lone"])
    cr1 = hybrid_cluster(["lone"], emb1, ScriptedKernel(), cfg)
    assert len(cr1.classes) == 1
    assert cr1.classes[0].members == [0]


# ---------------------------------------------------------------------------
# semantic density
# ---------------------------------------------------------------------------

def test_sd_higher_for_denser_cluster():
    """Two responses in a tight 4-way agreement should score higher SD than
    a singleton against the same set.
    """
    # responses 0-3 entail each other strongly; response 4 is alone.
    responses = [f"agree_{i}" for i in range(4)] + ["lone_answer"]

    pairs: dict[tuple[str, str], float] = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pairs[(responses[i], responses[j])] = 0.95
        # weak relation to lone:
        pairs[(responses[i], responses[4])] = 0.05

    kernel = ScriptedKernel(pairs=pairs)
    cfg = _cfg()
    gens = [_gen(r) for r in responses]
    res = score_subproblem(
        gens,
        embedder=LexicalEmbedder(),
        kernel=kernel,
        verifier=NullVerifier(value=0.5),
        subproblem="?",
        cfg=cfg,
        supports_logprobs=False,
    )

    # SD of any sample in the dense cluster should beat the lone sample.
    dense_sd = max(c.SD.max() for c in res.classes if 0 in c.klass.members
                   or 1 in c.klass.members or 2 in c.klass.members
                   or 3 in c.klass.members)
    lone_sd = max(c.SD.max() for c in res.classes if 4 in c.klass.members)
    assert dense_sd > lone_sd


# ---------------------------------------------------------------------------
# sample-weight oracle (both modes)
# ---------------------------------------------------------------------------

def test_whitebox_weights_length_normalized():
    from core.confidence.sample_weight import whitebox_weights

    short = _gen("yes", lp=[-0.1])              # mean -0.1
    long_  = _gen("yes " * 5, lp=[-0.1] * 5)    # mean -0.1, same per-token
    w = whitebox_weights([short, long_])
    # Length normalization: identical per-token logprob -> equal weights.
    assert abs(w[0] - w[1]) < 1e-9


def test_whitebox_weights_favor_higher_confidence():
    from core.confidence.sample_weight import whitebox_weights

    high = _gen("a", lp=[-0.1, -0.1])      # mean -0.1
    low  = _gen("b", lp=[-3.0, -3.0])      # mean -3.0
    w = whitebox_weights([high, low])
    assert w[0] > w[1]


def test_blackbox_class_mass_laplace():
    from core.confidence.sample_weight import blackbox_class_mass

    # 3 classes with sizes [4, 1, 0] out of K=5
    m = blackbox_class_mass([[0, 1, 2, 3], [4], []], K=5, laplace_a=1.0)
    # denom = 5 + 1*3 = 8; numerators 5, 2, 1
    assert m[0] == pytest.approx(5/8)
    assert m[1] == pytest.approx(2/8)
    assert m[2] == pytest.approx(1/8)
    # never zero (the empty-class no-divide-by-zero guarantee from §7)
    assert (m > 0).all()


def test_both_oracle_modes_produce_valid_u_s():
    """Same responses, same kernel — verify both modes return U_s in [0, 1]."""
    responses = ["foo", "foo", "bar"]
    pairs = {
        (responses[0], responses[1]): 0.95,
        (responses[0], responses[2]): 0.1,
        (responses[1], responses[2]): 0.1,
    }
    kernel = ScriptedKernel(pairs=pairs)
    cfg = _cfg()

    # white-box
    gens_wb = [_gen(r, lp=[-0.5, -0.5]) for r in responses]
    res_wb = score_subproblem(
        gens_wb, embedder=LexicalEmbedder(), kernel=kernel,
        verifier=NullVerifier(0.7), subproblem="?", cfg=cfg,
        supports_logprobs=True,
    )
    assert res_wb.oracle_mode == "whitebox"
    for c in res_wb.classes:
        assert 0.0 <= c.U_s <= 1.0

    # black-box
    gens_bb = [_gen(r) for r in responses]
    res_bb = score_subproblem(
        gens_bb, embedder=LexicalEmbedder(), kernel=kernel,
        verifier=NullVerifier(0.7), subproblem="?", cfg=cfg,
        supports_logprobs=False,
    )
    assert res_bb.oracle_mode == "blackbox"
    for c in res_bb.classes:
        assert 0.0 <= c.U_s <= 1.0
    # dominant class is the 2-sample one in both modes
    assert sorted(res_wb.dominant.klass.members) == [0, 1]
    assert sorted(res_bb.dominant.klass.members) == [0, 1]


# ---------------------------------------------------------------------------
# complexity / cost bound
# ---------------------------------------------------------------------------

def test_entailment_queries_stay_subquadratic():
    """The hybrid pipeline must NOT issue K*(K-1)/2 entailment calls.

    Plan §6: 'If the entailment cost shows O(K^2) scaling in profiling, the
    hybrid pre-filter is broken.'
    """
    # Build K=24 responses across 3 tight surface-similar buckets so cosine
    # pre-filter does most of the work.
    responses = (
        [f"alpha bravo charlie {i}" for i in range(8)] +
        [f"delta echo foxtrot {i}" for i in range(8)] +
        [f"golf hotel india {i}" for i in range(8)]
    )
    K = len(responses)

    # Use the lexical kernel — it's deterministic and we just want to count
    # how many times it's called.
    class CountingKernel:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0
        def kernel(self, a, b):
            self.calls += 1
            return self.inner.kernel(a, b)

    ck = CountingKernel(LexicalKernel())
    cfg = _cfg().confidence
    emb = LexicalEmbedder().encode(responses)
    cr = hybrid_cluster(responses, emb, ck, cfg)

    # All-pairs would be K*(K-1)/2 = 276 calls. Hybrid should be << half that.
    all_pairs = K * (K - 1) // 2
    assert ck.calls < all_pairs // 2, (
        f"hybrid pre-filter issued {ck.calls} of {all_pairs} possible pairs"
    )
    assert cr.queried_pairs == ck.calls


# ---------------------------------------------------------------------------
# verifier plug-in surface
# ---------------------------------------------------------------------------

def test_verifier_protocol_swappable():
    """U_s should track the verifier signal when alpha is small (verifier-heavy)."""
    responses = ["x", "x"]
    pairs = {(responses[0], responses[1]): 0.95}
    cfg = _cfg()
    cfg.confidence.alpha = 0.0  # all weight on verifier

    high_v = score_subproblem(
        [_gen(r) for r in responses],
        embedder=LexicalEmbedder(), kernel=ScriptedKernel(pairs=pairs),
        verifier=NullVerifier(1.0), subproblem="?", cfg=cfg,
        supports_logprobs=False,
    )
    low_v = score_subproblem(
        [_gen(r) for r in responses],
        embedder=LexicalEmbedder(), kernel=ScriptedKernel(pairs=pairs),
        verifier=NullVerifier(0.0), subproblem="?", cfg=cfg,
        supports_logprobs=False,
    )
    assert high_v.dominant.U_s > low_v.dominant.U_s
