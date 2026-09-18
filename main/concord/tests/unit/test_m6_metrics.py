"""M6 gate — metrics + baselines round-trip.

Plan §7 gate: 'one full comparison table (Concord vs B0..B3) on a small
slice, with per-run cost logged'. We test the building blocks here; the
end-to-end run is exercised separately by `experiments/compare.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PKG_ROOT.parent))

from core.config import Config
from core.llm.mock import MockLLM

from experiments.baselines import BASELINES
from experiments.metrics import RunRecord, accuracy, auroc, coherence_rate, ece, summarize


# ---------------------------------------------------------------------------
# Metrics primitives
# ---------------------------------------------------------------------------

def test_accuracy_ignores_ungraded():
    recs = [
        RunRecord("q1", correct=True,  coherent=True, sigma=0.0,
                  rollouts=1, calls=1, tokens_in=0, tokens_out=0, usd=0, elapsed_s=0),
        RunRecord("q2", correct=False, coherent=True, sigma=0.0,
                  rollouts=1, calls=1, tokens_in=0, tokens_out=0, usd=0, elapsed_s=0),
        RunRecord("q3", correct=None,  coherent=False, sigma=1.0,
                  rollouts=1, calls=1, tokens_in=0, tokens_out=0, usd=0, elapsed_s=0),
    ]
    assert accuracy(recs) == 0.5


def test_coherence_rate_counts_all():
    recs = [
        RunRecord("q1", correct=True,  coherent=True,  sigma=0.0,
                  rollouts=1, calls=1, tokens_in=0, tokens_out=0, usd=0, elapsed_s=0),
        RunRecord("q2", correct=False, coherent=False, sigma=0.5,
                  rollouts=1, calls=1, tokens_in=0, tokens_out=0, usd=0, elapsed_s=0),
    ]
    assert coherence_rate(recs) == 0.5


def test_auroc_perfect_separation():
    # high-score positives, low-score negatives → AUROC 1.0
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [True, True, True, False, False]
    assert auroc(scores, labels) == 1.0


def test_auroc_random_around_half():
    # interleaved → ~0.5
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [True, False, True, False]
    a = auroc(scores, labels)
    assert 0.4 <= a <= 0.6


def test_auroc_returns_nan_when_one_class_missing():
    import math
    a = auroc([0.5, 0.6], [True, True])
    assert math.isnan(a)


def test_ece_perfect_calibration_is_near_zero():
    # confidence == empirical accuracy in each bin
    scores = [0.1] * 10 + [0.9] * 10
    labels = ([False] * 9 + [True]) + ([True] * 9 + [False])
    assert ece(scores, labels, n_bins=10) <= 0.05


def test_summarize_returns_full_dict():
    recs = [
        RunRecord("q1", correct=True, coherent=True, sigma=0.0,
                  rollouts=4, calls=8, tokens_in=100, tokens_out=80,
                  usd=0.01, elapsed_s=1.0, u_s_sample=0.9),
        RunRecord("q2", correct=False, coherent=False, sigma=0.6,
                  rollouts=12, calls=20, tokens_in=200, tokens_out=160,
                  usd=0.02, elapsed_s=2.5, u_s_sample=0.4),
    ]
    s = summarize(recs)
    assert s["n"] == 2
    assert s["accuracy"] == 0.5
    assert s["coherence_rate"] == 0.5
    assert s["cost"]["calls"] == 28
    assert s["calibration"]["n_calibration_points"] == 2


# ---------------------------------------------------------------------------
# Baselines run end-to-end on the mock LLM
# ---------------------------------------------------------------------------

EASY_PROBLEM = (
    "Problem node_0: Compute the value 5 + 5.\n"
    "Problem node_1: Use the answer from problem node_0 and add 2.\n"
)


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 6
    cfg.sampling.K_blackbox = 3
    return cfg


@pytest.mark.parametrize("arm_name", list(BASELINES.keys()))
def test_each_baseline_runs_end_to_end(arm_name: str, tmp_path: Path):
    cfg = _cfg(tmp_path)
    llm = MockLLM(responses={})
    solve_fn = BASELINES[arm_name]
    res = solve_fn(EASY_PROBLEM, cfg=cfg, llm=llm, domain="math")
    # No crash; returns a Result; cost is recorded.
    assert res is not None
    assert isinstance(res.rollouts, int)
    assert "calls" in res.cost
