"""Metrics for the experiment harness (plan §8.3).

Per-run metrics are written into the JSONL traces by the orchestrator and the
baselines; this module computes the *aggregate* metrics by reading those
traces back. Computed:

  - solve@budget: accuracy
  - coherence_rate: fraction of returned answers with sigma == 0
  - cost: total calls, tokens, USD, wall-clock (sum)
  - search_efficiency: rollouts mean, max depth (when reported)
  - U_s calibration: AUROC + ECE of U_s vs eventual correctness

ECE and AUROC are computed from per-rollout U_s and the final-answer
correctness for each problem. These are the two numbers the §10 acceptance
criteria explicitly call out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RunRecord:
    """Minimum data needed for aggregation."""

    question_id: str
    correct: bool | None
    coherent: bool
    sigma: float
    rollouts: int
    calls: int
    tokens_in: int
    tokens_out: int
    usd: float
    elapsed_s: float
    u_s_sample: float | None = None   # representative U_s for calibration


def accuracy(records: list[RunRecord]) -> float:
    graded = [r for r in records if r.correct is not None]
    if not graded:
        return float("nan")
    return sum(1 for r in graded if r.correct) / len(graded)


def coherence_rate(records: list[RunRecord]) -> float:
    return (sum(1 for r in records if r.coherent) / max(1, len(records)))


def total_cost(records: list[RunRecord]) -> dict[str, float]:
    return {
        "calls": sum(r.calls for r in records),
        "tokens_in": sum(r.tokens_in for r in records),
        "tokens_out": sum(r.tokens_out for r in records),
        "usd": round(sum(r.usd for r in records), 4),
        "elapsed_s": round(sum(r.elapsed_s for r in records), 1),
    }


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Mann-Whitney U / AUROC. Returns NaN with <2 distinct classes."""
    n_pos = sum(1 for l in labels if l)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    # rank averages for tied scores
    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        # ranks 1-indexed
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(scores: list[float], labels: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error on equally-spaced bins in [0, 1]."""
    if not scores:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for s, l in zip(scores, labels):
        s = max(0.0, min(1.0, s))
        b = min(n_bins - 1, int(s * n_bins))
        bins[b].append((s, l))
    total = len(scores)
    err = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(s for s, _ in bucket) / len(bucket)
        acc = sum(1 for _, l in bucket if l) / len(bucket)
        err += (len(bucket) / total) * abs(avg_conf - acc)
    return err


def summarize(records: list[RunRecord]) -> dict:
    correct_records = [r for r in records if r.correct is not None]
    cal_scores = [r.u_s_sample for r in records if r.u_s_sample is not None
                  and r.correct is not None]
    cal_labels = [bool(r.correct) for r in records if r.u_s_sample is not None
                  and r.correct is not None]
    return {
        "n": len(records),
        "n_graded": len(correct_records),
        "accuracy": accuracy(records),
        "coherence_rate": coherence_rate(records),
        "cost": total_cost(records),
        "rollouts_mean": (
            sum(r.rollouts for r in records) / max(1, len(records))
        ),
        "calibration": {
            "auroc": auroc(cal_scores, cal_labels) if cal_scores else float("nan"),
            "ece": ece(cal_scores, cal_labels) if cal_scores else float("nan"),
            "n_calibration_points": len(cal_scores),
        },
    }
