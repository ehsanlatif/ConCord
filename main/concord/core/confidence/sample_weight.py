"""Sample-weight oracle (§5.1).

This module — NOT semantic_density — owns the white-box / black-box branch.
SD consumes weights agnostically.

- white-box: w(r) = exp(sum(token_logprobs) / len(token_logprobs))
  (length normalization is mandatory: raw sums bias toward short answers.)
- black-box: w(r) = (count(class(r)) + a) / (K + a * num_classes)
  Laplace smoothing keeps small-K runs from giving zero mass to lone classes.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..types import Generation


def whitebox_weights(gens: list[Generation]) -> np.ndarray:
    """Per-sample weight = exp(mean logprob). Returns length-K float array.

    Length normalization is REQUIRED — short responses otherwise dominate
    on raw sum logprob. A generation with token_logprobs=None falls back to
    a uniform weight of 1.0 (caller is responsible for not mixing modes).
    """
    out = np.empty(len(gens), dtype=np.float64)
    for i, g in enumerate(gens):
        lp = g.token_logprobs
        if lp is None or len(lp) == 0:
            out[i] = 1.0
            continue
        out[i] = math.exp(sum(lp) / len(lp))
    return out


def blackbox_class_mass(classes_members: list[list[int]], K: int,
                        laplace_a: float = 1.0) -> np.ndarray:
    """Laplace-smoothed mass PER CLASS.

    Returns one weight per class, in input order. Sums to ~1 (exactly 1 when
    a=0, biased toward uniform as a grows).
    """
    n_classes = len(classes_members)
    if n_classes == 0:
        return np.zeros(0, dtype=np.float64)
    denom = K + laplace_a * n_classes
    return np.array(
        [(len(m) + laplace_a) / denom for m in classes_members],
        dtype=np.float64,
    )


def per_sample_weights_from_classes(K: int, classes_members: list[list[int]],
                                    class_mass: np.ndarray) -> np.ndarray:
    """Spread per-class mass uniformly across its members → per-sample weights.

    Used in black-box mode where the oracle output is class-level. Each
    sample inherits 1/|class| of its class's mass. The resulting K-length
    weight vector feeds SD the same way the white-box vector does.
    """
    w = np.zeros(K, dtype=np.float64)
    for cl, m in zip(classes_members, class_mass):
        if not cl:
            continue
        share = float(m) / len(cl)
        for idx in cl:
            w[idx] = share
    return w


def auto_mode(supports_logprobs: bool, ablation_override: str = "auto") -> str:
    """Decide which oracle to use given LLM capability + ablation flag."""
    if ablation_override == "whitebox":
        return "whitebox"
    if ablation_override == "blackbox":
        return "blackbox"
    return "whitebox" if supports_logprobs else "blackbox"
