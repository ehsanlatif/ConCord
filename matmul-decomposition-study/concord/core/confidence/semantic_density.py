"""Semantic Density (§5.2, spec §1).

SD(r_i) = Σ_j w_j · k(r_i, r_j)

This is the *intrinsic confidence* `c_i` in the spec; there is no separate
self-confidence call. SD is monotone in cluster size AND cluster tightness:
many co-confident samples that entail each other -> high SD; a lone sample
or one in a soft cluster -> low SD.
"""

from __future__ import annotations

import numpy as np


def semantic_density(i: int, weights: np.ndarray,
                     kernel_matrix: np.ndarray) -> float:
    """SD for response i."""
    return float(weights @ kernel_matrix[i])


def all_densities(weights: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray:
    """SD for every response in one matmul. Returns length-K array."""
    return kernel_matrix @ weights
