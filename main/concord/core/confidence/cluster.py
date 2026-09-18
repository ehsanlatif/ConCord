"""Hybrid meaning-class clustering (§5.2).

Pipeline:
  1. cosine_groups(emb, tau_pre)            -- O(K^2) but with cheap embeddings
  2. inside each cosine group: confirm by entailment >= threshold (single-link)
  3. across cosine groups: only run entailment on BORDERLINE cross-group pairs
     (cosine >= tau_pre * 0.75) to catch synonyms that the surface filter
     under-grouped. This is the "k neighbours not K^2" optimisation.

Output: list of MeaningClass with members (response indices), representative
text (the within-class medoid by avg-kernel), mass (will be set by the
sample-weight oracle in `score.py` — clustering itself does not know about
oracle mode), and the precomputed `kernel_matrix[i][j]` for the FULL K-by-K
matrix lazily filled in only at queried pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ConfidenceCfg
from .embed import Embedder, cosine_groups
from .entailment import EntailmentKernel


@dataclass
class MeaningClass:
    members: list[int]                   # indices into the response list
    representative: int                  # index of the medoid response
    mass: float = 0.0                    # set by sample_weight oracle
    representative_text: str = ""        # convenience copy


@dataclass
class ClusterResult:
    classes: list[MeaningClass]
    kernel_matrix: np.ndarray            # K x K, zeros where not queried
    queried_pairs: int = 0               # diagnostic: O(K*k) check


class _LazyKernel:
    """Wraps an EntailmentKernel + the response list so we can fill a K x K
    matrix only at indices that get queried. Symmetric storage.
    """

    def __init__(self, responses: list[str], kernel: EntailmentKernel):
        self.responses = responses
        self.kernel = kernel
        n = len(responses)
        self.mat = np.zeros((n, n), dtype=np.float32)
        # diagonal = 1.0 by convention
        for i in range(n):
            self.mat[i, i] = 1.0
        self.filled: set[tuple[int, int]] = {(i, i) for i in range(n)}
        self.queries = 0

    def get(self, i: int, j: int) -> float:
        if i == j:
            return 1.0
        key = (i, j) if i < j else (j, i)
        if key in self.filled:
            return float(self.mat[key])
        v = self.kernel.kernel(self.responses[i], self.responses[j])
        self.mat[key] = v
        self.mat[(key[1], key[0])] = v
        self.filled.add(key)
        self.queries += 1
        return v


def _confirm_within_group(group: list[int], lk: _LazyKernel,
                          threshold: float) -> list[list[int]]:
    """Single-link split of a cosine group via entailment.

    Inside one cosine group, items that fail bidirectional entailment
    (kernel < threshold to ALL current cluster members) split off.
    """
    if not group:
        return []
    clusters: list[list[int]] = [[group[0]]]
    for idx in group[1:]:
        placed = False
        for cl in clusters:
            # link to ANY existing member (single-link)
            if any(lk.get(idx, m) >= threshold for m in cl):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


def _merge_across_groups(clusters: list[list[int]], emb: np.ndarray,
                          lk: _LazyKernel, cfg: ConfidenceCfg) -> list[list[int]]:
    """Merge two clusters when borderline cross-group pairs entail.

    Borderline = cosine in [0.75 * tau_pre, tau_pre). Far-apart pairs don't
    get re-checked; closer-than-tau_pre pairs were already in the same
    cosine group. Keeps the entailment budget at O(K * k).
    """
    if not clusters:
        return clusters
    threshold = cfg.entail_threshold
    lo = cfg.tau_pre * 0.75
    hi = cfg.tau_pre
    n_clusters = len(clusters)
    parent = list(range(n_clusters))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # for each pair of clusters, check borderline candidate
    for ci in range(n_clusters):
        for cj in range(ci + 1, n_clusters):
            if find(ci) == find(cj):
                continue
            # find one borderline pair (members closest in cosine)
            best = (None, -1.0)
            for a in clusters[ci]:
                for b in clusters[cj]:
                    sim = float(emb[a] @ emb[b])
                    if lo <= sim < hi and sim > best[1]:
                        best = ((a, b), sim)
            if best[0] is None:
                continue
            a, b = best[0]
            if lk.get(a, b) >= threshold:
                union(ci, cj)

    # collect merged
    merged: dict[int, list[int]] = {}
    for ci, members in enumerate(clusters):
        merged.setdefault(find(ci), []).extend(members)
    return [sorted(v) for v in merged.values()]


def _medoid(cluster: list[int], lk: _LazyKernel) -> int:
    """Member with the highest average kernel to the rest of the cluster."""
    if len(cluster) == 1:
        return cluster[0]
    best_idx, best_score = cluster[0], -1.0
    for i in cluster:
        s = sum(lk.get(i, j) for j in cluster if j != i) / max(1, len(cluster) - 1)
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx


def hybrid_cluster(responses: list[str], emb: np.ndarray, kernel: EntailmentKernel,
                   cfg: ConfidenceCfg) -> ClusterResult:
    """Run cosine pre-filter then entailment confirmation.

    Returns ClusterResult with mass=0 (the oracle fills it). The full K x K
    kernel matrix is returned (zeros where not queried), so callers that
    want the lazy matrix for SD calculation can use it directly.
    """
    lk = _LazyKernel(responses, kernel)
    if not responses:
        return ClusterResult(classes=[], kernel_matrix=lk.mat, queried_pairs=0)

    cosine_g = cosine_groups(emb, cfg.tau_pre)

    # confirm within groups
    refined: list[list[int]] = []
    for g in cosine_g:
        refined.extend(_confirm_within_group(g, lk, cfg.entail_threshold))

    # merge borderline cross-group pairs
    final = _merge_across_groups(refined, emb, lk, cfg)

    classes: list[MeaningClass] = []
    for members in final:
        rep = _medoid(members, lk)
        classes.append(MeaningClass(
            members=sorted(members),
            representative=rep,
            representative_text=responses[rep],
        ))
    # stable order: descending size, then by lowest member index
    classes.sort(key=lambda c: (-len(c.members), min(c.members)))
    return ClusterResult(classes=classes, kernel_matrix=lk.mat,
                         queried_pairs=lk.queries)
