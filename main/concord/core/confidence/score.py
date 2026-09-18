"""Unified score U_s and the full score-a-subproblem pipeline.

U_s(q) = mean over the DOMINANT meaning class of [ α · SD_i + (1-α) · v_i ]

This module owns the end-to-end "given K samples, give me the class with the
highest U_s and its mass" computation: it ties together the cosine pre-filter,
entailment confirmation, sample-weight oracle, semantic density, and verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..types import Generation
from .cluster import ClusterResult, MeaningClass, hybrid_cluster
from .embed import Embedder
from .entailment import EntailmentKernel
from .sample_weight import (
    auto_mode,
    blackbox_class_mass,
    per_sample_weights_from_classes,
    whitebox_weights,
)
from .semantic_density import all_densities
from .verifier import Verifier


@dataclass
class ScoredClass:
    klass: MeaningClass
    U_s: float
    SD_mean: float
    v_mean: float
    mass: float
    SD: np.ndarray              # per-member SD
    v: np.ndarray               # per-member verifier


@dataclass
class ScoreResult:
    responses: list[str]
    classes: list[ScoredClass]
    weights: np.ndarray         # per-sample weights (K,)
    kernel_matrix: np.ndarray   # K x K
    queried_pairs: int          # entailment queries actually issued
    oracle_mode: str            # "whitebox" | "blackbox"

    @property
    def dominant(self) -> ScoredClass | None:
        return self.classes[0] if self.classes else None


def _normalize_to_sum_one(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    if s <= 0:
        return np.full_like(x, 1.0 / max(1, len(x)))
    return x / s


def score_subproblem(gens: list[Generation], *, embedder: Embedder,
                     kernel: EntailmentKernel, verifier: Verifier,
                     subproblem: str, cfg: Config,
                     supports_logprobs: bool) -> ScoreResult:
    """Cluster, weight, and score the K samples for one subproblem.

    Returns classes sorted by U_s descending — `result.dominant` is the
    class that PUCT would use as the child's prior.
    """
    responses = [g.text for g in gens]
    K = len(responses)
    mode = auto_mode(supports_logprobs, cfg.ablation.weight_oracle)

    # 1) embed + cluster (cluster is mode-agnostic; mass comes after)
    if K == 0:
        return ScoreResult(responses=[], classes=[], weights=np.zeros(0),
                           kernel_matrix=np.zeros((0, 0)), queried_pairs=0,
                           oracle_mode=mode)
    emb = embedder.encode(responses)
    cr: ClusterResult = hybrid_cluster(responses, emb, kernel, cfg.confidence)

    # 2) sample-weight oracle
    if mode == "whitebox":
        w = whitebox_weights(gens)
        w = _normalize_to_sum_one(w)
        # class mass = sum of member weights
        class_mass = np.array([float(w[c.members].sum()) for c in cr.classes])
    else:
        cm = blackbox_class_mass([c.members for c in cr.classes], K=K,
                                 laplace_a=cfg.sampling.laplace_a)
        w = per_sample_weights_from_classes(K, [c.members for c in cr.classes], cm)
        class_mass = cm

    # 3) semantic density (per response) — uses normalized weights
    sd = all_densities(w, cr.kernel_matrix)

    # 4) verifier per response
    v = np.array([verifier.verify(r, subproblem) for r in responses],
                 dtype=np.float64)

    # 5) U_s per class = mean over members of α·SD + (1-α)·v
    alpha = cfg.confidence.alpha
    if cfg.ablation.confidence == "cosine_centroid":
        # Ablation arm: use cosine-to-centroid as the confidence signal.
        # Centroid is the mean of normalized embeddings within the class.
        ablated = np.zeros(K)
        for cls in cr.classes:
            mem = cls.members
            cent = emb[mem].mean(axis=0)
            cent /= max(1e-9, float(np.linalg.norm(cent)))
            for i in mem:
                ablated[i] = float(emb[i] @ cent)
        sd = ablated  # swap in for SD when ablating

    scored: list[ScoredClass] = []
    for cls, m in zip(cr.classes, class_mass):
        mem = np.array(cls.members, dtype=int)
        sd_m = sd[mem]
        v_m = v[mem]
        u_per = alpha * sd_m + (1 - alpha) * v_m
        u = float(u_per.mean()) if len(u_per) else 0.0
        cls.mass = float(m)
        scored.append(ScoredClass(
            klass=cls, U_s=u, SD_mean=float(sd_m.mean()) if len(sd_m) else 0.0,
            v_mean=float(v_m.mean()) if len(v_m) else 0.0,
            mass=float(m), SD=sd_m, v=v_m,
        ))

    # rank by U_s; mass acts as the PUCT prior elsewhere
    scored.sort(key=lambda s: (-s.U_s, -s.mass))

    return ScoreResult(
        responses=responses, classes=scored, weights=w,
        kernel_matrix=cr.kernel_matrix, queried_pairs=cr.queried_pairs,
        oracle_mode=mode,
    )
