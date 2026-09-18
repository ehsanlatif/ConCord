"""Embeddings + cosine pre-filter for the hybrid clustering pipeline.

The pre-filter (§5.2) is a CANDIDATE GROUPER, not a clustering decision: it
forms loose cosine groups so the entailment kernel only runs on borderline
cross-group pairs. That's how we keep the per-expansion cost at O(K*k)
instead of O(K**2).

Two backends are shipped:

- `LexicalEmbedder`: deterministic, no model downloads. Uses character 3-gram
  hash counts -> sparse-but-fixed-width vector. Sufficient for the M1 unit
  gate (synonyms cluster, contradictions don't — with carefully crafted text)
  and for any test that should run in CI without network.
- `STEmbedder`: lazy wrapper around `sentence-transformers`. Imported only on
  first call so the package stays importable without that heavy dep.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dim) L2-normalized embeddings."""
        ...


# ---------------------------------------------------------------------------
# Lexical (cheap, deterministic, no downloads)
# ---------------------------------------------------------------------------

class LexicalEmbedder:
    """Hashed character-3gram bag. Cosine on this proxies surface similarity."""

    def __init__(self, dim: int = 256, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def _hash(self, tok: str) -> int:
        h = hashlib.blake2b(tok.encode(), digest_size=4).digest()
        return int.from_bytes(h, "big") % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = t.strip().lower()
            if not s:
                continue
            padded = f"  {s}  "
            for j in range(len(padded) - self.ngram + 1):
                tok = padded[j : j + self.ngram]
                vecs[i, self._hash(tok)] += 1.0
        # L2 normalize (rows with zero norm stay zero)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ---------------------------------------------------------------------------
# Sentence-Transformers (lazy)
# ---------------------------------------------------------------------------

class STEmbedder:
    """sentence-transformers wrapper. Loads the model on first encode()."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = 384  # MiniLM-L6 default; overwritten after first load

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        v = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)


# ---------------------------------------------------------------------------
# Cosine pre-filter — groups candidates for entailment confirmation
# ---------------------------------------------------------------------------

def cosine_groups(emb: np.ndarray, tau_pre: float) -> list[list[int]]:
    """Greedy single-link grouping at cosine threshold `tau_pre`.

    Returns a partition: list of groups, each a list of row indices into `emb`.
    Order within groups preserves input order. This is a CANDIDATE filter —
    the entailment kernel later confirms or merges across groups. With a loose
    `tau_pre` (e.g. 0.6) we accept false-merges that entailment can split, and
    avoid the O(K**2) cost of all-pairs NLI.
    """
    n = emb.shape[0]
    if n == 0:
        return []
    assigned = [-1] * n
    groups: list[list[int]] = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        gid = len(groups)
        assigned[i] = gid
        groups.append([i])
        # link-by-similarity to existing seed (single-link greedy)
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            sim = float(emb[i] @ emb[j])
            if sim >= tau_pre:
                assigned[j] = gid
                groups[gid].append(j)
    return groups


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine matrix for already-normalized embeddings."""
    if emb.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return (emb @ emb.T).astype(np.float32)
