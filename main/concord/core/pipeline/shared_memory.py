"""Shared memory with both text and vector retrieval.

Two access modes:

  1. **Exact lookup by key.** `get("node_5")` → the answer literally
     committed for block `node_5`. Used for explicit `answer from
     problem node_K` cross-references.

  2. **Semantic retrieval by free-text query.** `search("graph theory
     incidence matrix bridge", k=3)` → top-k stored entries whose
     `question` text is most cosine-similar to the query. Used by
     atomic units that ask "have we solved anything related to X?"
     without knowing the upstream node id.

The embedder is the existing `LexicalEmbedder` (deterministic,
character-3-gram, no model download). The store is in-memory per run
and persisted to `shared_memory.json` alongside the manifest.

Entries are timestamped + scope-tagged (block id) so callers can filter
to "answers from blocks I depend on, not random siblings."
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..confidence.embed import LexicalEmbedder


@dataclass
class MemoryEntry:
    """One write to the scratchpad.

    `scope` is a hierarchical id like "block:node_5" or
    "atom:node_5/d2/atom_1" so callers can filter by level.
    `question` is the natural-language question the answer responded to —
    that's what gets embedded for retrieval.
    `extras` holds any provenance fields (confidence score, source
    rollout, etc.).
    """

    scope: str
    question: str
    answer: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class VectorSharedMemory:
    """In-memory text + lexical-vector scratchpad.

    Thread-unsafe on purpose (the orchestrator is single-threaded).
    The embedder is held loosely so tests can swap in a stub.
    """

    def __init__(self, embedder: LexicalEmbedder | None = None):
        self._entries: list[MemoryEntry] = []
        # `_index[scope]` → index into `_entries` (last write wins).
        self._index: dict[str, int] = {}
        self._embedder = embedder or LexicalEmbedder()
        # Lazy-built embedding matrix; invalidated on every write.
        self._emb: np.ndarray | None = None

    # ----- writes ---------------------------------------------------------

    def put(self, scope: str, question: str, answer: str,
            **extras: Any) -> MemoryEntry:
        """Write a (scope, question, answer) triple. Last write wins for a
        given scope. Returns the stored entry."""
        entry = MemoryEntry(scope=scope, question=question,
                            answer=answer, extras=dict(extras))
        if scope in self._index:
            self._entries[self._index[scope]] = entry
        else:
            self._index[scope] = len(self._entries)
            self._entries.append(entry)
        self._emb = None      # invalidate cached embeddings
        return entry

    # ----- exact / scope queries -----------------------------------------

    def get(self, scope: str) -> MemoryEntry | None:
        """Exact lookup by scope id (e.g. `block:node_5`)."""
        idx = self._index.get(scope)
        return None if idx is None else self._entries[idx]

    def get_block(self, node_id: str) -> str | None:
        """Convenience: return the committed answer for `block:<node_id>`
        or None if not yet committed."""
        e = self.get(f"block:{node_id}")
        return None if e is None else e.answer

    def all_blocks(self) -> dict[str, str]:
        """Snapshot of every committed block answer keyed by node_id."""
        out: dict[str, str] = {}
        for e in self._entries:
            if e.scope.startswith("block:"):
                out[e.scope.split(":", 1)[1]] = e.answer
        return out

    def __len__(self) -> int:
        return len(self._entries)

    # ----- semantic search ------------------------------------------------

    def _ensure_emb(self) -> np.ndarray:
        if self._emb is None or self._emb.shape[0] != len(self._entries):
            if not self._entries:
                self._emb = np.zeros((0, self._embedder.dim))
            else:
                self._emb = self._embedder.encode(
                    [e.question for e in self._entries])
        return self._emb

    def search(self, query: str, k: int = 3,
               scope_prefix: str | None = None) -> list[tuple[MemoryEntry, float]]:
        """Top-k entries by cosine similarity on the QUESTION text.

        Returns a list of (entry, similarity) in descending order.
        `scope_prefix` filters to entries whose scope starts with the prefix
        (e.g. `"block:"` to exclude atoms).
        """
        if not self._entries or k <= 0:
            return []
        # Filter first so the embedding subset matches the candidates.
        cand_idx = [i for i, e in enumerate(self._entries)
                    if scope_prefix is None or e.scope.startswith(scope_prefix)]
        if not cand_idx:
            return []
        emb = self._ensure_emb()
        q = self._embedder.encode([query])[0]
        sims = (emb[cand_idx] @ q).astype(float)
        # Sort descending by similarity.
        order = np.argsort(-sims)[: min(k, len(cand_idx))]
        return [(self._entries[cand_idx[int(j)]], float(sims[int(j)]))
                for j in order]

    # ----- persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-safe dump of the whole scratchpad."""
        return {
            "n_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def write_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, blob: dict,
                  embedder: LexicalEmbedder | None = None) -> "VectorSharedMemory":
        mem = cls(embedder=embedder)
        for raw in blob.get("entries", []):
            mem.put(scope=raw["scope"], question=raw["question"],
                    answer=raw["answer"], **(raw.get("extras") or {}))
        return mem
