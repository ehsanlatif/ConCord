"""Contextual transposition table (§5.7).

A memoized expansion (the classes + their U_s scores) is reusable only when
the *constraint-relevant context* matches — not just the subproblem text.
A solution coherent in one context can violate constraints in another, so
the key includes both:

  - the subproblem signature, and
  - bindings of upstream resolutions that affect this subproblem's
    checkable constraints.

For M3 we ship a string-keyed dict with a `key()` helper. M4's coherence
gate identifies which bindings are constraint-relevant; until then, we
include all known bindings — over-keying is safe (more cache misses), under-
keying is not (wrong-context hits).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TTKey:
    subproblem: str
    binding_sig: str    # canonical-JSON hash of the binding subset

    @classmethod
    def make(cls, subproblem: str, bindings: dict[str, Any]) -> "TTKey":
        # canonical JSON for stable hashing
        blob = json.dumps(bindings, sort_keys=True, default=str).encode()
        sig = hashlib.blake2b(blob, digest_size=8).hexdigest()
        return cls(subproblem=subproblem, binding_sig=sig)


class TranspositionTable:
    """Stores expansion results (the per-class summaries) keyed by TTKey.

    `N`/`Q` are intentionally NOT cached — those stay per-path (§5.7
    "DAG-MCTS node sharing is allowed; N/Q stay per-path"). Only the
    *expensive* part — the K samples and their cluster/score — is reused.
    """

    def __init__(self):
        self._store: dict[TTKey, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: TTKey) -> Any | None:
        v = self._store.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return v

    def put(self, key: TTKey, value: Any) -> None:
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)
