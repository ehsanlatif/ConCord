"""Pipeline solver — Split → Solve → Combine → Verify.

Replaces the legacy `RewriteExpansionPolicy`'s "decompose template into one
clean question + sample K times" expansion with a four-phase pipeline:

    A) LLMSplitter         — recursive decomposition into atomic units
    B) (executor reused)   — K samples per atomic unit, clustered
    C) LLMCombiner         — atomic answers → candidate block answer
    D) LLMBlockVerifier    — score + issues + corrected? + backtrack signal

A global `LLMSynthesizer` (run by solve_multi after all blocks finish)
emits the final `solution = [v1, ..., vN]` list in the format the LongCoT
grader expects.

`VectorSharedMemory` is the cross-block scratchpad — text + lexical-vector
retrieval so an atomic unit can fetch related prior work even when no
explicit `node_K` reference exists.
"""

from __future__ import annotations

from .block_verifier import LLMBlockVerifier, VerifierVerdict
from .combiner import LLMCombiner
from .shared_memory import MemoryEntry, VectorSharedMemory
from .splitter import AtomicUnit, CallBudget, LLMSplitter, SplitTree
from .synthesizer import LLMSynthesizer, SynthResult

__all__ = [
    "AtomicUnit",
    "CallBudget",
    "LLMBlockVerifier",
    "LLMCombiner",
    "LLMSplitter",
    "LLMSynthesizer",
    "MemoryEntry",
    "SplitTree",
    "SynthResult",
    "VectorSharedMemory",
    "VerifierVerdict",
]
