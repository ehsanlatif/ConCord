"""B3: vanilla MCTS without SD and without the coherence gate.

Plan §8.2 baseline 3. Same engine as Concord, but two ablations stacked:
  - confidence ablation: cosine-to-centroid instead of Semantic Density
  - gate off

This isolates "how much of Concord's lift comes from SD + the gate vs. MCTS alone".
Implemented as a thin wrapper that flips two config flags and delegates to
the main orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import Config
from core.llm.client import LLMClient
from core.orchestrator import solve as v2_solve
from core.types import Result


def solve(problem: str, *, cfg: Config, llm: LLMClient, domain: str | None = None,
          tag: str | None = None) -> Result:
    # Mutate a copy of the config rather than the caller's instance.
    cfg = cfg.model_copy(deep=True)
    cfg.ablation.confidence = "cosine_centroid"
    cfg.coherence.mode = "off"
    cfg.ablation.gate = "off"
    res = v2_solve(problem, cfg=cfg, llm=llm, domain=domain,
                   tag=(tag or "") + "_b3")
    # Re-flag for downstream metric aggregation.
    res.flagged = "baseline_b3_mcts_no_sd_no_gate"
    return res
