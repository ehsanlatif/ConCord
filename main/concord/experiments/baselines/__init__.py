"""Baselines B0..B3 (plan §8.2) for the experiment harness.

Each module exposes a `solve(problem, *, cfg, llm, domain, tag) -> Result`
function with the same signature as the main orchestrator. The harness
treats them interchangeably.
"""

from . import b0_cot, b1_self_consistency, b2_concord_v1, b3_mcts_no_sd_no_gate

BASELINES = {
    "b0_cot": b0_cot.solve,
    "b1_self_consistency": b1_self_consistency.solve,
    "b2_concord_v1": b2_concord_v1.solve,
    "b3_mcts_no_sd_no_gate": b3_mcts_no_sd_no_gate.solve,
}

__all__ = ["BASELINES", "b0_cot", "b1_self_consistency", "b2_concord_v1",
           "b3_mcts_no_sd_no_gate"]
