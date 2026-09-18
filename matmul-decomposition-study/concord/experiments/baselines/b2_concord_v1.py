"""B2: Concord v1 — cosine-to-centroid + hard tau threshold + chronological
backtracking.

Plan §8.2 baseline 2 — the thing Concord claims to beat. We implement the v1
algorithm directly here (it's not a full re-port of an old codebase, just
its specification: cosine-centroid confidence, hard threshold τ, walk
subproblems in chronological order, backtrack if the current node's
confidence is below threshold).

Loop:
  for sub in topological_order(G'):
      sample K answers; compute centroid embedding; pick the answer
      closest to centroid; if its cosine-to-centroid >= τ, accept;
      else BACKTRACK by one and re-sample the previous node.

Bounded by `cfg.mcts.N` LLM calls (same budget as Concord).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import Config
from core.confidence import LexicalEmbedder
from core.expansion import build_subproblem_prompt, extract_answer
from core.llm.client import LLMClient
from core.structure import (
    ExplicitDecomposer,
    condense,
    extract_graph,
    is_atomic,
    topo_order,
)
from core.telemetry import Telemetry
from core.types import Result, SolutionState


def _cosine_to_centroid(emb: np.ndarray) -> tuple[int, float]:
    """Return (best_index, max_cosine_to_centroid)."""
    if emb.shape[0] == 0:
        return 0, 0.0
    centroid = emb.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        return 0, 0.0
    centroid /= norm
    sims = emb @ centroid
    i = int(np.argmax(sims))
    return i, float(sims[i])


def solve(problem: str, *, cfg: Config, llm: LLMClient, domain: str | None = None,
          tag: str | None = None) -> Result:
    telemetry = Telemetry.from_config(cfg, tag=tag)
    telemetry.event("problem", {"baseline": "b2_concord_v1"})

    G = extract_graph(problem)
    Gp = condense(G)
    decomposer = ExplicitDecomposer()
    embedder = LexicalEmbedder()
    K = cfg.sampling.K_blackbox if not llm.supports_logprobs else cfg.sampling.K_whitebox
    tau = cfg.confidence.tau_pre   # using tau_pre as the v1 hard threshold
    state = SolutionState()

    max_calls = cfg.mcts.N
    backtracks_remaining = 3   # bounded v1-style chronological backtracking

    visited_attempts: dict[str, int] = defaultdict(int)
    last_resolved_chain: list[tuple[str, str, float]] = []

    while not is_atomic(state, Gp) and llm.cost().calls < max_calls:
        sp = decomposer.next_subproblem(state, Gp)
        if sp == "<ATOMIC>" or sp is None:
            break

        prompt = build_subproblem_prompt(sp.text, {**state.bindings, **sp.bindings})
        # adjust K so we don't blow the budget mid-loop
        remaining = max_calls - llm.cost().calls
        k_use = min(K, max(1, remaining))
        gens = llm.generate(prompt, temperature=cfg.llm.temperature, n=k_use)
        responses = [g.text for g in gens]
        emb = embedder.encode(responses)
        i, conf = _cosine_to_centroid(emb)
        chosen_text = extract_answer(responses[i]) or responses[i]

        telemetry.event("v1_step", {
            "subproblem": sp.node_id,
            "confidence": conf,
            "tau": tau,
            "chosen_text": chosen_text[:80],
            "calls_used": llm.cost().calls,
        })

        if conf >= tau:
            state = state.extend(sp.node_id, chosen_text,
                                 **{sp.node_id: chosen_text})
            last_resolved_chain.append((sp.node_id, chosen_text, conf))
        else:
            # below threshold -> chronological backtrack one step
            visited_attempts[sp.node_id] += 1
            if last_resolved_chain and backtracks_remaining > 0:
                bt = last_resolved_chain.pop()
                # unwind one resolution from state
                state = SolutionState(
                    resolved=[(s, a) for (s, a) in state.resolved
                              if s != bt[0]],
                    bindings={k: v for k, v in state.bindings.items()
                              if k != bt[0]},
                )
                backtracks_remaining -= 1
                telemetry.event("v1_backtrack", {
                    "from": sp.node_id, "to": bt[0],
                    "backtracks_remaining": backtracks_remaining,
                })
            else:
                # no backtracks left — accept the under-threshold answer flagged
                state = state.extend(sp.node_id, chosen_text,
                                     **{sp.node_id: chosen_text})
                last_resolved_chain.append((sp.node_id, chosen_text, conf))
                if visited_attempts[sp.node_id] >= 2:
                    break

    answer = ""
    if state.resolved:
        answer = state.resolved[-1][1]
    answer = extract_answer(answer) or answer

    cost = llm.cost()
    coherent = is_atomic(state, Gp) and bool(last_resolved_chain) and \
        all(c >= tau for _, _, c in last_resolved_chain)
    res = Result(
        answer=answer,
        coherent=coherent,
        sigma=0.0 if coherent else 1.0,
        rollouts=cost.calls,
        cost=cost.to_dict(),
        flagged="baseline_b2_concord_v1" if not coherent else None,
        trace_path=str(telemetry.path),
    )
    telemetry.finish({"answer": answer, "coherent": coherent}, cost)
    return res
