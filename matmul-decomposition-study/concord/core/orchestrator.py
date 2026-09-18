"""Concord orchestrator — Phase 1 (characterization) + Phase 2 (MCTS).

This module wires every component built in M0-M4 together against a real
LLMClient. The flow follows the spec's two phases:

  Phase 1:  extract G -> condense to G' -> compute d_init / d_max
  Phase 2:  MCTS loop with the real ExpansionPolicy + incremental gate +
            marginal-gain stop, then solution selection (coherent path
            preferred; soft-relax fallback if none).

The function below is the only public entry point used by experiment runners.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import networkx as nx

from . import classifier as _classifier
from .config import Config
from .confidence import (
    Embedder,
    EntailmentKernel,
    LLMJudgeVerifier,
    LexicalEmbedder,
    LexicalKernel,
    NullVerifier,
    Verifier,
    default_verifier_for,
)
from .decomposer import LLMDecomposer
from .expansion import RewriteExpansionPolicy, extract_answer
from .llm.client import LLMClient
from .llm.factory import RoleClients
from .search import (
    Node,
    SearchResult,
    depth_cap,
    search,
    should_stop,
    soft_relax,
)
from .search.depth import StopState
from .search.mcts import reset_node_ids
from .structure import (
    Constraint,
    ExplicitDecomposer,
    condense,
    evaluate_gate,
    extract_graph,
    integer_answer_constraint,
    is_atomic,
    predecessor_consistency_constraint,
)
from .telemetry import BudgetExceeded, Telemetry
from .tracer import Tracer
from .types import Result, SolutionState


# ---------------------------------------------------------------------------
# Terminal predicate
# ---------------------------------------------------------------------------

class _Terminal:
    def __init__(self, Gp: nx.DiGraph, d_max: int):
        self.Gp = Gp
        self.d_max = d_max
    def is_terminal(self, state: SolutionState, depth: int) -> bool:
        if depth >= self.d_max:
            return True
        return is_atomic(state, self.Gp)


# ---------------------------------------------------------------------------
# Constraint sets per domain
# ---------------------------------------------------------------------------

def _constraints_for(domain: str | None, Gp: nx.DiGraph) -> list[Constraint]:
    base = [predecessor_consistency_constraint(Gp)]
    if domain == "math":
        # eval-set math nodes resolve to integers
        base.append(integer_answer_constraint())
    return base


# ---------------------------------------------------------------------------
# solve()
# ---------------------------------------------------------------------------

def solve(problem: str, *, cfg: Config,
          llm: LLMClient | None = None,
          clients: RoleClients | None = None,
          domain: str | None = None,
          embedder: Embedder | None = None,
          kernel: EntailmentKernel | None = None,
          verifier: Verifier | None = None,
          tag: str | None = None) -> Result:
    """Run Concord on `problem` and return a Result.

    Two entry shapes are supported:

      - `clients=` (preferred when `cfg.models.*` is set): the orchestrator
        consumes per-role LLMClients from a `RoleClients` container —
        execution / verification / classification / decomposition can each
        be a different model. Costs are aggregated across roles.
      - `llm=` (legacy/simple): one client used for every role; equivalent
        to building a `RoleClients` where every role points at the same
        underlying client.

    `domain` is used to pick a default verifier when one is not passed in.
    Missing `domain` triggers the LLM classifier (one cheap call to the
    `classification` role's model) when `clients` is supplied.
    """
    # ---- normalize the LLM access surface ----
    if clients is None:
        if llm is None:
            raise TypeError("solve() needs either `clients=...` or `llm=...`")
        # Wrap the single LLMClient as a RoleClients so the rest of the
        # function can use one code path. All roles share the same client.
        # Include the pipeline-only roles too so cfg.solver="pipeline"
        # works with the single-LLM convenience entrypoint.
        from .llm.factory import RoleClients as _RC
        clients = _RC(
            execution=llm, decomposition=llm,
            classification=llm, verification=llm,
            splitter=llm, combiner=llm,
            synthesizer=llm, synth_verifier=llm,
            specs={
                "execution": cfg.llm, "decomposition": cfg.llm,
                "classification": cfg.llm, "verification": cfg.llm,
                "splitter": cfg.llm, "combiner": cfg.llm,
                "synthesizer": cfg.llm, "synth_verifier": cfg.llm,
            },
        )
    exec_llm = clients.execution

    # Snapshot the cumulative LLM-call count at solve() START so the
    # per-block budget check measures DELTA, not raw cumulative. In
    # solve_multi every block shares one RoleClients; without this
    # baseline, the second block onwards would inherit the first block's
    # cost and trip cfg.mcts.N on its first rollout.
    _baseline_calls = clients.total_cost().calls

    telemetry = Telemetry.from_config(cfg, tag=tag,
                                       baseline_calls=_baseline_calls)
    telemetry.event("problem", {"domain": domain, "text_head": problem[:600]})
    telemetry.event("role_models", {
        role: {"provider": spec.provider, "model": spec.model,
               "temperature": spec.temperature}
        for role, spec in clients.specs.items()
    })

    # Fresh node-id counter per solve() so tree.json ids are deterministic
    # and don't bleed across runs (otherwise the viewer's "n0" would refer
    # to whatever question happened to land first).
    reset_node_ids()

    # Tracer: writes <run-id>_rollouts.jsonl line-by-line and a final
    # <run-id>_tree.json snapshot for the viewer.
    tracer = Tracer(telemetry.path, meta={
        "run_id": telemetry.run_id,
        "domain": domain,
        "tag": tag,
        "role_models": {
            role: f"{spec.provider}/{spec.model}"
            for role, spec in clients.specs.items()
        },
        "config": {
            "N": cfg.mcts.N, "K_bb": cfg.sampling.K_blackbox,
            "K_wb": cfg.sampling.K_whitebox,
            "c_puct": cfg.mcts.c_puct, "C": cfg.mcts.C, "beta": cfg.mcts.beta,
            "n_min": cfg.mcts.n_min, "alpha": cfg.confidence.alpha,
            "coherence_mode": cfg.coherence.mode,
            "ablation": cfg.ablation.model_dump(),
        },
        "problem_excerpt": problem[:600],
    })

    # ---- Phase 1a: classify domain if not provided ----
    if domain is None and clients.classification is not exec_llm:
        # only call the classifier when it's distinct (avoid wasting the
        # execution model's calls on a category lookup)
        domain = _classifier.classify(
            problem,
            llm=clients.classification,
            tracer=tracer,
            model_name=clients.specs["classification"].model,
        )
        telemetry.event("classified", {"domain": domain})
    elif domain is None:
        # Cheap heuristic: skip classifier when only one shared client is
        # available — same model can just be invoked at expansion time.
        domain = None

    # ---- Phase 1b: structure & depth ----
    G = extract_graph(problem)
    Gp = condense(G)
    decomposer = ExplicitDecomposer()
    K = cfg.sampling.K_blackbox if not exec_llm.supports_logprobs else cfg.sampling.K_whitebox
    d_max = depth_cap(Gp,
                      budget_calls=cfg.mcts.N,
                      per_expansion_cost=max(1, K),
                      slack=cfg.depth.slack,
                      fixed=None if cfg.ablation.depth == "adaptive"
                            else (cfg.depth.budget_cap or 4))
    telemetry.event("phase1", {
        "graph_nodes": list(G.nodes),
        "condensed_nodes": list(Gp.nodes),
        "critical_path": d_max - cfg.depth.slack,
        "d_max": d_max,
        "K": K,
    })

    # ---- Components ----
    embedder = embedder or LexicalEmbedder()
    kernel = kernel or LexicalKernel()

    # Verifier: explicit override wins; else, if a dedicated verification
    # client distinct from execution is configured, use the LLM judge;
    # else fall back to the rule-based domain verifier.
    if verifier is None:
        if clients.verification is not exec_llm:
            verifier = LLMJudgeVerifier(
                clients.verification,
                tracer=tracer,
                model_name=clients.specs["verification"].model,
            )
            telemetry.event("verifier", {"kind": "llm_judge",
                                          "model": clients.specs["verification"].model})
        else:
            verifier = default_verifier_for(domain)
            telemetry.event("verifier", {"kind": "rule_based",
                                          "domain": domain})

    # LLM-driven sub-question synthesizer (the "PROPOSER" / decomposer role).
    # We wire it only when a DISTINCT decomposition model is configured —
    # otherwise the same model is making the cleanup call AND solving the
    # question, which both wastes calls and lets the cleanup model leak
    # answer-shaped content into the question. In single-model mode we fall
    # back to raw substitution (which is fine for templates with no parent
    # references, and for backward-compat with the legacy mock tests).
    sub_decomposer = None
    if clients.decomposition is not exec_llm:
        sub_decomposer = LLMDecomposer(
            llm=clients.decomposition,
            temperature=clients.specs["decomposition"].temperature,
            tracer=tracer,
            model_name=clients.specs["decomposition"].model,
        )
        telemetry.event("decomposer", {
            "kind": "llm",
            "model": clients.specs["decomposition"].model,
            "temperature": sub_decomposer.temperature,
        })
    else:
        telemetry.event("decomposer", {
            "kind": "raw_substitution",
            "reason": "decomposition role shares the execution client",
        })

    # ---- Solver dispatch ----
    # `cfg.solver` selects the MCTS expansion policy. "mcts" is the legacy
    # RewriteExpansionPolicy (single rewrite + K samples). "pipeline" is
    # the new Split → Solve → Combine → Verify pipeline.
    if getattr(cfg, "solver", "mcts") == "pipeline":
        from .pipeline import (
            LLMBlockVerifier,
            LLMCombiner,
            LLMSplitter,
            VectorSharedMemory,
        )
        from .pipeline.expansion import PipelineExpansionPolicy

        # Shared memory may be supplied by solve_multi (so blocks see
        # each other's answers) or constructed fresh for a stand-alone
        # solve. The orchestrator looks for a `shared_memory` attribute
        # on cfg.pipeline at runtime — solve_multi attaches it there.
        shared_mem = getattr(cfg.pipeline, "_runtime_shared_memory", None)
        if shared_mem is None:
            shared_mem = VectorSharedMemory(embedder=embedder)

        splitter = LLMSplitter(
            llm=clients.splitter, cfg=cfg,
            temperature=clients.specs["splitter"].temperature,
            tracer=tracer,
            model_name=clients.specs["splitter"].model,
        )
        combiner = LLMCombiner(
            llm=clients.combiner, cfg=cfg,
            temperature=clients.specs["combiner"].temperature,
            tracer=tracer,
            model_name=clients.specs["combiner"].model,
        )
        block_verifier = LLMBlockVerifier(
            llm=clients.verification, cfg=cfg,
            temperature=clients.specs["verification"].temperature,
            tracer=tracer,
            model_name=clients.specs["verification"].model,
        )
        telemetry.event("solver", {
            "kind": "pipeline",
            "splitter": clients.specs["splitter"].model,
            "combiner": clients.specs["combiner"].model,
            "block_verifier": clients.specs["verification"].model,
            "max_split_depth": cfg.pipeline.max_split_depth,
            "thresholds": {
                "accept": cfg.pipeline.verifier_accept,
                "retry": cfg.pipeline.verifier_retry,
                "backtrack": cfg.pipeline.verifier_backtrack,
            },
        })
        policy = PipelineExpansionPolicy(
            llm=exec_llm, Gp=Gp, decomposer=decomposer,
            embedder=embedder, kernel=kernel, verifier=verifier,
            cfg=cfg, rng=random.Random(cfg.seed), tracer=tracer,
            original_problem=problem,
            splitter=splitter, combiner=combiner,
            block_verifier=block_verifier,
            shared_memory=shared_mem,
        )
    else:
        telemetry.event("solver", {"kind": "mcts"})
        policy = RewriteExpansionPolicy(
            llm=exec_llm, Gp=Gp, decomposer=decomposer,
            embedder=embedder, kernel=kernel, verifier=verifier,
            cfg=cfg, rng=random.Random(cfg.seed), tracer=tracer,
            sub_decomposer=sub_decomposer,
            original_problem=problem,
        )

    # ---- Gate + stop ----
    constraints = _constraints_for(domain, Gp)

    def gate_fn(state: SolutionState, depth: int) -> float:
        if cfg.coherence.mode == "off":
            return 0.0
        if cfg.coherence.mode == "terminal_only" and not is_atomic(state, Gp):
            return 0.0
        gr = evaluate_gate(state, depth, constraints)
        # Hand the gate detail to the tracer so the next on_rollout writes it.
        tracer.stash_gate(sigma=gr.sigma, violated=gr.violated,
                           all_checked=gr.all_checked, depth=depth)
        return gr.sigma

    stop_state = StopState(best_at_depth={})
    # Soft budget: when LLM cost exceeds N we DON'T crash — we stop the
    # search cleanly so solution selection can still pick the best partial.
    # `budget_truncated` propagates into the Result so callers know the
    # search did not run to completion.
    budget_state = {"truncated": False}

    def on_rollout(i: int, leaf: Node, record=None) -> None:
        stop_state.update(leaf.depth, leaf.U_s)
        cost = clients.total_cost()
        telemetry.rollout(cost)
        try:
            telemetry.check_budget(cost)
        except BudgetExceeded as e:
            # Soft-stop: log once, mark, let stop_fn end the loop next tick.
            if not budget_state["truncated"]:
                telemetry.event("budget_exceeded", {
                    "calls": cost.calls,
                    "limit": cfg.mcts.N,
                    "at_rollout": i,
                })
            budget_state["truncated"] = True
        if record is not None:
            tracer.on_rollout(i, leaf, record)

    def stop_fn(i: int, leaf: Node) -> bool:
        # Soft budget always wins — once exceeded, stop ASAP.
        if budget_state["truncated"]:
            return True
        if cfg.ablation.depth != "adaptive":
            return False
        if max(stop_state.best_at_depth, default=0) < d_max:
            return False
        if i + 1 < max(8, cfg.mcts.N // 4):
            return False
        return should_stop(stop_state, current_depth=leaf.depth,
                            lam=cfg.depth.lam,
                            cost_next=max(1.0, float(K)))

    # ---- Phase 2: MCTS ----
    reached: list[Node] = []
    sr: SearchResult = search(
        SolutionState(), cfg=cfg, policy=policy,
        terminal_check=_Terminal(Gp, d_max),
        gate_fn=gate_fn, stop_fn=stop_fn, K=K,
        reached_terminals=reached,
        on_rollout=on_rollout,
    )

    # ---- Solution selection ----
    coherent = [n for n in reached if not n.gated_fail]
    flagged: str | None = None
    is_coherent: bool = False
    all_rejected = False
    if coherent:
        chosen = max(coherent, key=lambda n: (n.Q, -n.depth))
        sigma = 0.0
        is_coherent = True
    else:
        fr = soft_relax(reached)
        if fr is None:
            best_child = max(
                (sr.root.children.values() if sr.root.children else []),
                key=lambda n: n.Q, default=None,
            )
            chosen = best_child or sr.root
            sigma = chosen.sigma
            flagged = "no terminal reached; returning best partial"
        else:
            chosen = fr.node
            sigma = fr.sigma
            flagged = fr.flag
            all_rejected = fr.all_rejected

    if all_rejected:
        # Every terminal was gated-fail; the "winner" is just whatever
        # node beat the tiebreak. Returning its last resolved answer
        # would be misleading — that text is exactly what the gate
        # rejected. Surface an empty answer so downstream graders /
        # aggregators don't treat the diagnostic node as a real
        # prediction. The chosen node + sigma are still surfaced via
        # the summary for inspection.
        answer = ""
    else:
        answer = chosen.state.resolved[-1][1] if chosen.state.resolved else ""
        answer = extract_answer(answer) or answer

    # Surface budget truncation in `flagged` so the per-question status line
    # makes it obvious. Compose with any existing flag (e.g. soft-relax).
    if budget_state["truncated"]:
        bn = "budget_truncated"
        flagged = (f"{bn}; {flagged}") if flagged else bn

    total_cost = clients.total_cost()
    cost_dict = total_cost.to_dict()
    cost_dict["per_role"] = clients.per_role_cost()
    # Pipeline-policy provenance: when the new solver ran, attach its
    # per-block split tree + combiner attempts + verifier verdicts so
    # solve_multi can assemble the combine_tree.json.
    if hasattr(policy, "build_combine_tree_payload"):
        try:
            cost_dict["pipeline_provenance"] = policy.build_combine_tree_payload()
        except Exception:                                           # noqa: BLE001
            pass
    result = Result(
        answer=answer,
        coherent=is_coherent,        # purely on σ=0 + a coherent terminal,
                                      # NOT diluted by budget-truncation
        sigma=sigma,
        rollouts=sr.rollouts,
        cost=cost_dict,
        flagged=flagged,
        trace_path=str(telemetry.path),
    )
    summary = {
        "answer": answer,
        "coherent": result.coherent,
        "sigma": sigma,
        "rollouts": sr.rollouts,
        "flagged": flagged,
        "per_role_cost": clients.per_role_cost(),
        "best_terminal_id": sr.best_terminal.node_id if sr.best_terminal else None,
        "chosen_node_id": chosen.node_id if hasattr(chosen, "node_id") else None,
        "n_reached_terminals": len(reached),
        "d_max": d_max,
        "budget_truncated": budget_state["truncated"],
    }

    # Write the search tree + close the tracer's rollouts file.
    try:
        tracer.write_tree(sr.root, summary=summary)
    finally:
        tracer.close()

    # Expose viz paths on the result so the runner can announce them.
    result.cost["tree_path"] = str(tracer.tree_path)
    result.cost["rollouts_path"] = str(tracer.rollouts_path)

    telemetry.finish(summary, total_cost)
    return result


def solve_with_config(problem: str, *, cfg: Config,
                       domain: str | None = None,
                       tag: str | None = None) -> Result:
    """Convenience wrapper that instantiates per-role clients from `cfg.models`.

    This is the entry point you want when your YAML supplies a `models:`
    block. Roles that aren't pinned in the YAML fall back to the top-level
    `llm:` block.
    """
    clients = RoleClients.from_config(cfg)
    return solve(problem, cfg=cfg, clients=clients, domain=domain, tag=tag)
