"""Multi-block solver with shared answer memory.

When a problem text contains multiple independent (or partially-dependent)
`Problem node_K:` blocks, feeding the whole text to `solve()` is wrong:
`extract_graph` parses every block as a node in one giant DAG, and the
MCTS then tries to solve 30+ unrelated questions as a single sequence.
Output token caps blow up, the gate rejects everything, and soft-relax
returns garbage.

`solve_multi()` is the right entry point for that shape. It:

  1. Splits the input on `Problem node_K:` headers.
  2. Builds the dependency graph of cross-references
     (`answer from problem node_M`) between blocks.
  3. Walks the graph in topological order, so each block is solved AFTER
     every block it depends on.
  4. Maintains a `shared_memory: dict[node_id, answer]` of solved-so-far
     answers, and threads the relevant prior answers into each block as
     both (a) an inline substitution of `answer from problem node_M`
     references and (b) a short "Context:" preamble. The executor LLM
     sees only the consolidated, self-contained block plus the facts it
     needs.
  5. Aggregates per-block results into a single `Result`. The
     "consolidated answer" is the answer of the topological sink
     (the block everything else feeds into); when there are multiple
     sinks, the answers are returned as a JSON map.

When the input contains 0 or 1 blocks, `solve_multi` is identical to
`solve` — so callers can use it unconditionally.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import networkx as nx

from .config import Config
from .llm.client import LLMClient
from .llm.factory import RoleClients
from .orchestrator import solve as _single_solve
from .pipeline.shared_memory import VectorSharedMemory
from .pipeline.synthesizer import (
    LLMSynthesizer,
    SynthResult,
    heuristic_expected_length,
)
from .structure.graph import NODE_REF_RE, extract_graph, parse_explicit_nodes
from .structure.llm_graph import LLMGraphResolver
from .types import CostTally, Result


# ---------------------------------------------------------------------------
# Block ordering
# ---------------------------------------------------------------------------

def _build_block_graph(blocks) -> nx.DiGraph:
    """One node per parsed block; edges dep_id -> dependent_id."""
    G: nx.DiGraph = nx.DiGraph()
    for b in blocks:
        G.add_node(b.node_id, text=b.text, refs=b.refs)
    for b in blocks:
        for r in b.refs:
            if r in G.nodes and r != b.node_id:
                G.add_edge(r, b.node_id)
    return G


def _natural_key(node_id: str) -> tuple:
    """Sort `node_5` before `node_10` (numeric tiebreak), then fall back
    to the raw string for non-numeric ids (chess / chem / logic blocks)."""
    import re as _re
    m = _re.search(r"\d+", str(node_id))
    return (int(m.group()) if m else 0, str(node_id))


def _solve_order(G: nx.DiGraph) -> list[str]:
    """Layer-stratified topological order — *independents first*.

    Standard `nx.lexicographical_topological_sort` interleaves chains:
    after node_0 is taken, its successor node_1 becomes a source and is
    picked before the still-untouched sources node_4, node_5, node_10
    (because "1" < "4" lexicographically). That hurts the pipeline solver
    in two concrete ways:

      1. Shared memory is underpopulated when a dependent block runs —
         its sibling-graph independents haven't been solved yet, so the
         splitter can't pull related context via vector retrieval.
      2. Cost of an early budget trip leaks into chains we haven't even
         started yet.

    The fix is to drain layer 0 (all in-degree-0 sources) BEFORE layer 1
    (their direct successors), and so on. Within a layer we use natural
    numeric ordering (`node_5` before `node_10`) so the trace is
    human-scannable.

    On cycles, falls back to source order — the per-block SCC condensation
    inside `solve()` handles intra-block cycles itself.
    """
    if G.number_of_nodes() == 0:
        return []
    # Work on a copy because we mutate by peeling sources off.
    G2 = G.copy()
    order: list[str] = []
    safety_cap = G.number_of_nodes() + 1
    while G2.nodes and safety_cap > 0:
        safety_cap -= 1
        sources = sorted(
            (n for n in G2.nodes if G2.in_degree(n) == 0),
            key=_natural_key,
        )
        if not sources:
            # Cycle — append the rest in stable natural order and bail.
            order.extend(sorted(G2.nodes, key=_natural_key))
            break
        order.extend(sources)
        G2.remove_nodes_from(sources)
    return order


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

def _substitute_refs(text: str, memory: dict[str, str]) -> str:
    """Annotate every `... from problem node_M` reference with the resolved
    answer in a parenthetical, preserving the surrounding phrasing.

    The dataset uses richly-phrased cross-block references like
    "the denominator of the reduced form of the fraction from problem
    node_5 and add 14". The OLD substitution would have rewritten that
    into "the denominator of the reduced form of the fraction the answer
    1/3 (from problem node_5) and add 14" — grammatically broken. The
    new substitution preserves the phrasing and just inlines the answer
    as a parenthetical hint after the reference:
        "...the fraction from problem node_5 (whose committed answer was:
        1/3) and add 14"

    Unknown refs are left alone so the splitter LLM can still parse the
    original phrase.
    """
    def repl(m):
        nid = f"node_{m.group(1)}"
        ans = memory.get(nid)
        if ans is None or ans == "":
            return m.group(0)
        # Keep the original "from problem node_K" token, append an
        # answer-annotation immediately after it.
        return f"{m.group(0)} (whose committed answer was: {ans!s})"
    return NODE_REF_RE.sub(repl, text)


def _context_preamble(refs: list[str], memory: dict[str, str]) -> str:
    """Short `Context:` block listing the prior answers this block depends on.

    Only includes refs whose answer is non-empty in memory.
    """
    lines = []
    for r in refs:
        ans = memory.get(r)
        if ans:
            lines.append(f"- {r}: {ans}")
    if not lines:
        return ""
    return "Context (answers from earlier subproblems you should use):\n" + \
           "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Cost aggregation
# ---------------------------------------------------------------------------

def _add_cost(agg: CostTally, sub_cost: dict) -> None:
    agg.add(
        calls=int(sub_cost.get("calls", 0)),
        input_tokens=int(sub_cost.get("input_tokens", 0)),
        output_tokens=int(sub_cost.get("output_tokens", 0)),
        usd=float(sub_cost.get("usd", 0.0)),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    """Per-block diagnostics returned alongside the consolidated answer."""

    node_id: str
    answer: str
    coherent: bool
    sigma: float
    flagged: str | None
    rollouts: int
    cost: dict
    trace_path: str | None


ProgressCb = Callable[[dict], None]


def _default_progress(event: dict) -> None:
    """Default progress sink: pretty-print to stdout, flush eagerly so the
    operator sees per-block output in real time (important for hour-long
    runs where the only feedback is per-block).
    """
    kind = event.get("kind")
    if kind == "block_start":
        nid = event["node_id"]
        i = event["index"]
        n = event["total"]
        refs = event.get("refs") or []
        ref_note = f" (depends on: {', '.join(refs)})" if refs else ""
        q = event.get("subproblem_head", "")
        print(f"\n[block {i}/{n}] solving {nid}{ref_note}", flush=True)
        if q:
            print(f"  Q: {q}", flush=True)
    elif kind == "block_end":
        i = event["index"]
        n = event["total"]
        nid = event["node_id"]
        ans = event["answer"]
        coh = event["coherent"]
        sigma = event["sigma"]
        rollouts = event["rollouts"]
        usd = event["cost_usd"]
        elapsed = event["elapsed_s"]
        flagged = event.get("flagged")
        running_usd = event["running_usd"]
        running_calls = event["running_calls"]
        tok_in = event.get("input_tokens", 0)
        tok_out = event.get("output_tokens", 0)
        tok_total = event.get("total_tokens", tok_in + tok_out)
        running_tokens = event.get("running_tokens", 0)
        status = "OK " if coh else "INC"
        flag_note = f"  FLAG={flagged!r}" if flagged else ""
        # Truncate long answers so the log stays scannable.
        ans_short = (ans[:80] + "…") if len(ans) > 80 else ans
        print(
            f"  [{i}/{n}] {nid} -> {status} ans={ans_short!r} "
            f"σ={sigma:.2f} rollouts={rollouts} ${usd:.3f} "
            f"tok={tok_in}+{tok_out}={tok_total} {elapsed:.1f}s"
            f"  | running: ${running_usd:.2f}, {running_calls} calls, "
            f"{running_tokens} tok"
            f"{flag_note}",
            flush=True,
        )
        if event.get("trace_path"):
            print(f"        trace: {event['trace_path']}", flush=True)
    elif kind == "multi_start":
        rd = event.get("run_dir")
        gs = event.get("graph_source")
        print(
            f"\n[solve_multi] {event['n_blocks']} blocks parsed; "
            f"order: {event['order'][:8]}"
            + ("..." if event["n_blocks"] > 8 else ""),
            flush=True,
        )
        if gs:
            print(f"[solve_multi] dependency graph source: {gs}", flush=True)
        if rd:
            print(f"[solve_multi] run dir: {rd}", flush=True)
    elif kind == "multi_end":
        rd = event.get("run_dir")
        tt = event.get("total_tokens", 0)
        ti = event.get("total_input_tokens", 0)
        to = event.get("total_output_tokens", 0)
        print(
            f"\n[solve_multi] done. {event['coherent_count']}/{event['n_blocks']} "
            f"coherent. total ${event['total_usd']:.3f}, "
            f"{event['total_calls']} calls, "
            f"{tt} tokens (in {ti} / out {to}), "
            f"{event['elapsed_s']:.1f}s",
            flush=True,
        )
        fl = event.get("final_line")
        if fl:
            print(f"[solve_multi] final answer (graded line): {fl}", flush=True)
        if rd:
            print(f"[solve_multi] all artifacts in: {rd}", flush=True)
            print(f"[solve_multi] manifest:     {rd}/manifest.json", flush=True)
            print(f"[solve_multi] combine_tree: {rd}/combine_tree.json", flush=True)
            print(f"[solve_multi] shared mem:   {rd}/shared_memory.json", flush=True)


def solve_multi(problem_text: str, *, cfg: Config,
                llm: LLMClient | None = None,
                clients: RoleClients | None = None,
                domain: str | None = None,
                tag: str | None = None,
                progress: ProgressCb | None = _default_progress) -> Result:
    """Multi-block solve with shared answer memory.

    Identical to `solve()` when the input has zero or one parsed blocks.
    Otherwise splits the input and runs `solve()` once per block in
    topological order, sharing solved answers across blocks.

    `progress` is called with structured events as the run progresses:
      - {"kind": "multi_start", "n_blocks", "order"}
      - {"kind": "block_start", "index", "total", "node_id", "refs",
         "subproblem_head"}
      - {"kind": "block_end", "index", "total", "node_id", "answer",
         "coherent", "sigma", "rollouts", "cost_usd", "elapsed_s",
         "running_usd", "running_calls", "flagged", "trace_path"}
      - {"kind": "multi_end", "n_blocks", "coherent_count",
         "total_usd", "total_calls", "elapsed_s"}
    Pass `progress=None` to silence stdout, or pass your own callable to
    capture events programmatically (e.g. dump to JSONL, push to a UI).

    The returned `Result.answer` is:
      - the topological sink's answer when there is exactly one sink
        (the natural "final answer" of a dependency chain), or
      - a JSON map of `{node_id: answer}` when there are multiple sinks
        (independent siblings; no single consolidated answer exists).

    The returned `Result.cost` is the SUM of per-block costs, with
    per-block details on `result.cost["per_block"]`.
    """
    blocks = parse_explicit_nodes(problem_text)
    if len(blocks) <= 1:
        # Nothing to split — delegate to the single-problem path so we
        # don't pay the wrapping overhead.
        return _single_solve(problem_text, cfg=cfg, llm=llm, clients=clients,
                             domain=domain, tag=tag)

    # Wrap a bare LLMClient into a RoleClients so the pipeline-only
    # roles (synthesizer / synth_verifier) can be invoked at the end.
    if clients is None and llm is not None:
        clients = RoleClients(
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

    # ---- Token / cost accounting ---------------------------------------
    # Each per-block solve() returns a CUMULATIVE clients.total_cost()
    # snapshot, so naively summing those across blocks over-counts (a
    # triangular sum: block i is charged blocks 0..i). We instead measure
    # each block's DELTA via before/after snapshots, and report the whole
    # problem's true total as clients.total_cost() - baseline (which also
    # captures the graph-builder + synthesizer calls made outside the loop).
    def _cost_now() -> tuple[int, int, int, float]:
        if clients is None:
            return (0, 0, 0, 0.0)
        ct = clients.total_cost()
        return (ct.calls, ct.input_tokens, ct.output_tokens, ct.usd)

    run_cost_baseline = _cost_now()

    # ---- Dependency graph + cross-reference resolver -------------------
    # The graph (which block feeds which) and the per-block placeholder
    # resolution are both LLM-driven via the `decomposition` role — see
    # `structure/llm_graph.py`. We only activate the LLM path when a
    # decomposition client distinct from the executor is configured;
    # otherwise (single shared client / mock) we keep the deterministic
    # regex graph + parenthetical substitution so existing behaviour and
    # tests are preserved.
    resolver: LLMGraphResolver | None = None
    if clients is not None and clients.decomposition is not clients.execution:
        resolver = LLMGraphResolver(
            llm=clients.decomposition,
            temperature=clients.specs["decomposition"].temperature,
            model_name=clients.specs["decomposition"].model,
        )

    graph_source = "regex"
    if resolver is not None:
        spec = resolver.build_graph(problem_text, blocks)
        graph_source = spec.source
        G = spec.to_digraph()
    else:
        G = _build_block_graph(blocks)
    order = _solve_order(G)
    blocks_by_id = {b.node_id: b for b in blocks}
    # Resolved (placeholder-free) block bodies, recorded for the artifacts.
    resolved_texts: dict[str, str] = {}
    resolved_sources: dict[str, str] = {}

    shared_memory: dict[str, str] = {}
    per_block: list[BlockResult] = []
    agg_cost = CostTally()
    flags: list[str] = []
    coherent_count = 0
    sigma_max = 0.0
    rollouts_total = 0

    base_tag = tag or "multi"
    multi_t0 = time.time()

    # ---- Per-multi-run directory --------------------------------------
    # Each multi-block solve gets its OWN parent directory under
    # cfg.telemetry.log_dir. Every per-block sub-solve is then redirected
    # to write inside that parent so all 35 blocks (and their tree.json /
    # rollouts / agent_calls / telemetry files) live together. Avoids
    # littering results/concord/ with hundreds of sibling directories.
    multi_run_id = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}-{base_tag}"
    )
    base_log_dir = Path(cfg.telemetry.log_dir)
    multi_dir = base_log_dir / multi_run_id
    multi_dir.mkdir(parents=True, exist_ok=True)
    # Clone cfg so we can override `telemetry.log_dir` without leaking
    # the override back to the caller. Pydantic's model_copy(deep=True)
    # is the canonical way; fall back to a plain reassignment if not
    # available (older Pydantic v1 codepath).
    sub_cfg: Config = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else cfg
    sub_cfg.telemetry.log_dir = str(multi_dir)

    # Single VectorSharedMemory shared by every per-block sub-solve. The
    # PipelineExpansionPolicy reads it (cross-block deps + semantic context)
    # and writes back (committed block answers + atomic-unit answers). The
    # legacy MCTS solver ignores it.
    vector_mem = VectorSharedMemory()
    # We tunnel it to sub-solves via the cloned cfg.pipeline namespace.
    # The orchestrator picks it up from `cfg.pipeline._runtime_shared_memory`.
    try:
        sub_cfg.pipeline.__dict__["_runtime_shared_memory"] = vector_mem
    except Exception:                                               # noqa: BLE001
        # Pydantic v2 strict — fallback path: attach as plain attribute.
        object.__setattr__(sub_cfg.pipeline, "_runtime_shared_memory", vector_mem)

    if progress is not None:
        progress({"kind": "multi_start",
                  "n_blocks": len(blocks),
                  "order": list(order),
                  "graph_source": graph_source,
                  "run_dir": str(multi_dir)})

    for i, node_id in enumerate(order, 1):
        # Snapshot cost BEFORE this block's work (resolver call + solve) so
        # we can charge the block only its own delta, not the cumulative.
        block_cost_before = _cost_now()
        block = blocks_by_id[node_id]
        # Dependencies come from the (LLM-built or regex) graph, so the
        # resolver and the context preamble agree on what feeds this block.
        deps = (sorted(G.predecessors(node_id), key=_natural_key)
                if node_id in G.nodes else list(block.refs))
        dep_answers = {d: shared_memory.get(d, "") for d in deps}

        # Compose a single-block input: context preamble + the block body
        # with its `[For this value use ... from problem node_M ...]`
        # placeholders resolved to concrete values. We re-emit the
        # `Problem node_K:` header so the downstream `extract_graph` sees a
        # clean 1-node DAG.
        if resolver is not None:
            rb = resolver.resolve_block(node_id, block.text, dep_answers)
            body = rb.text
            resolved_sources[node_id] = rb.source
        else:
            body = _substitute_refs(block.text, shared_memory)
            resolved_sources[node_id] = "substitute"
        resolved_texts[node_id] = body
        preamble = _context_preamble(deps, shared_memory)
        single_text = f"{preamble}Problem {node_id}: {body}"

        if progress is not None:
            progress({
                "kind": "block_start",
                "index": i,
                "total": len(blocks),
                "node_id": node_id,
                "refs": list(deps),
                "subproblem_head": body[:160].replace("\n", " "),
            })

        t0 = time.time()
        sub = _single_solve(
            single_text, cfg=sub_cfg, llm=llm, clients=clients,
            domain=domain, tag=f"{base_tag}_{node_id}",
        )
        elapsed = time.time() - t0

        # Per-block DELTA cost: this block's own consumption (resolver +
        # solve), not the cumulative snapshot sub.cost carries. Keep the
        # non-numeric provenance keys (paths, per_role, ...) from sub.cost.
        block_cost_after = _cost_now()
        block_cost = dict(sub.cost)
        block_cost["calls"] = block_cost_after[0] - block_cost_before[0]
        block_cost["input_tokens"] = block_cost_after[1] - block_cost_before[1]
        block_cost["output_tokens"] = block_cost_after[2] - block_cost_before[2]
        block_cost["usd"] = round(block_cost_after[3] - block_cost_before[3], 6)
        block_cost["total_tokens"] = (block_cost["input_tokens"]
                                       + block_cost["output_tokens"])

        # Store this block's answer for downstream blocks to consume.
        shared_memory[node_id] = sub.answer

        per_block.append(BlockResult(
            node_id=node_id,
            answer=sub.answer,
            coherent=sub.coherent,
            sigma=sub.sigma,
            flagged=sub.flagged,
            rollouts=sub.rollouts,
            cost=block_cost,
            trace_path=sub.trace_path,
        ))
        _add_cost(agg_cost, block_cost)
        if sub.flagged:
            flags.append(f"{node_id}:{sub.flagged}")
        if sub.coherent:
            coherent_count += 1
        sigma_max = max(sigma_max, float(sub.sigma))
        rollouts_total += int(sub.rollouts)

        if progress is not None:
            progress({
                "kind": "block_end",
                "index": i,
                "total": len(blocks),
                "node_id": node_id,
                "answer": sub.answer,
                "coherent": sub.coherent,
                "sigma": float(sub.sigma),
                "rollouts": int(sub.rollouts),
                "cost_usd": float(block_cost.get("usd", 0.0)),
                "input_tokens": int(block_cost.get("input_tokens", 0)),
                "output_tokens": int(block_cost.get("output_tokens", 0)),
                "total_tokens": int(block_cost.get("total_tokens", 0)),
                "elapsed_s": elapsed,
                "running_usd": float(agg_cost.usd),
                "running_calls": int(agg_cost.calls),
                "running_tokens": int(agg_cost.input_tokens
                                      + agg_cost.output_tokens),
                "flagged": sub.flagged,
                "trace_path": sub.trace_path,
            })

    # Consolidated answer: sink of the dependency DAG. Heuristic only — the
    # synthesizer below produces the FINAL `solution = [...]` answer when
    # the pipeline solver was used. The heuristic falls back when no
    # synthesizer is available (e.g. legacy mcts solver, mock LLM).
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if len(sinks) == 1:
        consolidated = shared_memory.get(sinks[0], "")
    else:
        # Stable ordering (source order of blocks).
        consolidated = json.dumps({b.node_id: shared_memory.get(b.node_id, "")
                                   for b in blocks}, ensure_ascii=False)

    # ---- Optional global synthesis -----------------------------------
    # When the pipeline solver was used (cfg.solver == "pipeline") we run a
    # final synthesizer over the WHOLE problem to emit the grader-compatible
    # `solution = [v1, ..., vN]` list. The synthesizer is OPUS by default
    # (cfg.models.synthesizer); it sees the full original problem, every
    # committed block answer, and a heuristic candidate length derived
    # from the block-level DAG.
    synth_result: SynthResult | None = None
    if getattr(cfg, "solver", "mcts") == "pipeline" and clients is not None:
        try:
            full_graph = extract_graph(problem_text)
            length_hint = heuristic_expected_length(full_graph)
            synth = LLMSynthesizer(
                synth_llm=clients.synthesizer,
                verifier_llm=clients.synth_verifier,
                cfg=cfg,
                synth_temperature=clients.specs["synthesizer"].temperature,
                verifier_temperature=clients.specs["synth_verifier"].temperature,
                synth_model_name=clients.specs["synthesizer"].model,
                verifier_model_name=clients.specs["synth_verifier"].model,
            )
            synth_result = synth.synthesize(
                problem_text=problem_text,
                block_answers=dict(shared_memory),
                expected_length_hint=length_hint,
            )
            if synth_result and synth_result.final_line:
                # The synthesizer's FULL response (with the `solution = [...]`
                # line at the end) becomes the consolidated answer string
                # that downstream graders consume. The grader extracts the
                # last bracketed list, so emitting the whole response is
                # safe and gives the operator the reasoning trace too.
                consolidated = synth_result.full_response
        except Exception:                                           # noqa: BLE001
            # Synthesis must never break the multi-solve. Fall back to the
            # heuristic consolidated answer set above.
            pass

    # True problem total = everything consumed since the baseline: every
    # block (delta-measured above) PLUS the graph-builder and synthesizer
    # calls made outside the per-block loop. This is the accurate number to
    # "keep track" of — not the triangular sum of cumulative per-block
    # snapshots. Falls back to the per-block delta aggregate when no client
    # is available (e.g. some mock paths).
    cost_dict = agg_cost.to_dict()
    if clients is not None:
        end = _cost_now()
        cost_dict["calls"] = end[0] - run_cost_baseline[0]
        cost_dict["input_tokens"] = end[1] - run_cost_baseline[1]
        cost_dict["output_tokens"] = end[2] - run_cost_baseline[2]
        cost_dict["usd"] = round(end[3] - run_cost_baseline[3], 6)
    cost_dict["total_tokens"] = (cost_dict["input_tokens"]
                                  + cost_dict["output_tokens"])
    cost_dict["per_block"] = [
        {
            "node_id": br.node_id,
            "answer": br.answer,
            "coherent": br.coherent,
            "sigma": br.sigma,
            "flagged": br.flagged,
            "rollouts": br.rollouts,
            "cost": br.cost,
            "trace_path": br.trace_path,
        }
        for br in per_block
    ]
    cost_dict["shared_memory"] = dict(shared_memory)
    cost_dict["n_blocks"] = len(blocks)
    cost_dict["sinks"] = sinks
    cost_dict["run_dir"] = str(multi_dir)
    cost_dict["graph_source"] = graph_source
    cost_dict["resolved_blocks"] = dict(resolved_texts)

    # ---- resolved_blocks.json -----------------------------------------
    # The self-contained (placeholder-free) question that was actually fed
    # to the solver for each block, plus how it was produced. This is the
    # single most useful artifact for debugging dependency resolution.
    try:
        (multi_dir / "resolved_blocks.json").write_text(
            json.dumps({
                "graph_source": graph_source,
                "blocks": [
                    {
                        "node_id": nid,
                        "deps": sorted(G.predecessors(nid), key=_natural_key)
                                if nid in G.nodes else [],
                        "original": blocks_by_id[nid].text,
                        "resolved": resolved_texts.get(nid, ""),
                        "resolution": resolved_sources.get(nid, ""),
                    }
                    for nid in order
                ],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:                                               # noqa: BLE001
        pass

    # ---- Write manifest.json into the parent directory ----------------
    # One-stop summary of the multi-block run: which blocks live in which
    # subdirectory, with their final answer / coherence / cost. Useful for
    # `ls`-style discovery without crawling every block's tree.json.
    manifest = {
        "multi_run_id": multi_run_id,
        "run_dir": str(multi_dir),
        "started_at": datetime.utcfromtimestamp(multi_t0).isoformat() + "Z",
        "elapsed_s": time.time() - multi_t0,
        "n_blocks": len(blocks),
        "order": list(order),
        "sinks": sinks,
        "consolidated_answer": consolidated,
        "coherent_count": coherent_count,
        "total_cost": {
            "calls": cost_dict["calls"],
            "input_tokens": cost_dict["input_tokens"],
            "output_tokens": cost_dict["output_tokens"],
            "total_tokens": cost_dict["total_tokens"],
            "usd": cost_dict["usd"],
        },
        "blocks": [
            {
                "index": idx,
                "node_id": br.node_id,
                "answer": br.answer,
                "coherent": br.coherent,
                "sigma": br.sigma,
                "flagged": br.flagged,
                "rollouts": br.rollouts,
                "cost": br.cost,
                "resolved_text": resolved_texts.get(br.node_id, ""),
                "resolution": resolved_sources.get(br.node_id, ""),
                # Per-block paths are relative to multi_dir so the
                # manifest stays portable if you move the run directory.
                "rel_tree": (str(Path(br.cost.get("tree_path", "")).relative_to(multi_dir))
                              if br.cost.get("tree_path") else None),
                "rel_rollouts": (str(Path(br.cost.get("rollouts_path", "")).relative_to(multi_dir))
                                  if br.cost.get("rollouts_path") else None),
                "rel_trace": (str(Path(br.trace_path).relative_to(multi_dir))
                               if br.trace_path else None),
            }
            for idx, br in enumerate(per_block, 1)
        ],
    }
    try:
        (multi_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:                                               # noqa: BLE001
        pass

    # ---- shared_memory.json + combine_tree.json --------------------------
    # Persist the per-question scratchpad (text + provenance for vector
    # retrieval reconstruction) alongside the manifest.
    try:
        vector_mem.write_json(multi_dir / "shared_memory.json")
    except Exception:                                               # noqa: BLE001
        pass

    # Assemble the combine_tree.json — a single document covering the
    # full decompose-solve-combine recursion. Layer 0 is the block-level
    # DAG; Layer 1 is each block's pipeline provenance (split tree,
    # atomic answers, combiner attempts, verifier verdicts); Layer 2 is
    # the global synthesizer's final-list output.
    block_provenance: list[dict] = []
    for br in per_block:
        prov = (br.cost or {}).get("pipeline_provenance")
        if isinstance(prov, dict) and isinstance(prov.get("blocks"), list):
            # PipelineExpansionPolicy emits one entry per inner G' node it
            # saw. For a single-block solve that's typically one "scc_0"
            # entry; we relabel it back to the OUTER block node_id so the
            # combine_tree groups by the human-facing identifier.
            for entry in prov["blocks"]:
                relabeled = dict(entry)
                relabeled["block_node_id"] = br.node_id
                relabeled["inner_node_id"] = entry.get("block_node_id")
                block_provenance.append({
                    "solver": "pipeline",
                    **relabeled,
                })
        else:
            # Legacy MCTS or no provenance — still emit a stub so the
            # viewer can render the block at layer 1.
            block_provenance.append({
                "block_node_id": br.node_id,
                "solver": "mcts" if (br.cost or {}).get("per_role") else "unknown",
                "final_block_answer": br.answer,
                "split_tree": None,
                "atomic_answers": [],
                "combiner_attempts": [],
                "verifier_verdicts": [],
            })

    combine_tree = {
        "multi_run_id": multi_run_id,
        "run_dir": str(multi_dir),
        "problem_text": problem_text,
        "solver": getattr(cfg, "solver", "mcts"),
        # Layer 0: the block-level DAG (which block depends on which).
        "block_graph": {
            "source": graph_source,
            "nodes": [
                {"id": b.node_id,
                 "text_head": b.text[:200].replace("\n", " "),
                 "resolved_head": resolved_texts.get(b.node_id, "")[:200].replace("\n", " "),
                 "deps": (sorted(G.predecessors(b.node_id), key=_natural_key)
                          if b.node_id in G.nodes else list(b.refs)),
                 "refs": list(b.refs)}
                for b in blocks
            ],
            "edges": [{"from": u, "to": v} for u, v in G.edges()],
            "sinks": sinks,
            "topological_order": list(order),
        },
        # Layer 1: per-block pipeline provenance (split / atoms /
        # combiner / verifier verdicts / committed answer).
        "blocks": block_provenance,
        # Layer 2: the global synthesizer output (only when pipeline ran).
        "synthesis": (synth_result.to_dict() if synth_result is not None
                       else None),
    }
    try:
        (multi_dir / "combine_tree.json").write_text(
            json.dumps(combine_tree, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:                                               # noqa: BLE001
        pass

    # Surface the artifact paths + synth diagnostics on the cost dict so
    # callers / graders can find them without crawling the directory.
    cost_dict["combine_tree_path"] = str(multi_dir / "combine_tree.json")
    cost_dict["shared_memory_path"] = str(multi_dir / "shared_memory.json")
    if synth_result is not None:
        cost_dict["synth"] = {
            "final_line": synth_result.final_line,
            "values": list(synth_result.values),
            "heuristic_length": synth_result.heuristic_length,
            "actual_length": synth_result.actual_length,
            "verifier_score": synth_result.verifier_score,
            "verifier_consistent": synth_result.verifier_is_consistent,
            "verifier_issues": list(synth_result.verifier_issues),
            "n_retries": synth_result.n_retries,
        }

    if progress is not None:
        progress({
            "kind": "multi_end",
            "n_blocks": len(blocks),
            "coherent_count": coherent_count,
            "total_usd": float(cost_dict["usd"]),
            "total_calls": int(cost_dict["calls"]),
            "total_input_tokens": int(cost_dict["input_tokens"]),
            "total_output_tokens": int(cost_dict["output_tokens"]),
            "total_tokens": int(cost_dict["total_tokens"]),
            "elapsed_s": time.time() - multi_t0,
            "run_dir": str(multi_dir),
            "final_line": (synth_result.final_line
                           if synth_result is not None else None),
        })

    return Result(
        answer=consolidated,
        coherent=(coherent_count == len(blocks)),
        sigma=sigma_max,
        rollouts=rollouts_total,
        cost=cost_dict,
        flagged="; ".join(flags) if flags else None,
        trace_path=per_block[-1].trace_path if per_block else None,
    )
