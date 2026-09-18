"""Per-question search tracer.

Three artifacts per question, in addition to the existing JSONL telemetry:

  - `<run-id>_rollouts.jsonl` — one line per rollout, full provenance:
      * selected_path (root → leaf node ids)
      * expansion detail (subproblem text, K samples, classes formed,
        per-class U_s / mass / SD / verifier)
      * gate result (sigma, constraints checked, violated)
      * value backed up + new (N, Q) along ancestor chain
      * stop signal if the marginal-gain rule fired

  - `<run-id>_tree.json` — final tree snapshot:
      * meta block (config, role models, final answer, sigma, flagged)
      * nodes[] — every Node.to_dict() with parent_id wired
      * edges[] — convenience flat list of (parent, child) pairs

  - `<run-id>_agent_calls.jsonl` — one line per LLM call, in real time:
      * role: "decomposer" | "executor" | "classifier" | "verifier"
      * full prompt + full response (no truncation)
      * model, finish_reason, timestamps
      * cross-references (node_id, sample_idx, rollout_i) so a call
        can be tied back to a rollout/class in the tree
    `tail -F <run-id>_agent_calls.jsonl` gives a live transcript of
    every agent's input and output while the run is in flight.

These artifacts feed the HTML viewer in `viz/tree_viewer.html`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .search.mcts import Node, RolloutRecord


# ---------------------------------------------------------------------------
# JSON-safe coercion (CostTally / Generation / dataclasses)
# ---------------------------------------------------------------------------

def _to_json(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, dict):
        return {str(k): _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_to_json(x) for x in o]
    if is_dataclass(o):
        return _to_json(asdict(o))
    if hasattr(o, "to_dict"):
        return _to_json(o.to_dict())
    return str(o)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

@dataclass
class _Pending:
    """Side-channel for expansion details the policy gathers on the way."""

    rollout_i: int
    subproblem_id: str | None = None
    subproblem_text: str = ""
    prompt_excerpt: str = ""
    bindings: dict[str, str] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    classes: list[dict] = field(default_factory=list)
    oracle_mode: str | None = None
    queried_pairs: int = 0
    gate: dict | None = None


class Tracer:
    """Captures per-rollout events + builds the final tree snapshot.

    Construction is cheap; the orchestrator hands the tracer the open
    output paths and a reference to whatever expansion policy is used so
    the policy can stash its richer expansion details before search()'s
    on_rollout callback fires.
    """

    def __init__(self, base_path: Path, *, meta: dict | None = None):
        self.base_path = base_path
        self.rollouts_path = base_path.with_name(base_path.stem + "_rollouts.jsonl")
        self.tree_path = base_path.with_name(base_path.stem + "_tree.json")
        self.agent_calls_path = base_path.with_name(
            base_path.stem + "_agent_calls.jsonl")
        self.meta = meta or {}
        self._fp = open(self.rollouts_path, "w", buffering=1)
        # Real-time per-LLM-call log. Opened in append-aware mode but truncated
        # at start so the file matches THIS run. Line-buffered so `tail -F`
        # streams every call the moment it returns from the API.
        self._agent_fp = open(self.agent_calls_path, "w", buffering=1)
        # Counter so each agent call line carries a stable ordinal.
        self._agent_call_seq: int = 0
        # The currently-in-progress rollout index, used to annotate executor
        # samples that the policy generates inside a single rollout.
        self._current_rollout_i: int | None = None
        # Pending bag holds the in-progress rollout's expansion side-channel.
        # The orchestrator/policy pushes into it; on_rollout flushes.
        self._pending: _Pending | None = None

    # ----- expansion side-channel (called from policy.expand) ---------------

    def stash_expansion(self, *, subproblem_id: str, subproblem_text: str,
                        prompt_excerpt: str, bindings: dict[str, str],
                        samples: list[dict], classes: list[dict],
                        oracle_mode: str | None, queried_pairs: int) -> None:
        """Called by the policy after it finishes one expand() call. Always
        overwrites — the tracer consumes whatever is here when the matching
        on_rollout fires next."""
        # Preserve any gate detail stashed before this expansion call (the
        # gate fires in evaluate() AFTER expand for new nodes, but for
        # existing terminals the gate may have fired earlier).
        gate = self._pending.gate if self._pending is not None else None
        self._pending = _Pending(
            rollout_i=-1,    # filled by on_rollout
            subproblem_id=subproblem_id,
            subproblem_text=subproblem_text,
            prompt_excerpt=prompt_excerpt,
            bindings=bindings,
            samples=samples,
            classes=classes,
            oracle_mode=oracle_mode,
            queried_pairs=queried_pairs,
            gate=gate,
        )

    def stash_gate(self, *, sigma: float, violated: list[str],
                    all_checked: list[str], depth: int) -> None:
        """Record what the coherence gate saw during this rollout's evaluate."""
        if self._pending is None:
            self._pending = _Pending(rollout_i=-1)
        self._pending.gate = {
            "sigma": sigma,
            "violated": list(violated),
            "all_checked": list(all_checked),
            "depth": depth,
        }

    # ----- per-LLM-call log (decomposer / executor / classifier / verifier) -

    def log_agent_call(self, *, role: str, prompt: str, response: str,
                       model: str | None = None,
                       finish_reason: str | None = None,
                       node_id: str | None = None,
                       sample_idx: int | None = None,
                       extras: dict | None = None) -> None:
        """Write one line to the agent_calls JSONL.

        Captures every LLM round-trip (decomposer rewriting a subquestion,
        executor sampling K answers, classifier picking a domain, verifier
        scoring a response). Full `prompt` and `response` are stored
        verbatim — no truncation. Streaming-friendly: each call is flushed
        immediately so `tail -F` shows live progress.
        """
        if self._agent_fp is None:
            return
        self._agent_call_seq += 1
        line = {
            "seq": self._agent_call_seq,
            "t": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": model,
            "node_id": node_id,
            "sample_idx": sample_idx,
            "rollout_i": self._current_rollout_i,
            "finish_reason": finish_reason,
            "prompt": prompt,
            "response": response,
            "extras": extras or {},
        }
        try:
            self._agent_fp.write(json.dumps(_to_json(line),
                                            ensure_ascii=False) + "\n")
        except Exception:                                           # noqa: BLE001
            # Tracing must never break the search — swallow IO errors.
            pass

    def set_rollout_context(self, i: int | None) -> None:
        """Tell the tracer which rollout index the next LLM calls belong to.

        Called by `on_rollout`/the orchestrator so executor/verifier calls
        emitted DURING that rollout are correctly tagged. Pass None to
        clear after the rollout finishes.
        """
        self._current_rollout_i = i

    # ----- rollout event ----------------------------------------------------

    def on_rollout(self, i: int, leaf: Node, record: RolloutRecord) -> None:
        # Tag every executor/verifier LLM call that fired during THIS
        # rollout. The orchestrator's on_rollout wrapper calls us last,
        # so by the time we get here all the LLM calls for the rollout
        # have already been logged. Tag for the NEXT rollout.
        self._current_rollout_i = i
        row: dict[str, Any] = {
            "i": record.i,
            "selected_path": record.selected_path,
            "expanded_parent_id": record.expanded_parent_id,
            "new_child_id": record.new_child_id,
            "new_child_resolution": (
                {"subproblem": record.new_child_state_extend[0],
                 "answer": record.new_child_state_extend[1]}
                if record.new_child_state_extend else None
            ),
            "leaf": {
                "id": record.leaf_id,
                "depth": record.leaf_depth,
                "terminal": record.leaf_terminal,
                "gated_fail": record.leaf_gated_fail,
                "sigma": record.leaf_sigma,
                "U_s": record.leaf_U_s,
            },
            "value_backed_up": record.value,
            "backup_path": [
                {"id": nid, "N": n, "Q": q}
                for (nid, n, q) in record.backup_path
            ],
        }
        if self._pending is not None:
            if self._pending.subproblem_id is not None:
                row["expansion"] = {
                    "subproblem_id": self._pending.subproblem_id,
                    "subproblem_text": self._pending.subproblem_text[:400],
                    "prompt_excerpt": self._pending.prompt_excerpt[:600],
                    "bindings": self._pending.bindings,
                    "samples": self._pending.samples,
                    "classes": self._pending.classes,
                    "oracle_mode": self._pending.oracle_mode,
                    "queried_pairs": self._pending.queried_pairs,
                }
            if self._pending.gate is not None:
                row["gate"] = self._pending.gate
        self._pending = None
        self._fp.write(json.dumps(_to_json(row), ensure_ascii=False) + "\n")

    # ----- tree snapshot ----------------------------------------------------

    def write_tree(self, root: Node, *, summary: dict | None = None) -> None:
        nodes: list[dict] = []
        edges: list[dict] = []
        stack: list[Node] = [root]
        seen: set[str] = set()
        while stack:
            n = stack.pop()
            if n.node_id in seen:
                continue
            seen.add(n.node_id)
            nodes.append(n.to_dict())
            for c in n.children.values():
                edges.append({"from": n.node_id, "to": c.node_id,
                              "class_key": c.class_key})
                stack.append(c)
        # stable ordering — by id index
        nodes.sort(key=lambda d: int(d["id"][1:]) if d["id"].startswith("n")
                                                  else d["id"])
        edges.sort(key=lambda d: (d["from"], d["to"]))

        payload = {
            "meta": _to_json(self.meta),
            "summary": _to_json(summary or {}),
            "nodes": nodes,
            "edges": edges,
        }
        self.tree_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:                                           # noqa: BLE001
            pass
        try:
            if self._agent_fp is not None:
                self._agent_fp.close()
                self._agent_fp = None
        except Exception:                                           # noqa: BLE001
            pass
