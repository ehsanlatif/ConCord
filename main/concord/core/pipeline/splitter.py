"""LLMSplitter — recursive atomic-question decomposition.

Given a block-level question and the already-resolved upstream answers,
produce a small ordered list of atomic sub-questions that each can be
answered in one LLM call. Recursion is allowed (an atomic unit can be
split again) up to `cfg.pipeline.max_split_depth` levels.

Key invariants:

  - Output is ALWAYS a list of atoms. If the question is already atomic,
    return a list of length 1 wrapping the original question. The caller
    cannot distinguish "atomic" from "small split" — both look like a
    one-element list, and that's fine.

  - Atoms may declare cross-dependencies among themselves. The mini-DAG
    is encoded by each atom carrying `refs` — indices of preceding
    atoms whose answers it consumes.

  - The Splitter NEVER answers the question — it only restructures it.

Heuristic short-circuit: a question shorter than
`cfg.pipeline.atom_short_circuit_chars` AND free of obvious multi-step
markers ("first ... then", multiple "and"s separated by clauses, nested
bracket arithmetic) is treated atomic without an LLM call. This is
the cheap path for the leaf atoms.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..llm.client import LLMClient


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class CallBudget:
    """Per-node hard ceiling on agent (LLM) calls.

    Shared by the splitter recursion and the expansion policy's
    solve/combine phases so that ONE graph node's pipeline solve cannot
    exceed `limit` agent calls no matter how the recursive decomposition
    unfolds. `reserve` lets an early phase (splitting, leaf execution) leave
    headroom for a later, mandatory phase (the block combiner / verifier),
    so the node always produces a final answer.
    """

    limit: int
    used: int = 0

    def can_spend(self, n: int = 1, *, reserve: int = 0) -> bool:
        return self.used + n <= self.limit - reserve

    def spend(self, n: int = 1) -> None:
        self.used += max(0, int(n))

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


@dataclass
class AtomicUnit:
    """One leaf-or-non-leaf question in the split tree."""

    atom_id: str                          # stable id like "block_5/d0/a0"
    question: str
    depth: int                            # 0 == top of this block
    refs: list[str] = field(default_factory=list)
    # For non-leaf atoms (further split): the children produced by a
    # recursive splitter call. Leaves have `children=[]`.
    children: list["AtomicUnit"] = field(default_factory=list)
    # Heuristic / LLM decision: is this atom atomic (no further split)?
    is_atomic: bool = True
    # Provenance for the combine tree.
    split_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "question": self.question,
            "depth": self.depth,
            "refs": list(self.refs),
            "is_atomic": self.is_atomic,
            "split_reason": self.split_reason,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class SplitTree:
    """The full atomic decomposition of one block."""

    block_node_id: str
    root_question: str
    atoms: list[AtomicUnit]
    n_llm_calls: int = 0                  # cost provenance

    def flatten_leaves(self) -> list[AtomicUnit]:
        """Topo-ordered list of leaf atoms (the ones that actually get
        executed). Recursively descends into children when a non-atomic
        atom has them; otherwise yields the atom itself."""
        out: list[AtomicUnit] = []
        def visit(a: AtomicUnit):
            if a.children:
                for c in a.children:
                    visit(c)
            else:
                out.append(a)
        for a in self.atoms:
            visit(a)
        return out

    # -----------------------------------------------------------------
    # Bottom-up traversal (children-before-parent + refs-honored)
    # -----------------------------------------------------------------

    def topological_atom_order(self) -> list[AtomicUnit]:
        """Return ALL atoms (leaves AND non-leaves) in dependency order.

        Two kinds of edges contribute to the order:
          1. *Parent depends on its children* — a composite atom can only
             be synthesized AFTER each of its children has an answer.
          2. *Refs within a sibling list* — when the splitter emits
             `{"id": 1, "refs": [0]}`, atom-1 reads sibling 0's answer.
             Index 0 must be solved before index 1.

        The traversal therefore guarantees: when an atom is visited, all
        atoms it depends on (its children + its same-level ref targets)
        have already been visited. Leaves with no refs come first.

        On a cycle (which would be a splitter bug — refs only point
        backwards), falls back to `flatten_leaves()` so we at least
        execute SOMETHING and the verifier rejects later.
        """
        # Build flat index of all atoms with their parent + sibling list.
        all_by_id: dict[str, AtomicUnit] = {}
        siblings_at_level: dict[str, list[AtomicUnit]] = {}

        def walk(arr: list[AtomicUnit]) -> None:
            sib_list = list(arr)
            for a in arr:
                all_by_id[a.atom_id] = a
                siblings_at_level[a.atom_id] = sib_list
                if a.children:
                    walk(a.children)

        walk(self.atoms)

        if not all_by_id:
            return []

        # Build the dependency DAG.
        import networkx as _nx
        G = _nx.DiGraph()
        for aid in all_by_id:
            G.add_node(aid)
        for aid, atom in all_by_id.items():
            # (1) children must be solved before this atom
            for child in atom.children:
                G.add_edge(child.atom_id, aid)
            # (2) refs (integer indices into the SAME sibling list)
            sibs = siblings_at_level.get(aid) or []
            for ref in atom.refs:
                try:
                    idx = int(ref)
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(sibs) and sibs[idx].atom_id != aid:
                    G.add_edge(sibs[idx].atom_id, aid)

        try:
            order = list(_nx.lexicographical_topological_sort(G))
        except _nx.NetworkXUnfeasible:
            # Splitter produced a cycle — degrade gracefully.
            return self.flatten_leaves()
        return [all_by_id[aid] for aid in order]

    def siblings_map(self) -> dict[str, list[AtomicUnit]]:
        """Reverse-lookup: atom_id → its list of sibling atoms (i.e. the
        atom list that contains it). Top-level atoms map to `self.atoms`.
        Useful for inlining `refs` (which are indices into that list)."""
        out: dict[str, list[AtomicUnit]] = {}
        def walk(arr: list[AtomicUnit]) -> None:
            sibs = list(arr)
            for a in arr:
                out[a.atom_id] = sibs
                if a.children:
                    walk(a.children)
        walk(self.atoms)
        return out

    def to_dict(self) -> dict:
        return {
            "block_node_id": self.block_node_id,
            "root_question": self.root_question,
            "atoms": [a.to_dict() for a in self.atoms],
            "n_llm_calls": self.n_llm_calls,
        }


# ---------------------------------------------------------------------------
# Heuristic atomicity check
# ---------------------------------------------------------------------------

# Markers that strongly suggest a multi-step question even when short.
_MULTI_STEP_MARKERS = [
    re.compile(r"\bfirst\b.*\bthen\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bafter that\b", re.IGNORECASE),
    re.compile(r"\bfor each\b.*\b(compute|determine|find|count)\b",
                re.IGNORECASE | re.DOTALL),
    re.compile(r"\bgiven\b.*\band\b.*\bfind\b", re.IGNORECASE | re.DOTALL),
    # Nested bracket arithmetic — a leftover from raw-substitution flows.
    re.compile(r"\[[^\[\]]*\b(answer|value|node_)\b[^\[\]]*\]",
                re.IGNORECASE),
]


def is_likely_atomic(question: str, short_threshold: int) -> bool:
    """Heuristic: short + no multi-step markers ⇒ skip the splitter LLM."""
    q = question.strip()
    if len(q) <= short_threshold and not any(
            p.search(q) for p in _MULTI_STEP_MARKERS):
        return True
    return False


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SPLITTER_SYSTEM = (
    "You are a problem DECOMPOSER for a step-by-step math / reasoning "
    "solver. Break the given problem into the ORDERED sequence of "
    "intermediate sub-problems a careful solver would work through to "
    "reach the final answer — like the numbered steps of a worked "
    "solution. Each sub-problem becomes its own node that is solved on its "
    "own; later steps build on the answers of earlier ones.\n\n"
    "RULES:\n"
    "  1. DECOMPOSE into 2-{max_atoms} sub-problems whenever the problem "
    "needs more than one step of reasoning (e.g. set up / establish a fact, "
    "compute an intermediate quantity, handle a case, then combine and "
    "conclude). MOST contest-style problems do — prefer decomposing.\n"
    "  2. Each sub-problem must be self-contained and answerable by a "
    "single focused chain of reasoning. If a sub-problem is still complex "
    "it will be decomposed AGAIN automatically, so prefer genuinely small, "
    "concrete steps over broad ones.\n"
    "  3. Number atoms starting at 0. Use `refs: [i, j, ...]` to declare "
    "that a sub-problem consumes the ANSWERS of earlier sub-problems (by "
    "their 0-based index). The LAST sub-problem MUST yield the final answer "
    "to the whole problem and should ref the earlier steps it depends on. "
    "Do NOT introduce cycles (refs only point to smaller indices).\n"
    "  4. Return a SINGLE atom (one entry, the original question) ONLY when "
    "the problem is a single direct computation or lookup that a solver "
    "would answer in one step with no intermediate results.\n"
    "  5. Inline every known value from the resolved upstream context. "
    "NEVER leave placeholders like `node_K` or `[...]` in a sub-problem.\n"
    "  6. You NEVER answer the sub-problems — you only restructure.\n"
    "  7. Output strict JSON, no prose:\n"
    "       { \"atoms\": [ { \"id\": 0, \"question\": \"...\", "
    "\"refs\": [] }, ... ] }\n"
    "  8. Hard cap: at most {max_atoms} sub-problems at THIS level. Finer "
    "detail should come from the automatic recursive decomposition of each "
    "step, not from one long flat list.\n"
)


_USER_TEMPLATE = (
    "Problem to decompose (this block of the larger task):\n"
    "{block_text}\n\n"
    "Resolved upstream answers (use them by inlining the value):\n"
    "{resolved}\n\n"
    "{extra_context}"
    "Produce the JSON list of solution-step sub-problems now. Decompose "
    "unless the problem is a single direct computation; in that one case "
    "return a single atom whose `question` IS the original problem."
)


# ---------------------------------------------------------------------------
# LLMSplitter
# ---------------------------------------------------------------------------

@dataclass
class LLMSplitter:
    """Recursive splitter. Holds the LLM client; not stateful otherwise."""

    llm: LLMClient
    cfg: Config
    temperature: float = 0.3
    # Optional tracer for logging the splitter LLM calls.
    tracer: "object | None" = None
    model_name: str | None = None

    # -----------------------------------------------------------------

    def split(self, *, block_node_id: str, block_text: str,
              resolved: dict[str, str],
              retrieved_context: list[str] | None = None,
              budget: "CallBudget | None" = None,
              max_split_calls: int | None = None) -> SplitTree:
        """Build the SplitTree for one block.

        `budget` (when supplied) caps the agent calls the whole node may
        spend; `max_split_calls` further caps how many of those the splitter
        recursion alone may consume, so it can never starve leaf execution
        and combining. Both default to None (unbounded), preserving the
        legacy behaviour for callers / tests that don't pass them.
        """
        root = AtomicUnit(
            atom_id=f"{block_node_id}/d0/a0",
            question=block_text.strip(),
            depth=0,
        )
        tree = SplitTree(
            block_node_id=block_node_id,
            root_question=block_text.strip(),
            atoms=[root],
        )
        self._maybe_split(
            atom=root, block_node_id=block_node_id, depth=0,
            resolved=resolved, retrieved_context=retrieved_context,
            tree=tree, budget=budget, max_split_calls=max_split_calls,
        )
        # If the root was determined non-atomic and got children, flatten:
        # the "atoms" list becomes the level-0 atoms (root's children),
        # not the root itself.
        if root.children:
            tree.atoms = root.children
        return tree

    # -----------------------------------------------------------------

    def _maybe_split(self, *, atom: AtomicUnit, block_node_id: str,
                     depth: int, resolved: dict[str, str],
                     retrieved_context: list[str] | None,
                     tree: SplitTree,
                     budget: "CallBudget | None" = None,
                     max_split_calls: int | None = None) -> None:
        """Decide if this atom should be split further; if so, replace its
        children with the splitter's output and recurse."""
        # Depth cap (recursion).
        if depth >= self.cfg.pipeline.max_split_depth:
            atom.is_atomic = True
            atom.split_reason = "depth_cap_reached"
            return

        # Per-node call budget: stop decomposing once the splitter has used
        # its share of the node's agent-call budget (or the shared budget is
        # exhausted). Leaving the atom atomic means it gets solved directly.
        if budget is not None:
            split_cap_hit = (max_split_calls is not None
                             and tree.n_llm_calls >= max_split_calls)
            if split_cap_hit or not budget.can_spend(1):
                atom.is_atomic = True
                atom.split_reason = "call_budget_reached"
                return

        # Heuristic short-circuit — applied ONLY to recursive sub-steps
        # (depth > 0). A TOP-LEVEL block (depth 0) always gets a real
        # splitter decision so hard problems are genuinely decomposed into
        # solution steps instead of being solved in one shot. Character
        # length is a poor proxy for "atomic" on contest math — the model,
        # not a 300-char cutoff, should judge whether the block needs steps.
        # At deeper levels the heuristic still stops genuinely-small leaves
        # from burning further splitter calls.
        if depth > 0 and is_likely_atomic(
                atom.question, self.cfg.pipeline.atom_short_circuit_chars):
            atom.is_atomic = True
            atom.split_reason = "heuristic_atomic"
            return

        # Call the splitter LLM.
        children = self._call_splitter_llm(
            block_text=atom.question,
            resolved=resolved,
            retrieved_context=retrieved_context,
            depth=depth, block_node_id=block_node_id,
            parent_atom_id=atom.atom_id,
            tree=tree, budget=budget,
        )

        if len(children) <= 1:
            # Splitter agreed it's atomic (or call failed / returned 1).
            atom.is_atomic = True
            atom.split_reason = "llm_atomic"
            return

        atom.is_atomic = False
        atom.children = children
        atom.split_reason = f"llm_split_into_{len(children)}"

        # Recursively split each child (depth+1). The cap stops runaways.
        for child in children:
            self._maybe_split(
                atom=child, block_node_id=block_node_id, depth=depth + 1,
                resolved=resolved, retrieved_context=retrieved_context,
                tree=tree, budget=budget, max_split_calls=max_split_calls,
            )

    # -----------------------------------------------------------------

    def _call_splitter_llm(self, *, block_text: str,
                            resolved: dict[str, str],
                            retrieved_context: list[str] | None,
                            depth: int, block_node_id: str,
                            parent_atom_id: str,
                            tree: SplitTree,
                            budget: "CallBudget | None" = None
                            ) -> list[AtomicUnit]:
        """One LLM call. Returns a (possibly single-element) list of
        children. On failure returns [] which is interpreted upstream as
        "leave as atomic."""
        # Charge this splitter call against the per-node budget up front so
        # the recursion guard above sees it on the next descent.
        if budget is not None:
            budget.spend(1)
        max_atoms = self.cfg.pipeline.max_atoms_per_split
        sys_prompt = _SPLITTER_SYSTEM.replace("{max_atoms}", str(max_atoms))
        resolved_str = (
            "(none)" if not resolved else
            "\n".join(f"- {k}: {v}" for k, v in resolved.items())
        )
        extra = ""
        if retrieved_context:
            extra = (
                "Possibly-relevant prior results (from shared memory; "
                "use only if pertinent):\n"
                + "\n".join(f"- {c}" for c in retrieved_context)
                + "\n\n"
            )
        user = _USER_TEMPLATE.format(
            block_text=block_text, resolved=resolved_str,
            extra_context=extra,
        )
        prompt = sys_prompt + "\n\n" + user

        try:
            gens = self.llm.generate(prompt, temperature=self.temperature, n=1)
            text = gens[0].text if gens else ""
            finish = gens[0].finish_reason if gens else None
        except Exception as e:                                      # noqa: BLE001
            _log.exception("LLMSplitter call failed; treating as atomic")
            self._log_call(prompt=prompt, response="",
                            finish_reason=None, error=repr(e),
                            block_node_id=block_node_id, depth=depth)
            tree.n_llm_calls += 1
            return []

        tree.n_llm_calls += 1
        atoms = self._parse_splitter_output(
            text, parent_atom_id=parent_atom_id, depth=depth + 1,
        )
        self._log_call(prompt=prompt, response=text,
                        finish_reason=finish, error=None,
                        block_node_id=block_node_id, depth=depth,
                        n_atoms=len(atoms))
        return atoms

    # -----------------------------------------------------------------

    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    def _parse_splitter_output(self, text: str, *,
                                parent_atom_id: str,
                                depth: int) -> list[AtomicUnit]:
        """Extract `{"atoms": [...]}` from the LLM response. Tolerant of
        leading/trailing prose; takes the first balanced JSON object."""
        if not text:
            return []
        m = self._JSON_RE.search(text)
        if not m:
            return []
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return []
        raw_atoms = obj.get("atoms") if isinstance(obj, dict) else None
        if not isinstance(raw_atoms, list) or not raw_atoms:
            return []
        # If the LLM responded with one atom whose question equals the
        # input, treat as atomic.
        if len(raw_atoms) == 1:
            return [AtomicUnit(
                atom_id=f"{parent_atom_id}/a0",
                question=str(raw_atoms[0].get("question", "")).strip(),
                depth=depth,
            )]
        out: list[AtomicUnit] = []
        for i, raw in enumerate(raw_atoms):
            if not isinstance(raw, dict):
                continue
            q = str(raw.get("question", "")).strip()
            if not q:
                continue
            refs_raw = raw.get("refs") or []
            refs = [str(r) for r in refs_raw if isinstance(r, (int, str))]
            out.append(AtomicUnit(
                atom_id=f"{parent_atom_id}/a{i}",
                question=q,
                depth=depth,
                refs=refs,
            ))
        return out

    # -----------------------------------------------------------------

    def _log_call(self, *, prompt: str, response: str,
                  finish_reason: str | None,
                  block_node_id: str, depth: int,
                  error: str | None = None,
                  n_atoms: int = 0) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.log_agent_call(
                role="splitter",
                prompt=prompt,
                response=response,
                model=self.model_name,
                finish_reason=finish_reason,
                node_id=block_node_id,
                extras={"depth": depth, "n_atoms": n_atoms,
                        "error": error},
            )
        except Exception:                                           # noqa: BLE001
            pass
