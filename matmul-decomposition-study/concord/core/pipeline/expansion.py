"""PipelineExpansionPolicy — Split → Solve → Combine → Verify.

Drop-in replacement for `RewriteExpansionPolicy`. Same public surface:
`expand(state, depth, *, parent_u_s, K) -> ExpandedNodeInfo`. The
difference is what happens INSIDE the call:

  Phase A  Splitter LLM produces a SplitTree of atomic units.
  Phase B  For each atomic leaf (in topological order over the mini-DAG):
             - resolve cross-block deps from VectorSharedMemory
             - Executor LLM × K samples → cluster → best atomic answer
             - write atomic answer back into VectorSharedMemory
  Phase C  Combiner LLM × K_combiner samples → cluster → candidate
  Phase D  BlockVerifier scores the candidate.
             score ≥ verifier_accept (0.75)     → commit
             verifier_retry ≤ score < accept    → retry combiner with hint
             score < verifier_backtrack (0.25)  → emit low-U_s class so
                                                   MCTS backtracks
             else                                → commit with lower U_s

The K classes returned by `expand()` are derived from the COMBINER's
cluster (not the executor's), so the existing MCTS PUCT machinery still
works unchanged — it just sees the combined block answer as the
"expansion choice" at this depth.

The verifier's score is folded into U_s so that backtracking is automatic:
when the candidate scores low, MCTS down-weights this branch and
explores siblings on the next rollout.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..config import Config
from ..confidence import (
    Embedder,
    EntailmentKernel,
    Verifier,
    score_subproblem,
)
from ..expansion import build_executor_prompt, extract_answer
from ..llm.client import LLMClient
from ..search import ExpandedClass, ExpandedNodeInfo
from ..structure import ATOMIC, ExplicitDecomposer, granularity_for, is_atomic
from ..types import SolutionState
from .block_verifier import LLMBlockVerifier, VerifierVerdict
from .combiner import LLMCombiner
from .shared_memory import VectorSharedMemory
from .splitter import CallBudget, LLMSplitter, SplitTree


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional task-specific SOURCE RESOLVER (index-reference support)
# ---------------------------------------------------------------------------
# A task may register a resolver that, given an atom and its (ref-inlined)
# executor question, injects the exact SOURCE values the atom references by
# index (e.g. matmul's A_i..A_j) — copied by CODE from the frozen problem, not
# re-typed by the splitter LLM. Defaults to None (no-op), so all other tasks
# are unaffected. The matmul study sets/clears this around each solve.
_SOURCE_RESOLVER = None  # type: ignore[var-annotated]


def set_source_resolver(fn) -> None:
    """Register (or clear, with None) the active source resolver. Callable as
    `fn(atom, atom_question_text) -> str`. Set before a solve; clear after."""
    global _SOURCE_RESOLVER
    _SOURCE_RESOLVER = fn


def get_source_resolver():
    return _SOURCE_RESOLVER


# ---------------------------------------------------------------------------
# Optional task-specific COMBINE RESOLVER (deterministic aggregation)
# ---------------------------------------------------------------------------
# A task may register a resolver that, given a composite atom's children (each
# with source_span, question and already-resolved answer), returns the parent
# block's answer DETERMINISTICALLY — bypassing the LLM combiner. For a
# running-state chain (e.g. matmul: M_t = M_{t-1} . A_t) the block answer is
# simply its LAST child's answer (the child covering the highest source index),
# so an LLM re-combination is redundant AND error-prone (it silently re-does
# the matrix product and corrupts a correct chained result). Returning a value
# skips the combiner call entirely (also saving call budget); returning None
# falls back to the LLM combiner. Defaults to None (no-op) — all other tasks
# are unaffected. The matmul study sets/clears this around each solve.
_COMBINE_RESOLVER = None  # type: ignore[var-annotated]


def set_combine_resolver(fn) -> None:
    """Register (or clear, with None) the active combine resolver. Callable as
    `fn(children) -> str | None`, where `children` is a list of dicts with keys
    `source_span` (tuple|None), `question` (str) and `answer` (str), in the
    splitter's emitted order. Return the block answer, or None to fall back to
    the LLM combiner. Set before a solve; clear after."""
    global _COMBINE_RESOLVER
    _COMBINE_RESOLVER = fn


def get_combine_resolver():
    return _COMBINE_RESOLVER


def _resolve_combine(children, atom_answers) -> "str | None":
    """Apply the registered combine resolver to a composite atom's children.

    `children` is a list of AtomicUnit; `atom_answers` maps atom_id → answer.
    Returns the deterministic block answer, or None to fall back to the LLM
    combiner (resolver unregistered, raised, or declined). A resolver may also
    decline (return None) when it can't determine the answer — e.g. the
    designated last child has no answer yet (budget-truncated) — so we don't
    mask an incomplete chain with a stale partial.
    """
    cr = get_combine_resolver()
    if cr is None:
        return None
    try:
        kids = [{
            "source_span": getattr(c, "source_span", None),
            "question": getattr(c, "question", ""),
            "answer": atom_answers.get(c.atom_id, ""),
        } for c in children]
        out = cr(kids)
        return out if (out is not None and str(out).strip()) else None
    except Exception:                                               # noqa: BLE001
        _log.exception("combine resolver failed; falling back to LLM combiner")
        return None


# ---------------------------------------------------------------------------
# Provenance carried by the policy across rollouts for the combine tree
# ---------------------------------------------------------------------------

@dataclass
class _BlockTrace:
    """One block-level provenance record, keyed by G' node_id."""

    block_node_id: str
    split_tree: dict | None = None
    atomic_answers: list[dict] = field(default_factory=list)
    combiner_attempts: list[dict] = field(default_factory=list)
    verifier_verdicts: list[dict] = field(default_factory=list)
    final_block_answer: str | None = None
    final_score: float = 0.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class PipelineExpansionPolicy:
    """MCTS expansion via the 4-phase pipeline. Drop-in for
    RewriteExpansionPolicy.

    Phase B uses the SAME executor LLM and the SAME score_subproblem
    clustering as the legacy path — we just call it once per atomic unit
    instead of once per whole block.
    """

    # Core components (mirror RewriteExpansionPolicy)
    llm: LLMClient                       # executor (used in Phase B)
    Gp: nx.DiGraph
    decomposer: ExplicitDecomposer
    embedder: Embedder
    kernel: EntailmentKernel
    verifier: Verifier
    cfg: Config
    rng: random.Random
    tracer: "Any | None" = None
    original_problem: str = ""

    # New components for the pipeline
    splitter: LLMSplitter | None = None
    combiner: LLMCombiner | None = None
    block_verifier: LLMBlockVerifier | None = None
    shared_memory: VectorSharedMemory | None = None

    # Block-trace accumulator (keyed by block node_id) for the combine tree.
    block_traces: dict[str, _BlockTrace] = field(default_factory=dict)

    # Per-node agent-call budget. One PipelineExpansionPolicy is constructed
    # per graph node (per orchestrator.solve), so this counter is shared
    # across that node's MCTS rollouts and caps its TOTAL agent calls. Lazily
    # initialised on the first expand() from cfg.pipeline.max_node_calls.
    call_budget: "CallBudget | None" = None

    # -------------------------------------------------------------------

    def _combine_reserve(self) -> int:
        """Calls to hold back for the mandatory Phase C/D so the node always
        produces a final answer: worst-case combiner samples across all retry
        rounds + one verifier call per round."""
        rounds = int(self.cfg.pipeline.combiner_retries) + 1
        return rounds * int(self.cfg.pipeline.K_combiner) + rounds

    def expand(self, state: SolutionState, depth: int, *,
               parent_u_s: float, K: int) -> ExpandedNodeInfo:
        # Lazily arm the per-node call budget (shared across this node's
        # rollouts). cfg.pipeline.max_node_calls is the hard ceiling.
        if self.call_budget is None:
            self.call_budget = CallBudget(
                limit=int(getattr(self.cfg.pipeline, "max_node_calls", 100)))
        # 1) Pick next subproblem (same machinery as RewriteExpansionPolicy)
        sp = self.decomposer.next_subproblem(
            state, self.Gp, granularity=granularity_for(parent_u_s),
        )
        if sp == ATOMIC or is_atomic(state, self.Gp):
            final = state.resolved[-1][1] if state.resolved else ""
            return ExpandedNodeInfo(
                classes=[ExpandedClass(
                    class_key="__terminal__", answer_text=final,
                    mass=1.0, u_s=parent_u_s if parent_u_s > 0 else 1.0,
                    terminal_hint=True,
                )],
                subproblem_text="__atomic__", bindings={},
            )

        # 2) Resolve upstream context. The shared memory has the committed
        #    block answers from previously-solved blocks; state.bindings has
        #    the resolved set for THIS rollout's path.
        resolved_ctx = self._resolved_context(state, sp.bindings)

        # 3) Phase A — Splitter
        split_tree = self._phase_a_split(
            block_node_id=sp.node_id, block_text=sp.text,
            resolved=resolved_ctx,
        )

        # 4) Phase B — Solve each atomic leaf in order; write back to memory
        atomic_qa = self._phase_b_solve(
            split_tree=split_tree,
            resolved=resolved_ctx, K=K,
            block_node_id=sp.node_id, depth=depth,
        )

        # 5) Phases C+D — Combiner & Verifier (with retry loop)
        candidate_text, candidate_answer, verdict, attempts = \
            self._phase_cd_combine_verify(
                original_question=sp.text,
                atomic_qa=atomic_qa,
                node_id=sp.node_id,
                top_atoms=split_tree.atoms,
            )

        # 6) Commit the block answer to shared memory (whatever the verdict
        #    — backtracking is signalled via low U_s, not by skipping the
        #    write, so the rest of the run can still see something here).
        if self.shared_memory is not None and candidate_answer:
            self.shared_memory.put(
                scope=f"block:{sp.node_id}",
                question=sp.text,
                answer=candidate_answer,
                score=verdict.score,
                source="pipeline_policy",
                depth=depth,
            )

        # 7) Convert into ExpandedClass shape so MCTS can PUCT.
        out_classes = self._build_expanded_classes(
            candidate_text=candidate_text,
            candidate_answer=candidate_answer,
            verdict=verdict, depth=depth, node_id=sp.node_id,
        )

        # 8) Update block-trace for combine_tree.json
        bt = self.block_traces.setdefault(
            sp.node_id, _BlockTrace(block_node_id=sp.node_id))
        bt.split_tree = split_tree.to_dict()
        bt.atomic_answers = [
            {"atom_id": atom_id, "question": q, "answer": a}
            for atom_id, q, a in atomic_qa
        ]
        bt.combiner_attempts = attempts
        bt.verifier_verdicts.append(verdict.to_dict())
        bt.final_block_answer = candidate_answer
        bt.final_score = verdict.score

        # 9) Side-channel tracer dump (mirrors the legacy policy's shape so
        #    the existing rollouts JSONL keeps working).
        self._stash_tracer(
            sp_node_id=sp.node_id, sp_text=sp.text,
            split_tree=split_tree, atomic_qa=atomic_qa,
            candidate_text=candidate_text, verdict=verdict,
            out_classes=out_classes, attempts=attempts,
        )

        return ExpandedNodeInfo(
            classes=out_classes,
            subproblem_text=sp.node_id,
            bindings=sp.bindings,
        )

    # =====================================================================
    # Phase implementations
    # =====================================================================

    def _resolved_context(self, state: SolutionState,
                           sp_bindings: dict[str, str]) -> dict[str, str]:
        """Build the resolved-context dict the Splitter sees.

        Includes (in order of preference):
          1. The shared memory's committed block answers (cross-block).
          2. This rollout's state.bindings (path-local resolved set).
          3. The decomposer-supplied sp.bindings.
        """
        out: dict[str, str] = {}
        if self.shared_memory is not None:
            out.update(self.shared_memory.all_blocks())
        if state.bindings:
            for k, v in state.bindings.items():
                if v is not None:
                    out[str(k)] = str(v)
        for k, v in sp_bindings.items():
            if v is not None:
                out[str(k)] = str(v)
        return out

    # ---------------------------------------------------------------------

    def _phase_a_split(self, *, block_node_id: str, block_text: str,
                        resolved: dict[str, str]) -> SplitTree:
        if self.splitter is None:
            # Fallback: emit a single-atom split tree. Pipeline degrades
            # gracefully to "executor on the whole block."
            from .splitter import AtomicUnit, SplitTree
            return SplitTree(
                block_node_id=block_node_id,
                root_question=block_text,
                atoms=[AtomicUnit(
                    atom_id=f"{block_node_id}/d0/a0",
                    question=block_text, depth=0,
                    is_atomic=True, split_reason="no_splitter_configured",
                )],
                n_llm_calls=0,
            )
        retrieved = self._semantic_retrieve_context(block_text)
        # Cap splitter recursion at a quarter of the node budget so it can
        # never starve leaf execution + combining; the shared budget object
        # also keeps the splitter from exceeding the node total.
        max_split_calls = None
        if self.call_budget is not None:
            max_split_calls = max(2, self.call_budget.limit // 4)
        return self.splitter.split(
            block_node_id=block_node_id, block_text=block_text,
            resolved=resolved, retrieved_context=retrieved,
            budget=self.call_budget, max_split_calls=max_split_calls,
        )

    def _semantic_retrieve_context(self, query: str) -> list[str]:
        """Top-k similar prior block answers to inject as context."""
        if self.shared_memory is None:
            return []
        topk = max(0, int(self.cfg.pipeline.sharedmem_topk))
        hits = self.shared_memory.search(query, k=topk, scope_prefix="block:")
        return [f"{entry.scope}: {entry.answer}" for entry, _ in hits]

    # ---------------------------------------------------------------------

    def _phase_b_solve(self, *, split_tree: SplitTree,
                        resolved: dict[str, str], K: int,
                        block_node_id: str,
                        depth: int) -> list[tuple[str, str, str]]:
        """Solve every atomic unit bottom-up, propagating answers upward.

        Walks the FULL split tree (leaves AND composites) in topological
        order using `SplitTree.topological_atom_order()`:

          * For LEAF atoms (no children): execute K samples via the
            executor LLM, cluster, take the best-U_s class's answer.
          * For COMPOSITE atoms (have children): synthesize an answer
            from the already-resolved child answers using the combiner
            LLM with n=1 (mini-combine). This is what makes the
            decomposition recursive — a composite's answer is the
            *composition* of its sub-answers, exactly like the user
            asked: "for each node, decompose till atomic, then solve
            from bottom and pass it to the top."

        REFS are honored: when an atom declares `refs: [0, 1]`, those
        indices into its same-level sibling list are looked up and the
        corresponding answers are textually inlined into the atom's
        prompt before execution.

        Returns a list of (atom_id, atom_question, atom_answer) tuples
        in solve order — so the FIRST entry is the deepest leaf and
        the LAST entry is the top-level root atom whose answer is the
        block's final answer.
        """
        K_exec = max(1, int(self.cfg.pipeline.K_executor))
        order = split_tree.topological_atom_order()
        sib_map = split_tree.siblings_map()
        parent_map = split_tree.parent_map()

        # atom_id → answer; lets later atoms (refs + parents) look up
        # earlier atoms' answers.
        atom_answers: dict[str, str] = {}
        atomic_qa: list[tuple[str, str, str]] = []

        reserve = self._combine_reserve()
        resolver = get_source_resolver()
        for atom in order:
            atom_q = self._inline_atom_refs(
                atom, sib_map, atom_answers, parent_map)

            # Index-reference tasks: inject the exact source values this atom
            # references by index (e.g. matmul A_i..A_j), copied by code from
            # the frozen problem. No-op when no resolver is registered.
            if resolver is not None:
                try:
                    atom_q = resolver(atom, atom_q)
                except Exception:                                   # noqa: BLE001
                    _log.exception("source resolver failed; using unresolved atom_q")

            if atom.children:
                # Composite — synthesize from already-resolved children.
                # First try a deterministic combine resolver (e.g. running-state
                # chains: block answer = last child's answer). It needs no LLM
                # call, so it is exact AND budget-free. Falls back to the LLM
                # mini-combine (one call, budget-gated) when unregistered or the
                # resolver declines (returns None).
                ans = _resolve_combine(atom.children, atom_answers)
                if ans is None:
                    if (self.call_budget is not None
                            and not self.call_budget.can_spend(1, reserve=reserve)):
                        ans = " | ".join(
                            atom_answers.get(c.atom_id, "") for c in atom.children
                            if atom_answers.get(c.atom_id)) or ""
                    else:
                        if self.call_budget is not None:
                            self.call_budget.spend(1)
                        ans = self._mini_combine_composite(
                            atom=atom, atom_q=atom_q, atom_answers=atom_answers,
                            block_node_id=block_node_id,
                        )
            else:
                # Leaf — execute via the executor LLM (K_exec calls). Once
                # the node budget is spent, leave the atom unsolved (empty)
                # so the combiner still runs on whatever was solved.
                if (self.call_budget is not None
                        and not self.call_budget.can_spend(K_exec, reserve=reserve)):
                    ans = ""
                else:
                    if self.call_budget is not None:
                        self.call_budget.spend(K_exec)
                    ans = self._execute_leaf_atom(
                        atom=atom, atom_q=atom_q, K_exec=K_exec,
                        block_node_id=block_node_id,
                    )

            atom_answers[atom.atom_id] = ans
            atomic_qa.append((atom.atom_id, atom.question, ans))

            # Write atomic answer to shared memory so later blocks /
            # synthesizers can retrieve it semantically. Scope-tagged
            # so it's distinguishable from committed block answers.
            if self.shared_memory is not None:
                try:
                    self.shared_memory.put(
                        scope=f"atom:{atom.atom_id}",
                        question=atom.question, answer=ans,
                        block=block_node_id,
                        depth=atom.depth,
                        composite=bool(atom.children),
                    )
                except Exception:                                   # noqa: BLE001
                    pass

        return atomic_qa

    # ----- inline refs ------------------------------------------------

    @staticmethod
    def _inline_atom_refs(atom, sib_map, atom_answers, parent_map=None) -> str:
        """Append the actual VALUES this atom depends on to its question text,
        so the executor sees concrete numbers rather than placeholders like
        "M_4 (from the previous step)".

        Two sources of dependency, both inlined:

          (1) SAME-LEVEL sibling refs — `atom.refs` are indices into the
              atom's own sibling list (a step that consumes an earlier
              sibling's answer).

          (2) CARRIED RUNNING STATE from the nearest ancestor block — the
              fix for state being severed at recursion boundaries. The entry
              (leading-edge / first) atom of a recursively-split sub-chain
              inherits the incoming state of the ancestor block it belongs to
              (e.g. a leaf `M_5 = (M_4 . A_5)` two levels deep gets M_4, which
              lives as a ref on its grandparent block, not as a same-level
              sibling). We walk UP the index-0 chain to the nearest ancestor
              that carries a ref and inline that resolved value.

        We append rather than substitute (the splitter's question text is
        free-form with no reliable placeholder to replace).
        """
        injected: list[str] = []
        sibs = sib_map.get(atom.atom_id) or []

        # (1) same-level sibling refs
        for ref in (atom.refs or []):
            try:
                idx = int(ref)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(sibs):
                tgt = sibs[idx]
                if tgt.atom_id == atom.atom_id:
                    continue
                ans = atom_answers.get(tgt.atom_id)
                if ans:
                    injected.append(
                        f"[sub-answer #{idx} from earlier atom "
                        f"\"{tgt.question[:80]}\"]: {ans}")

        # (2) carried running state — only for an entry atom that has no
        # same-level ref of its own supplying the incoming state.
        if parent_map is not None and not injected:
            cur = atom
            guard = 0
            while cur is not None and guard < 128:
                guard += 1
                cur_sibs = sib_map.get(cur.atom_id) or []
                # If THIS block carries refs, they ARE the incoming running
                # state for its leading-edge descendant (`atom`) — inline and
                # stop. Read refs BEFORE the leftmost check: a continuation
                # block like [6..10] is itself index-1 among the top blocks,
                # yet its refs (→ M_5) are exactly what its first leaf needs.
                if cur.refs:
                    for ref in cur.refs:
                        try:
                            idx = int(ref)
                        except (ValueError, TypeError):
                            continue
                        if (0 <= idx < len(cur_sibs)
                                and cur_sibs[idx].atom_id != cur.atom_id):
                            ans = atom_answers.get(cur_sibs[idx].atom_id)
                            if ans:
                                injected.append(
                                    "[carried running state to start this "
                                    "step from (result of \""
                                    f"{cur_sibs[idx].question[:80]}\")]: {ans}")
                    break
                # No refs here: climb only while `cur` is the leftmost child of
                # its parent, so `atom` stays the leading-edge descendant of the
                # ancestor we move to. A non-leftmost atom with no ref gets its
                # incoming state from a same-level ref (handled in (1)).
                if cur_sibs and cur_sibs[0].atom_id != cur.atom_id:
                    break
                cur = parent_map.get(cur.atom_id)

        if not injected:
            return atom.question
        return atom.question + "\n\n" + "\n".join(injected)

    # ----- leaf execution ---------------------------------------------

    def _execute_leaf_atom(self, *, atom, atom_q: str, K_exec: int,
                            block_node_id: str) -> str:
        """K-sample executor LLM call + cluster + take best class's answer."""
        prompt = build_executor_prompt(atom_q)
        try:
            gens = self.llm.generate(
                prompt, temperature=self.cfg.llm.temperature, n=K_exec)
        except Exception as e:                                      # noqa: BLE001
            _log.exception("Executor call failed for atom %s", atom.atom_id)
            self._log_executor_calls(
                prompt=prompt, gens=[], leaf=atom,
                block_node_id=block_node_id, K=K_exec, error=repr(e))
            return ""

        self._log_executor_calls(
            prompt=prompt, gens=gens, leaf=atom,
            block_node_id=block_node_id, K=K_exec, error=None)

        scored = score_subproblem(
            gens, embedder=self.embedder, kernel=self.kernel,
            verifier=self.verifier, subproblem=atom.question,
            cfg=self.cfg, supports_logprobs=self.llm.supports_logprobs,
        )
        best = scored.classes[0] if scored.classes else None
        if best is None:
            return ""
        rep_idx = best.klass.representative
        rep_text = scored.responses[rep_idx]
        rep_finish = (gens[rep_idx].finish_reason
                      if rep_idx < len(gens) else None)
        return (extract_answer(rep_text, finish_reason=rep_finish)
                or (rep_text if rep_finish != "max_tokens" else ""))

    # ----- composite synthesis (recursive bottom-up) ------------------

    def _mini_combine_composite(self, *, atom, atom_answers: dict,
                                  block_node_id: str,
                                  atom_q: str | None = None) -> str:
        """Synthesize a composite atom's answer from its already-resolved
        children. Uses the existing combiner LLM with n=1 (no clustering
        — the composite layer is cheap aggregation, not a search step).

        `atom_q` is the ref-inlined question (carrying any upstream running
        state); it is used as the combiner's `original_question` so the
        composition step is state-aware, not just a blind join of children.

        If no combiner is configured, falls back to a textual join of
        the children's answers (so callers still get *something*).
        """
        original_question = atom_q or atom.question
        child_qas: list[tuple[str, str]] = []
        for c in atom.children:
            ans = atom_answers.get(c.atom_id, "")
            child_qas.append((c.question, ans))

        if self.combiner is None or not child_qas:
            # Fallback: textual join. Better than returning empty.
            joined = " | ".join(a for _, a in child_qas if a)
            return joined or ""

        # One combiner sample is enough for the composite layer; the
        # full K_combiner only runs at the BLOCK-level Phase C. We do
        # this by temporarily forcing K=1 around the combiner call.
        saved_K = self.cfg.pipeline.K_combiner
        try:
            object.__setattr__(self.cfg.pipeline, "K_combiner", 1)
            gens = self.combiner.combine(
                original_question=original_question,
                atomic_answers=child_qas,
                verifier_hint=None,
                node_id=block_node_id,
            )
        except Exception:                                           # noqa: BLE001
            gens = []
        finally:
            object.__setattr__(self.cfg.pipeline, "K_combiner", saved_K)

        if not gens:
            return " | ".join(a for _, a in child_qas if a) or ""
        text = gens[0].text or ""
        finish = gens[0].finish_reason
        return (extract_answer(text, finish_reason=finish)
                or (text if finish != "max_tokens" else ""))

    # ---------------------------------------------------------------------

    def _phase_cd_combine_verify(
        self, *, original_question: str,
        atomic_qa: list[tuple[str, str, str]],
        node_id: str,
        top_atoms: list | None = None,
    ) -> tuple[str, str, VerifierVerdict, list[dict]]:
        """Combiner + verifier loop. Returns
        (best_candidate_full_text, extracted_answer, verdict, attempts_log)."""
        # Deterministic block combine (e.g. running-state chains): the block
        # answer is a code-computable function of the top-level atoms' answers
        # (for matmul, the last one), so we bypass the LLM combiner + verifier
        # entirely — exact, and it frees the block-level K_combiner budget for
        # execution. Only fires when a combine resolver is registered AND
        # resolves; otherwise the normal combiner/verifier loop runs.
        if top_atoms:
            answers_by_id = {aid: a for aid, _, a in atomic_qa}
            det = _resolve_combine(top_atoms, answers_by_id)
            if det is not None:
                verdict = VerifierVerdict(score=1.0, is_consistent=True,
                                           issues=[])
                attempts = [{"attempt": 0, "n_samples": 0,
                             "deterministic_combine": True,
                             "verdict": verdict.to_dict()}]
                return det, det, verdict, attempts

        atoms_for_combiner = [(q, a) for _, q, a in atomic_qa]

        attempts: list[dict] = []
        verifier_hint: str | None = None
        candidate_text = ""
        candidate_answer = ""
        verdict = VerifierVerdict(score=0.0, is_consistent=False)

        accept_th = self.cfg.pipeline.verifier_accept
        retry_th = self.cfg.pipeline.verifier_retry
        max_retries = self.cfg.pipeline.combiner_retries

        K_comb = int(self.cfg.pipeline.K_combiner)
        for attempt in range(max_retries + 1):
            # Combiner: K samples (budget-gated). When the node's call
            # budget can't afford another combiner round, stop retrying and
            # keep the best candidate produced so far.
            if self.combiner is None:
                gens = []
            elif (self.call_budget is not None
                    and not self.call_budget.can_spend(K_comb)):
                break
            else:
                if self.call_budget is not None:
                    self.call_budget.spend(K_comb)
                gens = self.combiner.combine(
                    original_question=original_question,
                    atomic_answers=atoms_for_combiner,
                    verifier_hint=verifier_hint,
                    node_id=node_id,
                )
            if not gens:
                candidate_text = ""
                candidate_answer = ""
                verdict = VerifierVerdict(score=0.0, is_consistent=False,
                                           issues=["combiner returned no candidates"])
                attempts.append({"attempt": attempt, "n_samples": 0,
                                 "verdict": verdict.to_dict()})
                break

            # Cluster combiner samples via existing semantic-density.
            scored = score_subproblem(
                gens, embedder=self.embedder, kernel=self.kernel,
                verifier=self.verifier, subproblem=original_question,
                cfg=self.cfg, supports_logprobs=self.llm.supports_logprobs,
            )
            best = scored.classes[0] if scored.classes else None
            if best is None:
                candidate_text = (gens[0].text or "") if gens else ""
            else:
                candidate_text = scored.responses[best.klass.representative]
            rep_finish = (gens[best.klass.representative].finish_reason
                          if best is not None else None)
            candidate_answer = (
                extract_answer(candidate_text, finish_reason=rep_finish)
                or (candidate_text if rep_finish != "max_tokens" else "")
            )

            # Verifier
            if self.block_verifier is None:
                # Without a verifier, treat the candidate as accepted at
                # the existing semantic-density score (no retry loop).
                verdict = VerifierVerdict(score=0.7, is_consistent=True)
                attempts.append({"attempt": attempt, "n_samples": len(gens),
                                 "candidate_answer": candidate_answer,
                                 "verdict": verdict.to_dict()})
                break

            if (self.call_budget is not None
                    and not self.call_budget.can_spend(1)):
                # No budget left to verify — accept the candidate as-is.
                verdict = VerifierVerdict(
                    score=0.7, is_consistent=True,
                    issues=["verifier skipped: node call budget exhausted"])
                attempts.append({"attempt": attempt, "n_samples": len(gens),
                                 "candidate_answer": candidate_answer,
                                 "verdict": verdict.to_dict()})
                break

            if self.call_budget is not None:
                self.call_budget.spend(1)
            verdict = self.block_verifier.verify(
                original_question=original_question,
                atomic_answers=atoms_for_combiner,
                candidate_answer=candidate_answer,
                node_id=node_id,
            )
            attempts.append({"attempt": attempt, "n_samples": len(gens),
                             "candidate_answer": candidate_answer,
                             "verdict": verdict.to_dict()})

            # Accept?
            if verdict.score >= accept_th:
                # If verifier offered a trivial fix, honor it.
                if verdict.corrected_answer and not verdict.is_consistent:
                    candidate_answer = verdict.corrected_answer
                break

            # Retry combiner with hint?
            if verdict.score >= retry_th and attempt < max_retries:
                verifier_hint = verdict.hint_text()
                continue

            # Below retry threshold — accept this attempt as-is. MCTS will
            # see a low U_s and naturally explore siblings on next rollout
            # (which gives the splitter / executor a fresh chance via the
            # transposition table miss on different bindings).
            break

        return candidate_text, candidate_answer, verdict, attempts

    # ---------------------------------------------------------------------

    def _build_expanded_classes(
        self, *, candidate_text: str, candidate_answer: str,
        verdict: VerifierVerdict, depth: int, node_id: str,
    ) -> list[ExpandedClass]:
        """Project the (candidate, verdict) pair into MCTS classes.

        We emit a SINGLE class per expansion — the combiner-selected
        answer. Its U_s is the verifier score (so backtracking is
        automatic when the score is low).
        """
        u_s = float(verdict.score)
        mass = 1.0
        # Reserve a low U_s tail when the verifier is below backtrack
        # threshold so MCTS strongly prefers any alternative sibling.
        if verdict.score < self.cfg.pipeline.verifier_backtrack:
            u_s = max(0.0, u_s * 0.5)
        return [ExpandedClass(
            class_key=f"d{depth}_{node_id}_pipeline_v{int(verdict.score*100):02d}",
            answer_text=candidate_answer or "",
            mass=mass,
            u_s=u_s,
        )]

    # =====================================================================
    # Tracer plumbing
    # =====================================================================

    def _log_executor_calls(self, *, prompt: str, gens: list,
                             leaf, block_node_id: str, K: int,
                             error: str | None) -> None:
        if self.tracer is None:
            return
        exec_model = getattr(self.cfg.llm, "model", None)
        if not gens:
            try:
                self.tracer.log_agent_call(
                    role="executor", prompt=prompt, response="",
                    model=exec_model, finish_reason=None,
                    node_id=block_node_id,
                    extras={"atom_id": leaf.atom_id, "K": K, "error": error},
                )
            except Exception:                                       # noqa: BLE001
                pass
            return
        for sidx, g in enumerate(gens):
            try:
                self.tracer.log_agent_call(
                    role="executor", prompt=prompt,
                    response=g.text or "", model=exec_model,
                    finish_reason=g.finish_reason,
                    node_id=block_node_id, sample_idx=sidx,
                    extras={"atom_id": leaf.atom_id,
                             "atom_depth": leaf.depth,
                             "atom_is_atomic": leaf.is_atomic,
                             "K": K},
                )
            except Exception:                                       # noqa: BLE001
                pass

    def _stash_tracer(self, *, sp_node_id: str, sp_text: str,
                       split_tree: SplitTree,
                       atomic_qa: list[tuple[str, str, str]],
                       candidate_text: str, verdict: VerifierVerdict,
                       out_classes: list[ExpandedClass],
                       attempts: list[dict]) -> None:
        if self.tracer is None:
            return
        try:
            samples_dump = [
                {
                    "idx": i,
                    "text": (text or ""),
                    "extracted_answer": ans,
                    "weight": 1.0 / max(1, len(atomic_qa)),
                    "class_idx": 0,
                    "finish_reason": "atom",
                    "truncated": False,
                    "atom_id": atom_id,
                }
                for i, (atom_id, text, ans) in enumerate(atomic_qa)
            ]
            classes_dump = [
                {
                    "idx": 0,
                    "members": list(range(len(atomic_qa))),
                    "representative_idx": 0,
                    "representative_text": (candidate_text or "")[:400],
                    "mass": 1.0,
                    "U_s": float(verdict.score),
                    "SD_mean": 0.0,
                    "v_mean": float(verdict.score),
                    "n_members": len(atomic_qa),
                }
            ]
            self.tracer.stash_expansion(
                subproblem_id=sp_node_id,
                subproblem_text=sp_text,
                prompt_excerpt="(pipeline policy — see agent_calls.jsonl "
                                "for full executor/splitter/combiner prompts)",
                bindings={"__pipeline__": "true",
                          "__n_atoms__": str(len(atomic_qa)),
                          "__verifier_score__": f"{verdict.score:.2f}",
                          "__verifier_consistent__":
                              "yes" if verdict.is_consistent else "no",
                          "__combiner_attempts__": str(len(attempts))},
                samples=samples_dump,
                classes=classes_dump,
                oracle_mode="pipeline",
                queried_pairs=0,
            )
        except Exception:                                           # noqa: BLE001
            pass

    # =====================================================================
    # Combine-tree builder
    # =====================================================================

    def build_combine_tree_payload(self) -> dict:
        """Return a JSON-safe payload describing every block's pipeline
        provenance — split tree, atomic answers, combiner attempts,
        verifier verdicts, final committed answer.

        Consumed by `solve_multi` to assemble the per-question
        `combine_tree.json`.
        """
        return {
            "blocks": [
                {
                    "block_node_id": bt.block_node_id,
                    "split_tree": bt.split_tree,
                    "atomic_answers": list(bt.atomic_answers),
                    "combiner_attempts": list(bt.combiner_attempts),
                    "verifier_verdicts": list(bt.verifier_verdicts),
                    "final_block_answer": bt.final_block_answer,
                    "final_score": bt.final_score,
                }
                for bt in self.block_traces.values()
            ],
        }
