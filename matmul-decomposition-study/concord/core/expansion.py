"""Concrete ExpansionPolicy: decomposer + executor + confidence stack.

Role split (decomposer + executor):

  - The DECOMPOSER (uses `clients.decomposition` LLM) produces a CLEAN,
    self-contained sub-question from the raw G' template + the resolved
    state. It inlines parent answers, resolves any embedded arithmetic,
    and strips all `node_K` / `[For this value...]` meta-syntax.

  - The EXECUTOR (uses `clients.execution` LLM) receives ONLY the clean
    sub-question — no original problem, no parent facts, no scaffolding
    beyond a one-line instruction. It returns K candidate solutions.

  - `score_subproblem` clusters those K samples, computes U_s per class,
    and returns them as ExpandedClass children for PUCT. mass(class)·U_s
    is the prior (spec line 21).

The result: each subproblem the executor sees is "independently solvable,
isolated, and has an answer" — the executor cannot accidentally produce
a new question instead of a solution because it never sees the meta-
structure of the parent problem.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .config import Config
from .confidence import (
    Embedder,
    EntailmentKernel,
    Verifier,
    score_subproblem,
)
from .decomposer import LLMDecomposer, template_needs_decomposing
from .llm.client import LLMClient
from .search import ExpandedClass, ExpandedNodeInfo
from .structure import ATOMIC, ExplicitDecomposer, granularity_for, is_atomic
from .types import SolutionState


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_NODE_REF_PAT = re.compile(
    r"\[\s*for\s+this\s+value\s+use\s+the\s+answer\s+from\s+problem\s+node_(\d+)([^\]]*)\]",
    flags=re.IGNORECASE,
)
_NODE_REF_PLAIN = re.compile(
    r"answer\s+from\s+problem\s+node_(\d+)",
    flags=re.IGNORECASE,
)


def _resolve_node_refs(subproblem_text: str, bindings: dict[str, str]) -> str:
    """Substitute bracketed `[For this value use the answer from problem
    node_K and ...]` references with the actual parent answer. The trailing
    arithmetic instruction inside the bracket is left intact for the LLM
    to apply; we only replace the *reference* itself.
    """
    def replace(m: re.Match[str]) -> str:
        k = m.group(1)
        tail = m.group(2).strip()
        key = f"node_{k}"
        val = bindings.get(key)
        if val is None:
            # try SCC-prefixed key (M2 condense renames to scc_N containing node_K)
            for binding_key, binding_val in bindings.items():
                if binding_key.startswith("scc_") and key in binding_key:
                    val = binding_val
                    break
        if val is None:
            return m.group(0)   # leave unresolved if we don't have it
        if tail:
            return f"(taking the value {val} from problem node_{k} {tail})"
        return f"(the value {val})"

    out = _NODE_REF_PAT.sub(replace, subproblem_text)
    # Plain references like "the answer from problem node_2" outside brackets.
    def replace2(m: re.Match[str]) -> str:
        k = m.group(1)
        key = f"node_{k}"
        val = bindings.get(key)
        if val is None:
            return m.group(0)
        return f"value {val} from problem node_{k}"
    out = _NODE_REF_PLAIN.sub(replace2, out)
    return out


def build_executor_prompt(clean_question: str) -> str:
    """The prompt the executor sees.

    Principle: the answerer gets ONLY the question. We add
    a one-line instruction so the answer is reliably extractable, and that's
    it — no original-problem context, no parent facts, no `node_K` meta-syntax.
    """
    return (
        f"{clean_question.strip()}\n\n"
        f"Provide your final answer on its own line in exactly this format:\n"
        f"solution = <your final answer>"
    )


# Kept under its old name for any external caller that still imports it,
# but the policy itself uses `build_executor_prompt` now.
def build_subproblem_prompt(subproblem_text: str,
                            bindings: dict[str, str]) -> str:
    """Legacy single-shot prompt (raw substitution + scaffolding).

    Used as a fallback when no LLMDecomposer is configured. New callers
    should use `build_executor_prompt(clean_question)` instead.
    """
    resolved = _resolve_node_refs(subproblem_text, bindings)
    return build_executor_prompt(resolved)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_SOLUTION_LINE_RE = re.compile(r"solution\s*=\s*(.+)", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:=]\s*(.+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*answer\s*[:=]\s*(.+)", re.IGNORECASE | re.MULTILINE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

# Phrases that indicate the model is still asking, not answering — used to
# detect when extraction would otherwise return prompt-like garbage.
_PROMPT_TELLS = (
    "solve the following",
    "subproblem:",
    "your final answer",
    "<mock:",
    "for this value use the answer",
)


def _looks_like_a_prompt(s: str) -> bool:
    lo = s.lower()
    return any(tell in lo for tell in _PROMPT_TELLS)


def extract_answer(text: str, finish_reason: str | None = None) -> str:
    """Best-effort answer extraction from a model response.

    Tries, in order:
      1. The LAST `solution = X` line (preferred; compatible with longcot).
      2. A `\\boxed{X}` LaTeX answer.
      3. A `final answer: X` line.
      4. An `answer: X` line at the start of a line.
      5. The last non-empty line — BUT only if it doesn't look like prompt
         scaffolding being echoed back (mock LLM, raw template text, etc.).
      6. Empty string when nothing usable is found — caller should treat
         this as an extraction failure, not as a wrong answer.

    When `finish_reason == "max_tokens"`, clauses 5 is skipped: a truncated
    response that produced no structured answer line is treated as extraction
    failure, not as "whatever was on the last cut-off line." This prevents
    mid-derivation fragments from being promoted to "the answer."
    """
    if not text:
        return ""

    truncated = (finish_reason == "max_tokens")

    # 1. solution = X
    for line in reversed(text.splitlines()):
        m = _SOLUTION_LINE_RE.search(line)
        if m:
            ans = m.group(1).strip().rstrip(". ")
            if ans and not _looks_like_a_prompt(ans):
                return ans

    # 2. \boxed{X}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    # 3. final answer: X
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(". ")

    # 4. answer: X
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(". ")

    # 5. last non-empty line that doesn't look like prompt echo — but ONLY
    # if the response actually finished. A truncated response that never
    # produced a structured answer line is an extraction failure: the
    # "last line" is just whatever the model was mid-writing.
    if truncated:
        return ""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if not _looks_like_a_prompt(line):
            return line

    # 6. extraction failed
    return ""


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------

@dataclass
class RewriteExpansionPolicy:
    """The production expansion policy: decomposer + executor + confidence stack."""

    llm: LLMClient                     # executor LLM (clients.execution)
    Gp: nx.DiGraph
    decomposer: ExplicitDecomposer
    embedder: Embedder
    kernel: EntailmentKernel
    verifier: Verifier
    cfg: Config
    rng: random.Random
    tracer: "Any | None" = None
    # LLM-driven sub-question synthesizer. When None we fall back to the
    # raw substitution (legacy behavior). The orchestrator constructs an
    # LLMDecomposer from `clients.decomposition` when wiring this up.
    sub_decomposer: "LLMDecomposer | None" = None
    # Full original problem text — fed to the decomposer so it can write
    # self-contained questions without echoing the original problem to the
    # executor. Stripped from executor prompts.
    original_problem: str = ""

    def expand(self, state: SolutionState, depth: int, *,
               parent_u_s: float, K: int) -> ExpandedNodeInfo:
        # 1. Pick next subproblem
        sp = self.decomposer.next_subproblem(
            state, self.Gp, granularity=granularity_for(parent_u_s),
        )
        if sp == ATOMIC or is_atomic(state, self.Gp):
            # Wrap the already-resolved final answer as a single-class
            # terminal so PUCT has something to back up through.
            final = state.resolved[-1][1] if state.resolved else ""
            return ExpandedNodeInfo(
                classes=[ExpandedClass(
                    class_key="__terminal__", answer_text=final,
                    mass=1.0, u_s=parent_u_s if parent_u_s > 0 else 1.0,
                    terminal_hint=True,
                )],
                subproblem_text="__atomic__", bindings={},
            )

        # 2a. DECOMPOSE — produce a clean, self-contained question for the
        # executor. Either via the LLM decomposer (preferred when the template
        # has references to resolved subtasks, or when an LLM decomposer is
        # configured) or via legacy raw substitution (fallback).
        raw_template = sp.text
        bindings_full = {**state.bindings, **sp.bindings}
        if self.sub_decomposer is not None and (
            template_needs_decomposing(raw_template) or state.resolved
        ):
            # Provide the decomposer with the (label, answer) tuples it can
            # inline. Labels are the G' node ids the bindings were keyed on.
            clean_question = self.sub_decomposer.make_subquestion(
                template_text=raw_template,
                resolved=list(state.resolved),
                original_problem=self.original_problem,
                node_id=sp.node_id,
            )
        else:
            # No decomposer or no parent references — do legacy substitution.
            clean_question = _resolve_node_refs(raw_template, bindings_full)

        # 2b. EXECUTE — answerer sees ONLY the clean question.
        prompt = build_executor_prompt(clean_question)
        gens = self.llm.generate(prompt, temperature=self.cfg.llm.temperature,
                                  n=K)

        # 2c. Log EVERY executor sample to agent_calls.jsonl — one line per
        # call, with the full prompt and full response so `tail -F` of that
        # file gives an unredacted, real-time view of what the executor is
        # producing for each subproblem.
        if self.tracer is not None:
            exec_model = getattr(self.cfg.llm, "model", None)
            for sidx, g in enumerate(gens):
                try:
                    self.tracer.log_agent_call(
                        role="executor",
                        prompt=prompt,
                        response=g.text or "",
                        model=exec_model,
                        finish_reason=g.finish_reason,
                        node_id=sp.node_id,
                        sample_idx=sidx,
                        extras={
                            "clean_question": clean_question,
                            "raw_template": raw_template,
                            "depth": depth,
                            "K": K,
                        },
                    )
                except Exception:                                   # noqa: BLE001
                    pass

        # 3. Score — cluster + SD + verifier
        scored = score_subproblem(
            gens, embedder=self.embedder, kernel=self.kernel,
            verifier=self.verifier, subproblem=sp.text, cfg=self.cfg,
            supports_logprobs=self.llm.supports_logprobs,
        )

        # 4. Class -> ExpandedClass; answer_text uses extracted "solution ="
        out_classes: list[ExpandedClass] = []
        for sc in scored.classes:
            rep_idx = sc.klass.representative
            rep_text = scored.responses[rep_idx]
            rep_finish = (gens[rep_idx].finish_reason
                          if rep_idx < len(gens) else None)
            answer = extract_answer(rep_text, finish_reason=rep_finish)
            # A truncated sample with no structured answer line is NOT a
            # valid answer — surface it as the empty marker so downstream
            # gates / aggregation can drop it instead of treating the
            # mid-derivation fragment as the chosen answer.
            if not answer and rep_finish == "max_tokens":
                answer_text = ""
            else:
                answer_text = answer or rep_text
            out_classes.append(ExpandedClass(
                class_key=f"d{depth}_{sp.node_id}_{rep_idx}",
                answer_text=answer_text,
                mass=float(sc.mass),
                u_s=float(sc.U_s),
            ))

        # 5. Side-channel for the tracer — full expansion provenance.
        if self.tracer is not None:
            # Map each sample to its class index for the viewer.
            sample_class_idx: dict[int, int] = {}
            for ci, sc in enumerate(scored.classes):
                for m in sc.klass.members:
                    sample_class_idx[m] = ci
            samples_dump = [
                {
                    "idx": idx,
                    # Full executor response — no truncation. The earlier
                    # 400-char cap meant long contest-math derivations
                    # never appeared in the trace. The agent_calls log
                    # also gets a copy with full prompt+response.
                    "text": (g.text or ""),
                    "extracted_answer": extract_answer(
                        g.text or "", finish_reason=g.finish_reason),
                    "weight": float(scored.weights[idx])
                              if idx < len(scored.weights) else None,
                    "class_idx": sample_class_idx.get(idx),
                    "finish_reason": g.finish_reason,
                    "truncated": (g.finish_reason == "max_tokens"),
                }
                for idx, g in enumerate(gens)
            ]
            classes_dump = [
                {
                    "idx": ci,
                    "members": sc.klass.members,
                    "representative_idx": sc.klass.representative,
                    "representative_text": sc.klass.representative_text[:300],
                    "mass": float(sc.mass),
                    "U_s": float(sc.U_s),
                    "SD_mean": float(sc.SD_mean),
                    "v_mean": float(sc.v_mean),
                    "n_members": len(sc.klass.members),
                }
                for ci, sc in enumerate(scored.classes)
            ]
            try:
                self.tracer.stash_expansion(
                    subproblem_id=sp.node_id,
                    # Carry the CLEAN executor-facing question — that's what
                    # the model actually saw. The raw template is shown
                    # alongside in bindings for diagnosis.
                    subproblem_text=clean_question,
                    prompt_excerpt=prompt,
                    bindings={
                        **bindings_full,
                        "__raw_template__": raw_template[:400],
                        "__decomposed__": "yes" if (
                            self.sub_decomposer is not None
                            and (template_needs_decomposing(raw_template)
                                  or state.resolved)
                        ) else "no",
                    },
                    samples=samples_dump,
                    classes=classes_dump,
                    oracle_mode=scored.oracle_mode,
                    queried_pairs=scored.queried_pairs,
                )
            except Exception:                                       # noqa: BLE001
                pass

        return ExpandedNodeInfo(
            classes=out_classes,
            subproblem_text=sp.node_id,    # store the G' node id
            bindings=sp.bindings,
        )
