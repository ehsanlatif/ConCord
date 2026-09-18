"""LLM-driven dependency graph + cross-reference resolution.

This module replaces two brittle regex steps that the multi-block solver
used to depend on:

  * **dependency discovery** — the old ``from\\s+problem\\s+node_(\\d+)``
    regex only caught references phrased *exactly* that way and silently
    dropped paraphrased ones, and

  * **cross-reference substitution** — the old ``_substitute_refs`` merely
    appended ``(whose committed answer was: X)`` after a reference while
    leaving the literal ``[For this value use ... and subtract N]``
    placeholder *bracket* sitting inside the LaTeX. The executor could not
    parse that (``\\sqrt{[For this value use ...]}`` is not a number), so
    every terminal was gated-fail and the block came back with
    ``"no coherent solution; all terminals rejected"``.

``LLMGraphResolver`` does both jobs with the ``decomposition`` role model:

  ``build_graph(problem_text, blocks) -> GraphSpec``
      One LLM call reads the whole multi-block problem and returns a clean
      DAG: every node id plus the exact list of node ids it depends on.
      The result is reconciled against the deterministically-parsed block
      ids (so the LLM can neither invent nor drop a node) and unioned with
      the regex-detected references as a safety floor. If the call fails or
      yields a cyclic / unusable graph, we fall back to the pure regex
      graph.

  ``resolve_block(node_id, block_text, dep_answers) -> ResolvedBlock``
      One LLM call per block whose body references an earlier node. Given
      the committed answers of the block's dependencies, it computes each
      placeholder's concrete value (feature extraction + arithmetic) and
      returns the fully self-contained question with every placeholder
      replaced by a literal value. Falls back to the parenthetical
      annotation when the call fails.

Both calls degrade gracefully: a mock LLM (or any provider that returns
unparseable text) simply triggers the regex / substitution fallback, so
the existing mock-based tests keep their semantics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import networkx as nx

from ..llm.client import LLMClient
from .graph import NODE_REF_RE, SubproblemNode

_log = logging.getLogger(__name__)


# A block "needs resolution" if it mentions another node anywhere in its
# body — every cross-reference in the dataset is phrased "... from problem
# node_K ...", usually inside a `[For this value use ...]` bracket.
def _mentions_node_ref(text: str) -> bool:
    return bool(NODE_REF_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GraphSpec:
    """A clean dependency graph for a multi-block problem.

    ``deps`` maps every block id to the list of block ids it depends on
    (the ids whose committed answer must be known first). ``summaries`` is
    an optional one-line description per node (for the trace / viewer).
    ``source`` records whether the edges came from the LLM or the regex
    fallback.
    """

    deps: dict[str, list[str]]
    summaries: dict[str, str] = field(default_factory=dict)
    source: str = "llm"

    def to_digraph(self) -> nx.DiGraph:
        """Build a DiGraph with edges dep -> dependent (solve order)."""
        G: nx.DiGraph = nx.DiGraph()
        for nid, deps in self.deps.items():
            G.add_node(nid, summary=self.summaries.get(nid, ""))
        for nid, deps in self.deps.items():
            for d in deps:
                if d in G.nodes and d != nid:
                    G.add_edge(d, nid)
        return G


@dataclass
class ResolvedBlock:
    """Result of resolving one block's cross-references.

    ``text`` is the rewritten, self-contained question body (placeholders
    replaced by literal values). ``source`` is "llm" when the resolver LLM
    produced it, "fallback" when we fell back to parenthetical annotation,
    and "noop" when the block had nothing to resolve.
    """

    text: str
    source: str = "llm"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GRAPH_SYSTEM = (
    "You are analysing a multi-part problem made of blocks labelled\n"
    "`Problem node_K:`. Some blocks reference the committed ANSWER (or a\n"
    "feature of the answer) of an earlier block — almost always via a\n"
    "bracketed placeholder like `[For this value use ... from problem\n"
    "node_K ...]`.\n\n"
    "Build the DEPENDENCY GRAPH: for every node, list the node ids whose\n"
    "answer must be known before that node can be solved.\n\n"
    "Reply with STRICT JSON only (no markdown fences, no prose) of shape:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"id": "node_0", "depends_on": [], "summary": "<= 10 words"},\n'
    '    {"id": "node_1", "depends_on": ["node_0"], "summary": "..."}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "  - Include EVERY node exactly once, using the exact id from its header.\n"
    "  - `depends_on` may only contain ids that also appear as nodes.\n"
    "  - A node never depends on itself.\n"
    "  - Only list a dependency when the block genuinely consumes that\n"
    "    node's answer. Do not invent transitive edges.\n"
)

_RESOLVE_SYSTEM = (
    "You rewrite ONE sub-problem so that it becomes fully self-contained.\n\n"
    "The sub-problem text contains one or more PLACEHOLDERS that reference\n"
    "the committed answers of earlier sub-problems. A placeholder looks like\n"
    "  [For this value use <description> from problem node_K ...]\n"
    "or any bracketed instruction mentioning `from problem node_K`.\n\n"
    "You are given the committed answer of every referenced node. For EACH\n"
    "placeholder you must:\n"
    "  1. Identify which node(s) it references.\n"
    "  2. Extract the specific feature it asks for — e.g. 'the exponent of\n"
    "     2', 'the denominator of the reduced fraction', 'the x-coordinate\n"
    "     of the second ordered pair', or simply 'the answer'.\n"
    "  3. Apply any arithmetic the placeholder states — e.g. 'and subtract\n"
    "     51', 'and add 42'.\n"
    "  4. Replace the ENTIRE bracketed placeholder (including its `[` and\n"
    "     `]`) with the single resulting literal value (a number or short\n"
    "     expression). Leave NO bracket fragments behind.\n\n"
    "Hard rules:\n"
    "  - Do NOT solve the sub-problem itself.\n"
    "  - Do NOT change any other text, LaTeX, or numbers. Only the\n"
    "    placeholders change.\n"
    "  - If a referenced answer is missing/unknown, still produce your best\n"
    "    literal value but never leave a `[...]` placeholder in the output.\n\n"
    "Output ONLY the rewritten sub-problem between these markers, with\n"
    "nothing before or after them:\n"
    "<<<RESOLVED>>>\n"
    "<rewritten sub-problem>\n"
    "<<<END>>>\n"
)


def _format_dep_answers(dep_answers: dict[str, str]) -> str:
    if not dep_answers:
        return "(none)"
    def _key(k: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", k)
        return (int(m.group(1)) if m else 1_000_000, k)
    lines = []
    for k in sorted(dep_answers, key=_key):
        v = dep_answers[k]
        lines.append(f"  {k}: {v if v not in (None, '') else '(could not be determined)'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

@dataclass
class LLMGraphResolver:
    """LLM-backed dependency graph builder + cross-reference resolver.

    A thin wrapper around one ``LLMClient`` (the ``decomposition`` role).
    Both public methods are total: they never raise — on any error they
    return a regex / substitution fallback so the multi-block solve keeps
    running.
    """

    llm: LLMClient
    temperature: float = 0.0
    model_name: str | None = None
    tracer: "object | None" = None

    # ----- graph building -------------------------------------------------

    def build_graph(self, problem_text: str,
                    blocks: list[SubproblemNode]) -> GraphSpec:
        """Return a clean dependency GraphSpec for the parsed blocks."""
        ids = [b.node_id for b in blocks]
        id_set = set(ids)
        regex_deps = {b.node_id: [r for r in b.refs if r in id_set and r != b.node_id]
                      for b in blocks}

        llm_deps: dict[str, list[str]] | None = None
        summaries: dict[str, str] = {}
        try:
            text = self._call(_GRAPH_SYSTEM + "\n\nProblem:\n" + problem_text.strip()
                              + "\n\nReturn ONLY the JSON object.", role="graph")
            parsed = self._parse_graph_json(text, id_set)
            if parsed is not None:
                llm_deps, summaries = parsed
        except Exception:                                            # noqa: BLE001
            _log.exception("LLMGraphResolver.build_graph failed; using regex")
            llm_deps = None

        if llm_deps is None:
            return GraphSpec(deps=regex_deps, summaries={}, source="regex_fallback")

        # Union the LLM edges with the regex-detected refs as a safety floor:
        # the regex can't drop a literal `from problem node_K`, and the LLM
        # can add paraphrased edges the regex missed.
        merged: dict[str, list[str]] = {}
        for nid in ids:
            seen: list[str] = []
            for d in list(llm_deps.get(nid, [])) + list(regex_deps.get(nid, [])):
                if d in id_set and d != nid and d not in seen:
                    seen.append(d)
            merged[nid] = seen

        spec = GraphSpec(deps=merged, summaries=summaries, source="llm")
        # Guard: if the merged graph is cyclic, the LLM almost certainly
        # added a bad edge — fall back to the regex graph (which, for this
        # dataset, is acyclic by construction).
        if not nx.is_directed_acyclic_graph(spec.to_digraph()):
            _log.warning("LLM dependency graph was cyclic; using regex graph")
            return GraphSpec(deps=regex_deps, summaries=summaries,
                             source="regex_fallback_cyclic")
        return spec

    @staticmethod
    def _parse_graph_json(text: str,
                          id_set: set[str]) -> tuple[dict[str, list[str]],
                                                     dict[str, str]] | None:
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        nodes = obj.get("nodes") if isinstance(obj, dict) else None
        if not isinstance(nodes, list):
            return None
        deps: dict[str, list[str]] = {}
        summaries: dict[str, str] = {}
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            nid = str(entry.get("id", "")).strip()
            if nid not in id_set:
                continue
            raw = entry.get("depends_on") or []
            if isinstance(raw, str):
                raw = [raw]
            deps[nid] = [str(d).strip() for d in raw
                         if str(d).strip() in id_set and str(d).strip() != nid]
            summ = entry.get("summary")
            if summ:
                summaries[nid] = str(summ).strip()
        # Require that the LLM covered (almost) all nodes; otherwise the
        # response is too partial to trust over the regex.
        if not deps or len(deps) < max(1, len(id_set) // 2):
            return None
        return deps, summaries

    # ----- per-block resolution ------------------------------------------

    def resolve_block(self, node_id: str, block_text: str,
                      dep_answers: dict[str, str]) -> ResolvedBlock:
        """Rewrite ``block_text`` with every placeholder replaced by a value.

        ``dep_answers`` maps each dependency node id to its committed
        answer. Returns the original text unchanged when the block has no
        cross-references; falls back to a parenthetical annotation when the
        LLM call fails or returns no usable text.
        """
        if not _mentions_node_ref(block_text):
            return ResolvedBlock(text=block_text, source="noop")

        # Nothing to resolve against — keep the parenthetical fallback so the
        # downstream solver at least sees the (empty) reference handled.
        if not any(v for v in dep_answers.values()):
            return ResolvedBlock(text=self._fallback_substitute(block_text, dep_answers),
                                 source="fallback")

        prompt = (
            _RESOLVE_SYSTEM
            + "\n\nCommitted answers of referenced problems:\n"
            + _format_dep_answers(dep_answers)
            + "\n\nSub-problem to rewrite:\n"
            + block_text.strip()
            + "\n\nReturn the rewritten sub-problem between the markers now."
        )
        try:
            text = self._call(prompt, role="resolve")
        except Exception:                                            # noqa: BLE001
            _log.exception("LLMGraphResolver.resolve_block failed; substituting")
            return ResolvedBlock(text=self._fallback_substitute(block_text, dep_answers),
                                 source="fallback")

        resolved = self._extract_between_markers(text)
        if not resolved:
            return ResolvedBlock(text=self._fallback_substitute(block_text, dep_answers),
                                 source="fallback")
        # Safety: if the model left an unresolved `[... from problem node ...]`
        # placeholder behind, the rewrite is not self-contained — prefer the
        # annotated fallback so at least the answer value is visible.
        if _PLACEHOLDER_RE.search(resolved):
            return ResolvedBlock(text=self._fallback_substitute(block_text, dep_answers),
                                 source="fallback")
        return ResolvedBlock(text=resolved.strip(), source="llm")

    @staticmethod
    def _fallback_substitute(text: str, dep_answers: dict[str, str]) -> str:
        """Old behaviour: annotate each `from problem node_M` reference with
        the committed answer in a parenthetical, leaving phrasing intact."""
        def repl(m):
            nid = f"node_{m.group(1)}"
            ans = dep_answers.get(nid)
            if ans in (None, ""):
                return m.group(0)
            return f"{m.group(0)} (whose committed answer was: {ans!s})"
        return NODE_REF_RE.sub(repl, text)

    _MARKER_RE = re.compile(r"<<<RESOLVED>>>(.*?)<<<END>>>", re.DOTALL)

    @classmethod
    def _extract_between_markers(cls, text: str) -> str:
        if not text:
            return ""
        m = cls._MARKER_RE.search(text)
        if m:
            return m.group(1).strip()
        # Be lenient: a model may emit the opening marker but no closing one.
        if "<<<RESOLVED>>>" in text:
            return text.split("<<<RESOLVED>>>", 1)[1].strip()
        return ""

    # ----- internals ------------------------------------------------------

    def _call(self, prompt: str, *, role: str) -> str:
        gens = self.llm.generate(prompt, temperature=self.temperature, n=1)
        text = gens[0].text if gens else ""
        finish = gens[0].finish_reason if gens else None
        self._log(role, prompt, text, finish)
        return text

    def _log(self, role: str, prompt: str, response: str,
             finish: str | None, error: str | None = None) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.log_agent_call(
                role=f"graph_{role}", prompt=prompt, response=response,
                model=self.model_name, finish_reason=finish,
                extras={"error": error},
            )
        except Exception:                                            # noqa: BLE001
            pass


# Detector for an unresolved `[ ... from problem node_K ... ]` placeholder.
# Used to reject rewrites that still leak a bracket. `[^\[\]]` forbids nested
# brackets, matching the dataset's flat placeholders.
_PLACEHOLDER_RE = re.compile(
    r"\[[^\[\]]*?from\s+problem\s+node_\d+[^\[\]]*?\]",
    flags=re.IGNORECASE | re.DOTALL,
)
