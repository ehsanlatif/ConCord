"""Tests for the LLM-driven dependency graph + cross-reference resolver.

Covers `structure/llm_graph.py` directly with a scripted fake LLM, plus the
`solve_multi` integration (resolver active when a distinct decomposition
client is configured; regex fallback otherwise).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.llm.factory import RoleClients
from core.llm.mock import MockLLM
from core.multi_solve import solve_multi
from core.structure.graph import parse_explicit_nodes
from core.structure.llm_graph import LLMGraphResolver
from core.types import Generation


# ---------------------------------------------------------------------------
# Fake LLM that recognises the graph + resolver prompts
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Branches on prompt content: emits graph JSON for the graph prompt,
    a marker-wrapped rewrite for the resolve prompt, garbage otherwise."""

    supports_logprobs = False

    def __init__(self):
        from core.types import CostTally
        self._cost = CostTally()

    def generate(self, prompt, *, temperature, n):
        self._cost.add(calls=n, input_tokens=1, output_tokens=1, usd=0.0)
        if "DEPENDENCY GRAPH" in prompt:
            text = (
                '{"nodes": ['
                '{"id": "node_0", "depends_on": [], "summary": "base"},'
                '{"id": "node_1", "depends_on": ["node_0"], "summary": "uses node_0"}'
                ']}'
            )
        elif "rewrite ONE sub-problem" in prompt:
            # Pull the block body out of the prompt and replace every
            # [...] placeholder with the literal 42.
            body = prompt.split("Sub-problem to rewrite:\n", 1)[1]
            body = body.split("\n\nReturn the rewritten", 1)[0].strip()
            body = re.sub(r"\[[^\[\]]*\]", "42", body)
            text = f"<<<RESOLVED>>>\n{body}\n<<<END>>>"
        else:
            text = f"<noise> {prompt[:20]}"
        return [Generation(text=text, token_logprobs=None, finish_reason="stop")
                for _ in range(n)]

    def cost(self):
        return self._cost


def _resolver_with(fake) -> LLMGraphResolver:
    return LLMGraphResolver(llm=fake, temperature=0.0, model_name="fake")


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

PROBLEM = (
    "Problem node_0: Compute the value 5 + 5.\n"
    "Problem node_1: Take the answer from problem node_0 and add 2.\n"
)


def test_build_graph_uses_llm_edges():
    blocks = parse_explicit_nodes(PROBLEM)
    spec = _resolver_with(_FakeLLM()).build_graph(PROBLEM, blocks)
    assert spec.source == "llm"
    assert spec.deps["node_0"] == []
    assert spec.deps["node_1"] == ["node_0"]
    assert spec.summaries["node_1"] == "uses node_0"
    G = spec.to_digraph()
    assert ("node_0", "node_1") in G.edges()


def test_build_graph_falls_back_to_regex_on_garbage():
    class Garbage:
        supports_logprobs = False
        def generate(self, prompt, *, temperature, n):
            return [Generation(text="not json at all", token_logprobs=None,
                               finish_reason="stop") for _ in range(n)]
        def cost(self):
            from core.types import CostTally
            return CostTally()

    blocks = parse_explicit_nodes(PROBLEM)
    spec = _resolver_with(Garbage()).build_graph(PROBLEM, blocks)
    assert spec.source == "regex_fallback"
    # Regex still recovers the explicit `from problem node_0` edge.
    assert spec.deps["node_1"] == ["node_0"]


def test_build_graph_unions_regex_floor_even_if_llm_misses_edge():
    """LLM that returns node_1 with NO deps must not drop the explicit
    `from problem node_0` reference — the regex floor re-adds it."""
    class MissesEdge:
        supports_logprobs = False
        def generate(self, prompt, *, temperature, n):
            text = ('{"nodes": [{"id": "node_0", "depends_on": []},'
                    '{"id": "node_1", "depends_on": []}]}')
            return [Generation(text=text, token_logprobs=None,
                               finish_reason="stop") for _ in range(n)]
        def cost(self):
            from core.types import CostTally
            return CostTally()

    blocks = parse_explicit_nodes(PROBLEM)
    spec = _resolver_with(MissesEdge()).build_graph(PROBLEM, blocks)
    assert spec.deps["node_1"] == ["node_0"]


# ---------------------------------------------------------------------------
# resolve_block
# ---------------------------------------------------------------------------

def test_resolve_block_replaces_placeholder():
    block = (r"How many integers are greater than $\sqrt{15}$ and less than "
             r"$\sqrt{[For this value use the exponent of 2 in the answer "
             r"from problem node_0 and subtract 51]}$?")
    rb = _resolver_with(_FakeLLM()).resolve_block(
        "node_1", block, {"node_0": "2^101 - 1"})
    assert rb.source == "llm"
    assert "[" not in rb.text and "]" not in rb.text
    assert "from problem node_0" not in rb.text


def test_resolve_block_noop_without_reference():
    block = "Compute 2 + 2."
    rb = _resolver_with(_FakeLLM()).resolve_block("node_0", block, {})
    assert rb.source == "noop"
    assert rb.text == block


def test_resolve_block_fallback_when_marker_missing():
    class NoMarkers:
        supports_logprobs = False
        def generate(self, prompt, *, temperature, n):
            return [Generation(text="here is your answer, no markers",
                               token_logprobs=None, finish_reason="stop")
                    for _ in range(n)]
        def cost(self):
            from core.types import CostTally
            return CostTally()

    block = "Use the answer from problem node_0 and add 2."
    rb = _resolver_with(NoMarkers()).resolve_block(
        "node_1", block, {"node_0": "10"})
    assert rb.source == "fallback"
    # Parenthetical annotation preserved as the safety net.
    assert "whose committed answer was: 10" in rb.text


def test_resolve_block_rejects_leftover_placeholder():
    """A rewrite that still contains a `[... from problem node ...]` bracket
    is not self-contained — the resolver must fall back."""
    class LeavesBracket:
        supports_logprobs = False
        def generate(self, prompt, *, temperature, n):
            text = ("<<<RESOLVED>>>\nUse [the answer from problem node_0] "
                    "and add 2.\n<<<END>>>")
            return [Generation(text=text, token_logprobs=None,
                               finish_reason="stop") for _ in range(n)]
        def cost(self):
            from core.types import CostTally
            return CostTally()

    block = "Use [the answer from problem node_0] and add 2."
    rb = _resolver_with(LeavesBracket()).resolve_block(
        "node_1", block, {"node_0": "10"})
    assert rb.source == "fallback"


# ---------------------------------------------------------------------------
# solve_multi integration
# ---------------------------------------------------------------------------

DEP_PROBLEM = (
    "Problem node_0: Compute the value 5 + 5.\n"
    "Problem node_1: Find $\\sqrt{[For this value use the answer from "
    "problem node_0 and add 2]}$.\n"
)


def _clients_with_distinct_decomposition(cfg, exec_llm, deco_llm):
    return RoleClients(
        execution=exec_llm, decomposition=deco_llm,
        classification=exec_llm, verification=exec_llm,
        splitter=exec_llm, combiner=exec_llm,
        synthesizer=exec_llm, synth_verifier=exec_llm,
        specs={r: cfg.llm for r in RoleClients.ROLE_NAMES},
    )


def test_solve_multi_activates_resolver_with_distinct_decomposition(tmp_path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    clients = _clients_with_distinct_decomposition(cfg, MockLLM(), _FakeLLM())
    res = solve_multi(DEP_PROBLEM, cfg=cfg, clients=clients, domain="math",
                      progress=None)

    assert res.cost["graph_source"] == "llm"
    resolved = res.cost["resolved_blocks"]["node_1"]
    # The placeholder bracket is gone in the question fed to the solver.
    assert "[" not in resolved and "]" not in resolved
    assert "from problem node_0" not in resolved


def test_solve_multi_keeps_regex_path_with_shared_client(tmp_path):
    """With one shared client (the mock convenience path) the resolver stays
    off and the deterministic regex substitution is used."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    res = solve_multi(DEP_PROBLEM, cfg=cfg, llm=MockLLM(), domain="math",
                      progress=None)
    assert res.cost["graph_source"] == "regex"
