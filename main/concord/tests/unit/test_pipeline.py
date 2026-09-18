"""Tests for the new Split → Solve → Combine → Verify pipeline.

These pin down:
  - VectorSharedMemory put/get/all_blocks/search basics
  - LLMSplitter: heuristic short-circuit + JSON parsing + recursion cap
  - LLMCombiner: prompt shape + retry-with-hint
  - LLMBlockVerifier: JSON parse + threshold-friendly verdicts
  - LLMSynthesizer: solution=[...] extraction + corrected_final_line retry
  - heuristic_expected_length: sinks-count behavior
  - End-to-end solve_multi with cfg.solver="pipeline" on a mock LLM that
    scripts every component's expected output. Verifies the combine_tree.json
    + shared_memory.json artifacts land in the run dir.

We DO NOT touch the legacy MCTS path here — those tests live in
test_decomposer_executor.py / test_m3_mcts.py / test_m5_walking_skeleton.py
and continue to use cfg.solver="mcts" (the default).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config
from core.llm.mock import MockLLM
from core.multi_solve import solve_multi
from core.pipeline import (
    LLMBlockVerifier,
    LLMCombiner,
    LLMSplitter,
    LLMSynthesizer,
    MemoryEntry,
    VectorSharedMemory,
)
from core.pipeline.shared_memory import VectorSharedMemory as VSM
from core.pipeline.splitter import is_likely_atomic
from core.pipeline.synthesizer import (
    extract_solution_list,
    heuristic_expected_length,
)
from core.types import Generation


# ===========================================================================
# VectorSharedMemory
# ===========================================================================

def test_shared_memory_put_get_overwrites_on_same_scope():
    m = VSM()
    m.put(scope="block:node_0", question="Compute X", answer="42")
    m.put(scope="block:node_0", question="Compute X", answer="43")
    assert m.get_block("node_0") == "43"
    assert len(m) == 1


def test_shared_memory_all_blocks_filters_atoms():
    m = VSM()
    m.put(scope="block:node_0", question="Q0", answer="A0")
    m.put(scope="atom:node_0/d0/a0", question="atomic", answer="A0-atom")
    m.put(scope="block:node_1", question="Q1", answer="A1")
    blocks = m.all_blocks()
    assert blocks == {"node_0": "A0", "node_1": "A1"}


def test_shared_memory_search_returns_topk_by_cosine():
    m = VSM()
    m.put(scope="block:node_0", question="graph theory bridgeless 3-regular",
          answer="42")
    m.put(scope="block:node_1", question="kite geometry perpendicular bisector",
          answer="20")
    m.put(scope="block:node_2", question="prime numbers consecutive integers",
          answer="2017")
    hits = m.search("graph theory incidence matrix bridge", k=2,
                     scope_prefix="block:")
    assert len(hits) == 2
    # The graph-theory entry should be the top hit.
    assert hits[0][0].scope == "block:node_0"
    assert hits[0][1] >= hits[1][1]


def test_shared_memory_persistence_roundtrip(tmp_path):
    m = VSM()
    m.put(scope="block:node_0", question="Compute X", answer="42",
          score=0.95)
    m.write_json(tmp_path / "mem.json")
    blob = json.loads((tmp_path / "mem.json").read_text())
    m2 = VSM.from_dict(blob)
    assert m2.get_block("node_0") == "42"
    assert m2.get("block:node_0").extras["score"] == 0.95


# ===========================================================================
# LLMSplitter
# ===========================================================================

def test_splitter_heuristic_short_circuits_on_short_atomic_question():
    """A short, multi-step-free question must skip the LLM call entirely
    and be returned as a single-atom tree."""
    assert is_likely_atomic("What is 2 + 2?", short_threshold=300)
    # And does NOT short-circuit on multi-step markers:
    assert not is_likely_atomic(
        "First compute X, then use it to determine Y, finally combine.",
        short_threshold=300)


def test_splitter_top_level_block_always_calls_llm_even_when_short():
    """A SHORT top-level block must still consult the splitter LLM (no
    heuristic short-circuit at depth 0) so hard-but-short problems get
    decomposed into solution steps. The short sub-steps it returns DO
    short-circuit at depth 1, so the LLM is consulted exactly once."""
    cfg = Config()
    cfg.pipeline.atom_short_circuit_chars = 300   # block is < 300 chars
    cfg.pipeline.max_split_depth = 3

    calls = {"n": 0}

    class _SplitOnce(MockLLM):
        def generate(self, prompt, *, temperature, n):
            calls["n"] += 1
            payload = ('{"atoms": ['
                       '{"id": 0, "question": "Step A.", "refs": []},'
                       '{"id": 1, "question": "Step B.", "refs": [0]}]}')
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    splitter = LLMSplitter(llm=_SplitOnce(), cfg=cfg)
    tree = splitter.split(block_node_id="node_X",
                          block_text="Short but hard.",   # < 300 chars
                          resolved={})
    assert calls["n"] == 1, "splitter LLM must be consulted at depth 0"
    assert len(tree.atoms) == 2, "short top-level block must still decompose"
    # The short sub-steps short-circuited at depth 1 (no extra LLM calls).
    assert all(a.split_reason == "heuristic_atomic" for a in tree.atoms)


def test_splitter_recursion_caps_at_max_split_depth():
    """No matter what the splitter LLM returns, recursion must not exceed
    cfg.pipeline.max_split_depth."""
    cfg = Config()
    cfg.pipeline.max_split_depth = 2
    cfg.pipeline.max_atoms_per_split = 4
    cfg.pipeline.atom_short_circuit_chars = 10  # force LLM use

    # MockLLM that ALWAYS splits into 2 atoms — would recurse forever
    # if the depth cap weren't enforced.
    class _SplitForever(MockLLM):
        def generate(self, prompt, *, temperature, n):
            payload = ('{"atoms": ['
                       '{"id": 0, "question": "subq A with long enough text to keep splitting", "refs": []},'
                       '{"id": 1, "question": "subq B with long enough text to keep splitting", "refs": []}'
                       ']}')
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    splitter = LLMSplitter(llm=_SplitForever(), cfg=cfg, temperature=0.3)
    tree = splitter.split(
        block_node_id="node_X",
        block_text="some long enough block text to bypass the heuristic",
        resolved={},
    )

    # Walk to find the maximum depth of any atom — must be ≤ max_split_depth.
    from core.pipeline.splitter import AtomicUnit
    def max_depth(atoms: list[AtomicUnit]) -> int:
        if not atoms:
            return 0
        return max(
            max(a.depth, max_depth(a.children) if a.children else a.depth)
            for a in atoms
        )
    assert max_depth(tree.atoms) <= cfg.pipeline.max_split_depth


def test_splitter_parses_json_atoms_with_refs():
    cfg = Config()
    cfg.pipeline.atom_short_circuit_chars = 10
    cfg.pipeline.max_split_depth = 1

    payload = (
        'Some preamble that should be skipped.\n'
        '{"atoms": ['
        '{"id": 0, "question": "Step one: compute X.", "refs": []},'
        '{"id": 1, "question": "Step two: use X.", "refs": [0]}'
        ']}'
    )

    class _Once(MockLLM):
        def generate(self, prompt, *, temperature, n):
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    splitter = LLMSplitter(llm=_Once(), cfg=cfg)
    tree = splitter.split(block_node_id="node_X",
                          block_text="a long enough block text to bypass heuristic",
                          resolved={})
    assert len(tree.atoms) == 2
    assert "Step one" in tree.atoms[0].question
    # Second atom carries refs.
    assert "0" in tree.atoms[1].refs


# ---------------------------------------------------------------------------
# SplitTree.topological_atom_order — children-before-parent + refs-honored
# ---------------------------------------------------------------------------

def _make_tree(atoms):
    """Build a SplitTree with the given top-level atoms."""
    from core.pipeline.splitter import SplitTree
    return SplitTree(block_node_id="node_X",
                     root_question="root q", atoms=atoms)


def _atom(atom_id, *, refs=None, children=None, is_atomic=None, q=None):
    from core.pipeline.splitter import AtomicUnit
    children = list(children or [])
    if is_atomic is None:
        is_atomic = not children
    return AtomicUnit(
        atom_id=atom_id,
        question=q if q is not None else f"q for {atom_id}",
        depth=0,
        refs=list(refs or []),
        children=children,
        is_atomic=is_atomic,
    )


def test_topological_atom_order_honors_refs_within_a_level():
    """Sibling refs must order: ref-target before ref-source."""
    # Three siblings at the top level. Atom 2 refs 0, atom 1 refs 2.
    # Expected order: 0, 2, 1 (no parent dep, just refs).
    a0 = _atom("a0")
    a1 = _atom("a1", refs=["2"])
    a2 = _atom("a2", refs=["0"])
    tree = _make_tree([a0, a1, a2])
    order = [a.atom_id for a in tree.topological_atom_order()]
    assert order.index("a0") < order.index("a2"), \
        f"a2 refs a0 — a0 must come first; got {order}"
    assert order.index("a2") < order.index("a1"), \
        f"a1 refs a2 — a2 must come first; got {order}"


def test_topological_atom_order_puts_children_before_parents():
    """A composite atom must come AFTER all its children."""
    # parent has two children — children must precede parent.
    c0 = _atom("p/c0")
    c1 = _atom("p/c1")
    parent = _atom("p", children=[c0, c1])
    tree = _make_tree([parent])
    order = [a.atom_id for a in tree.topological_atom_order()]
    assert order.index("p/c0") < order.index("p")
    assert order.index("p/c1") < order.index("p")


def test_topological_atom_order_falls_back_on_ref_cycle():
    """If the splitter emitted a cyclic ref (splitter bug), fall back to
    flatten_leaves so we still execute *something* instead of crashing."""
    # Cycle: a0 refs a1, a1 refs a0.
    a0 = _atom("a0", refs=["1"])
    a1 = _atom("a1", refs=["0"])
    tree = _make_tree([a0, a1])
    # Doesn't raise — must produce SOMETHING.
    order = tree.topological_atom_order()
    assert {a.atom_id for a in order} >= {"a0", "a1"}


def test_siblings_map_indexes_each_atom_to_its_parent_list():
    a0 = _atom("a0")
    c0 = _atom("a1/c0")
    a1 = _atom("a1", children=[c0])
    tree = _make_tree([a0, a1])
    sm = tree.siblings_map()
    # Top-level atoms share the same sibling list.
    assert {x.atom_id for x in sm["a0"]} == {"a0", "a1"}
    assert sm["a0"] is sm["a1"]
    # The lone child has a singleton sibling list.
    assert [x.atom_id for x in sm["a1/c0"]] == ["a1/c0"]


# ---------------------------------------------------------------------------
# PipelineExpansionPolicy._phase_b_solve bottom-up behavior
# ---------------------------------------------------------------------------

def _build_pipeline_policy(cfg, exec_llm, combiner_llm=None):
    """Construct a minimal PipelineExpansionPolicy for unit testing
    _phase_b_solve in isolation."""
    import random
    import networkx as nx
    from core.confidence import (
        LexicalEmbedder, LexicalKernel, default_verifier_for,
    )
    from core.pipeline.combiner import LLMCombiner
    from core.pipeline.expansion import PipelineExpansionPolicy
    from core.structure import ExplicitDecomposer
    Gp = nx.DiGraph()
    Gp.add_node("scc_0")
    policy = PipelineExpansionPolicy(
        llm=exec_llm,
        Gp=Gp,
        decomposer=ExplicitDecomposer(),
        embedder=LexicalEmbedder(),
        kernel=LexicalKernel(),
        verifier=default_verifier_for("math"),
        cfg=cfg, rng=random.Random(0),
        tracer=None, original_problem="",
        splitter=None,
        combiner=(LLMCombiner(llm=combiner_llm, cfg=cfg)
                  if combiner_llm is not None else None),
        block_verifier=None, shared_memory=None,
    )
    return policy


class _ScriptedExecutor(MockLLM):
    """MockLLM that maps an exact prompt-substring to a scripted
    `solution = X` response. Used to verify which atomic question the
    executor actually saw."""
    def __init__(self, bank):
        super().__init__(responses={})
        self.bank = bank
        self.seen_prompts: list[str] = []
    def generate(self, prompt, *, temperature, n):
        self.seen_prompts.append(prompt)
        from core.types import Generation
        match = "solution = unmatched"
        for needle, payload in self.bank:
            if needle in prompt:
                match = payload
                break
        self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
        return [Generation(text=match, token_logprobs=None,
                            finish_reason="stop") for _ in range(n)]


def test_phase_b_solve_inlines_ref_targets_into_dependent_atom_prompt():
    """A1 declares refs=[0]. When phase B reaches A1, A0's answer
    ("42") must be visible in the prompt the executor receives.

    NOTE: the executor's bank lookup is first-match-wins, so the more
    specific needle ("BBB") is listed BEFORE the more generic one
    ("AAA") — because after ref-inlining, A1's prompt contains BOTH
    needles (A0's question text is appended into A1's prompt as the
    ref-substitution provenance line)."""
    cfg = Config()
    cfg.pipeline.K_executor = 1
    cfg.pipeline.K_combiner = 1
    # Unique tokens so prompt matching is unambiguous.
    bank = [
        ("ATOM_BBB", "solution = 84"),
        ("ATOM_AAA", "solution = 42"),
    ]
    exec_llm = _ScriptedExecutor(bank)
    policy = _build_pipeline_policy(cfg, exec_llm)
    tree = _make_tree([
        _atom("a0", q="ATOM_AAA first question"),
        _atom("a1", q="ATOM_BBB second question", refs=["0"]),
    ])
    qa = policy._phase_b_solve(
        split_tree=tree, resolved={}, K=1,
        block_node_id="node_X", depth=0,
    )
    # Verify the executor saw A0's answer "42" inlined into A1's prompt.
    a1_prompts = [p for p in exec_llm.seen_prompts if "ATOM_BBB" in p]
    assert a1_prompts, "executor never saw the second atom"
    assert "42" in a1_prompts[0], (
        f"A0's answer was NOT inlined into A1's prompt:\n{a1_prompts[0]!r}"
    )
    # And atomic_qa preserved both answers, matched by their unique needles.
    answers = {atom_id: ans for atom_id, _, ans in qa}
    assert answers["a0"] == "42"
    assert answers["a1"] == "84"


def test_phase_b_solve_synthesizes_composite_from_children_via_combiner():
    """A composite atom (`p` with children `c0`, `c1`) must NOT be
    executed by the executor — it must be SYNTHESIZED by the combiner
    from its children's already-resolved answers."""
    cfg = Config()
    cfg.pipeline.K_executor = 1
    cfg.pipeline.K_combiner = 1
    # Leaves get matched by their question text; the composite's question
    # never reaches the executor at all.
    exec_bank = [
        ("child zero", "solution = 7"),
        ("child one",  "solution = 11"),
        ("composite parent", "solution = SHOULD_NOT_BE_RUN_BY_EXECUTOR"),
    ]
    exec_llm = _ScriptedExecutor(exec_bank)

    # Combiner returns a synthesized answer from the children.
    combiner_bank = [
        ("composite parent",
         "Composite combines 7 + 11.\nsolution = 18"),
    ]
    combiner_llm = _ScriptedExecutor(combiner_bank)

    policy = _build_pipeline_policy(cfg, exec_llm, combiner_llm=combiner_llm)
    c0 = _atom("p/c0", q="child zero")
    c1 = _atom("p/c1", q="child one")
    p = _atom("p", q="composite parent", children=[c0, c1])
    tree = _make_tree([p])

    qa = policy._phase_b_solve(
        split_tree=tree, resolved={}, K=1,
        block_node_id="node_X", depth=0,
    )
    answers = {atom_id: ans for atom_id, _, ans in qa}
    # Leaves executed.
    assert answers["p/c0"] == "7"
    assert answers["p/c1"] == "11"
    # Composite came from the COMBINER, not the executor — should be 18
    # (the synthesized value), NOT "SHOULD_NOT_BE_RUN_BY_EXECUTOR".
    assert answers["p"] == "18", (
        f"composite was executed instead of synthesized — got {answers['p']!r}"
    )
    # The executor must NOT have been asked about the composite question.
    composite_to_exec = [p for p in exec_llm.seen_prompts
                         if "composite parent" in p]
    assert not composite_to_exec, (
        f"executor was incorrectly invoked on the composite atom: "
        f"{composite_to_exec!r}"
    )


def test_pipeline_expand_decomposes_block_end_to_end():
    """Through `expand()`: a top-level block is decomposed into >1 atoms by
    the splitter (depth-0 always splits), each atom is solved bottom-up, and
    the result is merged into a single block answer. The provenance records
    the decomposition — proving 'decompose → solve → merge' actually fires
    at the node level."""
    import random
    import networkx as nx
    from core.confidence import (
        LexicalEmbedder, LexicalKernel, default_verifier_for,
    )
    from core.pipeline.combiner import LLMCombiner
    from core.pipeline.expansion import PipelineExpansionPolicy
    from core.pipeline.splitter import LLMSplitter
    from core.structure import ExplicitDecomposer
    from core.types import SolutionState

    cfg = Config()
    cfg.pipeline.K_executor = 1
    cfg.pipeline.K_combiner = 1
    cfg.pipeline.max_split_depth = 2
    cfg.pipeline.atom_short_circuit_chars = 300

    class _Splitter(MockLLM):
        def generate(self, prompt, *, temperature, n):
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            payload = ('{"atoms": ['
                       '{"id":0,"question":"leaf A short","refs":[]},'
                       '{"id":1,"question":"leaf B short","refs":[0]}]}')
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    # "leaf B" listed first: atom B refs atom A, so A's question text is
    # inlined into B's prompt — the more-specific needle must win.
    exec_llm = _ScriptedExecutor([("leaf B", "solution = 5"),
                                   ("leaf A", "solution = 3")])
    combiner_llm = _ScriptedExecutor([("", "Combine the steps.\nsolution = 8")])

    Gp = nx.DiGraph()
    Gp.add_node("scc_0",
                text="A genuinely hard multi-step problem needing steps.")
    policy = PipelineExpansionPolicy(
        llm=exec_llm, Gp=Gp, decomposer=ExplicitDecomposer(),
        embedder=LexicalEmbedder(), kernel=LexicalKernel(),
        verifier=default_verifier_for("math"), cfg=cfg,
        rng=random.Random(0), tracer=None, original_problem="",
        splitter=LLMSplitter(llm=_Splitter(), cfg=cfg),
        combiner=LLMCombiner(llm=combiner_llm, cfg=cfg),
        block_verifier=None, shared_memory=None,
    )

    info = policy.expand(SolutionState(), 0, parent_u_s=0.5, K=1)
    assert info.classes, "expand produced no class"

    blk = policy.build_combine_tree_payload()["blocks"][0]
    # Decomposed into 2 atoms (depth-0 block was NOT short-circuited).
    assert len(blk["split_tree"]["atoms"]) == 2
    # Both atoms were solved bottom-up.
    answers = {a["atom_id"]: a["answer"] for a in blk["atomic_answers"]}
    assert "3" in answers.values() and "5" in answers.values()
    # Merged into a single committed block answer via the combiner.
    assert blk["final_block_answer"] == "8"


def test_pipeline_expand_respects_per_node_call_budget():
    """A splitter that tries to explode the tree must NOT blow past the
    per-node agent-call budget. The total LLM calls across every role
    (splitter + executor + combiner) must stay <= cfg.pipeline.max_node_calls."""
    import random
    import networkx as nx
    from core.confidence import (
        LexicalEmbedder, LexicalKernel, default_verifier_for,
    )
    from core.pipeline.combiner import LLMCombiner
    from core.pipeline.expansion import PipelineExpansionPolicy
    from core.pipeline.splitter import LLMSplitter
    from core.structure import ExplicitDecomposer
    from core.types import SolutionState

    cfg = Config()
    cfg.pipeline.max_node_calls = 20            # tight cap for the test
    cfg.pipeline.max_split_depth = 3
    cfg.pipeline.max_atoms_per_split = 3
    cfg.pipeline.atom_short_circuit_chars = 1   # force every atom to call LLM
    cfg.pipeline.K_executor = 2
    cfg.pipeline.K_combiner = 2
    cfg.pipeline.combiner_retries = 1

    class _AlwaysSplit(MockLLM):
        """Splitter that always returns 3 sub-atoms — would recurse to the
        depth cap and produce 27 leaves if the budget didn't stop it."""
        def generate(self, prompt, *, temperature, n):
            payload = ('{"atoms": ['
                       '{"id":0,"question":"alpha step that is long enough","refs":[]},'
                       '{"id":1,"question":"beta step that is long enough","refs":[0]},'
                       '{"id":2,"question":"gamma step that is long enough","refs":[1]}]}')
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    class _Answer(MockLLM):
        def __init__(self, text):
            super().__init__(responses={})
            self._text = text
        def generate(self, prompt, *, temperature, n):
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=self._text, token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    splitter_llm = _AlwaysSplit()
    exec_llm = _Answer("solution = 1")
    combiner_llm = _Answer("solution = 2")

    Gp = nx.DiGraph()
    Gp.add_node("scc_0", text="A hard recursively-decomposable problem.")
    policy = PipelineExpansionPolicy(
        llm=exec_llm, Gp=Gp, decomposer=ExplicitDecomposer(),
        embedder=LexicalEmbedder(), kernel=LexicalKernel(),
        verifier=default_verifier_for("math"), cfg=cfg,
        rng=random.Random(0), tracer=None, original_problem="",
        splitter=LLMSplitter(llm=splitter_llm, cfg=cfg),
        combiner=LLMCombiner(llm=combiner_llm, cfg=cfg),
        block_verifier=None, shared_memory=None,
    )

    policy.expand(SolutionState(), 0, parent_u_s=0.5, K=2)

    total_calls = (splitter_llm.cost().calls + exec_llm.cost().calls
                   + combiner_llm.cost().calls)
    assert total_calls <= cfg.pipeline.max_node_calls, (
        f"per-node budget breached: {total_calls} > "
        f"{cfg.pipeline.max_node_calls}")
    # The internal counter agrees with the real calls spent.
    assert policy.call_budget.used <= cfg.pipeline.max_node_calls
    # And the splitter actually got curtailed (it would have recursed to 27
    # leaves unbounded) — at least one atom is flagged budget-limited.
    reasons = _collect_split_reasons(policy.build_combine_tree_payload())
    assert "call_budget_reached" in reasons or "depth_cap_reached" in reasons


def _collect_split_reasons(payload):
    reasons = set()
    def walk(atoms):
        for a in atoms or []:
            reasons.add(a.get("split_reason"))
            walk(a.get("children"))
    for blk in payload["blocks"]:
        st = blk.get("split_tree")
        if st:
            walk(st.get("atoms"))
    return reasons


def test_phase_b_solve_order_processes_children_before_parents():
    """The ROOT composite must be the LAST entry in the returned
    atomic_qa list (children solved first, root last)."""
    cfg = Config()
    cfg.pipeline.K_executor = 1
    cfg.pipeline.K_combiner = 1
    bank = [
        ("leaf A", "solution = 1"),
        ("leaf B", "solution = 2"),
        ("root q", "solution = root_via_combiner"),
    ]
    exec_llm = _ScriptedExecutor(bank)
    combiner_llm = _ScriptedExecutor([("root q",
                                         "solution = root_synthesized")])
    policy = _build_pipeline_policy(cfg, exec_llm, combiner_llm=combiner_llm)
    leafA = _atom("r/A", q="leaf A")
    leafB = _atom("r/B", q="leaf B")
    root = _atom("r", q="root q", children=[leafA, leafB])
    tree = _make_tree([root])

    qa = policy._phase_b_solve(
        split_tree=tree, resolved={}, K=1,
        block_node_id="node_X", depth=0,
    )
    order = [atom_id for atom_id, _, _ in qa]
    assert order.index("r/A") < order.index("r")
    assert order.index("r/B") < order.index("r")
    # Root atom is the last to be resolved (it's the top of the tree).
    assert order[-1] == "r"


# ===========================================================================
# LLMCombiner
# ===========================================================================

def test_combiner_prompt_includes_atomic_answers():
    cfg = Config()
    cfg.pipeline.K_combiner = 1

    seen_prompts: list[str] = []

    class _Capture(MockLLM):
        def generate(self, prompt, *, temperature, n):
            seen_prompts.append(prompt)
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text="ok\nsolution = 7", token_logprobs=None,
                                finish_reason="stop") for _ in range(n)]

    combiner = LLMCombiner(llm=_Capture(), cfg=cfg)
    gens = combiner.combine(
        original_question="What is the answer?",
        atomic_answers=[("first atom", "3"), ("second atom", "4")],
    )
    assert gens and gens[0].text.endswith("solution = 7")
    assert "first atom" in seen_prompts[0]
    assert "second atom" in seen_prompts[0]


def test_combiner_retry_injects_verifier_hint():
    cfg = Config()
    cfg.pipeline.K_combiner = 1
    seen: list[str] = []

    class _Capture(MockLLM):
        def generate(self, prompt, *, temperature, n):
            seen.append(prompt)
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text="solution = 8", token_logprobs=None,
                                finish_reason="stop")]

    combiner = LLMCombiner(llm=_Capture(), cfg=cfg)
    combiner.combine(
        original_question="Q",
        atomic_answers=[("a", "1")],
        verifier_hint="- value should be 9 not 8",
    )
    assert "verifier" in seen[0].lower() or "previous attempt" in seen[0].lower()
    assert "value should be 9 not 8" in seen[0]


# ===========================================================================
# LLMBlockVerifier
# ===========================================================================

def test_block_verifier_parses_score_and_issues():
    cfg = Config()
    payload = ('{"score": 0.82, "is_consistent": true, '
                '"issues": ["minor: rounding"], '
                '"corrected_answer": null}')

    class _Once(MockLLM):
        def generate(self, prompt, *, temperature, n):
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=payload, token_logprobs=None,
                                finish_reason="stop")]

    v = LLMBlockVerifier(llm=_Once(), cfg=cfg)
    verdict = v.verify(original_question="Q", atomic_answers=[],
                        candidate_answer="7")
    assert verdict.score == 0.82
    assert verdict.is_consistent is True
    assert verdict.issues == ["minor: rounding"]
    assert verdict.parse_failed is False


def test_block_verifier_defaults_neutral_on_garbage():
    cfg = Config()

    class _Garbage(MockLLM):
        def generate(self, prompt, *, temperature, n):
            from core.types import Generation
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text="I refuse.", token_logprobs=None,
                                finish_reason="stop")]

    v = LLMBlockVerifier(llm=_Garbage(), cfg=cfg)
    verdict = v.verify(original_question="Q", atomic_answers=[],
                        candidate_answer="7")
    assert verdict.parse_failed is True
    assert verdict.score == 0.5


# ===========================================================================
# LLMSynthesizer
# ===========================================================================

def test_extract_solution_list_balanced_list():
    text = ("Some reasoning.\n"
            "node_0 contributes 3480.\n"
            "solution = [3480, 76, 36, 7]")
    line, parts = extract_solution_list(text)
    assert line == "solution = [3480, 76, 36, 7]"
    assert parts == ["3480", "76", "36", "7"]


def test_extract_solution_list_handles_nested_brackets():
    text = "solution = [a, (b, c), [d, e]]"
    line, parts = extract_solution_list(text)
    # Top-level CSV split: ["a", "(b, c)", "[d, e]"]
    assert parts == ["a", "(b, c)", "[d, e]"]


def test_extract_solution_list_returns_empty_on_no_marker():
    line, parts = extract_solution_list("no list here")
    assert parts == []


def test_heuristic_expected_length_counts_sinks():
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("a", "c")])  # b, c are sinks
    G.add_node("z")                              # z is a sink
    assert heuristic_expected_length(G) == 3
    assert heuristic_expected_length(nx.DiGraph()) == 1


def test_synthesizer_extracts_and_respects_verifier(tmp_path):
    cfg = Config()
    # Synth returns a valid list; verifier confirms.
    synth_text = ("OK. node_0=3480, node_1=76, node_2=36, node_3=7.\n"
                  "solution = [3480, 76, 36, 7]")
    verifier_text = ('{"score": 0.9, "is_consistent": true, '
                      '"issues": [], "corrected_final_line": null}')

    class _Synth(MockLLM):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def generate(self, prompt, *, temperature, n):
            self.calls += 1
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=synth_text, token_logprobs=None,
                                finish_reason="stop")]

    class _Verif(MockLLM):
        def generate(self, prompt, *, temperature, n):
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=verifier_text, token_logprobs=None,
                                finish_reason="stop")]

    synth = LLMSynthesizer(synth_llm=_Synth(), verifier_llm=_Verif(),
                            cfg=cfg)
    res = synth.synthesize(
        problem_text="Big problem",
        block_answers={"node_0": "3480", "node_1": "76",
                       "node_2": "36", "node_3": "7"},
        expected_length_hint=4,
    )
    assert res.values == ["3480", "76", "36", "7"]
    assert res.actual_length == 4
    assert res.verifier_is_consistent is True
    assert res.n_retries == 0


def test_synthesizer_applies_verifier_corrected_final_line():
    cfg = Config()
    # Synth returns a 3-element list; verifier asks for a fixed 4-element line.
    synth_text = "solution = [3480, 76, 36]"
    verifier_text = ('{"score": 0.4, "is_consistent": false, '
                      '"issues": ["missing fourth value"], '
                      '"corrected_final_line": "solution = [3480, 76, 36, 7]"}')

    class _Synth(MockLLM):
        def generate(self, prompt, *, temperature, n):
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            return [Generation(text=synth_text, token_logprobs=None,
                                finish_reason="stop")]
    accept = '{"score": 0.9, "is_consistent": true, "issues": []}'

    class _Verif(MockLLM):
        def __init__(self):
            super().__init__()
            self.n = 0
        def generate(self, prompt, *, temperature, n):
            self.n += 1
            self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
            text = verifier_text if self.n == 1 else accept
            return [Generation(text=text, token_logprobs=None,
                                finish_reason="stop")]

    synth = LLMSynthesizer(synth_llm=_Synth(), verifier_llm=_Verif(),
                            cfg=cfg, max_verifier_retries=1)
    res = synth.synthesize(
        problem_text="prob",
        block_answers={"node_0": "3480"},
        expected_length_hint=4,
    )
    assert res.values == ["3480", "76", "36", "7"]
    assert res.n_retries == 1
    assert res.verifier_is_consistent is True


# ===========================================================================
# End-to-end: cfg.solver="pipeline" via solve_multi on a mock LLM
# ===========================================================================

def _scripted_pipeline_mock() -> MockLLM:
    """A mock LLM that produces sensible-looking outputs for EVERY pipeline
    role. Any unknown prompt falls back to the default `<mock:i>` echo,
    which is enough for the heuristic-atomic / splitter-failure paths."""
    # We use a tiny payload bank keyed by prompt-substrings. The MockLLM
    # itself does exact-key matching on prompts; for substring matching
    # we subclass below.
    return _SubstringMock()


class _SubstringMock(MockLLM):
    """MockLLM variant that looks up responses by substring rather than
    exact prompt equality. Useful because pipeline prompts are large
    and exact matching would be brittle."""

    BANK: list[tuple[str, str]] = [
        # Splitter: emit a single-atom JSON to avoid recursion.
        ("You are a problem decomposer",
         '{"atoms": [{"id": 0, "question": "atomic q", "refs": []}]}'),
        # Combiner: emit a clean solution = ...
        ("You are a problem solver",
         "Combining atoms.\nsolution = 42"),
        # Block verifier: accept.
        ("You are a careful answer auditor",
         '{"score": 0.9, "is_consistent": true, "issues": []}'),
        # Synthesizer: emit a list with the heuristic length.
        ("You are producing the FINAL answer",
         "Stitching block answers.\nsolution = [42, 42]"),
        # Synth-verifier: accept.
        ("You are auditing a FINAL",
         '{"score": 0.9, "is_consistent": true, "issues": []}'),
    ]

    def generate(self, prompt, *, temperature, n):
        match: str | None = None
        for needle, payload in self.BANK:
            if needle in prompt:
                match = payload
                break
        if match is None:
            return super().generate(prompt, temperature=temperature, n=n)
        out = []
        for _ in range(n):
            out.append(Generation(text=match, token_logprobs=None,
                                   finish_reason="stop"))
        self._cost.add(calls=n, input_tokens=1, output_tokens=n, usd=0.0)
        return out


def test_solve_multi_pipeline_end_to_end_writes_combine_tree(tmp_path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.solver = "pipeline"
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 4
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2
    cfg.pipeline.K_executor = 1
    cfg.pipeline.K_combiner = 1
    cfg.pipeline.combiner_retries = 0

    problem = ("Problem node_0: First subtask.\n"
               "Problem node_1: Second subtask uses the answer from problem node_0.")

    llm = _SubstringMock()
    res = solve_multi(problem, cfg=cfg, llm=llm, domain="math",
                      progress=None)

    run_dir = Path(res.cost["run_dir"])
    # The new artifacts MUST be present.
    assert (run_dir / "combine_tree.json").exists(), \
        "pipeline run did not write combine_tree.json"
    assert (run_dir / "shared_memory.json").exists(), \
        "pipeline run did not write shared_memory.json"

    ct = json.loads((run_dir / "combine_tree.json").read_text())
    # Layer 0 (block graph) populated.
    assert ct["solver"] == "pipeline"
    assert {n["id"] for n in ct["block_graph"]["nodes"]} == {"node_0", "node_1"}
    # Layer 1 (per-block provenance) populated.
    assert len(ct["blocks"]) >= 2
    for entry in ct["blocks"]:
        # Each block should carry at least the final block answer field.
        assert "final_block_answer" in entry
    # Layer 2 (synthesizer) populated.
    assert ct["synthesis"] is not None
    assert "solution = [" in ct["synthesis"]["final_line"]

    # The Result.answer also carries the synthesizer's solution = [...] line.
    assert "solution = [" in res.answer


def test_solve_multi_legacy_solver_does_not_break_combine_tree(tmp_path):
    """When cfg.solver stays 'mcts' the combine_tree.json must still be
    produced (with synthesis=None), so the viewer doesn't 404."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    assert cfg.solver == "mcts"
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 2
    cfg.sampling.K_blackbox = 2
    cfg.sampling.K_whitebox = 2

    problem = ("Problem node_0: trivial.\n"
               "Problem node_1: also trivial.")

    res = solve_multi(problem, cfg=cfg, llm=MockLLM(responses={}),
                      domain="math", progress=None)
    run_dir = Path(res.cost["run_dir"])
    assert (run_dir / "combine_tree.json").exists()
    ct = json.loads((run_dir / "combine_tree.json").read_text())
    assert ct["solver"] == "mcts"
    assert ct["synthesis"] is None
