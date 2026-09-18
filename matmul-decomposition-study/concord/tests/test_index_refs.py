"""Unit tests for the index-reference splitter fix (matmul).

Covers: source_span coercion + serialization, split-output parsing, the code
source-resolver (exact value injection, no LLM), the grader recovering slices
from source_span, and grade_block bridging source_span from the split tree.
No API calls.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # concord/ package root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from core.config import Config                                    # noqa: E402
from core.pipeline.splitter import AtomicUnit, LLMSplitter, _coerce_span  # noqa: E402
import matmul_grader as G                                          # noqa: E402
from matmul_task import (                                          # noqa: E402
    GroundTruth, build_source_resolver, build_combine_resolver,
    span_from_text, matrix_str, identity, matmul_mod, MATMUL_SPLITTER_SYSTEM,
)
from core.pipeline.expansion import (                              # noqa: E402
    set_combine_resolver, get_combine_resolver, _resolve_combine,
)


def _gt():
    p = 97
    A = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 0], [1, 3]], [[9, 1], [4, 2]]]
    gold, run = [], identity(2)
    for a in A:
        run = matmul_mod(run, a, p)
        gold.append(run)
    return GroundTruth(dim=2, modulus=p, matrices=A, golden=gold, sample_id=0)


# ---- coercion + serialization -------------------------------------------

def test_coerce_span():
    assert _coerce_span([101, 200]) == (101, 200)
    assert _coerce_span((3, 3)) == (3, 3)
    assert _coerce_span(["5", "9"]) == (5, 9)
    assert _coerce_span([200, 100]) is None   # j < i
    assert _coerce_span([0, 5]) is None        # i < 1
    assert _coerce_span("garbage") is None
    assert _coerce_span(None) is None


def test_atom_to_dict_roundtrips_span():
    assert AtomicUnit("a", "q", 0, source_span=(1, 5)).to_dict()["source_span"] == [1, 5]
    assert AtomicUnit("a", "q", 0).to_dict()["source_span"] is None


def test_parse_splitter_output_reads_span():
    sp = LLMSplitter(llm=None, cfg=Config())
    text = ('{"atoms":['
            '{"id":0,"question":"product A_1..A_2","source_span":[1,2],"refs":[]},'
            '{"id":1,"question":"product A_3..A_4","source_span":[3,4],"refs":[0]}]}')
    atoms = sp._parse_splitter_output(text, parent_atom_id="blk/d0/a0", depth=1)
    assert [a.source_span for a in atoms] == [(1, 2), (3, 4)]
    assert atoms[1].refs == ["0"]


# ---- resolver: exact value injection, no LLM ----------------------------

def test_resolver_injects_exact_values():
    gt = _gt()
    resolve = build_source_resolver(gt)
    atom = AtomicUnit("x", "Compute product A_2..A_3", 0, source_span=(2, 3))
    out = resolve(atom, "Compute product A_2..A_3")
    assert matrix_str(gt.matrices[1]) in out   # A_2
    assert matrix_str(gt.matrices[2]) in out   # A_3
    assert matrix_str(gt.matrices[0]) not in out  # A_1 NOT injected
    assert "mod 97" in out


def test_resolver_fallback_to_text_span():
    gt = _gt()
    resolve = build_source_resolver(gt)
    atom = AtomicUnit("x", "running product of A_3..A_4", 0)  # no source_span
    out = resolve(atom, "running product of A_3..A_4")
    assert matrix_str(gt.matrices[2]) in out and matrix_str(gt.matrices[3]) in out
    assert matrix_str(gt.matrices[0]) not in out


def test_span_from_text():
    assert span_from_text("compute A_3 .. A_7 now") == (3, 7)
    assert span_from_text("no indices here") is None


# ---- grader: recover slice from source_span -----------------------------

def test_grade_atom_by_span_pass_running_state():
    gt = _gt(); idx = gt.index_of()
    atom = {"atom_id": "x", "source_span": [1, 3],
            "answer": f"solution = {matrix_str(gt.golden[2])}"}  # M_3
    rec = G.grade_atom(atom, gt, idx)
    assert rec["status"] == "PASS" and rec["style"] == "running_state"
    assert rec["slice"] == [1, 3] and rec["contiguous"]


def test_grade_atom_by_span_partial_product():
    gt = _gt(); idx = gt.index_of()
    partial = identity(2)
    for t in (2, 3):
        partial = matmul_mod(partial, gt.matrices[t - 1], gt.modulus)
    atom = {"atom_id": "x", "source_span": [2, 3],
            "answer": f"solution = {matrix_str(partial)}"}
    rec = G.grade_atom(atom, gt, idx)
    assert rec["status"] == "PASS" and rec["style"] == "partial_product"


def test_grade_atom_by_span_fail():
    gt = _gt(); idx = gt.index_of()
    atom = {"atom_id": "x", "source_span": [1, 3],
            "answer": "solution = [[0,0],[0,0]]"}
    assert G.grade_atom(atom, gt, idx)["status"] == "FAIL"


def test_grade_block_bridges_span_from_tree():
    gt = _gt(); idx = gt.index_of()
    # atomic_answers has NO source_span (as persisted); split_tree carries it.
    block = {
        "split_tree": {"atoms": [
            {"atom_id": "b/d0/a0", "source_span": [1, 2], "children": []},
            {"atom_id": "b/d0/a1", "source_span": [3, 4], "children": []},
        ]},
        "atomic_answers": [
            {"atom_id": "b/d0/a0", "question": "product A_1..A_2",
             "answer": f"solution = {matrix_str(gt.golden[1])}"},
            {"atom_id": "b/d0/a1", "question": "product A_3..A_4",
             "answer": f"solution = {matrix_str(gt.golden[3])}"},
        ],
    }
    out = G.grade_block(block, gt, idx)
    statuses = [r["status"] for r in out["records"]]
    assert statuses == ["PASS", "PASS"], statuses   # bridged + graded, not UNGRADEABLE


# ---- config override plumbing -------------------------------------------

def test_splitter_override_field():
    cfg = Config()
    assert cfg.pipeline.splitter_system_override is None   # default: generic
    cfg.pipeline.splitter_system_override = MATMUL_SPLITTER_SYSTEM
    base = (getattr(cfg.pipeline, "splitter_system_override", None) or "GENERIC")
    assert "source_span" in base.replace("{max_atoms}", "5")


# ---- combine resolver (Issue A: LLM combiner corrupts running-state) ------

def test_combine_resolver_picks_last_child_by_span():
    combine = build_combine_resolver(_gt())
    # children out of order; the one covering the highest index wins
    kids = [
        {"source_span": (7, 9), "question": "A_7..A_9", "answer": "[[70,38],[19,2]]"},
        {"source_span": (1, 3), "question": "A_1..A_3", "answer": "[[5,31],[60,41]]"},
        {"source_span": (4, 6), "question": "A_4..A_6", "answer": "[[33,10],[71,43]]"},
    ]
    assert combine(kids) == "[[70,38],[19,2]]"


def test_combine_resolver_falls_back_to_question_span():
    combine = build_combine_resolver(_gt())
    kids = [
        {"source_span": None, "question": "product A_1..A_5", "answer": "AA"},
        {"source_span": None, "question": "product A_6..A_9", "answer": "BB"},
    ]
    assert combine(kids) == "BB"


def test_combine_resolver_last_emitted_when_no_span():
    combine = build_combine_resolver(_gt())
    kids = [{"source_span": None, "question": "x", "answer": "P"},
            {"source_span": None, "question": "y", "answer": "Q"}]
    assert combine(kids) == "Q"


def test_combine_resolver_declines_when_last_child_empty():
    # Budget-truncated: designated last child ([7,9]) has no answer yet →
    # return None so the incomplete chain is NOT masked by an earlier partial.
    combine = build_combine_resolver(_gt())
    kids = [{"source_span": (1, 3), "question": "A_1..A_3", "answer": "[[1,2],[3,4]]"},
            {"source_span": (7, 9), "question": "A_7..A_9", "answer": ""}]
    assert combine(kids) is None


class _Kid:                       # minimal stand-in for AtomicUnit
    def __init__(self, atom_id, question, source_span):
        self.atom_id = atom_id
        self.question = question
        self.source_span = source_span


def test_resolve_combine_uses_registered_resolver():
    try:
        set_combine_resolver(build_combine_resolver(_gt()))
        children = [_Kid("c0", "A_1..A_3", (1, 3)), _Kid("c1", "A_4..A_6", (4, 6))]
        answers = {"c0": "[[5,31],[60,41]]", "c1": "[[33,10],[71,43]]"}
        assert _resolve_combine(children, answers) == "[[33,10],[71,43]]"
    finally:
        set_combine_resolver(None)


def test_resolve_combine_noop_when_unregistered():
    assert get_combine_resolver() is None      # default: no resolver (e.g. chess)
    children = [_Kid("c0", "A_1..A_3", (1, 3))]
    assert _resolve_combine(children, {"c0": "[[1,2],[3,4]]"}) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
