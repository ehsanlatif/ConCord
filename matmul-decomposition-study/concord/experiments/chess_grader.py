"""Multi-level grader for the `uci_to_fen` chess problem.

The pipeline solver (Split → Solve → Combine → Verify) decomposes a single
`uci_to_fen` block into a recursive tree of *atomic units*. Each atomic unit is
a self-contained sub-question of the form "apply these UCI moves to this
position; what is the resulting FEN?", and its answer is a FEN string. This
module grades the run at four levels, all anchored on a python-chess
ground-truth trace (one verified FEN per ply):

  L0  OVERALL        — does the final answer FEN equal the gold FEN?
  L1  PER-SUBPROBLEM — for every atomic unit, re-derive the correct output FEN
                       from the *unit's own* input position + moves with
                       python-chess and compare it to the unit's answer. This
                       is the "verify the individual subproblem solutions"
                       check the study is built around.
  L2  DECOMPOSITION  — is the move slice the splitter handed each unit actually
      FIDELITY         the correct contiguous slice of the real game? (Catches
                       a splitter that carves wrong boundaries even when the
                       executor then "correctly" solves a wrong sub-question.)
  L3  PER-LEVEL       — aggregate the L1 pass/fail rate by split-tree DEPTH, so
                       you can see at which decomposition level things break.

The grader never reasons about chess itself — every expected position comes
from python-chess replay, exactly like the trace.

It can be used three ways:
  * imported  — `grade_run(trace, blocks, final_answer, gold_fen)`
  * as ground truth for the study runner (chess_study.py)
  * standalone — grade an existing combine_tree.json:
        python -m concord.experiments.chess_grader \\
            --trace verification/uci_to_fen_easy_6_trace.json \\
            --combine-tree results/.../combine_tree.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import chess


# ---------------------------------------------------------------------------
# FEN / UCI extraction
# ---------------------------------------------------------------------------

# A FEN *board* field: 8 ranks of [pnbrqkPNBRQK1-8] separated by '/'.
_FEN_BOARD = r"(?:[pnbrqkPNBRQK1-8]{1,8}/){7}[pnbrqkPNBRQK1-8]{1,8}"
# Optional trailing fields: side, castling, en-passant, halfmove, fullmove.
_FEN_TAIL = r"(?:\s+[wb]\s+(?:-|[KQkq]{1,4})\s+(?:-|[a-h][36])\s+\d+\s+\d+)?"
_FEN_RE = re.compile(_FEN_BOARD + _FEN_TAIL)

# A UCI move: from-square, to-square, optional promotion piece.
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def extract_fens(text: str) -> list[str]:
    """All FEN-like tokens (board field, optionally with the 5 trailing
    fields) appearing in `text`, in order."""
    if not text:
        return []
    return [m.group(0).strip() for m in _FEN_RE.finditer(text)]


def extract_uci_moves(text: str) -> list[str]:
    """UCI moves in `text`, AFTER masking out any FEN substrings.

    Masking matters: a FEN rank such as ``1b1b2`` contains the literal
    substring ``b1b2`` which looks like a UCI move. We blank out every FEN
    we can find first so board content can never masquerade as moves. (Files
    a,c,d,e,f,g,h are not FEN piece letters, so only the all-`b` family of
    pseudo-moves is at risk, but masking removes the whole class cleanly.)
    """
    if not text:
        return []
    masked = _FEN_RE.sub(" ", text)
    return [m.group(1) for m in _UCI_RE.finditer(masked)]


def board_field(fen: str) -> str:
    """The piece-placement field of a FEN (everything before the first space)."""
    return fen.strip().split()[0] if fen and fen.strip() else ""


def _full_fen(fen_token: str) -> str | None:
    """Return a fully-specified FEN python-chess can parse, or None when the
    token is board-only (side-to-move unknown -> ambiguous to replay)."""
    if not fen_token:
        return None
    parts = fen_token.split()
    if len(parts) >= 2:                 # has at least board + side
        try:
            chess.Board(fen_token)
            return fen_token
        except Exception:               # noqa: BLE001
            return None
    return None                          # board-only -> caller resolves via trace


# ---------------------------------------------------------------------------
# Ground-truth trace helpers
# ---------------------------------------------------------------------------

class GroundTruth:
    """Wraps the per-ply trace produced by build the verification step.

    Provides: the gold FEN at any ply, a board-placement -> ply-index lookup
    (to anchor a sub-problem's starting position onto the real game), and the
    canonical move list for decomposition-fidelity checks.
    """

    def __init__(self, trace: dict):
        self.trace = trace
        self.steps = trace["steps"]
        self.moves: list[str] = trace.get("moves_uci") or [s["move_uci"] for s in self.steps]
        self.starting_fen: str = trace["starting_fen"]
        # board placement -> sorted list of ply indices whose fen_after has it.
        self._board_to_plies: dict[str, list[int]] = {}
        self._register(board_field(self.starting_fen), 0)
        for s in self.steps:
            self._register(board_field(s["fen_after"]), s["ply"])
        # ply -> full fen_after (ply 0 == starting position)
        self._fen_at: dict[int, str] = {0: self.starting_fen}
        for s in self.steps:
            self._fen_at[s["ply"]] = s["fen_after"]

    def _register(self, board: str, ply: int) -> None:
        self._board_to_plies.setdefault(board, []).append(ply)

    def fen_at_ply(self, ply: int) -> str | None:
        return self._fen_at.get(ply)

    def gold_fen(self, max_plies: int | None = None) -> str:
        """The gold answer FEN: final position, or the position after the
        first `max_plies` half-moves when the study truncates the sequence."""
        if max_plies is None:
            return self._fen_at[self.steps[-1]["ply"]]
        return self._fen_at.get(max_plies, self._fen_at[self.steps[-1]["ply"]])

    def plies_for_board(self, board: str) -> list[int]:
        return self._board_to_plies.get(board, [])

    def full_fen_for_board(self, board: str) -> str | None:
        """A fully-specified gold FEN whose placement matches `board` (uses the
        earliest matching ply). Lets us resolve a board-only sub-problem start
        to a real side-to-move + counters."""
        plies = self.plies_for_board(board)
        if not plies:
            return None
        return self._fen_at.get(plies[0])


# ---------------------------------------------------------------------------
# FEN comparison
# ---------------------------------------------------------------------------

def fen_equal(a: str, b: str) -> tuple[bool, bool]:
    """Return (board_match, exact_match) for two FEN strings.

    board_match compares only piece placement (robust: a sub-problem may
    legitimately reset halfmove/fullmove counters or side-to-move framing).
    exact_match compares the full normalized FEN.
    """
    if not a or not b:
        return (False, False)
    ba, bb = board_field(a), board_field(b)
    board_match = (ba == bb)
    try:
        exact = chess.Board(a).fen() == chess.Board(b).fen()
    except Exception:                   # noqa: BLE001
        exact = (a.strip() == b.strip())
    return (board_match, exact)


# ---------------------------------------------------------------------------
# Split-tree walk: atom_id -> depth / structure
# ---------------------------------------------------------------------------

def walk_split_tree(split_tree: dict | None) -> dict[str, dict]:
    """Map every atom_id in a SplitTree.to_dict() payload to its metadata
    (depth, is_atomic, whether it has children, question, refs)."""
    out: dict[str, dict] = {}
    if not split_tree:
        return out

    def visit(atom: dict) -> None:
        out[atom.get("atom_id")] = {
            "depth": atom.get("depth"),
            "is_atomic": atom.get("is_atomic"),
            "has_children": bool(atom.get("children")),
            "question": atom.get("question", ""),
            "refs": atom.get("refs", []),
            "split_reason": atom.get("split_reason", ""),
        }
        for c in atom.get("children", []) or []:
            visit(c)

    for a in split_tree.get("atoms", []) or []:
        visit(a)
    return out


def _depth_from_atom_id(atom_id: str) -> int | None:
    m = re.search(r"/d(\d+)/", atom_id or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-subproblem grading
# ---------------------------------------------------------------------------

# Status values
PASS = "PASS"
FAIL = "FAIL"
UNGRADEABLE = "UNGRADEABLE"


def grade_atom(question: str, answer: str, gt: GroundTruth) -> dict:
    """Verify one atomic unit's answer with python-chess.

    Strategy:
      1. Parse the unit's INPUT position from its question (first FEN). If no
         FEN, assume the standard start. A board-only FEN is resolved to a
         full FEN via the trace (to recover side-to-move + counters).
      2. Parse the UCI MOVES from the question (FEN-masked).
      3. Re-derive the correct output FEN by replaying with python-chess.
      4. Compare to the FEN the unit actually answered (last FEN in answer).

    Also reports decomposition fidelity: whether the parsed (start, moves)
    correspond to a correct contiguous slice of the real game.
    """
    res: dict[str, Any] = {
        "status": UNGRADEABLE,
        "reason": "",
        "n_moves": 0,
        "input_ply": None,
        "move_range": None,
        "slice_faithful": None,
        "expected_fen": None,
        "answer_fen": None,
        "board_match": False,
        "exact_match": False,
    }

    q_fens = extract_fens(question)
    moves = extract_uci_moves(question)
    res["n_moves"] = len(moves)

    # --- resolve the input position -------------------------------------
    if q_fens:
        token = q_fens[0]
        full = _full_fen(token)
        if full is None:
            # board-only -> resolve via the trace's matching ply
            full = gt.full_fen_for_board(board_field(token))
        if full is None:
            res["reason"] = "input position board-only and not found in trace"
            return res
        input_fen = full
    else:
        input_fen = gt.starting_fen      # no FEN quoted -> start of game

    input_ply = None
    plies = gt.plies_for_board(board_field(input_fen))
    if plies:
        input_ply = plies[0]
        res["input_ply"] = input_ply

    if not moves:
        res["reason"] = "no UCI moves found in the sub-question"
        return res

    # --- re-derive the correct output via python-chess ------------------
    try:
        board = chess.Board(input_fen)
        for mv in moves:
            board.push_uci(mv)
        expected = board.fen()
    except Exception as e:               # noqa: BLE001
        # The parsed moves are illegal from the parsed start: we cannot
        # establish ground truth for this unit (almost always a parse issue
        # or a malformed split). Flag, don't score it as a wrong answer.
        res["reason"] = f"replay_illegal ({e})"
        return res
    res["expected_fen"] = expected

    # --- decomposition fidelity: are these the real game's moves? -------
    if input_ply is not None:
        real_slice = gt.moves[input_ply: input_ply + len(moves)]
        res["move_range"] = [input_ply + 1, input_ply + len(moves)]
        res["slice_faithful"] = (real_slice == moves)

    # --- compare to the unit's answer -----------------------------------
    a_fens = extract_fens(answer or "")
    if not a_fens:
        res["status"] = FAIL
        res["reason"] = "answer contains no FEN"
        return res
    answer_fen = a_fens[-1]              # last FEN = the unit's final answer
    res["answer_fen"] = answer_fen
    board_match, exact = fen_equal(expected, answer_fen)
    res["board_match"] = board_match
    res["exact_match"] = exact
    res["status"] = PASS if board_match else FAIL
    if not board_match:
        res["reason"] = "answer FEN != python-chess expected FEN"
    return res


# ---------------------------------------------------------------------------
# Block / run grading
# ---------------------------------------------------------------------------

def grade_block(block: dict, gt: GroundTruth) -> dict:
    """Grade one pipeline block's atomic answers + final block answer."""
    id_meta = walk_split_tree(block.get("split_tree"))
    atom_results: list[dict] = []
    for entry in block.get("atomic_answers", []) or []:
        atom_id = entry.get("atom_id", "")
        meta = id_meta.get(atom_id, {})
        depth = meta.get("depth")
        if depth is None:
            depth = _depth_from_atom_id(atom_id)
        g = grade_atom(entry.get("question", ""), entry.get("answer", ""), gt)
        g.update({
            "atom_id": atom_id,
            "depth": depth,
            "is_atomic": meta.get("is_atomic"),
            "has_children": meta.get("has_children"),
        })
        atom_results.append(g)

    # final committed block answer vs gold
    final_answer = block.get("final_block_answer") or ""
    a_fens = extract_fens(final_answer)
    block_board_match = block_exact = False
    if a_fens:
        block_board_match, block_exact = fen_equal(gt.gold_fen(), a_fens[-1])

    return {
        "block_node_id": block.get("block_node_id"),
        "final_block_answer": final_answer,
        "final_block_board_match": block_board_match,
        "final_block_exact_match": block_exact,
        "final_score": block.get("final_score"),
        "atom_results": atom_results,
    }


def summarize_levels(atom_results: list[dict]) -> dict[str, dict]:
    """Aggregate per-subproblem pass/fail by split-tree depth (L3)."""
    by_depth: dict[Any, dict] = {}
    for a in atom_results:
        d = a.get("depth")
        key = "unknown" if d is None else d
        b = by_depth.setdefault(key, {"n": 0, "gradeable": 0, "pass": 0,
                                       "fail": 0, "ungradeable": 0,
                                       "slice_faithful": 0, "slice_checked": 0})
        b["n"] += 1
        if a["status"] == PASS:
            b["pass"] += 1
            b["gradeable"] += 1
        elif a["status"] == FAIL:
            b["fail"] += 1
            b["gradeable"] += 1
        else:
            b["ungradeable"] += 1
        if a.get("slice_faithful") is not None:
            b["slice_checked"] += 1
            if a["slice_faithful"]:
                b["slice_faithful"] += 1
    for b in by_depth.values():
        b["pass_rate"] = (b["pass"] / b["gradeable"]) if b["gradeable"] else None
        b["slice_fidelity"] = (b["slice_faithful"] / b["slice_checked"]
                               if b["slice_checked"] else None)
    # stringify keys so JSON is stable/sorted
    return {str(k): by_depth[k] for k in sorted(by_depth, key=lambda x: (x == "unknown", x))}


def grade_run(*, trace: dict, blocks: list[dict], final_answer: str,
              max_plies: int | None = None) -> dict:
    """Top-level grade for one solver run.

    Args:
      trace:        the per-ply ground-truth trace dict.
      blocks:       pipeline provenance blocks
                    (res.cost["pipeline_provenance"]["blocks"] or
                    combine_tree.json["blocks"]).
      final_answer: the solver's final answer string (res.answer).
      max_plies:    if the move sequence was truncated for the run, the gold
                    FEN is taken at this ply.
    """
    gt = GroundTruth(trace)
    gold = gt.gold_fen(max_plies)

    # L0 overall
    fa_fens = extract_fens(final_answer or "")
    overall_board = overall_exact = False
    answer_fen = fa_fens[-1] if fa_fens else None
    if answer_fen:
        overall_board, overall_exact = fen_equal(gold, answer_fen)

    block_grades = [grade_block(b, gt) for b in blocks]
    all_atoms = [a for bg in block_grades for a in bg["atom_results"]]

    n_grade = sum(1 for a in all_atoms if a["status"] in (PASS, FAIL))
    n_pass = sum(1 for a in all_atoms if a["status"] == PASS)
    n_fail = sum(1 for a in all_atoms if a["status"] == FAIL)
    n_unjudged = sum(1 for a in all_atoms if a["status"] == UNGRADEABLE)
    n_slice_checked = sum(1 for a in all_atoms if a.get("slice_faithful") is not None)
    n_slice_ok = sum(1 for a in all_atoms if a.get("slice_faithful") is True)

    return {
        "gold_fen": gold,
        "final_answer_fen": answer_fen,
        "overall_correct": overall_exact,        # dataset criterion (full FEN)
        "overall_board_match": overall_board,
        "n_atoms": len(all_atoms),
        "n_gradeable": n_grade,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_ungradeable": n_unjudged,
        "atom_pass_rate": (n_pass / n_grade) if n_grade else None,
        "decomposition_fidelity": (n_slice_ok / n_slice_checked
                                   if n_slice_checked else None),
        "n_slice_checked": n_slice_checked,
        "n_slice_faithful": n_slice_ok,
        "levels": summarize_levels(all_atoms),
        "blocks": block_grades,
    }


# ---------------------------------------------------------------------------
# Provenance extraction (single-block solve OR multi combine_tree.json)
# ---------------------------------------------------------------------------

def blocks_from_result_cost(cost: dict) -> list[dict]:
    """Pull the pipeline provenance blocks out of a Result.cost dict.

    Single-block solves carry them inline at cost["pipeline_provenance"];
    multi-block runs persist them to combine_tree.json (path on
    cost["combine_tree_path"])."""
    prov = (cost or {}).get("pipeline_provenance")
    if isinstance(prov, dict) and prov.get("blocks"):
        return prov["blocks"]
    ctp = (cost or {}).get("combine_tree_path")
    if ctp and Path(ctp).exists():
        try:
            return json.loads(Path(ctp).read_text()).get("blocks", [])
        except Exception:               # noqa: BLE001
            return []
    return []


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_grade(grade: dict) -> str:
    lines: list[str] = []
    oc = grade["overall_correct"]
    lines.append(
        f"OVERALL: {'CORRECT' if oc else 'WRONG'} "
        f"(exact={oc}, board_match={grade['overall_board_match']})")
    lines.append(
        f"  final answer FEN: {grade['final_answer_fen']}")
    lines.append(
        f"  gold FEN:         {grade['gold_fen']}")
    apr = grade["atom_pass_rate"]
    lines.append(
        f"SUBPROBLEMS: {grade['n_pass']}/{grade['n_gradeable']} passed"
        + (f" (pass_rate={apr:.3f})" if apr is not None else "")
        + f"  |  {grade['n_ungradeable']} ungradeable of {grade['n_atoms']} atoms")
    df = grade["decomposition_fidelity"]
    if df is not None:
        lines.append(
            f"DECOMPOSITION FIDELITY: {grade['n_slice_faithful']}/"
            f"{grade['n_slice_checked']} sub-questions used a correct move "
            f"slice (fidelity={df:.3f})")
    lines.append("PER DECOMPOSITION LEVEL:")
    for depth, b in grade["levels"].items():
        pr = b["pass_rate"]
        pr_s = f"{pr:.3f}" if pr is not None else "n/a"
        lines.append(
            f"  depth {depth}: {b['pass']}/{b['gradeable']} pass "
            f"(rate={pr_s}), {b['ungradeable']} ungradeable, n={b['n']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: grade an existing combine_tree.json offline
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Grade a uci_to_fen pipeline run against the ground-truth trace.")
    p.add_argument("--trace", type=Path,
                   default=Path("verification/uci_to_fen_easy_6_trace.json"))
    p.add_argument("--combine-tree", type=Path, required=True,
                   help="combine_tree.json from a multi run, OR a JSON file "
                        "with a top-level 'blocks' list and optional "
                        "'final_answer'/'synthesis'.")
    p.add_argument("--max-plies", type=int, default=None,
                   help="If the run truncated the move list, gold is taken "
                        "at this ply.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the full grade JSON here.")
    args = p.parse_args()

    trace = json.loads(args.trace.read_text())
    ct = json.loads(args.combine_tree.read_text())
    blocks = ct.get("blocks", [])
    # final answer: prefer synthesis final line, else single block answer.
    final_answer = ""
    synth = ct.get("synthesis")
    if isinstance(synth, dict):
        final_answer = synth.get("full_response") or synth.get("final_line") or ""
    if not final_answer and blocks:
        final_answer = blocks[-1].get("final_block_answer", "") or ""

    grade = grade_run(trace=trace, blocks=blocks, final_answer=final_answer,
                      max_plies=args.max_plies)
    print(format_grade(grade))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(grade, indent=2))
        print(f"\nFull grade JSON -> {args.out}")


if __name__ == "__main__":
    main()
