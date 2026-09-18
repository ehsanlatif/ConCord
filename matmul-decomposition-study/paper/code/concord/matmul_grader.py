"""Grade a Concord run on the matrix-chain task at every decomposition level.

Mirrors `chess_grader.py`. Four levels (see MATMUL_STUDY.md):

  L0 overall     — is the final answer M_T exactly the golden M_T?
  L1 per-atom    — did each atomic unit compute its slice correctly? We match
                   the matrices appearing in the atom's question back to the
                   known chain to recover the contiguous slice [i..j] it
                   covers, then compare the atom's answer against BOTH the
                   running-state result M_j (golden) and the standalone
                   partial product A_i..A_j. PASS if it matches either style.
  L2 fidelity    — did the splitter hand each atom a correct CONTIGUOUS slice,
                   and do the atoms tile the whole chain 1..T?
  L3 per-depth   — L1 pass-rate aggregated by split-tree depth.

`UNGRADEABLE`: no chain matrices found in the atom (e.g. a pure "combine the
sub-answers" step) or the answer is unparseable — reported separately, never
counted as wrong (identical policy to the chess grader).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matmul_task import (
    GroundTruth,
    find_matrices,
    identity,
    last_matrix,
    matmul_mod,
    matrices_equal,
)


# --------------------------------------------------------------------------- #
# Provenance extraction (task-agnostic; same shape as chess_grader)
# --------------------------------------------------------------------------- #

def blocks_from_result_cost(cost: dict) -> list[dict]:
    """Pull the pipeline `blocks` list out of a solve result's cost dict.

    Single-block runs carry it inline at cost["pipeline_provenance"]["blocks"];
    multi-block runs persist it to the combine_tree file referenced by
    cost["combine_tree_path"]. Returns [] if neither is present (e.g. the
    legacy MCTS solver, which emits no pipeline provenance)."""
    cost = cost or {}
    prov = cost.get("pipeline_provenance")
    if isinstance(prov, dict) and isinstance(prov.get("blocks"), list):
        return prov["blocks"]
    ctp = cost.get("combine_tree_path")
    if ctp and Path(ctp).exists():
        try:
            data = json.loads(Path(ctp).read_text())
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, dict) and isinstance(data.get("blocks"), list):
            return data["blocks"]
    return []


# --------------------------------------------------------------------------- #
# Split-tree depth map
# --------------------------------------------------------------------------- #

def _depth_from_atom_id(atom_id: str) -> int:
    """Fallback depth: recursion adds one `/aN` segment per level below the
    block root `.../d0/a0`, so depth = (#'/a' segments) - 1, floored at 0."""
    return max(0, atom_id.count("/a") - 1)


def _walk_split_tree(atoms: list[dict], out: dict[str, int]) -> None:
    for a in atoms or []:
        aid = a.get("atom_id")
        if aid is not None:
            out[aid] = int(a.get("depth", 0))
        kids = a.get("children") or []
        if kids:
            _walk_split_tree(kids, out)


def _depth_map(block: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    st = block.get("split_tree") or {}
    _walk_split_tree(st.get("atoms") or [], out)
    return out


# --------------------------------------------------------------------------- #
# Per-atom grading
# --------------------------------------------------------------------------- #

def _contiguous_run(indices: list[int]) -> tuple[int, int] | None:
    """If the sorted unique indices form a gap-free run, return (i, j); else
    None. Non-contiguous ⇒ the atom mixed non-adjacent chain positions."""
    if not indices:
        return None
    s = sorted(set(indices))
    if s == list(range(s[0], s[-1] + 1)):
        return (s[0], s[-1])
    return None


def grade_atom(atom: dict, gt: GroundTruth, idx: dict[tuple, int]) -> dict:
    """Grade one atomic unit. Returns a record with status PASS/FAIL/UNGRADEABLE.

    Strategy: identify WHICH chain matrices the atom uses by matching every
    matrix in its question against the known chain (by value). If they form a
    contiguous slice [i..j], compare the atom's answer against the exact golden
    result for that slice — so we grade the atom's arithmetic against frozen
    labels, not against its own (possibly wrong) framing.
    """
    q = atom.get("question", "") or ""
    ans_text = atom.get("answer", "") or ""

    # Preferred (index-reference mode): the atom declares the input range it
    # covers via `source_span` [i, j], so we grade against that slice directly.
    hit_indices: list[int] = []
    span = atom.get("source_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            i0, j0 = int(span[0]), int(span[1])
            if i0 >= 1 and j0 >= i0:
                hit_indices = list(range(max(1, i0), min(gt.T, j0) + 1))
        except (ValueError, TypeError):
            hit_indices = []
    # Fallback (legacy inline mode): match matrices in the question by value.
    if not hit_indices:
        for m in find_matrices(q, gt.dim):
            key = tuple(tuple(row) for row in m)
            t = idx.get(key)
            if t is not None:
                hit_indices.append(t)

    ans = last_matrix(ans_text, gt.dim)

    if not hit_indices:
        return {"atom_id": atom.get("atom_id"), "status": "UNGRADEABLE",
                "reason": "no source_span or chain matrices in question",
                "slice": None, "contiguous": False}
    if ans is None:
        return {"atom_id": atom.get("atom_id"), "status": "UNGRADEABLE",
                "reason": "no parseable answer matrix",
                "slice": [min(hit_indices), max(hit_indices)],
                "contiguous": _contiguous_run(hit_indices) is not None}

    run = _contiguous_run(hit_indices)
    contiguous = run is not None
    if run is None:
        # We can still grade non-contiguous atoms as FAIL of fidelity, but the
        # "expected" slice is ill-defined — grade against the running-state
        # result at the max index reached (best-effort) and flag non-contiguity.
        i, j = min(hit_indices), max(hit_indices)
    else:
        i, j = run

    # Candidate 1: running-state framing — answer should be M_j = golden[j-1].
    expected_running = gt.golden[j - 1]
    # Candidate 2: standalone partial product A_i..A_j from identity.
    partial = identity(gt.dim)
    for t in range(i, j + 1):
        partial = matmul_mod(partial, gt.matrices[t - 1], gt.modulus)

    if matrices_equal(ans, expected_running):
        status, style = "PASS", "running_state"
    elif matrices_equal(ans, partial):
        status, style = "PASS", "partial_product"
    else:
        status, style = "FAIL", None

    return {"atom_id": atom.get("atom_id"), "status": status, "style": style,
            "slice": [i, j], "contiguous": contiguous,
            "n_factors": len(hit_indices)}


def _collect_spans(atoms: list[dict], out: dict[str, list]) -> None:
    """atom_id -> source_span, walking the split tree (incl. children)."""
    for a in atoms or []:
        aid = a.get("atom_id")
        if aid and a.get("source_span"):
            out[aid] = a["source_span"]
        kids = a.get("children") or []
        if kids:
            _collect_spans(kids, out)


def grade_block(block: dict, gt: GroundTruth, idx: dict[tuple, int]) -> dict:
    depth_map = _depth_map(block)
    # source_span lives on the split tree (serialized), not on atomic_answers;
    # bridge it across by atom_id so grade_atom can grade by index range.
    span_map: dict[str, list] = {}
    _collect_spans((block.get("split_tree") or {}).get("atoms") or [], span_map)
    atoms = block.get("atomic_answers") or []
    records = []
    covered: dict[int, int] = {}          # chain index -> times covered
    for a in atoms:
        aid = a.get("atom_id", "")
        if a.get("source_span") is None and aid in span_map:
            a = {**a, "source_span": span_map[aid]}   # copy; don't mutate source
        rec = grade_atom(a, gt, idx)
        aid = rec.get("atom_id") or a.get("atom_id", "")
        rec["depth"] = depth_map.get(aid, _depth_from_atom_id(aid))
        records.append(rec)
        sl = rec.get("slice")
        if rec["status"] in ("PASS", "FAIL") and rec.get("contiguous") and sl:
            for t in range(sl[0], sl[1] + 1):
                covered[t] = covered.get(t, 0) + 1
    return {"records": records, "covered": covered}


# --------------------------------------------------------------------------- #
# Run-level grading
# --------------------------------------------------------------------------- #

def grade_run(gt: GroundTruth, blocks: list[dict],
              final_answer: str) -> dict:
    idx = gt.index_of()

    # L0 — overall final answer.
    final_mat = last_matrix(final_answer or "", gt.dim)
    overall = matrices_equal(final_mat, gt.gold_final())

    # L1/L3 — per-atom across all blocks.
    all_recs: list[dict] = []
    covered: dict[int, int] = {}
    for b in blocks:
        gb = grade_block(b, gt, idx)
        all_recs.extend(gb["records"])
        for t, c in gb["covered"].items():
            covered[t] = covered.get(t, 0) + c

    gradeable = [r for r in all_recs if r["status"] in ("PASS", "FAIL")]
    n_pass = sum(1 for r in gradeable if r["status"] == "PASS")
    n_fail = sum(1 for r in gradeable if r["status"] == "FAIL")
    n_ung = sum(1 for r in all_recs if r["status"] == "UNGRADEABLE")
    apr = (n_pass / len(gradeable)) if gradeable else None

    # L2 — decomposition fidelity: contiguous-slice rate + chain coverage.
    matrix_bearing = [r for r in all_recs if r.get("slice") is not None]
    contig = [r for r in matrix_bearing if r.get("contiguous")]
    contiguity_rate = (len(contig) / len(matrix_bearing)
                       if matrix_bearing else None)
    covered_once = sum(1 for t in range(1, gt.T + 1) if covered.get(t, 0) >= 1)
    chain_coverage = covered_once / gt.T if gt.T else None
    overlap = sum(1 for t in range(1, gt.T + 1) if covered.get(t, 0) > 1)
    # A single fidelity scalar for the summary table (mean of the two signals
    # when both exist), keeping the finer parts in the record too.
    if contiguity_rate is not None and chain_coverage is not None:
        fidelity = round((contiguity_rate + chain_coverage) / 2, 4)
    else:
        fidelity = chain_coverage if chain_coverage is not None else None

    # L3 — per-depth aggregation.
    levels: dict[int, dict] = {}
    for r in all_recs:
        d = int(r.get("depth", 0))
        lv = levels.setdefault(d, {"pass": 0, "gradeable": 0, "ungradeable": 0})
        if r["status"] == "PASS":
            lv["pass"] += 1
            lv["gradeable"] += 1
        elif r["status"] == "FAIL":
            lv["gradeable"] += 1
        else:
            lv["ungradeable"] += 1
    for lv in levels.values():
        lv["pass_rate"] = (lv["pass"] / lv["gradeable"]
                           if lv["gradeable"] else None)
    levels = {k: levels[k] for k in sorted(levels)}

    return {
        "overall_correct": bool(overall),
        "overall_board_match": bool(overall),   # no partial-credit analogue
        "final_answer_matrix": final_mat,
        "gold_final": gt.gold_final(),
        "atom_pass_rate": apr,
        "n_atoms": len(all_recs),
        "n_gradeable": len(gradeable),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_ungradeable": n_ung,
        "decomposition_fidelity": fidelity,
        "contiguity_rate": contiguity_rate,
        "chain_coverage": chain_coverage,
        "overlap_positions": overlap,
        "levels": levels,
        "atom_records": all_recs,
    }


# --------------------------------------------------------------------------- #
# CLI: grade a persisted combine_tree.json offline
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Grade a matmul Concord run.")
    p.add_argument("--combine-tree", type=Path, required=True)
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/matmul"))
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--max-turns", type=int, required=True)
    p.add_argument("--final-answer", type=str, default="")
    args = p.parse_args()

    gt = GroundTruth.load(args.data_dir, args.dim, args.sample_id,
                          args.max_turns)
    data = json.loads(args.combine_tree.read_text())
    blocks = data.get("blocks", []) if isinstance(data, dict) else []
    grade = grade_run(gt, blocks, args.final_answer)
    print(json.dumps({k: v for k, v in grade.items()
                      if k not in ("atom_records", "final_answer_matrix",
                                   "gold_final")}, indent=2))


if __name__ == "__main__":
    main()
