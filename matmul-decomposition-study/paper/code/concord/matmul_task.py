"""Matrix-chain long-horizon task for Concord (the matmul analogue of the
`uci_to_fen` chess study).

The task: given an ordered sequence of `T` integer matrices A_1..A_T (each
`d x d`, entries in [0, p-1]), compute the running product modulo p

    M_0 = I_d ,   M_t = (M_{t-1} . A_t) mod p ,   report M_T.

This is a compounding, non-Markovian, long-horizon computation — one wrong
entry corrupts every later product — exactly the property the decomposition
study wants to stress. Concord's pipeline solver is handed the WHOLE chain as
one problem and must Split it into sub-chains, Solve each, and Combine, all
under a fixed per-node call/token budget.

This module provides:
  * `GroundTruth` — loads a golden chain from the static dataset in
    `data/matmul/` (produced by the parent repo's make_matmul_dataset.py),
    truncates it to `T` turns, and exposes exact modular arithmetic so the
    grader can verify any slice against the frozen labels.
  * `build_problem_prompt` — renders the chain as one self-contained problem
    for the solver, with matrices in an explicit ordered list.
  * matrix parsing / arithmetic helpers shared with the grader.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

Matrix = list[list[int]]


# --------------------------------------------------------------------------- #
# Exact modular matrix arithmetic (pure Python; d is small)
# --------------------------------------------------------------------------- #

def identity(dim: int) -> Matrix:
    return [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]


def matmul_mod(a: Matrix, b: Matrix, p: int) -> Matrix:
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][l] * b[l][j] for l in range(k)) % p for j in range(m)]
            for i in range(n)]


def matrices_equal(a: Matrix | None, b: Matrix | None) -> bool:
    return a is not None and b is not None and a == b


def matrix_str(m: Matrix) -> str:
    """Compact one-line JSON: [[1,2],[3,4]] — unambiguous to parse."""
    return "[" + ",".join(
        "[" + ",".join(str(x) for x in row) + "]" for row in m) + "]"


# --------------------------------------------------------------------------- #
# Matrix parsing out of free-form model / atom text
# --------------------------------------------------------------------------- #

def find_matrices(text: str, dim: int) -> list[Matrix]:
    """Return every `dim x dim` integer matrix that appears in `text`, in
    order. Scans for balanced `[...]` blocks and json-loads each, keeping the
    ones whose shape is exactly dim x dim with integer entries. Tolerant of
    surrounding prose, `STATE=`/`solution =` prefixes, and nested spacing."""
    text = text or ""
    out: list[Matrix] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            c = text[j]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        blob = text[i:j + 1]
        i = j + 1
        # Only attempt json on plausibly-matrix blobs (two open brackets in).
        if not blob.startswith("[["):
            continue
        try:
            m = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if (isinstance(m, list) and len(m) == dim
                and all(isinstance(r, list) and len(r) == dim for r in m)
                and all(isinstance(x, int) for r in m for x in r)):
            out.append(m)
    return out


def last_matrix(text: str, dim: int) -> Matrix | None:
    """The final `dim x dim` matrix in `text` (models sometimes echo inputs
    before the answer, so the last block is the reported result). Falls back
    to reshaping the trailing dim*dim integers if no clean block is found."""
    ms = find_matrices(text, dim)
    if ms:
        return ms[-1]
    nums = [int(x) for x in re.findall(r"-?\d+", text or "")]
    if len(nums) >= dim * dim:
        nums = nums[-dim * dim:]
        return [nums[r * dim:(r + 1) * dim] for r in range(dim)]
    return None


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

@dataclass
class GroundTruth:
    """One golden chain from the static dataset, truncated to `T` turns."""

    dim: int
    modulus: int
    matrices: list[Matrix]       # A_1..A_T   (matrices[t-1] == A_t)
    golden: list[Matrix]         # M_1..M_T   (golden[t-1]  == M_t)
    sample_id: int

    @property
    def T(self) -> int:
        return len(self.matrices)

    @classmethod
    def load(cls, data_dir: Path, dim: int, sample_id: int,
             max_turns: int) -> "GroundTruth":
        fpath = Path(data_dir) / f"chains_d{dim}.jsonl"
        if not fpath.exists():
            raise SystemExit(
                f"No dataset for d={dim} at {fpath}. Generate it with the "
                f"parent repo's make_matmul_dataset.py, or point --data-dir "
                f"at data/matmul/.")
        rec = None
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("id", 0) == sample_id:
                    rec = r
                    break
        if rec is None:
            raise SystemExit(
                f"sample id {sample_id} not found in {fpath}")
        T = min(max_turns, rec["n_turns"])
        return cls(dim=rec["dim"], modulus=rec["modulus"],
                   matrices=[rec["matrices"][i] for i in range(T)],
                   golden=[rec["golden"][i] for i in range(T)],
                   sample_id=sample_id)

    # -- exact-arithmetic accessors the grader uses -----------------------

    def gold_final(self) -> Matrix:
        """M_T — the correct final answer."""
        return self.golden[self.T - 1]

    def state_before(self, t: int) -> Matrix:
        """M_{t-1}: the running state entering step t (identity if t==1)."""
        return identity(self.dim) if t <= 1 else self.golden[t - 2]

    def index_of(self) -> dict[tuple, int]:
        """value(A_t) -> t  (1-indexed). Entries are ~uniform mod p, so
        collisions are negligible; on the rare tie we keep the first."""
        idx: dict[tuple, int] = {}
        for t, a in enumerate(self.matrices, start=1):
            key = tuple(tuple(row) for row in a)
            idx.setdefault(key, t)
        return idx


# --------------------------------------------------------------------------- #
# Problem prompt (the whole chain, handed to the solver as ONE problem)
# --------------------------------------------------------------------------- #

def build_problem_prompt(gt: GroundTruth) -> str:
    d, p, T = gt.dim, gt.modulus, gt.T
    ident = matrix_str(identity(d))
    lines = [
        f"You must compute a running matrix product modulo {p}.",
        "",
        f"You are given an ordered sequence of {T} matrices, each {d}x{d}, "
        f"with integer entries in 0..{p - 1}. Starting from the {d}x{d} "
        f"identity matrix, multiply them IN ORDER on the right, reducing "
        f"every entry modulo {p} after each multiplication:",
        f"    M_0 = {ident}",
        f"    M_t = (M_(t-1) . A_t) mod {p}     for t = 1, 2, ..., {T}",
        "",
        f"Compute the final matrix M_{T}. Rows are outer, columns inner.",
        "",
        "The matrices, in order:",
    ]
    for t, a in enumerate(gt.matrices, start=1):
        lines.append(f"A_{t} = {matrix_str(a)}")
    lines += [
        "",
        f"Report ONLY the final {d}x{d} matrix M_{T}, every entry in "
        f"0..{p - 1}, on its own line in this exact format:",
        "    solution = [[...],[...],...]",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Index-reference decomposition: splitter emits ranges, code injects values
# --------------------------------------------------------------------------- #

# Splitter system-prompt override for this task. The splitter partitions the
# contiguous INPUT RANGE into sub-ranges and references them by `source_span`
# — it must NEVER copy matrix values (that caused transcription errors and
# token blow-up). The exact values are injected at execution by the resolver.
MATMUL_SPLITTER_SYSTEM = (
    "You are a problem DECOMPOSER for a running MATRIX PRODUCT task. The "
    "problem post-multiplies an incoming state matrix S by an ordered sequence "
    "of input matrices A_1..A_T (modulo p) — result = S . A_1 . A_2 . ... . A_T "
    "(state on the LEFT, inputs applied on the RIGHT in ascending index order) "
    "— and reports the final matrix.\n\n"
    "Split the CONTIGUOUS input range this block covers into 2-{max_atoms} "
    "contiguous, non-overlapping sub-ranges that tile it in order.\n\n"
    "CRITICAL RULES:\n"
    "  1. Identify the input range [i, j] this block covers. If it is the "
    "whole problem, that is A_1..A_T (T is stated in the problem text). If the "
    "block text says 'inputs A_i..A_j', use exactly that range.\n"
    "  2. Partition [i, j] into 2-{max_atoms} contiguous sub-ranges "
    "[i,k1],[k1+1,k2],...,[..,j]. Each sub-range becomes one atom.\n"
    "  3. For each atom set \"source_span\": [lo, hi] (1-based inclusive) — the "
    "input indices it covers. DO NOT copy, transcribe, or invent the matrix "
    "VALUES; reference them ONLY by this range. The exact values are supplied "
    "to the solver automatically.\n"
    "  4. Each atom's \"question\" MUST state the ACTUAL numeric indices of its "
    "sub-range, e.g. 'Post-multiply the incoming state S by inputs A_26..A_50 "
    "in order (mod p): result = S . A_26 . A_27 . ... . A_50.' Use the real "
    "numbers (matching source_span) — NEVER the literal words 'lo'/'hi'. Put NO "
    "matrix values in the question. (Recursive splitting reads this range, so it "
    "must be concrete.)\n"
    "  5. State threading: atom 0 starts from this block's incoming state. "
    "Atom k (k>0) consumes atom (k-1)'s answer as its incoming state → set "
    "\"refs\": [k-1]. This chains the running product in order.\n"
    "  6. The LAST atom's answer IS this block's final matrix.\n"
    "  7. Return a SINGLE atom covering the whole [i, j] ONLY when i == j (a "
    "single input matrix).\n"
    "  8. You NEVER compute products — you only partition ranges.\n"
    "  9. Output strict JSON, no prose. Example for a block covering A_1..A_50 "
    "(use YOUR block's real numbers):\n"
    "       { \"atoms\": [ {\"id\": 0, \"question\": \"Post-multiply the incoming "
    "state S by inputs A_1..A_25 in order (mod p): result = S . A_1 . ... . "
    "A_25.\", \"source_span\": [1, 25], \"refs\": []}, {\"id\": 1, \"question\": "
    "\"Post-multiply the incoming state S by inputs A_26..A_50 in order (mod p): "
    "result = S . A_26 . ... . A_50.\", \"source_span\": [26, 50], \"refs\": "
    "[0]} ] }\n"
    "  10. Hard cap: at most {max_atoms} sub-ranges at THIS level; finer "
    "detail comes from automatic recursive splitting of each sub-range.\n"
)

_A_INDEX_RE = re.compile(r"A[_ ]?(\d+)")


def span_from_text(text: str) -> "tuple[int, int] | None":
    """Fallback: recover the [min, max] A-index range mentioned in text."""
    idxs = [int(m) for m in _A_INDEX_RE.findall(text or "")]
    if not idxs:
        return None
    return (min(idxs), max(idxs))


def build_source_resolver(gt: GroundTruth):
    """Return `resolve(atom, atom_q) -> str` that injects the EXACT input
    matrices an atom references by index (`source_span` or, failing that, the
    A-indices named in its text) — copied by code from the frozen chain, never
    re-typed by the splitter. Atoms with no discernible range pass through."""
    T = gt.T

    def resolve(atom, atom_q: str) -> str:
        span = getattr(atom, "source_span", None)
        if not span:
            span = span_from_text(getattr(atom, "question", "") or atom_q)
        if not span:
            return atom_q
        i, j = int(span[0]), int(span[1])
        i = max(1, i)
        j = min(T, j)
        if j < i:
            return atom_q
        lines = [f"A_{t} = {matrix_str(gt.matrices[t - 1])}" for t in range(i, j + 1)]
        chain = " . ".join(f"A_{t}" for t in range(i, j + 1))
        block = (
            f"\n\nInput matrices for this step (indices {i}..{j}, authoritative "
            f"values — use these EXACTLY):\n" + "\n".join(lines) + "\n\n"
            f"MULTIPLICATION ORDER (critical): the incoming state S is on the "
            f"LEFT; each input matrix is applied on the RIGHT (post-multiply), "
            f"in ascending index order. Compute:\n"
            f"    result = ((S . {chain}) mod {gt.modulus})\n"
            f"i.e. first S . A_{i}, then that . A_{i + 1}, and so on. "
            f"Do NOT compute A . S (that is the wrong order — matrix "
            f"multiplication is not commutative). Reduce mod {gt.modulus} "
            f"after each step."
        )
        return atom_q + block

    return resolve


def build_combine_resolver(gt: "GroundTruth | None" = None):
    """Return `combine(children) -> str | None` for a running-state matrix
    chain. Because M_t = (M_{t-1} . A_t), a block's answer is exactly its LAST
    child's answer — the child covering the highest input index. So we pick that
    child (by `source_span` hi, else by the A-index range parsed from its
    question, else by emitted order) and return its answer verbatim. The LLM
    combiner is not just redundant here but actively harmful (it re-multiplies
    and corrupts a correct chained result).

    Returns None to fall back to the LLM combiner when the designated last child
    has no answer yet (e.g. budget-truncated) — so an incomplete chain is never
    masked by a stale earlier partial. `gt` is accepted for symmetry with
    `build_source_resolver` but is not needed (this is pure aggregation)."""

    def _hi(child) -> "int | None":
        sp = child.get("source_span")
        if sp:
            try:
                return int(sp[1])
            except (TypeError, ValueError, IndexError):
                pass
        sp2 = span_from_text(child.get("question", "") or "")
        return sp2[1] if sp2 else None

    def combine(children) -> "str | None":
        if not children:
            return None
        his = [_hi(c) for c in children]
        if any(h is not None for h in his):
            # child covering the highest index; ties → later emitted wins
            best_idx = max(
                range(len(children)),
                key=lambda i: (his[i] if his[i] is not None else -1, i),
            )
        else:
            best_idx = len(children) - 1          # no span info → last emitted
        ans = (children[best_idx].get("answer") or "").strip()
        return ans or None

    return combine
