"""Shared helpers for the matrix-chain long-horizon *execution* benchmark.

Why this exists
---------------
`horizon_exec_common.py` measures long-horizon execution with a running *scalar*
sum: state is a single integer, per-step work is fixed at K lookups+adds, and the
only knob is HORIZON LENGTH (number of turns). It has no clean way to also dial
*per-step complexity* independently.

This module generalizes that task to a running *matrix product mod p*:

    M_0 = I_d                       (the d x d identity)
    M_t = (M_{t-1} . A_t) mod p     (A_t is a fresh d x d integer matrix / turn)

The model maintains the running matrix STATE across many turns; each turn it is
handed one matrix A_t, multiplies its current STATE on the right by it, reduces
mod p, and reports the whole matrix. This gives us TWO orthogonal difficulty
axes:

  * HORIZON  T = number of matrices in the chain (number of turns). Exactly the
    same compounding, state-carrying, non-Markovian horizon as the scalar task —
    one wrong entry corrupts every later product. Reading the accuracy curve at
    every prefix length of one long run gives H_s (see below).

  * COMPLEXITY  d = matrix dimension. Each step is a d x d by d x d matmul: d^2
    output entries, each a length-d dot product mod p, i.e. ~d^3 scalar
    multiply-adds per step. d is a clean per-step arithmetic-load dial that is
    independent of the horizon T. (A secondary complexity knob is the modulus p:
    larger p = harder individual multiplies.)

The scalar running-sum task is the degenerate d=1 case of this (a running
*product* of 1x1 matrices mod p), so results stay conceptually comparable while
adding the complexity axis.

Why this is a legitimate long-horizon test (same two properties as the paper's):
  * State is model-maintained and non-Markovian — the correct matrix at turn t
    depends on the model's own turn t-1 matrix, not on anything re-presented.
  * Errors compound — one wrong entry propagates through every later product.

We grade each turn two ways, mirroring horizon_exec_common:
  * task-correct: reported matrix == ground-truth cumulative product (compounds).
  * turn-correct: the single multiply the model applied this turn is right
    relative to the matrix it could actually SEE last turn, regardless of whether
    that prior matrix was itself correct (isolates single-step skill).

The headline metric is the horizon length H_s: the number of steps a model can
execute before whole-task accuracy across samples drops below s (default 0.5).
Both H_s (reused verbatim) and the per-sample/per-turn aggregation come from
horizon_exec_common so the two benchmarks report identical shapes.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

# Reuse the metric layer verbatim: these operate purely on per-turn grade dicts
# with 'task'/'turn' booleans, so they are task-agnostic.
from horizon_exec_common import horizon_length, summarize_samples  # noqa: F401

Matrix = list[list[int]]


# --------------------------------------------------------------------------- #
# Modular matrix arithmetic (pure Python so grading is exact; d is small)
# --------------------------------------------------------------------------- #

def identity(dim: int) -> Matrix:
    return [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]


def matmul_mod(a: Matrix, b: Matrix, p: int) -> Matrix:
    """Standard matrix product reduced entrywise into [0, p-1]."""
    n = len(a)
    k = len(b)
    m = len(b[0])
    return [
        [sum(a[i][l] * b[l][j] for l in range(k)) % p for j in range(m)]
        for i in range(n)
    ]


def random_matrix(dim: int, p: int, rng: random.Random,
                  nonzero: bool = False) -> Matrix:
    """A dense d x d matrix with entries uniform in [0, p-1] (or [1, p-1] when
    `nonzero`). For d=1 with a prime modulus, nonzero entries keep the running
    scalar product in the multiplicative group Z_p* — it never hits the absorbing
    0, so every turn stays a genuine multiplication (no trivial 0-tail)."""
    lo = 1 if nonzero else 0
    return [[rng.randrange(lo, p) for _ in range(dim)] for _ in range(dim)]


def matrices_equal(a: Matrix | None, b: Matrix | None) -> bool:
    return a is not None and b is not None and a == b


def hamming(a: Matrix | None, b: Matrix | None) -> int | None:
    """Number of differing entries (a scalar drift signal for live logging)."""
    if a is None or b is None:
        return None
    if len(a) != len(b) or any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        return None
    return sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y)


# --------------------------------------------------------------------------- #
# Episode structure
# --------------------------------------------------------------------------- #

@dataclass
class MatTurn:
    """One step of the chain: the matrix to multiply in, and the true state after."""

    index: int
    matrix: Matrix          # A_t handed to the model this turn
    cumulative: Matrix      # ground-truth running product M_t through this turn


@dataclass
class MatEpisode:
    """A full independent run: fixed (dim, modulus) + a predetermined schedule."""

    dim: int
    modulus: int
    turns: list[MatTurn] = field(default_factory=list)

    @property
    def max_turns(self) -> int:
        return len(self.turns)


def make_chain(dim: int, n_turns: int, modulus: int, seed: int,
               nonzero: bool = False) -> MatEpisode:
    """Build an `n_turns`-long chain of d x d matrices with ground-truth products.

    Deterministic in `seed`: the same seed yields the same matrix schedule, so
    every model sees an identical task instance (apples-to-apples comparison).
    `nonzero` draws entries from [1, modulus-1] — use it for d=1 to avoid the
    absorbing-0 degeneracy (see random_matrix)."""
    rng = random.Random(seed)
    running = identity(dim)
    turns: list[MatTurn] = []
    for i in range(1, n_turns + 1):
        a = random_matrix(dim, modulus, rng, nonzero=nonzero)
        running = matmul_mod(running, a, modulus)   # fresh object each iteration
        turns.append(MatTurn(index=i, matrix=a, cumulative=running))
    return MatEpisode(dim=dim, modulus=modulus, turns=turns)


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #

def matrix_str(m: Matrix) -> str:
    """Compact one-line JSON rendering, e.g. [[1,2],[3,4]] — unambiguous to parse."""
    return "[" + ",".join("[" + ",".join(str(x) for x in row) + "]" for row in m) + "]"


def system_prompt(dim: int, modulus: int) -> str:
    ident = matrix_str(identity(dim))
    return (
        f"You are executing a long, multi-turn matrix-accumulation task. You "
        f"maintain a single running {dim}x{dim} integer matrix STATE. It starts "
        f"as the identity matrix:\n"
        f"    STATE = {ident}\n\n"
        f"On each turn I give you one {dim}x{dim} matrix A. You must:\n"
        f"  1. Multiply your current STATE on the RIGHT by A: new = STATE . A "
        f"(ordinary matrix multiplication).\n"
        f"  2. Reduce EVERY entry of the result modulo {modulus}, into the range "
        f"0..{modulus - 1}.\n"
        f"  3. Reply with ONLY the new STATE, as a JSON-style nested list on its "
        f"own line, in exactly this format:\n"
        f"     STATE=[[...],[...],...]\n\n"
        f"Rows are outer, columns are inner. Do not add commentary after the "
        f"STATE line. The modulus {modulus} and size {dim}x{dim} are fixed for "
        f"the whole session."
    )


def turn_prompt(turn: MatTurn, modulus: int) -> str:
    return (
        f"Turn {turn.index}. Multiply the running STATE on the right by this "
        f"matrix, then reduce mod {modulus}:\n"
        f"A={matrix_str(turn.matrix)}\n"
        f"Reply with only: STATE=[[...],...]"
    )


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #

_STATE_RE = re.compile(r"STATE\s*=\s*(\[.*\])", re.DOTALL)


def _reshape(nums: list[int], dim: int) -> Matrix | None:
    if len(nums) < dim * dim:
        return None
    nums = nums[-dim * dim:]          # models sometimes echo A; trust the last block
    return [nums[i * dim:(i + 1) * dim] for i in range(dim)]


def parse_state(text: str, dim: int) -> Matrix | None:
    """Extract the last STATE=<matrix> reply as a dim x dim int matrix.

    Prefers a clean JSON nested list after the last `STATE=`; falls back to
    grabbing the trailing dim*dim integers and reshaping (handles minor
    formatting drift without silently accepting the wrong shape).
    """
    text = text or ""
    matches = _STATE_RE.findall(text)
    if matches:
        blob = matches[-1]
        # Trim to the first balanced [...] so trailing prose doesn't break json.
        depth = 0
        end = None
        for i, ch in enumerate(blob):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = blob[:end] if end else blob
        try:
            m = json.loads(candidate)
            if (isinstance(m, list) and len(m) == dim
                    and all(isinstance(r, list) and len(r) == dim for r in m)
                    and all(isinstance(x, int) for r in m for x in r)):
                return m
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: last dim*dim integers in the whole reply.
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    return _reshape(nums, dim)


# --------------------------------------------------------------------------- #
# Self-conditioning corruption
# --------------------------------------------------------------------------- #

def corrupt(matrix: Matrix, modulus: int, seed: int) -> Matrix:
    """Return a wrong-but-plausible matrix (one entry perturbed) for injection."""
    rng = random.Random(seed)
    dim = len(matrix)
    out = [row[:] for row in matrix]
    i, j = rng.randrange(dim), rng.randrange(dim)
    delta = 0
    while delta % modulus == 0:
        delta = rng.randint(1, modulus - 1)
    out[i][j] = (out[i][j] + delta) % modulus
    return out
