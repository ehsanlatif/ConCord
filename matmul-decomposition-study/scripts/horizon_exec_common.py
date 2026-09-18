"""Shared helpers for the long-horizon *execution* benchmark.

Why this exists
---------------
Chess turned out to be a poor long-horizon testbed because it is Markovian and
fully observable: every ply re-presents the complete ground-truth board, so a
model never has to *carry* state across steps and early mistakes never compound.
Horizon length therefore does not inflate difficulty, and SOTA models saturate
pass@1.

This task fixes that. It reproduces the "retrieve-then-compose" paradigm from
Sinha et al., "The Illusion of Diminishing Returns: Measuring Long Horizon
Execution in LLMs" (arXiv:2509.09677). The model maintains a *running sum*
across many turns:

    S_0 = 0
    S_t = S_{t-1} + sum(D[k] for k in plan_t)

where `D` is a fixed word->int dictionary handed to the model up front and
`plan_t` is a short list of keys given each turn. Knowledge (the dictionary) and
planning (which keys) are provided, so the ONLY thing under test is the ability
to execute a long dependent chain without drifting. Each single step is trivial;
difficulty comes purely from horizon length.

Two properties make this a real long-horizon test:
  * State is model-maintained and non-Markovian — the correct answer at turn t
    depends on the model's own turn t-1 output, not on a re-presented board.
  * Errors compound — one wrong sum corrupts every later turn.

We grade each turn two ways:
  * task-correct: reported sum == ground-truth cumulative sum (this compounds).
  * turn-correct: the *increment* the model applied this turn is right,
    regardless of whether its prior state was correct (isolates single-step
    skill; this is how you show single-step accuracy stays high while task
    accuracy collapses).

The headline metric is the horizon length H_s: the number of steps a model can
execute before whole-task accuracy across samples drops below s (default 0.5).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

_CONSONANTS = "bcdfghjklmnpqrstvwz"
_VOWELS = "aeiou"


def make_dictionary(size: int, seed: int) -> dict[str, int]:
    """Deterministic word->int dictionary. Values are nonzero ints in [-99, 99].

    Words are pronounceable 5-letter CVCVC tokens so the prompt reads like the
    paper's setup, but the exact spellings don't matter — only that they are
    unique, stable keys the model can look up.
    """
    rng = random.Random(seed)
    words: dict[str, int] = {}
    while len(words) < size:
        w = (
            rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)
            + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)
        )
        if w in words:
            continue
        v = 0
        while v == 0:  # keep values nonzero so a dropped term is always detectable
            v = rng.randint(-99, 99)
        words[w] = v
    return words


@dataclass
class Turn:
    """One step of the chain: which keys to add, and the true state after."""

    index: int
    keys: list[str]
    increment: int          # sum of this turn's key values
    cumulative: int         # ground-truth running total through this turn


@dataclass
class Episode:
    """A full independent run: fixed dictionary + a predetermined key schedule."""

    dictionary: dict[str, int]
    turns: list[Turn] = field(default_factory=list)

    @property
    def max_turns(self) -> int:
        return len(self.turns)


def make_episode(
    dictionary: dict[str, int],
    max_turns: int,
    keys_per_turn: int,
    seed: int,
) -> Episode:
    """Build a `max_turns`-long schedule of key lookups with ground-truth sums."""
    rng = random.Random(seed)
    vocab = list(dictionary)
    turns: list[Turn] = []
    running = 0
    for i in range(1, max_turns + 1):
        keys = [rng.choice(vocab) for _ in range(keys_per_turn)]
        inc = sum(dictionary[k] for k in keys)
        running += inc
        turns.append(Turn(index=i, keys=keys, increment=inc, cumulative=running))
    return Episode(dictionary=dictionary, turns=turns)


def dictionary_block(dictionary: dict[str, int]) -> str:
    """Render the dictionary as a compact, unambiguous key=value list."""
    return "\n".join(f"{k} = {v}" for k, v in dictionary.items())


def system_prompt(dictionary: dict[str, int]) -> str:
    return (
        "You are executing a long, multi-turn accumulation task. You maintain a "
        "single running integer TOTAL that starts at 0.\n\n"
        "On each turn I give you a short list of KEYS. You must:\n"
        "  1. Look up each key's integer value in the dictionary below.\n"
        "  2. Add those values to the running TOTAL from the previous turn.\n"
        "  3. Reply with ONLY the new total, in exactly this format on its own "
        "line:\n"
        "     STATE=<integer>\n\n"
        "Do not restate the dictionary. Do not add commentary after the STATE "
        "line. The dictionary is fixed for the whole session:\n\n"
        f"{dictionary_block(dictionary)}"
    )


def turn_prompt(turn: Turn) -> str:
    return (
        f"Turn {turn.index}. Add these keys to the running total: "
        f"{', '.join(turn.keys)}\n"
        "Reply with only: STATE=<integer>"
    )


_STATE_RE = re.compile(r"STATE\s*=\s*(-?\d+)")


def parse_state(text: str) -> int | None:
    """Extract the last STATE=<int> in the reply (models sometimes echo)."""
    matches = _STATE_RE.findall(text or "")
    if matches:
        return int(matches[-1])
    # Fallback: a bare trailing integer.
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def corrupt(value: int, seed: int) -> int:
    """Return a wrong-but-plausible state for self-conditioning injection."""
    rng = random.Random(seed)
    delta = 0
    while delta == 0:
        delta = rng.randint(-20, 20)
    return value + delta


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def horizon_length(task_accuracy_by_len: list[float], threshold: float) -> int:
    """First-crossing horizon H_s.

    `task_accuracy_by_len[i]` is the fraction of samples correct on ALL turns
    1..(i+1). H_s is the largest prefix length L whose accuracy is still >=
    threshold, counting from turn 1 (so a model that already fails turn 1 has
    H_s = 0). Uses first-crossing (not last) so a late fluke recovery doesn't
    inflate the score.
    """
    h = 0
    for length, acc in enumerate(task_accuracy_by_len, start=1):
        if acc >= threshold:
            h = length
        else:
            break
    return h


def summarize_samples(
    per_sample_turns: list[list[dict]], threshold: float
) -> dict:
    """Aggregate per-sample, per-turn grades into curves + the H_s metric.

    `per_sample_turns[s][t]` is a dict with keys 'task' and 'turn' (bools) for
    sample s, turn t. Samples may have different lengths (a run can die early on
    an API error); we aggregate over the shortest common prefix for the curves
    and note the coverage.
    """
    if not per_sample_turns:
        return {"horizon_length": 0, "n_samples": 0}

    min_len = min(len(s) for s in per_sample_turns)
    n = len(per_sample_turns)

    task_acc, turn_acc = [], []
    for t in range(min_len):
        # task accuracy = correct on every turn up to and including t
        survived = sum(
            1 for s in per_sample_turns if all(s[j]["task"] for j in range(t + 1))
        )
        task_acc.append(survived / n)
        turn_acc.append(sum(1 for s in per_sample_turns if s[t]["turn"]) / n)

    return {
        "n_samples": n,
        "evaluated_length": min_len,
        "threshold": threshold,
        "horizon_length": horizon_length(task_acc, threshold),
        "task_accuracy_by_len": [round(x, 4) for x in task_acc],
        "turn_accuracy_by_turn": [round(x, 4) for x in turn_acc],
        "final_turn_accuracy": round(turn_acc[-1], 4) if turn_acc else 0.0,
    }
