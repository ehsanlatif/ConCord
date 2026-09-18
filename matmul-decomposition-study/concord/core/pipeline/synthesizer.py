"""LLMSynthesizer — block answers → `solution = [v1, ..., vN]` list.

Runs once per QUESTION at the end of `solve_multi`. Reads the full
problem text, every committed block answer (from shared memory), and a
heuristic candidate `expected_length` derived from the block-level DAG
(default: number of graph sinks). Emits the consolidated final answer
in the format the LongCoT grader expects:

    solution = [v1, v2, ..., vN]

The synthesizer is allowed to OVERRIDE the heuristic length if the
problem text makes the true list length unambiguous (e.g. "find all
of {A, B, C, D}" → N=4).

A separate synth-verifier reads the produced list and either
acknowledges or proposes a structural fix (e.g. swap two positions,
re-extract a numeric value). The orchestrator commits the final list
after the verifier returns is_consistent=True or after one retry.

Both are sampled n=1 (deterministic enough for a final step) using the
opus role models pinned in wide_and_deep.yaml.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import networkx as nx

from ..config import Config
from ..llm.client import LLMClient


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic: expected list length from the block-level DAG
# ---------------------------------------------------------------------------

def heuristic_expected_length(G: nx.DiGraph) -> int:
    """Best-effort guess of how many values the grader expects.

    Heuristic: count nodes with out-degree 0 (graph sinks). These are
    the "terminal" blocks whose answers don't feed any downstream
    block — typical "final answer" positions in the eval set's
    backtracking templates.

    Returns at least 1.
    """
    if G is None or G.number_of_nodes() == 0:
        return 1
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    return max(1, len(sinks))


# ---------------------------------------------------------------------------
# Synthesizer prompt
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = (
    "You are producing the FINAL answer for a multi-part problem.\n\n"
    "You are given:\n"
    "  (1) the ORIGINAL whole-problem text,\n"
    "  (2) each sub-block's committed answer in the form\n"
    "        node_K: <answer>\n"
    "  (3) a HEURISTIC candidate length for the final answer list. You\n"
    "      may override it if the problem text dictates otherwise.\n\n"
    "OUTPUT CONTRACT — read carefully, this is graded literally:\n"
    "  The very LAST line of your response MUST be exactly\n"
    "        solution = [v1, v2, ..., vN]\n"
    "  where each v_i is a SINGLE expression (a number, a fraction, a\n"
    "  short symbolic form). NO prose inside the list. Use COMMAS to\n"
    "  separate. The OUTER brackets are LITERAL `[` and `]`.\n\n"
    "Before that final line you may write a few short lines explaining\n"
    "which sub-block answers fill which positions of the list. Keep it\n"
    "concise (≤ 10 lines).\n\n"
    "Choose the values from the sub-block answers — do not re-derive\n"
    "them unless the sub-block answer is clearly malformed. Order MUST\n"
    "follow whatever ordering the original problem implies.\n"
)


def _format_blocks(block_answers: dict[str, str]) -> str:
    if not block_answers:
        return "(none — no blocks were solved)"
    # Sort by node id where possible (node_0 < node_1 < ... < node_10 < ...).
    def _key(k: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", k)
        return (int(m.group(1)) if m else 999_999, k)
    return "\n".join(f"  {k}: {v}" for k, v in sorted(block_answers.items(),
                                                       key=lambda kv: _key(kv[0])))


def _build_synth_prompt(*, problem_text: str,
                         block_answers: dict[str, str],
                         expected_length_hint: int) -> str:
    return (
        _SYNTH_SYSTEM
        + "\n\nOriginal problem:\n"
        + problem_text.strip()
        + "\n\nSub-block answers:\n"
        + _format_blocks(block_answers)
        + f"\n\nHeuristic candidate length: {expected_length_hint}"
        + "\n\nWrite the consolidated answer now."
    )


# ---------------------------------------------------------------------------
# Synth-verifier prompt
# ---------------------------------------------------------------------------

_SYNTH_VERIFIER_SYSTEM = (
    "You are auditing a FINAL `solution = [...]` answer for a multi-part "
    "problem.\n\n"
    "Check:\n"
    "  - Does the final line have exactly `solution = [...]`?\n"
    "  - Does the list length make sense given the problem and the "
    "sub-block answers? (The heuristic length is a hint, not a rule.)\n"
    "  - Are the values in the list well-formed (each is a single "
    "expression, no prose, no nested labels)?\n"
    "  - Do the listed values correspond to plausible sub-block answers?\n\n"
    "Reply in strict JSON (no markdown, no prose), with this shape:\n"
    "{\n"
    "  \"score\": <float 0..1>,\n"
    "  \"is_consistent\": <bool>,\n"
    "  \"issues\": [\"...\", ...],\n"
    "  \"corrected_final_line\": \"solution = [v1, v2, ..., vN]\"  "
    "<- optional, only if you can fix it cleanly>\n"
    "}\n"
)


def _build_synth_verifier_prompt(*, problem_text: str,
                                  block_answers: dict[str, str],
                                  candidate_full_text: str,
                                  expected_length_hint: int) -> str:
    return (
        _SYNTH_VERIFIER_SYSTEM
        + "\n\nOriginal problem:\n"
        + problem_text.strip()
        + "\n\nSub-block answers:\n"
        + _format_blocks(block_answers)
        + f"\n\nHeuristic candidate length: {expected_length_hint}"
        + "\n\nCandidate final response (last line is the graded line):\n"
        + candidate_full_text
        + "\n\nReturn ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# SynthResult
# ---------------------------------------------------------------------------

@dataclass
class SynthResult:
    """Output of LLMSynthesizer.synthesize()."""

    full_response: str                    # the synthesizer's whole response
    final_line: str                       # the literal `solution = [...]`
    values: list[str] = field(default_factory=list)
    heuristic_length: int = 1
    actual_length: int = 0
    verifier_score: float = 0.0
    verifier_issues: list[str] = field(default_factory=list)
    verifier_is_consistent: bool = False
    n_retries: int = 0

    def to_dict(self) -> dict:
        return {
            "full_response": self.full_response,
            "final_line": self.final_line,
            "values": list(self.values),
            "heuristic_length": self.heuristic_length,
            "actual_length": self.actual_length,
            "verifier_score": self.verifier_score,
            "verifier_issues": list(self.verifier_issues),
            "verifier_is_consistent": self.verifier_is_consistent,
            "n_retries": self.n_retries,
        }


# ---------------------------------------------------------------------------
# Helpers — parse `solution = [a, b, c]`
# ---------------------------------------------------------------------------

_SOLUTION_LIST_RE = re.compile(
    r"solution\s*=\s*\[(.*?)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_solution_list(text: str) -> tuple[str, list[str]]:
    """Return (final_line, list_components) extracted from the text.

    If no `solution = [...]` line is found, returns (text-tail, []).
    """
    if not text:
        return "", []
    # Search from end. Process each candidate line that contains `solution =`.
    lines = text.strip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        m = _SOLUTION_LIST_RE.search(line)
        if m:
            inner = m.group(1)
            # Top-level comma split (respect brackets/parens).
            parts: list[str] = []
            cur: list[str] = []
            depth = 0
            for ch in inner:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append("".join(cur).strip())
                    cur = []
                else:
                    cur.append(ch)
            if cur:
                parts.append("".join(cur).strip())
            parts = [p for p in parts if p]
            return line.strip(), parts
    return lines[-1].strip() if lines else "", []


# ---------------------------------------------------------------------------
# LLMSynthesizer
# ---------------------------------------------------------------------------

@dataclass
class LLMSynthesizer:
    """Two LLMs: synthesizer (opus) + synth_verifier (opus, deterministic).

    Both are passed in; falling back to defaults is the caller's job.
    """

    synth_llm: LLMClient
    verifier_llm: LLMClient
    cfg: Config
    synth_temperature: float = 0.3
    verifier_temperature: float = 0.0
    max_verifier_retries: int = 1
    tracer: "object | None" = None
    synth_model_name: str | None = None
    verifier_model_name: str | None = None

    def synthesize(self, *, problem_text: str,
                   block_answers: dict[str, str],
                   expected_length_hint: int) -> SynthResult:
        # 1) Synthesizer call.
        prompt = _build_synth_prompt(
            problem_text=problem_text,
            block_answers=block_answers,
            expected_length_hint=expected_length_hint,
        )
        full_text = self._call_synth(prompt)
        final_line, values = extract_solution_list(full_text)

        result = SynthResult(
            full_response=full_text,
            final_line=final_line,
            values=values,
            heuristic_length=expected_length_hint,
            actual_length=len(values),
        )

        # 2) Synth-verifier — score the produced list. Up to 1 retry.
        for attempt in range(self.max_verifier_retries + 1):
            verdict = self._call_verifier(
                problem_text=problem_text,
                block_answers=block_answers,
                candidate_full_text=full_text,
                expected_length_hint=expected_length_hint,
            )
            result.verifier_score = verdict["score"]
            result.verifier_issues = verdict["issues"]
            result.verifier_is_consistent = verdict["is_consistent"]

            # Accept on consistent + reasonable score.
            if verdict["is_consistent"] and verdict["score"] >= 0.6:
                return result

            corrected = verdict.get("corrected_final_line")
            if corrected and attempt == 0:
                # Apply the verifier's corrected last line and re-extract.
                result.n_retries += 1
                full_text = self._apply_corrected_last_line(
                    full_text, corrected)
                final_line, values = extract_solution_list(full_text)
                result.full_response = full_text
                result.final_line = final_line
                result.values = values
                result.actual_length = len(values)
                continue
            break

        return result

    # ----- internals -----------------------------------------------------

    def _call_synth(self, prompt: str) -> str:
        try:
            gens = self.synth_llm.generate(
                prompt, temperature=self.synth_temperature, n=1)
            text = gens[0].text if gens else ""
            finish = gens[0].finish_reason if gens else None
        except Exception as e:                                      # noqa: BLE001
            _log.exception("LLMSynthesizer call failed")
            self._log("synthesizer", prompt, "", None, error=repr(e))
            return ""
        self._log("synthesizer", prompt, text, finish)
        return text

    def _call_verifier(self, *, problem_text: str,
                        block_answers: dict[str, str],
                        candidate_full_text: str,
                        expected_length_hint: int) -> dict:
        prompt = _build_synth_verifier_prompt(
            problem_text=problem_text, block_answers=block_answers,
            candidate_full_text=candidate_full_text,
            expected_length_hint=expected_length_hint,
        )
        try:
            gens = self.verifier_llm.generate(
                prompt, temperature=self.verifier_temperature, n=1)
            text = gens[0].text if gens else ""
            finish = gens[0].finish_reason if gens else None
        except Exception as e:                                      # noqa: BLE001
            _log.exception("Synth-verifier call failed")
            self._log("synth_verifier", prompt, "", None, error=repr(e))
            return {"score": 0.5, "is_consistent": False,
                    "issues": ["synth-verifier call failed"],
                    "corrected_final_line": None}
        self._log("synth_verifier", prompt, text, finish)
        return self._parse_verifier_json(text)

    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    def _parse_verifier_json(self, text: str) -> dict:
        if not text:
            return {"score": 0.5, "is_consistent": False, "issues": [],
                    "corrected_final_line": None}
        m = self._JSON_RE.search(text)
        if not m:
            return {"score": 0.5, "is_consistent": False, "issues": [],
                    "corrected_final_line": None}
        import json as _json
        try:
            obj = _json.loads(m.group(0))
        except _json.JSONDecodeError:
            return {"score": 0.5, "is_consistent": False, "issues": [],
                    "corrected_final_line": None}
        if not isinstance(obj, dict):
            return {"score": 0.5, "is_consistent": False, "issues": [],
                    "corrected_final_line": None}
        try:
            score = float(obj.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        return {
            "score": max(0.0, min(1.0, score)),
            "is_consistent": bool(obj.get("is_consistent", False)),
            "issues": [str(x).strip() for x in (obj.get("issues") or [])
                        if str(x).strip()],
            "corrected_final_line": obj.get("corrected_final_line"),
        }

    @staticmethod
    def _apply_corrected_last_line(full_text: str, corrected: str) -> str:
        """Replace the LAST line of `full_text` with `corrected`."""
        lines = full_text.splitlines()
        if not lines:
            return corrected.strip()
        lines[-1] = corrected.strip()
        return "\n".join(lines)

    def _log(self, role: str, prompt: str, response: str,
             finish: str | None, error: str | None = None) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.log_agent_call(
                role=role, prompt=prompt, response=response,
                model=(self.synth_model_name if role == "synthesizer"
                       else self.verifier_model_name),
                finish_reason=finish,
                extras={"error": error},
            )
        except Exception:                                           # noqa: BLE001
            pass
