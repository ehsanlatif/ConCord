#!/usr/bin/env python3
"""
Matrix-chain long-horizon EXECUTION benchmark: HORIZON x COMPLEXITY.

This is the two-axis generalization of run_lhe_dataset.py / run_horizon_exec.py.
The model maintains a running d x d integer matrix STATE (starting at the
identity) and multiplies in one fresh matrix per turn, mod p:

    M_0 = I_d ,   M_t = (M_{t-1} . A_t) mod p ,   report M_t each turn.

Two orthogonal knobs (see matmul_exec_common.py for the full rationale):

  * HORIZON     T = number of turns (matrices in the chain). Compounding,
                state-carrying, non-Markovian — the same long-horizon axis as the
                scalar running-sum task. One long run yields the accuracy curve at
                every shorter prefix, so we don't re-run per horizon.

  * COMPLEXITY  d = matrix dimension (--dims). Each step is ~d^3 scalar
                multiply-adds, independent of T. d=1 recovers a scalar running
                product mod p. Because a single run fixes d, the complexity axis
                IS swept with separate runs (one per d), while the horizon axis
                comes free from prefixes. (Secondary knob: --modulus p.)

Grading (identical shape to the LHE benchmark):
  * turn accuracy — single-step correctness (this multiply, given the matrix the
    model could actually see last turn). Stays high if per-step skill holds.
  * task accuracy — correct on the ENTIRE chain up to length L. Collapses with T.
  * H_s           — horizon length: turns before task accuracy across samples
    drops below s (default 0.5). Reported per complexity level d.

Fully synthetic (deterministic in --seed) — no dataset download. Providers routed
by model id (claude-* -> Anthropic, gpt-*/o* -> OpenAI); keys from .env. Reuses
the API clients from run_horizon_exec.py and the metric layer from
horizon_exec_common.py.

Parallel, one-model-per-process design (same as run_lhe_dataset.py): launch each
model in its own terminal into the SAME --wandb-project and --wandb-group so W&B
overlays them as one line per model. Within a run, curves are namespaced by d.

Usage
-----
    # one model per process, sweeping complexity d in {1,2,3}
    python run_matmul_dataset.py --models claude-opus-4-8 --dims 1 2 3 --wandb
    python run_matmul_dataset.py --models claude-sonnet-5 --dims 1 2 3 --wandb
    python run_matmul_dataset.py --models gpt-5.5 --effort high --dims 1 2 3 --wandb

    # heavier per-step load, shorter horizon
    python run_matmul_dataset.py --models claude-opus-4-8 --dims 2 3 4 \
        --modulus 97 --max-turns 100 --num-samples 10

    # self-conditioning (teacher-forced history, 30% of prior states corrupted)
    python run_matmul_dataset.py --inject-error-rate 0.3 --dims 2

W&B dashboards produced (per complexity level d):
  * live/*    — one streaming point per turn (turn-ok, task-ok, wrong-entry count)
                so you can watch drift accumulate live.
  * curve/*   — task + turn accuracy at EVERY prefix length, one line per d.
  * sweep/*   — task + turn accuracy at horizons 20,40,...,max, x=horizon, per d.
  * summary/  — H_s scalars (per d) + a d-vs-H_s complexity table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from matmul_exec_common import (
    MatEpisode,
    MatTurn,
    corrupt,
    hamming,
    make_chain,
    matmul_mod,
    matrices_equal,
    matrix_str,
    parse_state,
    summarize_samples,
    system_prompt,
    turn_prompt,
)
from run_horizon_exec import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_OPENAI_MODELS,
    build_clients,
    provider_for,
)

load_dotenv(Path(__file__).parent / ".env")

DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "matmul"

# Milestone lengths to print an accuracy readout at (clipped to what was run).
MILESTONES = [1, 5, 10, 25, 50, 100, 200, 400, 800, 1600]


# --------------------------------------------------------------------------- #
# Episode sourcing: static golden dataset (preferred) or on-the-fly generation
# --------------------------------------------------------------------------- #

def load_chains(data_dir: Path, dim: int, num_samples: int,
                max_turns: int) -> list[MatEpisode]:
    """Load the first `num_samples` chains for dimension `dim` from the static
    dataset, truncated to `max_turns` turns. Golden products come straight from
    the file — no recomputation — so runs use the frozen labels verbatim.
    """
    fpath = data_dir / f"chains_d{dim}.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(
            f"No static dataset for d={dim} at {fpath}. Generate it with "
            f"`python make_matmul_dataset.py --dims {dim}`, or pass --synthetic.")
    episodes: list[MatEpisode] = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec["modulus"]
            n = min(max_turns, rec["n_turns"])
            turns = [MatTurn(index=i + 1, matrix=rec["matrices"][i],
                             cumulative=rec["golden"][i]) for i in range(n)]
            episodes.append(MatEpisode(dim=rec["dim"], modulus=p, turns=turns))
            if len(episodes) >= num_samples:
                break
    if len(episodes) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples for d={dim} but the dataset only "
            f"has {len(episodes)}. Regenerate with a larger --num-samples.")
    return episodes


def build_episodes(args, dim: int) -> list[MatEpisode]:
    """Static dataset if present (and not --synthetic); else generate on the fly.

    Both paths use the identical per-chain seed, so --synthetic reproduces the
    static dataset exactly (given matching --modulus/--seed).
    """
    nonzero = getattr(args, "nonzero", False)
    # --nonzero requires on-the-fly generation (the static dataset allows zeros).
    if not args.synthetic and not nonzero:
        try:
            eps = load_chains(args.data_dir, dim, args.num_samples, args.max_turns)
            print(f"  [d={dim}] loaded {len(eps)} chains from {args.data_dir}")
            return eps
        except FileNotFoundError as e:
            print(f"  [d={dim}] {e}\n  falling back to synthetic generation.")
    if nonzero:
        print(f"  [d={dim}] generating {args.num_samples} NONZERO-entry chains "
              f"(no absorbing 0)")
    return [
        make_chain(dim, args.max_turns, args.modulus,
                   seed=args.seed * 100000 + dim * 1000 + sid, nonzero=nonzero)
        for sid in range(args.num_samples)
    ]


def sweep_horizons(turn_start: int, turn_step: int, max_turns: int) -> list[int]:
    """Horizon milestones to report the sweep at: start, start+step, ..., max."""
    hs = list(range(turn_start, max_turns + 1, turn_step))
    if not hs:
        hs = [max_turns]
    elif hs[-1] != max_turns:
        hs.append(max_turns)
    return hs


# --------------------------------------------------------------------------- #
# Model calls (single turn, given the full running message history)
# --------------------------------------------------------------------------- #

# Anthropic prompt caching needs a prefix of at least ~1024 tokens before a
# cache breakpoint is honored; below that the breakpoint is ignored (or errors).
# We only add one once the running transcript is comfortably past that.
_CACHE_MIN_CHARS = 5000          # ~1250 tokens at ~4 chars/token
CACHE_STATS = {"reads": 0, "writes": 0}   # cumulative, for a one-line report


def _cache_messages(messages):
    """Return a copy of `messages` with an ephemeral cache breakpoint on the LAST
    message's content, so Anthropic caches the entire prefix (system + all prior
    turns) up to it. Next turn shares that prefix and only pays full price for the
    two newly-appended messages — turning the quadratic history-resend into a
    near-linear cost. Returns messages unchanged when the prefix is still too
    small to cache."""
    total = sum(len(m["content"]) for m in messages if isinstance(m["content"], str))
    if total < _CACHE_MIN_CHARS or not messages:
        return messages
    out = [dict(m) for m in messages]
    last = out[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"],
                            "cache_control": {"type": "ephemeral"}}]
    return out


def ask_anthropic(client, model, system, messages, max_tokens,
                  use_cache=True) -> str:
    msgs = _cache_messages(messages) if use_cache else messages
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=msgs,
    ) as stream:
        message = stream.get_final_message()
    u = getattr(message, "usage", None)
    if u is not None:
        CACHE_STATS["reads"] += getattr(u, "cache_read_input_tokens", 0) or 0
        CACHE_STATS["writes"] += getattr(u, "cache_creation_input_tokens", 0) or 0
    return "".join(b.text for b in message.content if b.type == "text")


def ask_openai(client, model, system, messages, effort, max_tokens) -> str:
    # OpenAI caches identical prompt prefixes (>1024 tokens) AUTOMATICALLY and
    # bills the cached portion at a discount — no request-side flag needed, so
    # resending the stable history already benefits with no code change here.
    input_items = [{"role": "developer", "content": system}] + messages
    resp = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        max_output_tokens=max_tokens,
        input=input_items,
    )
    return resp.output_text or ""


def ask(clients, model, system, messages, effort, max_tokens,
        use_cache=True) -> str:
    if provider_for(model) == "anthropic":
        return ask_anthropic(clients["anthropic"], model, system, messages,
                             max_tokens, use_cache=use_cache)
    return ask_openai(clients["openai"], model, system, messages, effort,
                      max_tokens)


# Transient provider errors that should be retried, not treated as a real
# collapse. These killed ~41% of the first Opus/Sonnet samples at turn 0 (see
# the censoring analysis) when there was no retry. Matched case-insensitively
# against str(exc); covers Anthropic `overloaded_error` (HTTP 529), 429 rate
# limits, and the usual 5xx / timeout / connection blips.
_TRANSIENT_MARKERS = (
    "overloaded", "rate_limit", "rate limit", "429", "529", "503", "502",
    "500", "internal server", "service unavailable", "timeout", "timed out",
    "connection", "temporarily", "econnreset", "read timed out", "bad gateway",
)


def is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


def call_with_retries(fn, *, turn_index, max_retries, retry_base):
    """Call `fn()`, retrying transient provider errors with capped exponential
    backoff. Returns the reply string, or raises the last exception once retries
    are exhausted / the error is non-transient (the caller then ends the episode).

    Backoff: retry_base * 2**attempt seconds, capped at 120s (deterministic — no
    jitter needed since concurrent episodes run in separate processes).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_transient(e) or attempt >= max_retries:
                raise
            delay = min(retry_base * (2 ** attempt), 120.0)
            attempt += 1
            print(f"      turn {turn_index}: transient error "
                  f"({str(e)[:80]}); retry {attempt}/{max_retries} in {delay:.0f}s")
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# One episode (one long conversation) — matrix state
# --------------------------------------------------------------------------- #

def run_episode(clients, model, episode, effort, inject_rate, seed, max_tokens,
                on_turn=None, respond=None, max_retries=6,
                retry_base=5.0, use_cache=True) -> list[dict]:
    """Play one full matrix-chain conversation; return per-turn grades.

    Mirrors run_horizon_exec.run_episode but the carried state is a matrix.

    Free-running (inject_rate == 0): the assistant history contains the model's
    OWN prior STATE matrices, so natural drift and self-conditioning both show up.

    Teacher-forced (inject_rate > 0): the assistant history is rewritten to the
    ground-truth matrix each turn, except a `inject_rate` fraction of prior turns
    have one entry corrupted; turn-correctness is judged relative to the matrix we
    actually showed.

    `on_turn(turn_index, grade)` is called after each graded turn (live logging).
    `respond(system, messages) -> str` overrides the hosted-API path (local model
    or a test stub); `clients`/`effort`/`max_tokens` are then ignored.
    """
    dim, p = episode.dim, episode.modulus
    system = system_prompt(dim, p)
    messages: list[dict] = []
    prev_shown = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    grades: list[dict] = []
    rng = random.Random(seed)

    for turn in episode.turns:
        messages.append({"role": "user", "content": turn_prompt(turn, p)})
        try:
            reply = call_with_retries(
                (lambda: respond(system, messages)) if respond is not None
                else (lambda: ask(clients, model, system, messages, effort,
                                  max_tokens, use_cache)),
                turn_index=turn.index, max_retries=max_retries,
                retry_base=retry_base)
        except Exception as e:  # noqa: BLE001 - retries exhausted; end this episode
            print(f"      turn {turn.index}: ERROR after retries {e}")
            messages.pop()
            break

        pred = parse_state(reply, dim)
        task_ok = matrices_equal(pred, turn.cumulative)
        # Single-step correctness: right product of the matrix the model could see.
        expected_step = matmul_mod(prev_shown, turn.matrix, p)
        turn_ok = matrices_equal(pred, expected_step)
        grades.append({
            "turn": turn_ok, "task": task_ok,
            "predicted": pred, "expected": turn.cumulative,
            "wrong_entries": hamming(pred, turn.cumulative),
        })
        if on_turn is not None:
            on_turn(turn.index, grades[-1])

        if inject_rate > 0:
            shown = turn.cumulative
            if rng.random() < inject_rate:
                shown = corrupt(turn.cumulative, p, seed + turn.index)
            messages.append({"role": "assistant", "content": f"STATE={matrix_str(shown)}"})
            prev_shown = shown
        else:
            if pred is not None:
                messages.append({"role": "assistant",
                                 "content": f"STATE={matrix_str(pred)}"})
                prev_shown = pred
            else:
                messages.append({"role": "assistant", "content": reply})
                # keep prev_shown as-is when unparseable

    return grades


# --------------------------------------------------------------------------- #
# Weights & Biases logging (per complexity level d; no-op unless --wandb)
# --------------------------------------------------------------------------- #

class WandbLogger:
    """Streams per-turn progress, then logs horizon curves/sweep + H_s, all
    NAMESPACED BY COMPLEXITY d so a single run overlays its d-levels as separate
    lines. Model overlay across the parallel per-model processes is achieved (as
    in run_lhe_dataset.py) by NOT namespacing by model, so separate runs sharing
    key names overlay automatically.
    """

    def __init__(self, enabled, project, entity, run_name, group, config):
        self.enabled = enabled
        self.wandb = None
        self._step = 0
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            raise SystemExit(
                "--wandb requested but the 'wandb' package is not installed.\n"
                "Install it:  pip install wandb   (then run 'wandb login').")
        self.wandb = wandb
        wandb.init(project=project, entity=entity, name=run_name, group=group,
                   config=config)
        wandb.define_metric("live/global_step")
        wandb.define_metric("live/*", step_metric="live/global_step")
        wandb.define_metric("curve/horizon_length")
        wandb.define_metric("curve/*", step_metric="curve/horizon_length")
        wandb.define_metric("sweep/horizon")
        wandb.define_metric("sweep/*", step_metric="sweep/horizon")

    def live_turn(self, dim, turn_index, grade):
        if not self.enabled:
            return
        self._step += 1
        payload = {
            "live/global_step": self._step,
            f"live/turn_ok_d{dim}": int(grade["turn"]),
            f"live/task_ok_d{dim}": int(grade["task"]),
            f"live/turn_index_d{dim}": turn_index,
        }
        if grade["wrong_entries"] is not None:
            payload[f"live/wrong_entries_d{dim}"] = grade["wrong_entries"]
        self.wandb.log(payload)

    def log_curves(self, dim, summary):
        if not self.enabled:
            return
        task = summary.get("task_accuracy_by_len", [])
        turn = summary.get("turn_accuracy_by_turn", [])
        for i, (ta, tu) in enumerate(zip(task, turn), start=1):
            self.wandb.log({
                "curve/horizon_length": i,
                f"curve/task_acc_d{dim}": ta,
                f"curve/turn_acc_d{dim}": tu,
            })
        self.wandb.summary[f"H_s_d{dim}"] = summary["horizon_length"]
        self.wandb.summary[f"final_turn_acc_d{dim}"] = summary["final_turn_accuracy"]

    def log_sweep_point(self, dim, horizon, task_acc, turn_acc):
        if not self.enabled:
            return
        self.wandb.log({
            "sweep/horizon": horizon,
            f"sweep/task_accuracy_d{dim}": task_acc,
            f"sweep/turn_accuracy_d{dim}": turn_acc,
        })

    def log_complexity_summary(self, dims, per_dim_summary, threshold):
        """A complexity table: d -> H_s. The headline 2-axis view."""
        if not self.enabled:
            return
        table = self.wandb.Table(
            columns=["dim", "H_s", "final_turn_acc", "n_samples"])
        for d in dims:
            s = per_dim_summary[d]
            table.add_data(d, s["horizon_length"], s["final_turn_accuracy"],
                           s["n_samples"])
        self.wandb.log({
            "summary/H_s_by_complexity": self.wandb.plot.bar(
                table, "dim", "H_s",
                title=f"Horizon H_{threshold} vs complexity d (higher = better)"),
            "summary/complexity_table": table,
        })

    def finish(self):
        if self.enabled and self.wandb is not None:
            self.wandb.finish()


# --------------------------------------------------------------------------- #
# Per-(model, complexity) driver
# --------------------------------------------------------------------------- #

def run_model_dim(model, dim, episodes, args, wb, *, clients=None,
                  respond=None) -> dict:
    """Run one long chain per sample at complexity `dim`; return summary+per_sample.

    `episodes` is the pre-sourced list of MatEpisode (from the static dataset or
    generated), one per sample.
    """
    mod = episodes[0].modulus if episodes else args.modulus
    print(f"--- {model} @ d={dim} (mod {mod}) ---")
    per_sample: list[list[dict]] = []
    for s_idx, episode in enumerate(episodes):
        # Injection RNG seed stays reproducible and independent of episode source.
        ep_seed = args.seed * 100000 + dim * 1000 + s_idx
        t0 = time.monotonic()
        grades = run_episode(
            clients, model, episode, args.effort, args.inject_error_rate,
            seed=ep_seed, max_tokens=args.max_tokens,
            on_turn=lambda ti, g, _d=dim: wb.live_turn(_d, ti, g),
            respond=respond,
            max_retries=getattr(args, "max_retries", 6),
            retry_base=getattr(args, "retry_base", 5.0),
            use_cache=not getattr(args, "no_prompt_cache", False))
        dt = time.monotonic() - t0
        per_sample.append(grades)
        first_bad = next((i + 1 for i, g in enumerate(grades)
                          if not g["task"]), None)
        n_turn_ok = sum(1 for g in grades if g["turn"])
        print(f"  sample {s_idx + 1}/{args.num_samples}: reached turn "
              f"{len(grades)}, turn-acc {n_turn_ok}/{len(grades)}, first task "
              f"error @ turn {first_bad if first_bad else 'none'}  ({dt:.0f}s)")

    summary = summarize_samples(per_sample, args.threshold)
    wb.log_curves(dim, summary)

    task_curve = summary.get("task_accuracy_by_len", [])
    turn_curve = summary.get("turn_accuracy_by_turn", [])
    sweep_rows = []
    for H in sweep_horizons(args.turn_start, args.turn_step, args.max_turns):
        if H > len(task_curve):
            break
        task_acc = task_curve[H - 1]
        turn_acc = round(sum(turn_curve[:H]) / H, 4)
        wb.log_sweep_point(dim, H, task_acc, turn_acc)
        sweep_rows.append({"horizon": H, "task_accuracy": task_acc,
                           "turn_accuracy": turn_acc})
    summary["sweep"] = sweep_rows
    summary["milestones"] = {
        L: round(task_curve[L - 1], 4) for L in MILESTONES if L <= len(task_curve)
    }
    ms = "  ".join(f"L{L}:{a:.0%}" for L, a in summary["milestones"].items())
    print(f"  -> d={dim}: H_{args.threshold} = {summary['horizon_length']} turns "
          f"| final turn-acc {summary['final_turn_accuracy']:.1%}")
    print(f"     task-acc  {ms}\n")
    # per-sample matrices are large; drop the stored matrices from the JSON but
    # keep the booleans + wrong-entry counts that every metric derives from.
    slim = [[{"turn": g["turn"], "task": g["task"],
              "wrong_entries": g["wrong_entries"]} for g in s]
            for s in per_sample]
    return {"summary": summary, "per_sample": slim}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def resolve_models(args) -> list[str]:
    if args.models:
        return args.models
    if args.provider == "openai":
        return DEFAULT_OPENAI_MODELS
    if args.provider == "both":
        return DEFAULT_ANTHROPIC_MODELS + DEFAULT_OPENAI_MODELS
    return DEFAULT_ANTHROPIC_MODELS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--provider", choices=["anthropic", "openai", "both"],
                   default="anthropic")
    p.add_argument("--effort", choices=["none", "low", "medium", "high", "xhigh"],
                   default="high", help="OpenAI reasoning effort (ignored by Claude).")
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3],
                   help="COMPLEXITY axis: matrix dimensions d to sweep. Each is a "
                        "separate set of runs (d is fixed within a run). d=1 is "
                        "the scalar running-product baseline.")
    p.add_argument("--modulus", type=int, default=97,
                   help="Prime-ish modulus p; entries live in [0,p-1]. Larger p = "
                        "harder per-multiply arithmetic (secondary complexity knob). "
                        "Used only with --synthetic; the static dataset carries "
                        "its own modulus.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Static golden dataset dir (from make_matmul_dataset.py). "
                        "Loaded by default; falls back to synthetic if absent.")
    p.add_argument("--synthetic", action="store_true",
                   help="Ignore the static dataset and generate chains on the fly "
                        "(reproduces the dataset given matching --modulus/--seed).")
    p.add_argument("--nonzero", action="store_true",
                   help="Generate chains with entries in [1,modulus-1] (implies "
                        "on-the-fly generation). For d=1 with a prime modulus this "
                        "removes the absorbing-0 degeneracy so every turn is a real "
                        "multiplication across the full horizon.")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Independent chains per (model, d). Held constant across "
                        "the horizon sweep.")
    p.add_argument("--max-turns", type=int, default=200,
                   help="HORIZON axis upper bound. One long run yields the accuracy "
                        "curve at every shorter prefix, so horizon is not re-run.")
    p.add_argument("--turn-start", type=int, default=20,
                   help="First horizon milestone reported in the sweep.")
    p.add_argument("--turn-step", type=int, default=20,
                   help="Increment between horizon milestones (20,40,...,max).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="s in H_s: task-accuracy cutoff defining the horizon.")
    p.add_argument("--inject-error-rate", type=float, default=0.0,
                   help="0 = free-running. >0 = teacher-forced history with this "
                        "fraction of prior states corrupted (self-conditioning).")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Per-turn output cap (leaves room for reasoning). Bump for "
                        "larger d, where the model must work through ~d^3 mults.")
    p.add_argument("--max-retries", type=int, default=6,
                   help="Retries on transient provider errors (overloaded_error, "
                        "429/5xx, timeouts) before giving up on a turn. Prevents "
                        "API blips from censoring samples at turn 0.")
    p.add_argument("--retry-base", type=float, default=5.0,
                   help="Base seconds for exponential backoff between retries "
                        "(retry_base * 2**attempt, capped at 120s).")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Disable prompt caching. By default the growing "
                        "conversation prefix is cached (Anthropic: explicit "
                        "ephemeral breakpoint; OpenAI: automatic), turning the "
                        "quadratic history-resend into near-linear cost with NO "
                        "change to results.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--wandb", action="store_true",
                   help="Stream metrics live to Weights & Biases.")
    p.add_argument("--wandb-project", default="matmul-horizon-execution")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None,
                   help="Ties the parallel per-model runs together; default "
                        "encodes the shared config (no model). Use the SAME value "
                        "across all terminals.")
    args = p.parse_args()

    models = resolve_models(args)
    clients = build_clients(models)
    uses_openai = any(provider_for(m) == "openai" for m in models)

    if args.out is None:
        slug = "+".join(m.replace("claude-", "") for m in models)
        eff = f"_{args.effort}" if uses_openai else ""
        inj = f"_inj{args.inject_error_rate}" if args.inject_error_rate > 0 else ""
        dims_slug = "-".join(str(d) for d in args.dims)
        args.out = Path(__file__).parent / (
            f"matmul_dataset__{slug}__T{args.max_turns}_n{args.num_samples}"
            f"_d{dims_slug}_p{args.modulus}{eff}{inj}.json")

    mode = (f"teacher-forced (inject {args.inject_error_rate})"
            if args.inject_error_rate > 0 else "free-running")
    print(f"Matrix-chain horizon x complexity: T={args.max_turns} turns x "
          f"{args.num_samples} samples, dims={args.dims}, mod={args.modulus}, "
          f"mode={mode}\nModels: {', '.join(models)}\n")

    wb_config = {
        "models": models, "num_samples": args.num_samples,
        "max_turns": args.max_turns, "turn_start": args.turn_start,
        "turn_step": args.turn_step, "dims": args.dims, "modulus": args.modulus,
        "threshold": args.threshold, "inject_error_rate": args.inject_error_rate,
        "max_tokens": args.max_tokens, "seed": args.seed,
        "openai_reasoning_effort": args.effort if uses_openai else None,
    }
    eff = f"_{args.effort}" if uses_openai else ""
    inj = f"_inj{args.inject_error_rate}" if args.inject_error_rate > 0 else ""
    dims_slug = "-".join(str(d) for d in args.dims)
    default_run = (f"matmul_{'+'.join(m.replace('claude-', '') for m in models)}"
                   f"_T{args.max_turns}_n{args.num_samples}_d{dims_slug}"
                   f"_p{args.modulus}{eff}{inj}")
    default_group = (f"matmul_T{args.max_turns}_n{args.num_samples}_d{dims_slug}"
                     f"_p{args.modulus}{eff}{inj}")
    wb = WandbLogger(args.wandb, args.wandb_project, args.wandb_entity,
                     args.wandb_run_name or default_run,
                     args.wandb_group or default_group, wb_config)
    if args.wandb:
        print(f"Streaming to Weights & Biases (project '{args.wandb_project}', "
              f"group '{args.wandb_group or default_group}').\n")

    all_results = {}
    for model in models:
        print(f"=== {model} ===")
        per_dim = {}
        for d in args.dims:
            episodes = build_episodes(args, d)
            per_dim[d] = run_model_dim(model, d, episodes, args, wb,
                                       clients=clients)
        all_results[model] = per_dim
        wb.log_complexity_summary(
            args.dims, {d: per_dim[d]["summary"] for d in args.dims},
            args.threshold)

    print("=" * 72)
    print(f"SUMMARY  (H_{args.threshold} = turns before task accuracy < "
          f"{args.threshold}); rows = model, cols = complexity d")
    print("=" * 72)
    header = f"{'model':22s}" + "".join(f"{'d=' + str(d):>10}" for d in args.dims)
    print(header)
    for model in models:
        row = f"{model:22s}"
        for d in args.dims:
            row += f"{all_results[model][d]['summary']['horizon_length']:>10}"
        print(row)

    args.out.write_text(json.dumps({
        "benchmark": "matmul_chain_horizon_execution",
        "description": ("Running matrix product mod p; horizon T = #turns, "
                        "complexity d = matrix dimension."),
        "parameters": {
            "models": models, "num_samples": args.num_samples,
            "max_turns": args.max_turns, "turn_start": args.turn_start,
            "turn_step": args.turn_step, "dims": args.dims,
            "modulus": args.modulus, "threshold": args.threshold,
            "inject_error_rate": args.inject_error_rate,
            "max_tokens": args.max_tokens, "seed": args.seed,
            "openai_reasoning_effort": args.effort if uses_openai else None,
        },
        "results": all_results,
    }, indent=2))
    print(f"\nDetailed results written to {args.out}")
    if not getattr(args, "no_prompt_cache", False) and CACHE_STATS["reads"]:
        tot = CACHE_STATS["reads"] + CACHE_STATS["writes"]
        print(f"Prompt cache: {CACHE_STATS['reads']:,} cached (read) input "
              f"tokens vs {CACHE_STATS['writes']:,} written "
              f"({100 * CACHE_STATS['reads'] / max(tot, 1):.0f}% of prefix served "
              f"from cache at ~0.1x cost).")
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
