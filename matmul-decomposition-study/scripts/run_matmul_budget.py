#!/usr/bin/env python3
"""
Budget-metered monolithic baseline for the matmul chain — the counterpart to the
Concord decomposition study, for a budget-controlled horizon×budget comparison.

Key idea: a monolithic run traces its ENTIRE budget–frontier from one long run.
Each turn we record the CUMULATIVE tokens + USD spent so far and whether the
whole chain is still correct through that turn. Then, offline, for ANY budget B:

    T*(B) = max turn L such that  cum_cost(L) <= B  AND  correct through all 1..L

so a single run per (model, d) gives solve-vs-horizon and cost-vs-horizon at once.
No budget cap is needed here (the frontier is derived in analyze_budget_frontier.py);
--budget-usd / --budget-tokens optionally hard-stop an episode early to save spend.

Budget currency: TOTAL TOKENS (input+output+cache) is the primary, provider-neutral
axis; USD is secondary (Anthropic prices are exact per Concord's cost model; the
GPT price is a documented estimate — override with --gpt-in/--gpt-out).

Reuses the validated task + client/retry/cache machinery from
run_matmul_dataset.py and matmul_exec_common.py; only the API calls are wrapped
with a token meter. Output JSON mirrors run_matmul_dataset so the analyzers load
it the same way, with two extra per-turn fields: cum_tokens, cum_usd.

Usage (from scripts/):
    python run_matmul_budget.py --models claude-opus-4-8 --dims 2 3 \
        --max-turns 600 --num-samples 10 --data-dir ../data/matmul --out ../runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from matmul_exec_common import (
    matmul_mod, matrices_equal, matrix_str, parse_state, system_prompt,
    turn_prompt, hamming, MatEpisode, MatTurn,
)
from run_matmul_dataset import (
    _cache_messages, call_with_retries, build_clients, provider_for,
    summarize_samples,
)

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

# $ per 1e6 tokens (input, output). Anthropic = exact (Concord cost model);
# gpt-5.5 = ESTIMATE, override with --gpt-in/--gpt-out.
PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.5": (1.25, 10.0),
}
CACHE_READ_MULT = {"anthropic": 0.1, "openai": 0.5}   # fraction of input price
CACHE_WRITE_MULT = 1.25                                # Anthropic write premium


class Meter:
    """Accumulates tokens, USD, and call count across an episode."""

    def __init__(self, gpt_in=None, gpt_out=None):
        self.tokens = 0
        self.usd = 0.0
        self.calls = 0
        self.prices = dict(PRICES)
        if gpt_in is not None:
            self.prices["gpt-5.5"] = (gpt_in, gpt_out)

    def add(self, model, prov, it, ot, cr, cw):
        pin, pout = self.prices.get(model, (3.0, 15.0))
        usd = (it * pin + ot * pout
               + cr * pin * CACHE_READ_MULT[prov]
               + cw * pin * CACHE_WRITE_MULT) / 1e6
        self.usd += usd
        self.tokens += it + ot + cr + cw
        self.calls += 1


def ask_anthropic_metered(client, model, system, messages, max_tokens,
                          use_cache, meter) -> str:
    msgs = _cache_messages(messages) if use_cache else messages
    with client.messages.stream(model=model, max_tokens=max_tokens,
                                thinking={"type": "adaptive"}, system=system,
                                messages=msgs) as stream:
        m = stream.get_final_message()
    u = m.usage
    meter.add(model, "anthropic",
              getattr(u, "input_tokens", 0) or 0,
              getattr(u, "output_tokens", 0) or 0,
              getattr(u, "cache_read_input_tokens", 0) or 0,
              getattr(u, "cache_creation_input_tokens", 0) or 0)
    return "".join(b.text for b in m.content if b.type == "text")


def ask_openai_metered(client, model, system, messages, effort, max_tokens,
                       meter) -> str:
    resp = client.responses.create(
        model=model, reasoning={"effort": effort}, max_output_tokens=max_tokens,
        input=[{"role": "developer", "content": system}] + messages)
    u = resp.usage
    it = getattr(u, "input_tokens", 0) or 0
    cr = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
    meter.add(model, "openai", max(it - cr, 0),
              getattr(u, "output_tokens", 0) or 0, cr, 0)
    return resp.output_text or ""


def ask_metered(clients, model, system, messages, effort, max_tokens, use_cache,
                meter):
    if provider_for(model) == "anthropic":
        return ask_anthropic_metered(clients["anthropic"], model, system,
                                     messages, max_tokens, use_cache, meter)
    return ask_openai_metered(clients["openai"], model, system, messages, effort,
                              max_tokens, meter)


def load_golden(data_dir: Path, dim: int, num_samples: int, max_turns: int):
    """Load first `num_samples` golden chains for dim, truncated to max_turns."""
    fp = data_dir / f"chains_d{dim}.jsonl"
    eps = []
    with open(fp) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n = min(max_turns, rec["n_turns"])
            turns = [MatTurn(index=i + 1, matrix=rec["matrices"][i],
                             cumulative=rec["golden"][i]) for i in range(n)]
            eps.append(MatEpisode(dim=rec["dim"], modulus=rec["modulus"],
                                  turns=turns))
            if len(eps) >= num_samples:
                break
    return eps


def run_episode_metered(clients, model, episode, args, meter) -> list[dict]:
    """One metered chain; per-turn grade carries cumulative tokens/usd."""
    dim, p = episode.dim, episode.modulus
    system = system_prompt(dim, p)
    messages, grades = [], []
    prev = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    for turn in episode.turns:
        messages.append({"role": "user", "content": turn_prompt(turn, p)})
        try:
            reply = call_with_retries(
                lambda: ask_metered(clients, model, system, messages, args.effort,
                                    args.max_tokens, not args.no_prompt_cache, meter),
                turn_index=turn.index, max_retries=args.max_retries, retry_base=5.0)
        except Exception as e:  # noqa: BLE001
            print(f"      turn {turn.index}: ERROR after retries {e}")
            messages.pop()
            break
        pred = parse_state(reply, dim)
        task_ok = matrices_equal(pred, turn.cumulative)
        turn_ok = matrices_equal(pred, matmul_mod(prev, turn.matrix, p))
        grades.append({"turn": turn_ok, "task": task_ok,
                       "wrong_entries": hamming(pred, turn.cumulative),
                       "cum_tokens": meter.tokens, "cum_usd": round(meter.usd, 6)})
        messages.append({"role": "assistant",
                         "content": f"STATE={matrix_str(pred)}" if pred is not None else reply})
        prev = pred if pred is not None else prev
        if args.budget_usd and meter.usd >= args.budget_usd:
            break
        if args.budget_tokens and meter.tokens >= args.budget_tokens:
            break
    return grades


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--dims", type=int, nargs="+", required=True)
    p.add_argument("--max-turns", type=int, default=600)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=64000)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--no-prompt-cache", action="store_true")
    p.add_argument("--budget-usd", type=float, default=0.0,
                   help="Optional hard per-episode USD stop (0 = run to max-turns).")
    p.add_argument("--budget-tokens", type=int, default=0,
                   help="Optional hard per-episode token stop (0 = off).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "matmul")
    p.add_argument("--gpt-in", type=float, default=None,
                   help="Override gpt-5.5 input $/Mtok (default estimate 1.25).")
    p.add_argument("--gpt-out", type=float, default=None)
    p.add_argument("--out", type=Path, default=REPO / "runs")
    args = p.parse_args()

    clients = build_clients(args.models)
    uses_openai = any(provider_for(m) == "openai" for m in args.models)
    args.out.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        results = {}
        for dim in args.dims:
            eps = load_golden(args.data_dir, dim, args.num_samples, args.max_turns)
            print(f"=== {model} d={dim}: {len(eps)} chains x <= {args.max_turns} turns ===")
            per_sample, costs = [], []
            for s, ep in enumerate(eps):
                meter = Meter(args.gpt_in, args.gpt_out)
                t0 = time.monotonic()
                grades = run_episode_metered(clients, model, ep, args, meter)
                per_sample.append(grades)
                costs.append((meter.tokens, meter.usd, meter.calls))
                first_bad = next((i + 1 for i, g in enumerate(grades)
                                  if not g["task"]), None)
                print(f"  s{s+1}/{len(eps)}: reached {len(grades)}, "
                      f"1st-err {first_bad}, {meter.tokens} tok, ${meter.usd:.2f}, "
                      f"{meter.calls} calls ({time.monotonic()-t0:.0f}s)")
            summary = summarize_samples(per_sample, args.threshold)
            summary["mean_total_tokens"] = sum(c[0] for c in costs) / len(costs)
            summary["mean_usd"] = sum(c[1] for c in costs) / len(costs)
            results[str(dim)] = {"summary": summary, "per_sample": per_sample}
            print(f"  -> H_{args.threshold}={summary['horizon_length']} | "
                  f"mean ${summary['mean_usd']:.2f}, {summary['mean_total_tokens']:.0f} tok\n")

        slug = model.replace("claude-", "")
        eff = f"_{args.effort}" if provider_for(model) == "openai" else ""
        dims_slug = "-".join(str(d) for d in args.dims)
        out = args.out / (f"matmul_budget__{slug}__T{args.max_turns}"
                          f"_n{args.num_samples}_d{dims_slug}{eff}.json")
        out.write_text(json.dumps({
            "benchmark": "matmul_budget_baseline",
            "parameters": {"model": model, "dims": args.dims,
                           "max_turns": args.max_turns,
                           "num_samples": args.num_samples,
                           "budget_currency": "total_tokens & usd (per-turn cumulative)",
                           "gpt_price_estimate": model == "gpt-5.5"},
            "results": {model: results},
        }, indent=2))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
