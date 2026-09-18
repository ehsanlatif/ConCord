"""Structured run logging.

Every run gets one JSONL file under `cfg.telemetry.log_dir`. Each line is one
event: {"t": iso8601, "kind": str, "data": {...}}. Cost is rolled forward and
written into every event so post-hoc analyses don't need to thread state.

`Telemetry.from_config` enforces the per-run budget ceiling: when total
LLM calls exceed `cfg.mcts.N`, `check_budget()` raises BudgetExceeded so a
runaway sweep can't burn the account (§8.5).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .types import CostTally


class BudgetExceeded(RuntimeError):
    pass


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
            timeout=2,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def _config_hash(cfg: Config) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()
    import hashlib
    return hashlib.sha256(blob).hexdigest()[:12]


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


class Telemetry:
    """Owns the JSONL log file and tracks run-wide cost vs the rollout budget.

    LLM clients update their own CostTally; the orchestrator passes that tally
    to Telemetry on each event so the log is self-contained.
    """

    def __init__(self, run_id: str, path: Path, cfg: Config,
                 baseline_calls: int = 0):
        self.run_id = run_id
        self.path = path
        self.cfg = cfg
        self._n_rollouts = 0
        # Cumulative-cost baseline at the moment THIS solve() started. The
        # budget check is per-block — it compares (current - baseline)
        # against cfg.mcts.N, not raw cumulative. Without this, every
        # block past the third in a multi-block run would trip the budget
        # on its first rollout because the shared RoleClients accumulates
        # cost across blocks.
        self.baseline_calls = int(baseline_calls)
        # touch + write run-meta header
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", buffering=1)  # line-buffered
        self._write({
            "kind": "run_start",
            "data": {
                "run_id": run_id,
                "git_sha": _git_sha(),
                "config_hash": _config_hash(cfg),
                "config": cfg.to_dict(),
                "baseline_calls": self.baseline_calls,
            },
        })

    @classmethod
    def from_config(cls, cfg: Config, *, tag: str | None = None,
                    baseline_calls: int = 0) -> "Telemetry":
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if tag:
            run_id = f"{run_id}-{tag}"
        # Each run gets its OWN subdirectory under log_dir. The Tracer
        # derives its sibling artifacts (tree.json, rollouts.jsonl,
        # agent_calls.jsonl) from `self.path`, so all four files land
        # in this same per-run directory — one folder per run, no more
        # 100s of loose files in results/concord/.
        run_dir = Path(cfg.telemetry.log_dir) / run_id
        path = run_dir / f"{run_id}.jsonl"
        return cls(run_id, path, cfg, baseline_calls=baseline_calls)

    # --- event API ---------------------------------------------------------

    def event(self, kind: str, data: dict[str, Any] | None = None,
              cost: CostTally | None = None) -> None:
        rec: dict[str, Any] = {"kind": kind, "data": _jsonable(data or {})}
        if cost is not None:
            rec["cost"] = cost.to_dict()
        self._write(rec)

    def rollout(self, cost: CostTally) -> None:
        self._n_rollouts += 1
        self.event("rollout", {"n": self._n_rollouts}, cost=cost)

    def finish(self, summary: dict[str, Any], cost: CostTally) -> None:
        self.event("run_end", summary, cost=cost)
        self._fp.close()

    # --- budget enforcement ------------------------------------------------

    def check_budget(self, cost: CostTally) -> None:
        """Enforce the hard call ceiling defined by `cfg.mcts.N`.

        Per the plan (§6 + §8.5) the rollout budget N is the cost cap. Tests
        assert total LLM calls never exceed it.

        The comparison is against `(cost.calls - self.baseline_calls)` so
        that, in multi-block runs where one RoleClients accumulates cost
        across blocks, each block still gets its OWN N-call budget.
        """
        delta = max(0, int(cost.calls) - self.baseline_calls)
        if delta > self.cfg.mcts.N:
            raise BudgetExceeded(
                f"LLM calls ({delta} this block, {cost.calls} total) "
                f"exceeded per-block budget N={self.cfg.mcts.N}"
            )

    # --- internals ---------------------------------------------------------

    def _write(self, rec: dict[str, Any]) -> None:
        rec = {"t": _iso(), **rec}
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
