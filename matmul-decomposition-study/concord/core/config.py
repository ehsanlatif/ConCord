"""Pydantic v2 config schema. Defaults match §9 of the implementation plan."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMCfg(BaseModel):
    """Specification for one LLM endpoint.

    Used both as the top-level fallback `llm:` block AND as the value of
    each role under `models:`. Per-role overrides win; anything left None
    in a role block inherits from the top-level `llm:` defaults at
    construction time (see `core.llm.factory`).
    """

    provider: Literal["mock", "anthropic", "openai"] = "mock"
    model: str = "mock-v0"
    temperature: float = 1.0
    # API knobs irrelevant for mock; used by real adapters in M5+.
    max_output_tokens: int = 4096
    request_timeout_s: float = 120.0


class RoleModelsCfg(BaseModel):
    """Per-role model assignments (§5.8 seam, plus user-facing per-role choice).

    Each role can pin a different model/provider. Roles:

      - `execution`   — the K-sample LLM calls inside MCTS expansion. This
        is the only role guaranteed to fire on every expansion; usually the
        strongest (and most expensive) model.
      - `decomposition` — used by the implicit decomposer when the problem
        does not expose explicit `Problem node_K:` markers. Asks the LLM
        for the next subproblem given the resolved state. Skipped entirely
        when the explicit decomposer fires.
      - `classification` — Phase-1 `CategorizeTask`. Asks the LLM to pick a
        domain when the caller doesn't supply one. Cheap; one call per run.
      - `verification`  — used by the LLM-judge verifier. Each scored
        response can be judged; cost scales with K and rollouts. Default
        is None (fall back to the rule-based domain verifier).

    Leaving a role None means "use the fallback `llm:` block".
    """

    execution: LLMCfg | None = None
    decomposition: LLMCfg | None = None
    classification: LLMCfg | None = None
    verification: LLMCfg | None = None
    # Pipeline-solver roles (introduced for the Split → Solve → Combine →
    # Verify pipeline). Each falls back to the top-level `llm:` block when
    # unset, exactly like the four classic roles above.
    splitter: LLMCfg | None = None
    combiner: LLMCfg | None = None
    synthesizer: LLMCfg | None = None
    synth_verifier: LLMCfg | None = None


class SamplingCfg(BaseModel):
    """K per subproblem differs by oracle mode — see §9."""

    K_whitebox: int = 8
    K_blackbox: int = 16
    laplace_a: float = 1.0


class ConfidenceCfg(BaseModel):
    """Hybrid clustering + scoring knobs."""

    tau_pre: float = 0.6                 # loose cosine pre-filter
    entail_threshold: float = 0.5        # bidirectional NLI threshold
    alpha: float = 0.25                  # SD vs verifier weight in U_s
    embed_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "microsoft/deberta-v3-base-mnli"


class MCTSCfg(BaseModel):
    c_puct: float = 3.0
    C: float = 2.0                       # progressive-widening coeff (was old beam M)
    beta: float = 0.5
    n_min: int = 3
    N: int = 128                         # rollout budget (hard ceiling)


class DepthCfg(BaseModel):
    slack: int = 2
    lam: float = 0.02                    # marginal-gain stop threshold
    # `budget_cap` derived at runtime from total $ budget and per-expansion cost.
    budget_cap: int | None = None


class CoherenceCfg(BaseModel):
    mode: Literal["incremental_hard", "terminal_only", "off"] = "incremental_hard"
    # Soft-relax: when no coherent terminal, return argmin sigma flagged.
    soft_relax: bool = True


class TelemetryCfg(BaseModel):
    log_dir: str = "results/concord"
    jsonl: bool = True
    wandb: bool = False
    mlflow: bool = False


class PipelineCfg(BaseModel):
    """Knobs for the Split → Solve → Combine → Verify pipeline solver.

    The pipeline replaces the legacy `RewriteExpansionPolicy`'s single
    "decompose then sample K" step with four phases. See
    `core/pipeline/expansion.py` for the full state machine.
    """

    # Recursive splitter — how deep can the atom tree grow?
    max_split_depth: int = 6
    # Maximum atomic units per splitter call. A *cap*, not a target.
    max_atoms_per_split: int = 6
    # Optional task-specific override for the splitter system prompt. When set,
    # it REPLACES the generic decomposition instructions (still `{max_atoms}`-
    # templated). Used by index-reference tasks (e.g. matmul) so the splitter
    # emits `source_span` ranges instead of copying values. None → generic.
    splitter_system_override: str | None = None
    # HARD ceiling on the number of agent (LLM) calls ONE node's pipeline
    # solve may spend — summed across splitter recursion, leaf execution,
    # composite synthesis, the block combiner, and the verifier. Without it
    # the recursive decomposition can explode: max_atoms_per_split **
    # max_split_depth leaves, each costing K_executor calls, easily reaches
    # thousands of calls for a single node. When the budget is exhausted the
    # splitter stops decomposing and any not-yet-solved atoms are left empty
    # so the combiner still produces a block answer from what was solved.
    max_node_calls: int = 100
    # Heuristic guard: a question shorter than this and free of obvious
    # multi-step markers is treated as atomic without burning a Splitter
    # LLM call.
    atom_short_circuit_chars: int = 300
    # Backtracking thresholds (block verifier score → action):
    #   verifier_accept ≤ score        → commit, no retry
    #   verifier_retry  ≤ score < accept → retry the combiner with hint
    #   else                            → MCTS backtrack (revise atomic)
    verifier_accept: float = 0.75
    verifier_retry: float = 0.50
    verifier_backtrack: float = 0.25
    # How many times can the combiner be retried with hints before we
    # escalate to an MCTS backtrack on the atomic answers.
    combiner_retries: int = 2
    # Number of samples drawn from the executor / combiner per call.
    K_executor: int = 3
    K_combiner: int = 3
    # Shared memory retrieval (text + vector). Top-k cosine neighbours
    # returned when an atom asks for context by free-text query.
    sharedmem_topk: int = 3


class AblationCfg(BaseModel):
    """Switches that select non-default variants for ablation runs (§8.4)."""

    confidence: Literal["semantic_density", "cosine_centroid"] = "semantic_density"
    cluster: Literal["entail_hybrid", "cosine_threshold"] = "entail_hybrid"
    backup: Literal["guarded_max", "average"] = "guarded_max"
    widening: Literal["progressive", "fixed_beam"] = "progressive"
    gate: Literal["incremental_hard", "terminal_only", "off"] = "incremental_hard"
    depth: Literal["adaptive", "fixed"] = "adaptive"
    weight_oracle: Literal["auto", "whitebox", "blackbox"] = "auto"


class Config(BaseModel):
    seed: int = 42
    # Which expansion policy MCTS uses. Default "mcts" preserves the
    # legacy RewriteExpansionPolicy so existing tests keep their semantics.
    # New runs (wide_and_deep.yaml) opt into "pipeline" for the
    # Split → Solve → Combine → Verify path.
    solver: Literal["mcts", "pipeline"] = "mcts"
    llm: LLMCfg = Field(default_factory=LLMCfg)
    models: RoleModelsCfg = Field(default_factory=RoleModelsCfg)
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)
    confidence: ConfidenceCfg = Field(default_factory=ConfidenceCfg)
    mcts: MCTSCfg = Field(default_factory=MCTSCfg)
    depth: DepthCfg = Field(default_factory=DepthCfg)
    coherence: CoherenceCfg = Field(default_factory=CoherenceCfg)
    telemetry: TelemetryCfg = Field(default_factory=TelemetryCfg)
    ablation: AblationCfg = Field(default_factory=AblationCfg)
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)

    def role_model(self, role: str) -> LLMCfg:
        """Return the LLMCfg for a role, falling back to the top-level `llm:`
        block when the role is unset. Unknown roles also fall back."""
        per_role = getattr(self.models, role, None) if hasattr(self.models, role) else None
        if per_role is None:
            return self.llm
        # role config wins on every field it explicitly sets; missing fields
        # inherit from the top-level `llm:` defaults.
        merged = self.llm.model_dump()
        merged.update({k: v for k, v in per_role.model_dump().items()
                       if v is not None})
        return LLMCfg.model_validate(merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_dict(self) -> dict:
        return self.model_dump()
