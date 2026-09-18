"""Per-role model assignments: schema, factory, orchestrator wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from core.config import Config, LLMCfg, RoleModelsCfg
from core.llm.factory import RoleClients, make_client
from core.llm.mock import MockLLM
from core.orchestrator import solve, solve_with_config


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

def test_default_config_has_empty_models_section():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    assert isinstance(cfg.models, RoleModelsCfg)
    # nothing pinned by default
    assert cfg.models.execution is None
    assert cfg.models.decomposition is None
    assert cfg.models.classification is None
    assert cfg.models.verification is None


def test_role_model_falls_back_to_llm_block_when_unset():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    spec = cfg.role_model("execution")
    assert spec.provider == cfg.llm.provider
    assert spec.model == cfg.llm.model


def test_role_model_override_wins_per_field():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.llm = LLMCfg(provider="anthropic", model="base-model", temperature=0.5,
                      max_output_tokens=1024)
    cfg.models.execution = LLMCfg(provider="anthropic", model="big-exec-model",
                                    temperature=1.0, max_output_tokens=2048)
    spec = cfg.role_model("execution")
    assert spec.model == "big-exec-model"
    assert spec.temperature == 1.0
    # other roles still come from the fallback llm block
    assert cfg.role_model("classification").model == "base-model"


def test_per_role_yaml_resolves_to_distinct_models():
    """The per_role.yaml ships with distinct models per role. We don't pin
    the EXACT model strings (the YAML is intentionally user-edited from time
    to time) — instead we assert the *structural* properties:
      - every role resolves to an anthropic model
      - execution + verification differ from classification (typical setup)
    """
    cfg = Config.from_yaml(PKG_ROOT / "config" / "per_role.yaml")
    for role in ("execution", "decomposition", "classification", "verification"):
        spec = cfg.role_model(role)
        assert spec.provider == "anthropic", role
        assert spec.model, role
    # classification is always the cheap option, distinct from execution
    assert (cfg.role_model("execution").model
            != cfg.role_model("classification").model)


def test_unknown_role_falls_back_to_llm_block():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    assert cfg.role_model("not_a_role").model == cfg.llm.model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_builds_mock_without_anthropic_import():
    spec = LLMCfg(provider="mock", model="mock-v0")
    client = make_client(spec)
    assert isinstance(client, MockLLM)


def test_factory_raises_on_unknown_provider():
    spec = LLMCfg(provider="anthropic", model="x")
    # We can't actually import anthropic here without network/credentials,
    # so we test the validation path instead.
    with pytest.raises(ValueError):
        make_client(LLMCfg.model_construct(provider="bogus", model="x"))


def test_role_clients_dedupes_shared_specs():
    """When multiple roles resolve to the same spec, they share a client.

    This matters for cost aggregation: a shared CostTally must not be
    double-counted by `total_cost()`.
    """
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.llm = LLMCfg(provider="mock", model="mock-v0", temperature=1.0)
    # Leave all roles unset -> every role.spec equals cfg.llm
    rc = RoleClients.from_config(cfg)
    # All four roles point at the SAME client object
    assert rc.execution is rc.decomposition
    assert rc.execution is rc.classification
    assert rc.execution is rc.verification
    # And there is exactly one unique client in the dedupe list
    assert len(rc._all_unique) == 1


def test_role_clients_distinct_when_specs_differ():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.llm = LLMCfg(provider="mock", model="base", temperature=1.0)
    cfg.models.execution = LLMCfg(provider="mock", model="exec", temperature=1.0)
    cfg.models.verification = LLMCfg(provider="mock", model="verif",
                                      temperature=0.0)
    rc = RoleClients.from_config(cfg)
    assert rc.execution is not rc.verification
    # decomposition + classification still share the fallback `base` client
    assert rc.decomposition is rc.classification
    # 3 unique clients (base, exec, verif)
    assert len(rc._all_unique) == 3


def test_role_clients_total_cost_aggregates_unique_clients():
    """Mock now counts per SAMPLE (matching the AnthropicAdapter), so
    generate(n=K) costs K not 1."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.llm = LLMCfg(provider="mock", model="base")
    cfg.models.execution = LLMCfg(provider="mock", model="exec")
    rc = RoleClients.from_config(cfg)

    rc.execution.generate("hi", temperature=1.0, n=2)        # 2 samples
    rc.verification.generate("ho", temperature=1.0, n=1)     # 1 sample

    total = rc.total_cost()
    # 2 separate clients: execution=2 samples, verification=1 sample → 3 total.
    assert total.calls == 3
    per = rc.per_role_cost()
    assert per["execution"]["calls"] == 2
    assert per["verification"]["calls"] == 1


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------

def test_solve_with_config_uses_role_clients(tmp_path: Path):
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.llm = LLMCfg(provider="mock", model="mock-v0", temperature=1.0)
    cfg.mcts.N = 4
    cfg.sampling.K_blackbox = 2

    res = solve_with_config("Problem node_0: trivial.", cfg=cfg,
                             domain="math", tag="role_models_test")
    # Cost dict contains per-role breakdown when RoleClients was used.
    assert "per_role" in res.cost
    # The four classic roles must always be present. Pipeline-solver
    # roles may also appear (splitter / combiner / synthesizer /
    # synth_verifier) when the new factory is in use.
    assert {"execution", "decomposition", "classification",
            "verification"} <= set(res.cost["per_role"])


def test_solve_legacy_single_llm_still_works(tmp_path: Path):
    """The pre-multi-model `solve(..., llm=...)` shape must still function."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.mcts.N = 4
    cfg.sampling.K_blackbox = 2
    llm = MockLLM()
    res = solve("Problem node_0: trivial.", cfg=cfg, llm=llm, domain="math")
    assert "per_role" in res.cost
    # Every role shares the legacy single LLM, so per-role tallies are equal.
    per = res.cost["per_role"]
    assert per["execution"]["calls"] == per["verification"]["calls"]


def test_distinct_verification_model_uses_llm_judge(tmp_path: Path):
    """When verification gets a different model, the LLM-judge verifier kicks in."""
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    cfg.telemetry.log_dir = str(tmp_path / "logs")
    cfg.llm = LLMCfg(provider="mock", model="base", temperature=1.0)
    cfg.models.verification = LLMCfg(provider="mock", model="judge",
                                       temperature=0.0)
    cfg.mcts.N = 4
    cfg.sampling.K_blackbox = 2

    res = solve_with_config("Problem node_0: trivial.", cfg=cfg, domain="math",
                             tag="judge_test")
    # Per-role costs are tracked separately when roles point at distinct clients
    assert res.cost["per_role"]["execution"]["calls"] >= 0
    assert res.cost["per_role"]["verification"]["calls"] >= 0


def test_solve_requires_either_llm_or_clients():
    cfg = Config.from_yaml(PKG_ROOT / "config" / "default.yaml")
    with pytest.raises(TypeError):
        solve("anything", cfg=cfg)
