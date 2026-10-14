from __future__ import annotations

import json
from types import SimpleNamespace

from agents.cognition import developmental_effective_projection
from simulation.env_hotpath_expression_diagnostics import EnvHotpathExpressionDiagnostics
from simulation.runner_pkg.runner_config import SimConfig


def test_env_hotpath_expression_diagnostics_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.env_hotpath_expression_diagnostics_enable is False
    assert cfg.env_hotpath_expression_diagnostics_schema_version == 1
    assert cfg.env_hotpath_expression_diagnostics_write_events is False
    assert cfg.env_hotpath_apply_decomposition_enable is False
    assert cfg.env_hotpath_apply_decomposition_schema_version == 1
    assert cfg.env_hotpath_phenotype_decomposition_enable is False
    assert cfg.env_hotpath_phenotype_decomposition_schema_version == 1
    assert cfg.env_hotpath_inactive_phenotype_defaults_enable is False
    assert cfg.env_hotpath_developmental_residual_diagnostics_enable is False
    assert cfg.env_hotpath_developmental_residual_diagnostics_schema_version == 1
    assert cfg.env_hotpath_developmental_cache_equality_diagnostics_enable is False
    assert cfg.env_hotpath_developmental_cache_equality_schema_version == 1
    assert cfg.env_hotpath_developmental_effective_projection_diagnostics_enable is False
    assert cfg.env_hotpath_developmental_effective_projection_schema_version == 1
    assert cfg.env_hotpath_budget_ledger_scalar_fastpath_enable is False

    runner = SimpleNamespace(sim_cfg=cfg, run_dir="/tmp/not-used")
    assert EnvHotpathExpressionDiagnostics.maybe_create(runner) is None


def test_env_hotpath_expression_diagnostics_summary_counts_repeats(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path))

    diag.record_dev_expression_request(
        agent_id=7,
        generation=1,
        tick=2,
        age_steps=3,
        cache_hit=False,
        has_genome=True,
    )
    diag.record_dev_expression_request(
        agent_id=7,
        generation=1,
        tick=2,
        age_steps=3,
        cache_hit=True,
        has_genome=True,
    )
    diag.record_phenotype_call(agent_id=7, generation=1, tick=2, age_steps=3, source="env_cost_charging")
    diag.record_dev_cache_clear(generation=1, tick=3, entries=1)

    out = diag.write_summary()
    payload = json.loads((tmp_path / "metrics" / "env_hotpath_expression_summary.json").read_text(encoding="utf-8"))

    assert out.endswith("env_hotpath_expression_summary.json")
    assert payload["schema_version"] == "env_hotpath_expression_summary_v1"
    assert payload["dev_expression"]["requests"] == 2
    assert payload["dev_expression"]["cache_hits"] == 1
    assert payload["dev_expression"]["repeated_same_agent_tick_age_keys"] == 1
    assert payload["phenotype"]["calls"] == 1
    assert payload["phenotype"]["source_counts"] == {"env_cost_charging": 1}
    assert payload["invalidation"]["dev_cache_clear_nonempty"] == 1


def test_env_apply_decomposition_summary_serializes_canonicalize_stats(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), decomposition_enabled=True)

    diag.record_decomposition_bucket("env_apply_total", 1.0)
    diag.record_decomposition_bucket("env_apply_cost_charging_total", 0.4)
    diag.record_decomposition_bucket("env_apply_phenotype_total", 0.3)
    diag.record_native_budget_cost_aggregation(
        {
            "backend": "numba",
            "hit_count": 4,
            "fallback_count": 1,
            "mismatch_count": 0,
            "pack_rows": 5,
            "native_kernel_s": 0.03,
            "input_packing_s": 0.02,
        }
    )
    diag.record_direct_cost_scalar(
        {
            "hit_count": 7,
            "ledger_fallback_count": 1,
            "guard_check_count": 3,
            "guard_mismatch_count": 0,
            "max_abs_diff": 0.0,
            "direct_scalar_s": 0.04,
            "guard_ledger_s": 0.03,
            "fallback_ledger_s": 0.02,
            "skipped_reason_counts": {"exception": 1},
        }
    )
    diag.record_canonicalize_call(payload_key='{"stages":[]}', payload_bytes=13, seconds=0.01)
    diag.record_canonicalize_call(payload_key='{"stages":[]}', payload_bytes=13, seconds=0.02)

    out = diag.write_decomposition_summary()
    payload = json.loads((tmp_path / "metrics" / "env_apply_decomposition_summary.json").read_text(encoding="utf-8"))

    assert out.endswith("env_apply_decomposition_summary.json")
    assert payload["schema_version"] == "env_apply_decomposition_summary_v1"
    assert payload["seconds"]["env_apply_total"] == 1.0
    assert payload["canonicalize"]["total_calls"] == 2
    assert payload["canonicalize"]["unique_payloads"] == 1
    assert payload["derived"]["canonicalize_repeat_fraction"] == 0.5
    assert payload["native_cognitive_budget_cost_aggregation"]["backend_counts"] == {"numba": 5}
    assert payload["native_cognitive_budget_cost_aggregation"]["hit_count"] == 4
    assert payload["native_cognitive_budget_cost_aggregation"]["fallback_count"] == 1
    assert payload["seconds"]["env_apply_native_budget_cost_kernel_total"] == 0.03
    assert payload["seconds"]["env_apply_native_budget_cost_input_packing_total"] == 0.02
    assert payload["direct_cost_scalar"]["hit_count"] == 7
    assert payload["direct_cost_scalar"]["ledger_fallback_count"] == 1
    assert payload["direct_cost_scalar"]["guard_check_count"] == 3
    assert payload["direct_cost_scalar"]["guard_mismatch_count"] == 0
    assert payload["direct_cost_scalar"]["skipped_reason_counts"] == {"exception": 1}
    assert payload["seconds"]["env_apply_direct_cost_scalar_total"] == 0.04
    assert payload["seconds"]["env_apply_direct_cost_scalar_guard_ledger_total"] == 0.03
    assert payload["seconds"]["env_apply_direct_cost_scalar_fallback_ledger_total"] == 0.02


def test_envhot3_phenotype_decomposition_summary_serializes_buckets(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), phenotype_decomposition_enabled=True)

    diag.record_phenotype_decomposition_bucket("phenotype_total", 1.0)
    diag.record_phenotype_decomposition_bucket("phenotype_perception_total", 0.25)
    diag.record_phenotype_decomposition_bucket("phenotype_directed_payload_from_dict_total", 0.05)
    diag.record_phenotype_inactive_path("directed_payload_non_dict_or_none")
    diag.record_phenotype_inactive_path("directed_payload_non_dict_or_none")
    diag.record_phenotype_inactive_default("factory:directed_payload_empty")

    out = diag.write_phenotype_decomposition_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot3_phenotype_decomposition_summary.json").read_text(encoding="utf-8")
    )

    assert out.endswith("envhot3_phenotype_decomposition_summary.json")
    assert payload["schema_version"] == "envhot3_phenotype_decomposition_summary_v1"
    assert payload["seconds"]["phenotype_total"] == 1.0
    assert payload["counts"]["phenotype_perception_total"] == 1
    assert payload["inactive_paths"]["directed_payload_non_dict_or_none"] == 2
    assert payload["inactive_default_strategy"]["factory:directed_payload_empty"] == 1
    assert payload["derived"]["phenotype_other_total"] == 0.7


def test_envhot3_phenotype_decomposition_flag_creates_diagnostics() -> None:
    cfg = SimConfig(env_hotpath_phenotype_decomposition_enable=True)
    runner = SimpleNamespace(sim_cfg=cfg, run_dir="/tmp/not-used")

    diag = EnvHotpathExpressionDiagnostics.maybe_create(runner)

    assert diag is not None
    assert diag.phenotype_decomposition_enabled is True
    assert diag.decomposition_enabled is False


def test_envhot7_developmental_residual_flag_creates_diagnostics() -> None:
    cfg = SimConfig(env_hotpath_developmental_residual_diagnostics_enable=True)
    runner = SimpleNamespace(sim_cfg=cfg, run_dir="/tmp/not-used")

    diag = EnvHotpathExpressionDiagnostics.maybe_create(runner)

    assert diag is not None
    assert diag.developmental_residual_enabled is True
    assert diag.decomposition_enabled is False
    assert diag.phenotype_decomposition_enabled is False


def test_envhot7_developmental_residual_summary_serializes_buckets(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), developmental_residual_enabled=True)

    diag.record_canonicalize_call(payload_key='{"stages":[]}', payload_bytes=13, seconds=0.01)
    diag.record_canonicalize_call(payload_key='{"stages":[]}', payload_bytes=13, seconds=0.02)
    diag.record_developmental_effective_lookup(source="genome_resolve", agent_id=7, age_steps=3, seconds=0.03)
    diag.record_developmental_stage_resolution(
        program_key='{"stages":[]}',
        age_steps=3,
        stage_id="default",
        seconds=0.04,
    )
    diag.record_developmental_module_active(
        program_key='{"stages":[]}',
        age_steps=3,
        module_id="m1",
        stage_id="default",
        seconds=0.05,
    )
    diag.record_developmental_module_active(
        program_key='{"stages":[]}',
        age_steps=3,
        module_id="m1",
        stage_id="default",
        seconds=0.06,
    )
    diag.record_developmental_genome_resolve(
        genome_key="g1",
        program_key='{"stages":[]}',
        age_steps=3,
        stage_id="default",
        module_count=1,
        seconds=0.07,
    )

    out = diag.write_developmental_residual_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot7_developmental_residual_summary.json").read_text(encoding="utf-8")
    )

    assert out.endswith("envhot7_developmental_residual_summary.json")
    assert payload["schema_version"] == "envhot7_developmental_residual_summary_v1"
    assert payload["source_counts"] == {"genome_resolve": 1}
    assert payload["canonicalize"]["total_calls"] == 2
    assert payload["canonicalize"]["unique_payloads"] == 1
    assert payload["derived"]["repeated_module_keys"] == 1
    assert payload["counts"]["genome_resolve_total"] == 1


def test_envhot8_developmental_cache_equality_flag_creates_diagnostics() -> None:
    cfg = SimConfig(env_hotpath_developmental_cache_equality_diagnostics_enable=True)
    runner = SimpleNamespace(sim_cfg=cfg, run_dir="/tmp/not-used")

    diag = EnvHotpathExpressionDiagnostics.maybe_create(runner)

    assert diag is not None
    assert diag.developmental_cache_equality_enabled is True
    assert diag.developmental_residual_enabled is False


def test_envhot8_developmental_cache_equality_summary_records_projection_match(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), developmental_cache_equality_enabled=True)
    runner_expr = {
        "stage_id": "adult",
        "active_by_module_id": {"m1": True},
        "inactive_reason_by_module_id": {},
        "effective": {"pred_lr": 0.2},
        "lock_state_by_module_id": {"m1": True},
        "lock_events": [],
        "lock_event_count": 0,
    }
    direct_expr = {
        "stage_id": "adult",
        "active_by_module_id": {"m1": True},
        "inactive_reason_by_module_id": {},
        "effective": {"pred_lr": 0.2},
    }
    candidate = {
        "agent_id": 7,
        "generation": 1,
        "tick": 2,
        "age_steps": 3,
        "lock_state_sha256": "abc",
        "expression": runner_expr,
    }

    diag.record_developmental_cache_candidate(
        agent_id=7,
        generation=1,
        tick=2,
        age_steps=3,
        lock_state_sha256="abc",
    )
    diag.record_developmental_cache_equality_check(
        candidate=candidate,
        direct=direct_expr,
        agent_id=7,
        age_steps=3,
        seconds=0.01,
    )

    out = diag.write_developmental_cache_equality_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot8_developmental_cache_equality_summary.json").read_text(encoding="utf-8")
    )

    assert out.endswith("envhot8_developmental_cache_equality_summary.json")
    assert payload["schema_version"] == "envhot8_developmental_cache_equality_summary_v1"
    assert payload["counts"]["candidate_reuse_checks"] == 1
    assert payload["counts"]["candidate_available"] == 1
    assert payload["counts"]["full_mismatches"] == 1
    assert payload["counts"]["effective_projection_matches"] == 1
    assert payload["derived"]["full_exact_match_fraction"] == 0.0
    assert payload["derived"]["effective_projection_match_fraction"] == 1.0
    assert payload["mismatch_fields"]["full:lock_state_by_module_id:missing_right"] == 1


def test_envhot8_developmental_cache_equality_summary_records_candidate_missing(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), developmental_cache_equality_enabled=True)

    diag.record_developmental_cache_equality_check(
        candidate=None,
        direct={"effective": {}},
        agent_id=7,
        age_steps=3,
        seconds=0.01,
    )

    diag.write_developmental_cache_equality_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot8_developmental_cache_equality_summary.json").read_text(encoding="utf-8")
    )

    assert payload["counts"]["candidate_reuse_checks"] == 1
    assert payload["fallback_reasons"] == {"candidate_missing": 1}


def test_envhot9_effective_projection_helper_filters_contract_fields() -> None:
    expression = {
        "effective": {
            "aggression_bias": 0.1,
            "action_temperature": 1.2,
            "drive_logit_gain": 1.3,
            "drive_inertia": 0.4,
            "drive_noise": 0.05,
            "pred_lr": 0.2,
            "reproduction_threshold": 0.8,
            "extra_effective_debug": "ignored",
        },
        "lock_state_by_module_id": {"m1": True},
        "lock_events": [{"module_id": "m1"}],
    }

    assert developmental_effective_projection(expression) == {
        "aggression_bias": 0.1,
        "action_temperature": 1.2,
        "drive_logit_gain": 1.3,
        "drive_inertia": 0.4,
        "drive_noise": 0.05,
        "pred_lr": 0.2,
        "reproduction_threshold": 0.8,
    }
    assert developmental_effective_projection({"pred_lr": 0.3, "lock_events": []}) == {"pred_lr": 0.3}
    assert developmental_effective_projection(None) == {}


def test_envhot9_developmental_effective_projection_flag_creates_diagnostics() -> None:
    cfg = SimConfig(env_hotpath_developmental_effective_projection_diagnostics_enable=True)
    runner = SimpleNamespace(sim_cfg=cfg, run_dir="/tmp/not-used")

    diag = EnvHotpathExpressionDiagnostics.maybe_create(runner)

    assert diag is not None
    assert diag.developmental_effective_projection_enabled is True
    assert diag.developmental_cache_equality_enabled is False


def test_envhot9_effective_projection_summary_ignores_runner_metadata(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), developmental_effective_projection_enabled=True)
    runner_expr = {
        "stage_id": "adult",
        "active_by_module_id": {"m1": True},
        "inactive_reason_by_module_id": {},
        "effective": {
            "pred_lr": 0.2,
            "drive_inertia": 0.4,
            "extra_effective_debug": "ignored",
        },
        "lock_state_by_module_id": {"m1": True},
        "lock_events": [],
        "lock_event_count": 0,
        "structural_change_ignored": False,
        "structural_change_reason": "",
    }
    direct_expr = {
        "stage_id": "adult",
        "active_by_module_id": {"m1": True},
        "inactive_reason_by_module_id": {},
        "effective": {
            "pred_lr": 0.2,
            "drive_inertia": 0.4,
        },
    }
    candidate = {
        "agent_id": 7,
        "generation": 1,
        "tick": 2,
        "age_steps": 3,
        "lock_state_sha256": "abc",
        "program_sha256": "def",
        "expression": runner_expr,
    }

    diag.record_developmental_cache_candidate(
        agent_id=7,
        generation=1,
        tick=2,
        age_steps=3,
        lock_state_sha256="abc",
    )
    diag.record_developmental_effective_projection_check(
        candidate=candidate,
        direct=direct_expr,
        direct_projection=developmental_effective_projection(direct_expr),
        agent_id=7,
        age_steps=3,
        seconds=0.02,
        projection_seconds=0.01,
    )

    out = diag.write_developmental_effective_projection_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot9_developmental_effective_projection_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert out.endswith("envhot9_developmental_effective_projection_summary.json")
    assert payload["schema_version"] == "envhot9_developmental_effective_projection_summary_v1"
    assert payload["counts"]["runner_cache_candidates_stored"] == 1
    assert payload["counts"]["candidate_reuse_checks"] == 1
    assert payload["counts"]["candidate_available"] == 1
    assert payload["counts"]["projection_exact_matches"] == 1
    assert payload["derived"]["projection_exact_match_fraction"] == 1.0
    assert payload["runner_only_metadata_fields"]["lock_state_by_module_id"] == 1
    assert payload["runner_only_metadata_fields"]["lock_events"] == 1
    assert payload["lock_signatures"]["unique"] == 1
    assert payload["program_signatures"]["unique"] == 1


def test_envhot9_effective_projection_summary_records_candidate_missing(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), developmental_effective_projection_enabled=True)

    diag.record_developmental_effective_projection_check(
        candidate=None,
        direct={"effective": {}},
        direct_projection={},
        agent_id=7,
        age_steps=3,
        seconds=0.01,
    )

    diag.write_developmental_effective_projection_summary()
    payload = json.loads(
        (tmp_path / "metrics" / "envhot9_developmental_effective_projection_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["counts"]["candidate_reuse_checks"] == 1
    assert payload["fallback_reasons"] == {"candidate_missing": 1}
