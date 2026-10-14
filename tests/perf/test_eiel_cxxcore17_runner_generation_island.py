from __future__ import annotations

from perf.cxxcore17_runner_generation_island import (
    DECISION,
    candidate_boundary_payload,
    compare_summary_tapes,
    config_from_sim_cfg,
    dependency_audit_payload,
    dependency_audit_rows,
    result_payload,
    run_native_summary_shadow,
    run_python_summary_reference,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture


def _inputs(seed: int = 17001, *, n: int = 8, width: int = 8, height: int = 8, ticks: int = 5):
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=n, width=width, height=height, ticks=ticks, seed=seed)
    return inputs


def test_cxxcore17_flags_default_off() -> None:
    cfg = config_from_sim_cfg(SimConfig())

    assert cfg.cxxcore17_runner_island_enable is False
    assert cfg.cxxcore17_runner_island_mode == "off"
    assert cfg.cxxcore17_runner_island_compare_enable is True
    assert cfg.cxxcore17_runner_island_summary_interval == 0
    assert cfg.cxxcore17_runner_island_fail_closed is True


def test_cxxcore17_dependency_audit_classifies_blockers() -> None:
    payload = dependency_audit_payload()
    rows = {row.dependency: row for row in dependency_audit_rows()}

    assert payload["decision"] == DECISION
    assert rows["env scalar state and positions"].classification == "resident_safe"
    assert rows["Agent.stats last deltas and survival/collection counters"].classification == "summary_safe"
    assert rows["memory_action_window instrumentation"].classification == "per_tick_python_required"
    assert rows["chemistry state"].blocks_resident_island is True
    assert payload["checkpoint_schema_changed"] is False
    assert payload["metrics_schema_changed"] is False
    assert payload["native_rng_added"] is False


def test_cxxcore17_candidate_boundary_payload_contains_required_candidates() -> None:
    payload = candidate_boundary_payload()
    names = [row["candidate"] for row in payload["candidates"]]

    assert any(name.startswith("A: env-only") for name in names)
    assert any(name.startswith("B: env + agent stats") for name in names)
    assert any(name.startswith("C: env + drives") for name in names)
    assert any(name.startswith("D: action-precomputed") for name in names)
    assert any(name.startswith("E: larger generation") for name in names)


def test_cxxcore17_python_summary_reference_parity_and_determinism() -> None:
    inputs = _inputs(17002, n=16, width=8, height=8, ticks=10)

    a = run_python_summary_reference(inputs, sparse_interval=5)
    b = run_python_summary_reference(inputs, sparse_interval=5)

    assert a.status == "ok"
    assert a.parity["ok"] is True
    assert a.summary_tape is not None
    assert b.summary_tape is not None
    assert a.summary_tape.digest == b.summary_tape.digest
    assert a.materializations == 1
    assert a.python_env_step_skipped is True


def test_cxxcore17_summary_tape_divergence_localizes() -> None:
    inputs = _inputs(17003, n=16, width=8, height=8, ticks=10)
    ok = run_python_summary_reference(inputs, sparse_interval=5)
    bad = run_python_summary_reference(inputs, sparse_interval=5, inject_divergence=True)

    assert ok.summary_tape is not None
    assert bad.summary_tape is not None
    cmp = compare_summary_tapes(ok.summary_tape, bad.summary_tape)
    assert cmp["ok"] is False
    assert cmp["first_divergence"] is not None
    assert cmp["first_divergence"]["field"] in {"digest", "energy_delta_total", "final_state_digest"}


def test_cxxcore17_missing_native_fallback() -> None:
    result = run_native_summary_shadow(_inputs(17004), native_backend=None)
    payload = result_payload(result)

    assert result.status == "fallback"
    assert payload["fallback_reason"] == "native_missing"
    assert payload["python_env_step_skipped"] is False


def test_cxxcore17_native_exception_fallback() -> None:
    result = run_native_summary_shadow(
        _inputs(17005),
        native_backend=lambda *args, **kwargs: {},
        raise_exception=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_exception"


def test_cxxcore17_native_shadow_with_python_backend_passes() -> None:
    def _backend(inputs, *, sparse_interval: int = 0, export_final: bool = True):
        from perf.cxxcore15_resident_soa_tick_island import (
            ResidentSoAConfig,
            as_resident_state,
            resident_result_payload,
            run_resident_python_reference,
        )

        result = run_resident_python_reference(
            as_resident_state(inputs),
            config=ResidentSoAConfig(sparse_interval=sparse_interval, export_final=export_final),
        )
        payload = resident_result_payload(result)
        return {
            "output": result.output,
            "agent_grid": result.agent_grid,
            "food_gained": result.food_gained,
            "moved_mask": result.moved_mask,
            "views_dirty": result.views_dirty,
            "sparse_ticks": result.sparse_ticks,
            "sparse_summaries": result.sparse_summaries,
            "final_summary": payload["final_summary"],
            "timing": {"backend_compute_s": 0.0},
        }

    result = run_native_summary_shadow(_inputs(17006), native_backend=_backend, sparse_interval=3)

    assert result.status == "ok"
    assert result.parity["ok"] is True
    assert result.parity["summary_tape"]["ok"] is True


def test_cxxcore17_injected_native_divergence_localizes() -> None:
    def _backend(inputs, *, sparse_interval: int = 0, export_final: bool = True):
        from perf.cxxcore15_resident_soa_tick_island import (
            ResidentSoAConfig,
            as_resident_state,
            resident_result_payload,
            run_resident_python_reference,
        )

        result = run_resident_python_reference(
            as_resident_state(inputs),
            config=ResidentSoAConfig(sparse_interval=sparse_interval, export_final=export_final),
        )
        payload = resident_result_payload(result)
        return {
            "output": result.output,
            "agent_grid": result.agent_grid,
            "food_gained": result.food_gained,
            "moved_mask": result.moved_mask,
            "views_dirty": result.views_dirty,
            "sparse_ticks": result.sparse_ticks,
            "sparse_summaries": result.sparse_summaries,
            "final_summary": payload["final_summary"],
            "timing": {"backend_compute_s": 0.0},
        }

    result = run_native_summary_shadow(
        _inputs(17007),
        native_backend=_backend,
        sparse_interval=3,
        inject_divergence=True,
    )

    assert result.status == "divergence"
    assert result.fallback_reason == "divergence_detected"
    assert result.first_divergence is not None

