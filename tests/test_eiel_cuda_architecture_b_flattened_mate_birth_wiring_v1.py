from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "engine" / "perf"
KERNELS = PERF / "cuda_architecture_b_flattened_controller_v1_kernels.cu"
FLAT_EVENTS = PERF / "cuda_architecture_b_flattened_birth_events_v2.cuh"
PERSISTENT_EVENTS = PERF / "cuda_architecture_b_persistent_birth_events.cuh"


def test_flattened_controller_has_explicit_mate_and_asexual_routes() -> None:
    source = KERNELS.read_text(encoding="utf-8")

    assert "mate_plan::plan_births_serial_v1" in source
    assert (
        "stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_"
        "external_scratch_v3"
    ) in source
    assert (
        "stage_ordinary_fixed_mlp_child_scalar_genomes_external_scratch_v2"
    ) in source
    assert "if (!birth_config.requires_mate)" in source
    assert (
        "if (arguments.birth_config.requires_mate && birth_capacity <= 2)"
        in source
    )
    assert "transaction_status[0] = 213;" in source


def test_mate_commit_updates_both_parent_authorities() -> None:
    source = KERNELS.read_text(encoding="utf-8")

    for required in (
        "committed_mate_agent_ids",
        "arguments.energy[mate_index]",
        "arguments.policy_energy_observation[mate_index]",
        "arguments.birth_state.repro_cooldown[mate_index]",
        "arguments.birth_state.offspring_count[mate_index] += 1",
        "cell_capacity_offset + mate_slot",
    ):
        assert required in source


def test_birth_event_contract_accepts_all_mate_planner_reasons() -> None:
    flattened = FLAT_EVENTS.read_text(encoding="utf-8")
    persistent = PERSISTENT_EVENTS.read_text(encoding="utf-8")

    for source in (flattened, persistent):
        assert "reason >= 5 && reason <= 6" in source
        assert "reason >= 7 && reason <= 11" in source
