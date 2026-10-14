"""R11 monotonic-ID/slot-token identity authority tests."""

from pathlib import Path

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_identity_r11 import (
    C4_IDENTITY_R11_DECISION_V1,
    C4_IDENTITY_R11_SCHEMA_VERSION_V1,
    C4_IDENTITY_R11_STATUS_V1,
    bind_c4_identity_authority_r11_v1,
    c4_identity_authority_r11_source_receipt_v1,
    make_c4_identity_staging_r11_v1,
    resolve_c4_identity_slot_r11_v1,
)
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    commit_c4_ordinary_births_oracle_v1,
    _rebuild_active_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def test_r11_identity_resolves_ids_above_slot_capacity_without_dense_lookup() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=4, active=2, width=8, world_local_identity=True
    )
    core = bundle.base.base.core
    core.agents.agent_id[0, 0] = 100
    core.agents.agent_id[0, 1] = 10_000
    bundle.population.next_agent_id[0] = 10_001
    _rebuild_active_projection(bundle, 0)

    authority = bind_c4_identity_authority_r11_v1(bundle)
    assert authority.capacity == 4
    assert resolve_c4_identity_slot_r11_v1(authority, world=0, agent_id=100) == 0
    assert resolve_c4_identity_slot_r11_v1(authority, world=0, agent_id=10_000) == 1
    assert resolve_c4_identity_slot_r11_v1(authority, world=0, agent_id=3) is None


def test_r11_birth_reuses_slot_with_monotonic_id_and_legacy_map_stays_compatibility_only() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=4, active=2, width=8, world_local_identity=True
    )
    core = bundle.base.base.core
    bundle.population.next_agent_id[0] = 100
    bundle.population.free_queue[0].fill_(-1)
    bundle.population.free_queue[0, 0] = 2
    bundle.population.free_queue_count[0] = 1
    bundle.population.parent_slots[0, 0] = 0
    bundle.population.child_slots[0, 0] = 2
    bundle.population.child_pos_y[0, 0] = 0
    bundle.population.child_pos_x[0, 0] = 0
    bundle.population.child_energy[0, 0] = 10.0
    bundle.population.planned_count[0] = 1
    bundle.population.decision_count[0] = 1
    bundle.population.decision_parent_slots[0, 0] = 0
    bundle.population.decision_reason_codes[0, 0] = 0
    core.agents.health[0, 0] = 12.0
    core.agents.energy[0, 0] = 20.0
    bundle.base.agent.policy_energy_observation[0, 0] = 20.0
    bundle.base.agent.repro_threshold[0, 0] = 0.0
    core.occupancy.agent_id_grid[0].fill_(-1)
    core.occupancy.slot_grid[0].fill_(-1)
    bundle.base.base.patch.agent_grid_i32[0].fill_(-1)

    commit_c4_ordinary_births_oracle_v1(
        bundle,
        TensorWorldC4BirthConfigV1(
            max_births=1,
            allow_mutation=False,
            require_spawn_rng_tape=False,
            generation=7,
        ),
    )

    assert int(core.agents.agent_id[0, 2]) == 100
    assert int(bundle.population.next_agent_id[0]) == 101
    assert int(bundle.population.active_agent_ids[0, 2]) == 100
    assert int(bundle.population.active_agent_slots[0, 2]) == 2
    # The old [B,C] map cannot represent ID 100 and must not be treated as
    # authoritative; the sorted active projection resolves it exactly.
    assert int(bundle.population.agent_id_to_slot[0, 0]) == -1
    authority = bind_c4_identity_authority_r11_v1(bundle)
    assert resolve_c4_identity_slot_r11_v1(authority, world=0, agent_id=100) == 2


def test_r11_identity_receipt_is_explicitly_not_a_gate_claim() -> None:
    receipt = c4_identity_authority_r11_source_receipt_v1()
    assert receipt["schema_version"] == C4_IDENTITY_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_IDENTITY_R11_DECISION_V1
    assert receipt["legacy_dense_map"] == "compatibility projection only; not authoritative"
    assert receipt["gate_claim"] is False


def test_r11_identity_staging_is_count_masked_and_status_codes_are_frozen() -> None:
    staging = make_c4_identity_staging_r11_v1(
        batch=48, capacity=32, device="cpu"
    )
    staging.validate(batch=48, capacity=32)
    assert staging.death_slot_tokens.eq(-1).all()
    assert staging.parent_slot_tokens.eq(-1).all()
    assert staging.mate_slot_tokens.eq(-1).all()
    assert staging.child_slot_tokens.eq(-1).all()
    assert staging.status.eq(0).all()
    assert C4_IDENTITY_R11_STATUS_V1[60] == "active_count_invalid"
    assert C4_IDENTITY_R11_STATUS_V1[64] == "compatibility_projection_mismatch"
    assert C4_IDENTITY_R11_STATUS_V1[65] == "slot_token_mismatch"
    assert C4_IDENTITY_R11_STATUS_V1[66] == "slot_token_overflow"


def test_r11_sorted_identity_survives_more_than_c512_slot_reuses() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=4, active=2, width=8, world_local_identity=True
    )
    core = bundle.base.base.core
    population = bundle.population
    for child_id in range(5, 518):
        core.agents.agent_id[0, 1] = child_id
        population.slot_tokens[0, 1] += 1
        population.next_agent_id[0] = child_id + 1
        population.agent_id_to_slot[0].fill_(-1)
        _rebuild_active_projection(bundle, 0)
    authority = bind_c4_identity_authority_r11_v1(bundle)
    assert int(population.next_agent_id[0]) == 518
    assert int(population.slot_tokens[0, 1]) == 514
    assert resolve_c4_identity_slot_r11_v1(
        authority, world=0, agent_id=517
    ) == 1
    assert population.agent_id_to_slot.eq(-1).all()


def test_r11_native_sources_gate_sorted_mode_to_composed_route() -> None:
    native = ROOT / "engine/tensor_world/native"
    r4 = (native / "tensor_world_c4_resident_mutation_r4.cu").read_text()
    r5 = (native / "tensor_world_c4_resident_population_r5.cu").read_text()
    r6 = (native / "tensor_world_c4_resident_preflight_r6.cu").read_text()
    r9 = (native / "tensor_world_c4_resident_population_r9.cu").read_text()
    r11 = (native / "tensor_world_c4_resident_graph_r11.cu").read_text()
    assert "kIdentitySortedMonotonic" in r4
    assert "sorted_identity_slot" in r5
    assert "kCompatibilityProjection" in r6
    assert "identity_mode" not in r9
    assert "kIdentitySortedMonotonic" in r11
    assert "capture_identity_tokens" in r11
    assert "validate_identity_tokens" in r11
    assert "publish_first_status" in r11
    assert "publish_r9_status<<<" not in r11
