from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
    ModularStorageError,
    build_modular_storage_layout,
    modular_storage_from_projection,
    modular_storage_generation_boundary_projection_payload,
    modular_storage_live_projection_payload,
    pack_modular_storage,
)
from perf.cuda_architecture_b_modular_state_bank import (
    build_modular_state_bank_manifest,
)
from perf.cuda_architecture_b_modular_tick import execute_modular_tick
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    LOCAL_SIZE,
    TICK_AGES,
    _fixed_inputs,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_tick import _packed_fixture


def _packed_profile(profile_id: str):
    expanded, _, _ = _packed_fixture()
    records = deepcopy(list(expanded.records_sidecar))
    for record in records:
        if record is not None:
            record["genome"]["b2_caps"] = deepcopy(
                B2_CAPACITY_PROFILES[profile_id]
            )
    return pack_modular_storage(
        records,
        agent_capacity=AGENT_COUNT,
        profile_id=profile_id,
    )


def _tick(planes, observations, tick: int):
    return execute_modular_tick(
        planes,
        torch.from_numpy(observations[tick].copy()),
        torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy()),
        torch.full((AGENT_COUNT,), tick, dtype=torch.int64),
    )


@pytest.mark.parametrize("profile_id", ("b2-12x32-v1", "b2-16x48-v1"))
def test_live_tick_checkpoint_resume_matches_uninterrupted_exactly(
    profile_id: str,
) -> None:
    authority = _packed_profile(profile_id)
    uninterrupted = {
        name: value.clone() for name, value in authority.tensors.items()
    }
    split = {name: value.clone() for name, value in authority.tensors.items()}
    observations, _, _, _ = _fixed_inputs()
    rng_before = torch.get_rng_state().clone()

    first_uninterrupted = _tick(uninterrupted, observations, 0)
    first_split = _tick(split, observations, 0)
    assert torch.equal(
        first_uninterrupted.action_residuals,
        first_split.action_residuals,
    )

    payload = modular_storage_live_projection_payload(authority, split)
    resumed = modular_storage_from_projection(
        payload["metadata"], payload["tensors"]
    )
    assert resumed is not None
    assert resumed.packed_sha256 != authority.packed_sha256
    assert resumed.identity_table == authority.identity_table
    assert resumed.layout.profile_id == profile_id
    assert resumed.layout.physical_module_slots == 16
    assert resumed.layout.physical_connection_slots == 48
    assert build_modular_state_bank_manifest(
        AGENT_COUNT, profile_id
    ).profile_id == profile_id

    resumed_planes = {
        name: value.clone() for name, value in resumed.tensors.items()
    }
    for tick in range(1, len(TICK_AGES)):
        continuous_result = _tick(uninterrupted, observations, tick)
        resumed_result = _tick(resumed_planes, observations, tick)
        assert torch.equal(
            resumed_result.action_residuals,
            continuous_result.action_residuals,
        )
        for name in uninterrupted:
            assert torch.equal(resumed_planes[name], uninterrupted[name]), name

    final_payload = modular_storage_live_projection_payload(
        resumed, resumed_planes
    )
    final_roundtrip = modular_storage_from_projection(
        final_payload["metadata"], final_payload["tensors"]
    )
    assert final_roundtrip is not None
    for name in uninterrupted:
        assert torch.equal(final_roundtrip.tensors[name], uninterrupted[name]), name
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_live_checkpoint_refresh_rejects_static_plane_change() -> None:
    authority = _packed_profile("b2-12x32-v1")
    live = {name: value.clone() for name, value in authority.tensors.items()}
    live["module_count"][0] += 1

    with pytest.raises(ModularStorageError, match="cold/static plane"):
        modular_storage_live_projection_payload(authority, live)


def test_live_checkpoint_refresh_changes_only_committed_runtime_authority() -> None:
    authority = _packed_profile("b2-12x32-v1")
    live = {name: value.clone() for name, value in authority.tensors.items()}
    observations, _, _, _ = _fixed_inputs()
    _tick(live, observations, 0)

    payload = modular_storage_live_projection_payload(authority, live)
    refreshed = modular_storage_from_projection(
        payload["metadata"], payload["tensors"]
    )
    assert refreshed is not None
    mutable = set(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
    for name in live:
        assert torch.equal(refreshed.tensors[name], live[name])
        if name not in mutable:
            assert torch.equal(refreshed.tensors[name], authority.tensors[name])


@pytest.mark.parametrize("profile_id", ("b2-12x32-v1", "b2-16x48-v1"))
def test_generation_boundary_death_rebinds_cold_owner_and_resumes_exactly(
    profile_id: str,
) -> None:
    authority = _packed_profile(profile_id)
    live = {name: value.clone() for name, value in authority.tensors.items()}
    observations, _, _, _ = _fixed_inputs()
    _tick(live, observations, 0)

    dead_slot = 0
    for field in build_modular_storage_layout(
        AGENT_COUNT, profile_id
    ).fields:
        live[field.name][dead_slot].fill_(field.fill_value)
    alive_mask = authority.tensors["agent_occupied"].clone()
    slot_to_agent_id = authority.tensors["agent_id"].clone()
    slot_tokens = authority.tensors["agent_slot_token"].clone()
    alive_mask[dead_slot] = False
    slot_to_agent_id[dead_slot] = -1

    payload = modular_storage_generation_boundary_projection_payload(
        authority,
        live,
        alive_mask=alive_mask,
        slot_to_agent_id=slot_to_agent_id,
        slot_tokens=slot_tokens,
    )
    resumed = modular_storage_from_projection(
        payload["metadata"], payload["tensors"]
    )
    assert resumed is not None
    assert resumed.records_sidecar[dead_slot] is None
    assert torch.equal(resumed.tensors["agent_occupied"], alive_mask)
    assert torch.equal(resumed.tensors["agent_id"], slot_to_agent_id)
    assert resumed.tensors["agent_slot_token"][dead_slot].item() == -1
    for field in resumed.layout.fields:
        assert torch.equal(
            resumed.tensors[field.name][dead_slot],
            torch.full_like(
                resumed.tensors[field.name][dead_slot], field.fill_value
            ),
        ), field.name

    resumed_planes = {
        name: value.clone() for name, value in resumed.tensors.items()
    }
    _tick(resumed_planes, observations, 1)
    assert resumed_planes["agent_occupied"][dead_slot].item() is False
    assert resumed_planes["agent_id"][dead_slot].item() == -1


def test_generation_boundary_rejects_uncleared_death_or_survivor_mutation() -> None:
    authority = _packed_profile("b2-12x32-v1")
    alive_mask = authority.tensors["agent_occupied"].clone()
    slot_to_agent_id = authority.tensors["agent_id"].clone()
    slot_tokens = authority.tensors["agent_slot_token"].clone()
    alive_mask[0] = False
    slot_to_agent_id[0] = -1

    uncleared = {
        name: value.clone() for name, value in authority.tensors.items()
    }
    uncleared["agent_occupied"][0] = False
    uncleared["agent_id"][0] = -1
    uncleared["agent_slot_token"][0] = -1
    with pytest.raises(ModularStorageError, match="dead modular row uncleared"):
        modular_storage_generation_boundary_projection_payload(
            authority,
            uncleared,
            alive_mask=alive_mask,
            slot_to_agent_id=slot_to_agent_id,
            slot_tokens=slot_tokens,
        )

    changed = {
        name: value.clone() for name, value in authority.tensors.items()
    }
    for field in authority.layout.fields:
        changed[field.name][0].fill_(field.fill_value)
    changed["module_count"][1] += 1
    with pytest.raises(ModularStorageError, match="surviving cold/static plane"):
        modular_storage_generation_boundary_projection_payload(
            authority,
            changed,
            alive_mask=alive_mask,
            slot_to_agent_id=slot_to_agent_id,
            slot_tokens=slot_tokens,
        )


@pytest.mark.parametrize(
    ("presence_name", "value_name", "value"),
    (
        ("module_state_present", "module_state", 0.875),
        ("module_output_present", "module_previous_output", -0.625),
        ("development_latch_present", "development_latch_state", True),
        (
            "development_effective_gain_present",
            "development_effective_gain",
            0.375,
        ),
    ),
)
def test_generation_boundary_rejects_hidden_runtime_value_without_presence(
    presence_name: str,
    value_name: str,
    value: float | bool,
) -> None:
    authority = _packed_profile("b2-12x32-v1")
    live = {name: tensor.clone() for name, tensor in authority.tensors.items()}
    if presence_name in {"module_state_present", "module_output_present"}:
        live["module_state_present"][1, 0] = False
        live["module_output_present"][1, 0] = False
    live[presence_name][1, 0] = False
    live[value_name][1, 0] = value

    with pytest.raises(ModularStorageError, match="not exactly sidecar-representable"):
        modular_storage_generation_boundary_projection_payload(
            authority,
            live,
            alive_mask=authority.tensors["agent_occupied"].clone(),
            slot_to_agent_id=authority.tensors["agent_id"].clone(),
            slot_tokens=authority.tensors["agent_slot_token"].clone(),
        )
