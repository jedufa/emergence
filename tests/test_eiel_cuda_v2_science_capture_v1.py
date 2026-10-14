from __future__ import annotations

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

import engine.native_runtime.cuda_v2_science_capture_v1 as capture_adapter
from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION,
    SchemaVersion,
    canonical_json_bytes,
    canonical_science_record_stream_sha256_v1,
    validate_science_observation_stream_v1,
)
from engine.campaigns import cuda_full_evolution_v2_provider as cuda_provider
from engine.campaigns.providers import CampaignProviderError
from engine.native_runtime.cuda_v2_science_capture_v1 import (
    build_native_cuda_v2_checkpoint_marker_output_v1,
    build_native_cuda_v2_completion_output_v1,
    build_native_cuda_v2_observation_outputs_v1,
    science_capture_producer_phase_registry_v1,
)
from engine.native_runtime.science_observation_projection_v1 import (
    NativeCudaV2ObservationContextV1,
    finalize_native_cuda_v2_observations_v1,
)
from engine.campaigns.science_cuda_v2_native_control_v1 import (
    ScienceCudaV2NativeControlV1,
)
from engine.tw_rngv2.philox import philox4x32_10


class _TensorList:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


def _capture(
    *,
    action_agent_id: int = 42,
    birth: tuple[int, int, int, int, int] | None = None,
    start_tick: int = 0,
) -> SimpleNamespace:
    zeros = [0] * 512
    action_ids = [action_agent_id, *zeros[1:]]
    child_ids = list(zeros)
    parent_a_ids = list(zeros)
    parent_b_ids = list(zeros)
    mutation_counts = list(zeros)
    sites = list(zeros)
    birth_count = 0
    if birth is not None:
        child_id, parent_a, parent_b, mutation_count, site = birth
        child_ids[0] = child_id
        parent_a_ids[0] = parent_a
        parent_b_ids[0] = parent_b
        mutation_counts[0] = mutation_count
        sites[0] = site
        birth_count = 1
    return SimpleNamespace(
        action_agent_ids=_TensorList([[action_ids]]),
        action_values=_TensorList([[[3, *zeros[1:]]]]),
        birth_child_ids=_TensorList([[child_ids]]),
        birth_parent_a_ids=_TensorList([[parent_a_ids]]),
        birth_parent_b_ids=_TensorList([[parent_b_ids]]),
        birth_mutation_counts=_TensorList([[mutation_counts]]),
        birth_child_slots=_TensorList([[zeros]]),
        birth_sites=_TensorList([[sites]]),
        birth_counts=_TensorList([[birth_count]]),
        start_next_absolute_tick=start_tick,
        end_next_absolute_tick=start_tick + 1,
        world_uids=(11,),
    )


def _agent(
    agent_id: int = 42,
    *,
    tick: int = 0,
    projection: dict[str, tuple[int, ...]] | None = None,
    alive: bool = True,
    position: int = 1,
    parent_a_id: int = 0,
    parent_b_id: int = 0,
    slot: int = 0,
    absolute_birth_tick: int = 0,
) -> dict[str, object]:
    if projection is None:
        projection = {"policy": (0,) * 224, "trait": (0,) * 32}
    genome_id = capture_adapter._genome_state_id(projection)
    return {
        "agent_state_id": f"state-{agent_id}-{tick}",
        "agent_id": agent_id,
        "alive": alive,
        "position": position,
        "energy": 10,
        "health": 10,
        "injury": 0,
        "age_ticks": 1,
        "genome_state_id": genome_id,
        "policy_id": f"policy-{agent_id}",
        "founder_id": f"founder-{agent_id}",
        "lineage_id": f"lineage-{agent_id}",
        "lineage_depth": 0,
        "hereditary_package_state_ids": [],
        "linkage_state_id": genome_id,
        "parent_a_id": parent_a_id,
        "parent_b_id": parent_b_id,
        "slot": slot,
        "absolute_birth_tick": absolute_birth_tick,
        "_native_genome_projection": projection,
    }


def _output(
    monkeypatch,
    requested: list[str] | None,
    *,
    include_requested_field: bool = True,
    snapshot_ticks: list[int] | None = None,
    dispatch_start_tick: int = 0,
    capture: SimpleNamespace | None = None,
    packed_events: bytes = b"",
    agent_decoder=None,
    applied_native_control: dict[str, object] | None = None,
    emit_control_binding: bool = True,
    state_before: bytes | None = None,
    state_after: bytes | None = None,
) -> dict[str, object]:
    if agent_decoder is None:

        def agent_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
            available = {42: _agent(42, tick=tick)}
            if selected_agent_ids is None:
                return available
            return {
                agent_id: row
                for agent_id, row in available.items()
                if agent_id in selected_agent_ids
            }

    monkeypatch.setattr(
        capture_adapter,
        "_agent_rows",
        agent_decoder,
    )
    monkeypatch.setattr(
        capture_adapter,
        "_state_summary",
        lambda raw, *, world_uid, tick: {
            "world_state_id": f"world-state-{tick}",
            "field_set_version": "test",
            "qualified_field_values": {},
            "snapshot_reason": "test",
        },
    )
    dispatch = SimpleNamespace(
        packed_events=packed_events,
        world_offsets=(0, len(packed_events) // 32),
        event_counts=(len(packed_events) // 32,),
        world_uids=(11,),
        start_next_absolute_tick=dispatch_start_tick,
        end_next_absolute_tick=dispatch_start_tick + 1,
        tick_count=1,
    )
    cell = {
        "seeds": {"task": 11},
        "science_semantic_inputs": {},
        "science_acquisition": (
            {} if snapshot_ticks is None else {"snapshot_ticks": snapshot_ticks}
        ),
    }
    if include_requested_field:
        cell["science_requested_observations"] = requested
    return build_native_cuda_v2_observation_outputs_v1(
        dispatch=dispatch,
        capture=_capture() if capture is None else capture,
        state_before=(
            bytes(capture_adapter._STATE_BYTES)
            if state_before is None
            else state_before
        ),
        state_after=(
            bytes(capture_adapter._STATE_BYTES)
            if state_after is None
            else state_after
        ),
        world_cells={"world-a": cell},
        applied_native_control=applied_native_control,
        emit_control_binding=emit_control_binding,
    )["world-a"]


def _native_event(
    *,
    agent_id: int,
    parent_a: int = 0,
    parent_b: int = 0,
    event_kind: int,
    tick_offset: int = 0,
    slot: int = 0,
    site: int = 0,
) -> bytes:
    packed = tick_offset | (event_kind << 6) | (slot << 8) | (site << 17)
    return struct.pack("<QQQQ", agent_id, parent_a, parent_b, packed)


def test_per_world_observation_contract_filters_native_rows(monkeypatch) -> None:
    action_output = _output(monkeypatch, ["decision", "ecological_consequence"])
    action_kinds = {
        row["record_kind"]
        for row in (
            *action_output["event_dispatch"],
            *action_output["state_projection"],
        )
    }
    assert action_kinds == {
        "control_binding",
        "world_state_snapshot",
        "decision_action",
        "ecological_transaction",
    }

    genome_output = _output(monkeypatch, ["genome_identity"])
    genome_kinds = {
        row["record_kind"]
        for row in (
            *genome_output["event_dispatch"],
            *genome_output["state_projection"],
        )
    }
    assert genome_kinds == {
        "control_binding",
        "world_state_snapshot",
        "agent_state_snapshot",
    }

    minimal_output = _output(monkeypatch, [])
    minimal_kinds = {
        row["record_kind"]
        for row in (
            *minimal_output["event_dispatch"],
            *minimal_output["state_projection"],
        )
    }
    assert minimal_kinds == {"control_binding", "world_state_snapshot"}


def test_provider_stream_emits_one_stable_control_binding_across_dispatches(
    monkeypatch,
) -> None:
    native_control = ScienceCudaV2NativeControlV1.from_values(
        mutation_enabled=True
    ).to_dict()
    cell = {
        "cell_id": "world-a",
        "seeds": {"task": 11},
        "science_semantic_inputs": {"control_bindings": []},
        "science_acquisition": {"snapshot_ticks": []},
        "science_requested_observations": [],
        "science_reducers": [],
        "science_semantic_world_key": {"world_id": "world-a"},
        "science_logical_world_id": 11,
        "science_acquisition_sha256": {"algorithm": "sha256", "hex": "3" * 64},
        "science_instrument_manifest_sha256": {
            "algorithm": "sha256",
            "hex": "1" * 64,
        },
        "science_crca_semantic_run_digest": {
            "algorithm": "sha256",
            "hex": "2" * 64,
        },
    }
    records = {"world-a": []}
    first = _output(
        monkeypatch,
        [],
        dispatch_start_tick=0,
        capture=_capture(start_tick=0),
        applied_native_control=native_control,
        state_after=bytes(capture_adapter._STATE_BYTES),
    )
    cuda_provider._science_append_output_v1(
        outputs={"world-a": first},
        records_by_cell=records,
        cells={"world-a": cell},
    )
    second_state = bytearray(capture_adapter._STATE_BYTES)
    second_state[-1] = 1
    second = _output(
        monkeypatch,
        [],
        dispatch_start_tick=1,
        capture=_capture(start_tick=1),
        applied_native_control=native_control,
        emit_control_binding=False,
        state_after=bytes(second_state),
    )
    cuda_provider._science_append_output_v1(
        outputs={"world-a": second},
        records_by_cell=records,
        cells={"world-a": cell},
    )

    rows = records["world-a"]
    controls = [
        row for row in rows if row.record_kind.value == "control_binding"
    ]
    assert len(controls) == 1
    assert {row.control_binding_sha256 for row in rows} == {
        controls[0].control_binding_sha256
    }
    assert cuda_provider._science_control_binding_digest_v1(
        rows,
        expected_native_control=native_control,
    ) == controls[0].control_binding_sha256.to_dict()


def test_agent_state_snapshots_honor_absolute_tick_schedule(monkeypatch) -> None:
    output = _output(
        monkeypatch,
        ["genome_identity"],
        snapshot_ticks=[1],
    )
    agent_rows = [
        row
        for row in output["state_projection"]
        if row["record_kind"] == "agent_state_snapshot"
    ]
    assert len(agent_rows) == 1
    assert agent_rows[0]["absolute_tick"] == 1


def test_capture_rejects_dispatch_start_tick_mismatch(monkeypatch) -> None:
    try:
        _output(monkeypatch, ["genome_identity"], dispatch_start_tick=7)
    except ValueError as exc:
        assert str(exc) == "science capture and sparse dispatch start ticks differ"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("dispatch/capture start mismatch was accepted")


def test_explicit_empty_contract_skips_agent_decoding(monkeypatch) -> None:
    def unexpected_decode(*args, **kwargs):
        raise AssertionError("empty acquisition contract decoded agent state")

    output = _output(monkeypatch, [], agent_decoder=unexpected_decode)

    assert output["event_dispatch"] == []
    assert {row["record_kind"] for row in output["state_projection"]} == {
        "control_binding",
        "world_state_snapshot",
    }


def test_sparse_contract_selects_only_causal_agent_ids(monkeypatch) -> None:
    selections: list[frozenset[int] | None] = []
    available_ids = {42, 43, 44, 45, 99}

    def selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        selected = None if selected_agent_ids is None else frozenset(selected_agent_ids)
        selections.append(selected)
        ids = (
            available_ids if selected is None else available_ids.intersection(selected)
        )
        return {agent_id: _agent(agent_id, tick=tick) for agent_id in ids}

    events = b"".join(
        (
            _native_event(agent_id=45, event_kind=1),
            _native_event(
                agent_id=44,
                parent_a=42,
                parent_b=43,
                event_kind=2,
                site=7,
            ),
        )
    )
    native_control, state_before, state_after = _strict_zero_native_setup(monkeypatch)

    def strict_selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        selected = None if selected_agent_ids is None else frozenset(selected_agent_ids)
        selections.append(selected)
        return _strict_birth_decoder(
            raw,
            world_uid=world_uid,
            tick=tick,
            selected_agent_ids=selected_agent_ids,
        )

    output = _output(
        monkeypatch,
        [
            "decision",
            "ecological_consequence",
            "death",
            "reproductive_funnel",
            "birth_parentage",
            "genome_identity",
            "mutation_provenance",
        ],
        snapshot_ticks=[],
        capture=_capture(birth=(44, 42, 43, 0, 7)),
        packed_events=events,
        agent_decoder=strict_selected_decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )

    assert selections == [frozenset({42, 43, 44, 45}), frozenset({42, 44})]
    birth_row = next(
        row
        for row in output["event_dispatch"]
        if row["record_kind"] == "birth_committed"
    )
    birth_payload = birth_row["payload"]["payload"]
    assert (
        birth_payload["child_pre_mutation_genome_state_id"]
        == birth_payload["child_post_birth_genome_state_id"]
    )


def test_scheduled_agent_snapshot_forces_only_scheduled_full_decode(
    monkeypatch,
) -> None:
    selections: list[frozenset[int] | None] = []

    def selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        selections.append(
            None if selected_agent_ids is None else frozenset(selected_agent_ids)
        )
        return {42: _agent(42, tick=tick)}

    _output(
        monkeypatch,
        ["genome_identity"],
        snapshot_ticks=[1],
        agent_decoder=selected_decoder,
    )

    assert selections == [None]


def test_missing_and_null_contract_retain_full_compatibility_decode(
    monkeypatch,
) -> None:
    selections: list[frozenset[int] | None] = []

    def selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        selections.append(
            None if selected_agent_ids is None else frozenset(selected_agent_ids)
        )
        return {42: _agent(42, tick=tick)}

    _output(
        monkeypatch,
        None,
        include_requested_field=False,
        agent_decoder=selected_decoder,
    )
    assert selections == [None, None]

    selections.clear()
    _output(monkeypatch, None, agent_decoder=selected_decoder)
    assert selections == [None, None]


def test_causal_lifecycle_wildcard_selects_action_subjects(monkeypatch) -> None:
    selections: list[frozenset[int] | None] = []

    def selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        selected = None if selected_agent_ids is None else frozenset(selected_agent_ids)
        selections.append(selected)
        return {42: _agent(42, tick=tick)}

    output = _output(
        monkeypatch,
        ["causal_lifecycle"],
        snapshot_ticks=[],
        agent_decoder=selected_decoder,
    )

    assert selections == [frozenset({42}), frozenset({42})]
    assert {row["record_kind"] for row in output["event_dispatch"]} == {
        "decision_action",
        "ecological_transaction",
    }


def test_zero_mutation_count_reconstructs_birth_and_emits_no_mutation_row(
    monkeypatch,
) -> None:
    native_control, state_before, state_after = _strict_zero_native_setup(monkeypatch)
    output = _output(
        monkeypatch,
        ["birth_parentage", "mutation_provenance"],
        snapshot_ticks=[],
        capture=_capture(birth=(44, 42, 43, 0, 7)),
        packed_events=_native_event(
            agent_id=44,
            parent_a=42,
            parent_b=43,
            event_kind=2,
            site=7,
        ),
        agent_decoder=_strict_birth_decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )

    assert not any(
        row["record_kind"] == "mutation_outcome"
        for row in output["event_dispatch"]
    )
    birth = next(
        row
        for row in output["event_dispatch"]
        if row["record_kind"] == "birth_committed"
    )
    payload = birth["payload"]["payload"]
    assert payload["child_pre_mutation_genome_state_id"] == payload[
        "child_post_birth_genome_state_id"
    ]


def test_positive_native_mutation_count_requires_applied_native_control(
    monkeypatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="native birth provenance requires|native .* provenance unavailable",
    ):
        _output(
            monkeypatch,
            ["mutation_provenance"],
            snapshot_ticks=[],
            capture=_capture(birth=(44, 42, 43, 1, 7)),
            packed_events=_native_event(
                agent_id=44,
                parent_a=42,
                parent_b=43,
                event_kind=2,
                site=7,
            ),
        )


def _native_header_state(
    *,
    world_uid: int = 11,
    rng_key: int = 0x1234,
    next_tick: int = 0,
    birth_capability_count: int = 0,
    mutation_capability_count: int = 0,
    recombination_capability_count: int = 0,
    birth_count: int = 0,
    mutation_count: int = 0,
    recombination_count: int = 0,
) -> bytes:
    raw = bytearray(capture_adapter._STATE_BYTES)
    header = capture_adapter._field_offset("scalar_header_counters")
    struct.pack_into("<Q", raw, header + 8 * 0, world_uid)
    struct.pack_into("<Q", raw, header + 8 * 1, next_tick)
    struct.pack_into("<Q", raw, header + 8 * 10, birth_capability_count)
    struct.pack_into("<Q", raw, header + 8 * 11, mutation_capability_count)
    struct.pack_into("<Q", raw, header + 8 * 12, recombination_capability_count)
    struct.pack_into("<Q", raw, header + 8 * 13, rng_key)
    struct.pack_into("<Q", raw, header + 8 * 14, 3)
    struct.pack_into("<Q", raw, header + 8 * 15, birth_count)
    struct.pack_into("<Q", raw, header + 8 * 17, mutation_count)
    struct.pack_into("<Q", raw, header + 8 * 18, recombination_count)
    return bytes(raw)


def _strict_zero_native_setup(monkeypatch):
    """Return a one-birth native vector with zero accepted mutation draws."""

    monkeypatch.setattr(
        capture_adapter,
        "_native_child_recombination_bits",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        capture_adapter,
        "_native_child_mutation_draws",
        lambda *args, **kwargs: (0x00000800, 0x00000800),
    )
    native_control = ScienceCudaV2NativeControlV1.from_values(
        mutation_enabled=True
    ).to_dict()
    return (
        native_control,
        _native_header_state(next_tick=0),
        _native_header_state(
            next_tick=1,
            birth_capability_count=1,
            recombination_capability_count=16,
            birth_count=1,
            recombination_count=16,
        ),
    )


def _strict_birth_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
    zero = {"policy": (0,) * 224, "trait": (0,) * 32}
    available = (
        {
            42: _agent(42, tick=tick, projection=zero),
            43: _agent(43, tick=tick, projection=zero),
            45: _agent(45, tick=tick, projection=zero),
        }
        if tick == 0
        else {
            42: _agent(42, tick=tick, projection=zero),
            43: _agent(43, tick=tick, projection=zero),
            44: _agent(
                44,
                tick=tick,
                projection=zero,
                parent_a_id=42,
                parent_b_id=43,
                position=7,
                slot=0,
                absolute_birth_tick=0,
            ),
            45: _agent(45, tick=tick, projection=zero),
        }
    )
    if selected_agent_ids is None:
        return available
    return {
        agent_id: row
        for agent_id, row in available.items()
        if agent_id in selected_agent_ids
    }


def test_positive_native_mutation_reconstructs_exact_child_delta(monkeypatch) -> None:
    zero = {"policy": (0,) * 224, "trait": (0,) * 32}
    changed = {"policy": (1,) + (0,) * 223, "trait": (0,) * 32}

    # Make this CPU contract vector deliberately small and literal: all
    # recombination blocks select zero-valued parent alleles, and only policy
    # haplotype 0/locus 0 receives one accepted +1 mutation draw.
    monkeypatch.setattr(
        capture_adapter,
        "_native_child_recombination_bits",
        lambda *args, **kwargs: 0,
    )

    def mutation_draws(tick_key, *, child_id, linked_locus):
        return (
            0x00010001 if linked_locus == 0 else 0x00000800,
            0x00000800,
        )

    monkeypatch.setattr(
        capture_adapter, "_native_child_mutation_draws", mutation_draws
    )

    def exact_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        available = (
            {
                42: _agent(42, tick=tick, projection=zero),
                43: _agent(43, tick=tick, projection=zero),
            }
            if tick == 0
            else {
                42: _agent(42, tick=tick, projection=zero),
                43: _agent(43, tick=tick, projection=zero),
                44: _agent(
                    44,
                    tick=tick,
                    projection=changed,
                    parent_a_id=42,
                    parent_b_id=43,
                    position=7,
                    slot=0,
                    absolute_birth_tick=0,
                ),
            }
        )
        if selected_agent_ids is None:
            return available
        return {
            agent_id: row
            for agent_id, row in available.items()
            if agent_id in selected_agent_ids
        }

    native_control = ScienceCudaV2NativeControlV1.from_values(
        mutation_enabled=True
    ).to_dict()
    output = _output(
        monkeypatch,
        ["birth_parentage", "mutation_provenance"],
        snapshot_ticks=[],
        capture=_capture(birth=(44, 42, 43, 1, 7)),
        packed_events=_native_event(
            agent_id=44,
            parent_a=42,
            parent_b=43,
            event_kind=2,
            site=7,
        ),
        agent_decoder=exact_decoder,
        applied_native_control=native_control,
        state_before=_native_header_state(next_tick=0),
        state_after=_native_header_state(
            next_tick=1,
            birth_capability_count=1,
            mutation_capability_count=1,
            recombination_capability_count=16,
            birth_count=1,
            mutation_count=1,
            recombination_count=16,
        ),
    )

    birth = next(
        row for row in output["event_dispatch"] if row["record_kind"] == "birth_committed"
    )
    birth_payload = birth["payload"]["payload"]
    mutation = next(
        row for row in output["event_dispatch"] if row["record_kind"] == "mutation_outcome"
    )
    mutation_payload = mutation["payload"]["payload"]
    assert mutation_payload["origin_agent_id"] == 44
    assert mutation_payload["mutation_class_id"] == "native_genome_v2_point_mutation"
    assert mutation_payload["reason_code"] == "native_exact_genome_delta"
    assert mutation_payload["pre_genome_state_id"] == birth_payload[
        "child_pre_mutation_genome_state_id"
    ]
    assert mutation_payload["post_genome_state_id"] == birth_payload[
        "child_post_birth_genome_state_id"
    ]
    assert mutation_payload["bounded_delta"] == {
        "mutation_count": 1,
        "accepted_draws": 1,
        "changed_allele_count": 1,
        "saturated_noop_count": 0,
    }
    assert mutation_payload["pre_value_or_state_digest"] == capture_adapter._canonical_digest(
        {"genome_state_id": mutation_payload["pre_genome_state_id"]}
    )
    assert mutation_payload["post_value_or_state_digest"] == capture_adapter._canonical_digest(
        {"genome_state_id": mutation_payload["post_genome_state_id"]}
    )
    assert mutation_payload["affected_coordinates"]["changed_alleles"] == [
        {
            "genome_region": 0,
            "haplotype": 0,
            "linked_locus": 0,
            "pre_value": 0,
            "post_value": 1,
        }
    ]
    assert mutation_payload["affected_coordinates"]["accepted_draw_details"] == [
        {
            "genome_region": 0,
            "haplotype": 0,
            "linked_locus": 0,
            "pre_value": 0,
            "post_value": 1,
            "requested_delta": 1,
            "effective_delta": 1,
            "saturated_noop": False,
        }
    ]


def test_native_philox_reconstruction_matches_shared_rngv2_oracle() -> None:
    vectors = (
        ((0, 0, 0, 0), (0, 0)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)),
        ((0xFFFFFFFF, 1, 0xABCDEF01, 0x10203040), (0x01020304, 0xA0B0C0D0)),
    )
    for counter, key in vectors:
        assert capture_adapter._native_philox4x32_10(counter, key) == philox4x32_10(
            counter, key
        )


def test_native_rng_recombination_and_mutation_literal_vectors() -> None:
    world_uid = 0x0102030405060708
    absolute_tick = 37
    raw = _native_header_state(
        world_uid=world_uid,
        rng_key=0x1122334455667788,
        next_tick=absolute_tick,
    )
    tick_key = capture_adapter._native_rng_tick_key(
        raw, world_uid=world_uid, absolute_tick=absolute_tick
    )
    assert tick_key == (0x39F49C17, 0x34192F17)
    assert capture_adapter._native_child_recombination_bits(
        tick_key,
        child_id=44,
        parent_a_id=42,
        parent_b_id=43,
        linked_block=0,
    ) == 0xAA2B14CB
    assert capture_adapter._native_child_recombination_bits(
        tick_key,
        child_id=44,
        parent_a_id=42,
        parent_b_id=43,
        linked_block=15,
    ) == 0x7395B2D1
    assert capture_adapter._native_child_mutation_draws(
        tick_key, child_id=44, linked_locus=0
    ) == (0xAED6A3AC, 0xAD09B368)
    assert capture_adapter._native_child_mutation_draws(
        tick_key, child_id=44, linked_locus=112
    ) == (0x3295362C, 0x1626C09F)
    assert capture_adapter._clamp_i8(-128) == -127
    assert capture_adapter._clamp_i8(128) == 127
    assert capture_adapter._clamp_i16(-32768) == -32767
    assert capture_adapter._clamp_i16(32768) == 32767


def test_positive_native_mutation_emits_explicit_saturated_noop(monkeypatch) -> None:
    saturated = {"policy": (-127,) * 224, "trait": (-32767,) * 32}
    monkeypatch.setattr(
        capture_adapter,
        "_native_child_recombination_bits",
        lambda *args, **kwargs: 0,
    )
    # Low 16 bits are below the native threshold and bit 16 requests -1;
    # every allele is already at the native lower mutation boundary.
    monkeypatch.setattr(
        capture_adapter,
        "_native_child_mutation_draws",
        lambda *args, **kwargs: (0x00000001, 0x00000001),
    )

    def decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        available = (
            {
                42: _agent(42, tick=tick, projection=saturated),
                43: _agent(43, tick=tick, projection=saturated),
            }
            if tick == 0
            else {
                42: _agent(42, tick=tick, projection=saturated),
                43: _agent(43, tick=tick, projection=saturated),
                44: _agent(
                    44,
                    tick=tick,
                    projection=saturated,
                    parent_a_id=42,
                    parent_b_id=43,
                    position=7,
                    slot=0,
                    absolute_birth_tick=0,
                ),
            }
        )
        if selected_agent_ids is None:
            return available
        return {
            agent_id: row
            for agent_id, row in available.items()
            if agent_id in selected_agent_ids
        }

    native_control = ScienceCudaV2NativeControlV1.from_values(
        mutation_enabled=True
    ).to_dict()
    state_before = _native_header_state(next_tick=0)
    state_after = _native_header_state(
        next_tick=1,
        birth_capability_count=1,
        mutation_capability_count=256,
        recombination_capability_count=16,
        birth_count=1,
        mutation_count=256,
        recombination_count=16,
    )
    output = _output(
        monkeypatch,
        ["birth_parentage", "mutation_provenance"],
        snapshot_ticks=[],
        capture=_capture(birth=(44, 42, 43, 256, 7)),
        packed_events=_native_event(
            agent_id=44,
            parent_a=42,
            parent_b=43,
            event_kind=2,
            site=7,
        ),
        agent_decoder=decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )
    mutation = next(
        row
        for row in output["event_dispatch"]
        if row["record_kind"] == "mutation_outcome"
    )
    payload = mutation["payload"]["payload"]
    coordinates = payload["affected_coordinates"]
    assert payload["result_class"] == "MUTATION_SATURATED_NOOP"
    assert coordinates["accepted_draws"] == 256
    assert coordinates["native_mutation_count"] == 256
    assert coordinates["changed_alleles"] == []
    assert coordinates["saturated_noop_count"] == 256
    assert len(coordinates["accepted_draw_details"]) == 256
    assert all(
        detail["requested_delta"] == -1
        and detail["effective_delta"] == 0
        and detail["saturated_noop"] is True
        for detail in coordinates["accepted_draw_details"]
    )


def test_native_birth_and_mutation_header_or_event_tamper_fails_closed(monkeypatch) -> None:
    native_control, state_before, state_after = _strict_zero_native_setup(monkeypatch)
    event = _native_event(
        agent_id=44,
        parent_a=42,
        parent_b=43,
        event_kind=2,
        site=7,
    )
    common = {
        "requested": ["birth_parentage", "mutation_provenance"],
        "snapshot_ticks": [],
        "capture": _capture(birth=(44, 42, 43, 0, 7)),
        "packed_events": event,
        "agent_decoder": _strict_birth_decoder,
        "applied_native_control": native_control,
        "state_before": state_before,
        "state_after": state_after,
    }
    bad_birth_header = bytearray(state_after)
    header = capture_adapter._field_offset("scalar_header_counters")
    struct.pack_into("<Q", bad_birth_header, header + 8 * 15, 2)
    with pytest.raises(ValueError, match="birth/recombination counter delta"):
        _output(monkeypatch, **{**common, "state_after": bytes(bad_birth_header)})

    bad_frontier = bytearray(state_after)
    struct.pack_into("<Q", bad_frontier, header + 8, 2)
    with pytest.raises(ValueError, match="next absolute tick"):
        _output(monkeypatch, **{**common, "state_after": bytes(bad_frontier)})

    bad_event = _native_event(
        agent_id=44,
        parent_a=42,
        parent_b=43,
        event_kind=2,
        site=8,
    )
    with pytest.raises(ValueError, match="exactly one sparse birth event"):
        _output(monkeypatch, **{**common, "packed_events": bad_event})

    bad_mutation_count = _capture(birth=(44, 42, 43, 0, 7))
    bad_mutation_count.birth_mutation_counts._value[0][0][0] = 1
    with pytest.raises(ValueError, match="native mutation counter delta"):
        _output(
            monkeypatch,
            **{**common, "capture": bad_mutation_count},
        )


def test_native_state_duplicate_agent_ids_fail_even_when_not_selected() -> None:
    raw = bytearray(capture_adapter._STATE_BYTES)
    offset = capture_adapter._field_offset("agent_id")
    struct.pack_into("<Q", raw, offset, 42)
    struct.pack_into("<Q", raw, offset + 8, 42)
    with pytest.raises(ValueError, match="duplicate agent identity"):
        capture_adapter._agent_rows(
            bytes(raw), world_uid=11, tick=0, selected_agent_ids={999}
        )


def test_multi_world_materialization_and_agent_selection_are_isolated(
    monkeypatch,
) -> None:
    raw = bytearray(capture_adapter._STATE_BYTES * 2)
    for world_index, agent_id in enumerate((11, 22)):
        base = world_index * capture_adapter._STATE_BYTES
        struct.pack_into(
            "<Q",
            raw,
            base + capture_adapter._field_offset("agent_id"),
            agent_id,
        )
    zeros = [0] * 512
    capture = SimpleNamespace(
        action_agent_ids=_TensorList([[[11, *zeros[1:]]], [[22, *zeros[1:]]]]),
        action_values=_TensorList([[[1, *zeros[1:]]], [[2, *zeros[1:]]]]),
        birth_child_ids=_TensorList([[zeros], [zeros]]),
        birth_parent_a_ids=_TensorList([[zeros], [zeros]]),
        birth_parent_b_ids=_TensorList([[zeros], [zeros]]),
        birth_mutation_counts=_TensorList([[zeros], [zeros]]),
        birth_child_slots=_TensorList([[zeros], [zeros]]),
        birth_sites=_TensorList([[zeros], [zeros]]),
        birth_counts=_TensorList([[0], [0]]),
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        world_uids=(101, 102),
    )
    dispatch = SimpleNamespace(
        packed_events=b"",
        world_offsets=(0, 0, 0),
        event_counts=(0, 0),
        world_uids=(101, 102),
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        tick_count=1,
    )
    materializations: list[str] = []
    original_tensor_bytes = capture_adapter._tensor_bytes

    def tracked_tensor_bytes(value, *, field):
        materializations.append(field)
        return original_tensor_bytes(value, field=field)

    monkeypatch.setattr(capture_adapter, "_tensor_bytes", tracked_tensor_bytes)
    outputs = build_native_cuda_v2_observation_outputs_v1(
        dispatch=dispatch,
        capture=capture,
        state_before=bytes(raw),
        state_after=bytes(raw),
        world_cells={
            "minimal-world": {
                "seeds": {"task": 101},
                "science_requested_observations": [],
                "science_semantic_inputs": {},
                "science_acquisition": {"snapshot_ticks": []},
            },
            "decision-world": {
                "seeds": {"task": 102},
                "science_requested_observations": ["decision"],
                "science_semantic_inputs": {},
                "science_acquisition": {"snapshot_ticks": []},
            },
        },
    )

    assert materializations == [
        "state_before_projection",
        "state_after_projection",
    ]
    assert outputs["minimal-world"]["event_dispatch"] == []
    decision_rows = outputs["decision-world"]["event_dispatch"]
    assert len(decision_rows) == 1
    assert decision_rows[0]["payload"]["payload"]["agent_id"] == 22


def test_sparse_selection_preserves_eager_observation_output(monkeypatch) -> None:
    available_ids = {42, 43, 44, 45, 99}

    def selected_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        ids = (
            available_ids
            if selected_agent_ids is None
            else available_ids.intersection(selected_agent_ids)
        )
        return {agent_id: _agent(agent_id, tick=tick) for agent_id in ids}

    events = b"".join(
        (
            _native_event(agent_id=45, event_kind=1),
            _native_event(
                agent_id=44,
                parent_a=42,
                parent_b=43,
                event_kind=2,
                site=7,
            ),
        )
    )
    requested = [
        "decision",
        "ecological_consequence",
        "death",
        "reproductive_funnel",
        "birth_parentage",
        "mutation_provenance",
    ]
    capture = _capture(birth=(44, 42, 43, 0, 7))
    native_control, state_before, state_after = _strict_zero_native_setup(monkeypatch)
    selected_output = _output(
        monkeypatch,
        requested,
        snapshot_ticks=[],
        capture=capture,
        packed_events=events,
        agent_decoder=_strict_birth_decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )

    def eager_decoder(raw, *, world_uid, tick, selected_agent_ids=None):
        return _strict_birth_decoder(raw, world_uid=world_uid, tick=tick)

    eager_output = _output(
        monkeypatch,
        requested,
        snapshot_ticks=[],
        capture=capture,
        packed_events=events,
        agent_decoder=eager_decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )

    assert selected_output == eager_output


def test_selected_agent_decode_matches_full_native_row() -> None:
    raw = bytearray(capture_adapter._STATE_BYTES)
    agent_id = 42
    slot = 7
    struct.pack_into(
        "<Q",
        raw,
        capture_adapter._field_offset("agent_id") + 8 * slot,
        agent_id,
    )
    struct.pack_into(
        "<Q",
        raw,
        capture_adapter._field_offset("parent_a_id") + 8 * slot,
        40,
    )
    struct.pack_into(
        "<Q",
        raw,
        capture_adapter._field_offset("parent_b_id") + 8 * slot,
        41,
    )
    struct.pack_into(
        "<H",
        raw,
        capture_adapter._field_offset("position") + 2 * slot,
        17,
    )
    struct.pack_into(
        "<h",
        raw,
        capture_adapter._field_offset("energy") + 2 * slot,
        9,
    )
    struct.pack_into(
        "<h",
        raw,
        capture_adapter._field_offset("health") + 2 * slot,
        8,
    )
    struct.pack_into(
        "<H",
        raw,
        capture_adapter._field_offset("age") + 2 * slot,
        6,
    )

    full = capture_adapter._agent_rows(bytes(raw), world_uid=11, tick=3)
    selected = capture_adapter._agent_rows(
        bytes(raw),
        world_uid=11,
        tick=3,
        selected_agent_ids={agent_id},
    )

    assert selected == {agent_id: full[agent_id]}
    assert selected[agent_id]["alive"] is False
    assert (
        capture_adapter._agent_rows(
            bytes(raw),
            world_uid=11,
            tick=3,
            selected_agent_ids={999},
        )
        == {}
    )


def test_capture_phase_identity_round_trips_through_canonical_stream_validation(
    monkeypatch,
) -> None:
    native_control, state_before, state_after = _strict_zero_native_setup(monkeypatch)

    output = _output(
        monkeypatch,
        [
            "decision",
            "ecological_consequence",
            "death",
            "reproductive_funnel",
            "birth_parentage",
            "genome_identity",
            "mutation_provenance",
        ],
        snapshot_ticks=[0, 1],
        capture=_capture(birth=(44, 42, 43, 0, 7)),
        packed_events=b"".join(
            (
                _native_event(agent_id=45, event_kind=1),
                _native_event(
                    agent_id=44,
                    parent_a=42,
                    parent_b=43,
                    event_kind=2,
                    site=7,
                ),
            )
        ),
        agent_decoder=_strict_birth_decoder,
        applied_native_control=native_control,
        state_before=state_before,
        state_after=state_after,
    )
    digest = ContentDigest("sha256", "1" * 64)
    rows = finalize_native_cuda_v2_observations_v1(
        output,
        context=NativeCudaV2ObservationContextV1(
            instrument_manifest_sha256=digest,
            crca_semantic_run_digest=ContentDigest("sha256", "2" * 64),
            science_semantic_acquisition_sha256=ContentDigest(
                "sha256", "3" * 64
            ),
            cell_id="world-a",
            logical_world_id=11,
            semantic_world_key=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_semantic_world_key_v1"),
                {"world_id": "world-a"},
            ),
        ),
        start_world_local_sequence=0,
    )

    validate_science_observation_stream_v1(
        rows,
        producer_phase_registry_ordinals=(
            science_capture_producer_phase_registry_v1()
        ),
    )
    registry = science_capture_producer_phase_registry_v1()
    agent_rows = [
        row
        for row in rows
        if row.record_kind.value == "agent_state_snapshot"
    ]
    assert agent_rows
    assert all(
        row.payload.schema_version
        == SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION
        for row in agent_rows
    )
    assert all("absolute_birth_tick" in row.payload.to_value() for row in agent_rows)
    assert len({row.producer_component_id for row in rows}) > 1
    assert set(row.producer_component_id for row in rows).issubset(registry)


def _provider_sidecar_fixture(monkeypatch, tmp_path):
    native_control = ScienceCudaV2NativeControlV1.from_values(
        mutation_enabled=True
    ).to_dict()
    context = NativeCudaV2ObservationContextV1(
        instrument_manifest_sha256=ContentDigest("sha256", "1" * 64),
        crca_semantic_run_digest=ContentDigest("sha256", "2" * 64),
        science_semantic_acquisition_sha256=ContentDigest("sha256", "3" * 64),
        cell_id="world-a",
        logical_world_id=11,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_world_key_v1"),
            {"world_id": "world-a"},
        ),
    )
    initial = finalize_native_cuda_v2_observations_v1(
        _output(
            monkeypatch,
            [],
            snapshot_ticks=[],
            applied_native_control=native_control,
        ),
        context=context,
        start_world_local_sequence=0,
    )
    control_digest = initial[0].control_binding_sha256.to_dict()
    marker = finalize_native_cuda_v2_observations_v1(
        build_native_cuda_v2_checkpoint_marker_output_v1(
            cell_id="world-a",
            absolute_tick=1,
            checkpoint_occurrence_id="checkpoint-000000000001",
            authoritative_state_sha256={"algorithm": "sha256", "hex": "4" * 64},
            observation_prefix_sha256=(
                canonical_science_record_stream_sha256_v1(initial).to_dict()
            ),
            next_world_local_sequence=len(initial),
            continuation_parent_identity=None,
            control_binding_sha256=control_digest,
        ),
        context=context,
        start_world_local_sequence=len(initial),
    )
    prefix = (*initial, *marker)
    completion = finalize_native_cuda_v2_observations_v1(
        build_native_cuda_v2_completion_output_v1(
            cell_id="world-a",
            absolute_tick=1,
            completion_class="TERMINAL_HORIZON",
            terminal_state_sha256={"algorithm": "sha256", "hex": "4" * 64},
            terminal_population_count=1,
            terminal_record_count=len(prefix) + 1,
            canonical_record_stream_sha256=(
                canonical_science_record_stream_sha256_v1(prefix).to_dict()
            ),
            checkpoint_occurrence_ids=["checkpoint-000000000001"],
            control_binding_sha256=control_digest,
        ),
        context=context,
        start_world_local_sequence=len(prefix),
    )
    rows = (*prefix, *completion)
    frontier = cuda_provider._science_frontier_v1(
        {"world-a": rows}, next_absolute_tick=1
    )
    body = {
        "schema_version": cuda_provider.SCIENCE_OBSERVATION_SIDECAR_SCHEMA_V1,
        "observation_abi": cuda_provider.SCIENCE_OBSERVATION_ABI_V1,
        "observation_mode": cuda_provider.SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
        "batch_id": "batch-a",
        "cell_ids": ["world-a"],
        "records_by_cell": {"world-a": [row.to_dict() for row in rows]},
        "frontier": frontier,
    }
    path = tmp_path / "science-observations.json"
    payload = canonical_json_bytes(body)
    path.write_bytes(payload)
    checkpoint_frontier = {
        **frontier,
        "records_path": str(path),
        "records_sha256": hashlib.sha256(payload).hexdigest(),
        "terminal": False,
    }
    request = SimpleNamespace(
        batch_id="batch-a",
        cell_ids=("world-a",),
        native_control=native_control,
    )
    return path, rows, checkpoint_frontier, request


def test_provider_continuation_loader_validates_canonical_stream_and_control_join(
    monkeypatch, tmp_path
) -> None:
    path, rows, frontier, request = _provider_sidecar_fixture(
        monkeypatch, tmp_path
    )
    loaded = cuda_provider._load_science_sidecar_v1(
        context=SimpleNamespace(run_root=tmp_path),
        request=request,
        frontier=frontier,
    )

    assert tuple(loaded["world-a"]) == rows
    assert {
        row.control_binding_sha256 for row in loaded["world-a"]
    } == {rows[0].control_binding_sha256}
    assert path.read_bytes() == canonical_json_bytes(json.loads(path.read_bytes()))


def test_provider_continuation_loader_rejects_noncanonical_bytes_and_symlinks(
    monkeypatch, tmp_path
) -> None:
    path, _rows, frontier, request = _provider_sidecar_fixture(
        monkeypatch, tmp_path
    )
    body = json.loads(path.read_bytes())
    pretty = tmp_path / "pretty.json"
    pretty_payload = json.dumps(body, indent=2, sort_keys=True).encode("ascii")
    pretty.write_bytes(pretty_payload)
    pretty_frontier = {
        **frontier,
        "records_path": str(pretty),
        "records_sha256": hashlib.sha256(pretty_payload).hexdigest(),
    }
    with pytest.raises(CampaignProviderError, match="bytes are not canonical"):
        cuda_provider._load_science_sidecar_v1(
            context=SimpleNamespace(run_root=tmp_path),
            request=request,
            frontier=pretty_frontier,
        )

    link = tmp_path / "sidecar-link.json"
    link.symlink_to(path)
    linked_frontier = {**frontier, "records_path": str(link)}
    with pytest.raises(CampaignProviderError, match="symbolic link"):
        cuda_provider._load_science_sidecar_v1(
            context=SimpleNamespace(run_root=tmp_path),
            request=request,
            frontier=linked_frontier,
        )
