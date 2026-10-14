from __future__ import annotations

import pytest

from engine.tensor_world.c4_state import make_c4_cpython_random_state_v1
from engine.tensor_world.c5_rngv2_checkpoint import (
    capture_v2_checkpoint,
    restore_v2_checkpoint,
)
from engine.tensor_world.c5_rngv2_cpu import mutate_planned_births_v2
from engine.tensor_world.c5_rngv2_route import (
    DEFAULT_RNG_SEMANTICS,
    RNG_SEMANTICS_V1,
    RNG_SEMANTICS_V2,
    V1_ROUTE_ID,
    V2_ROUTE_ID,
    assert_v2_no_host_tapes,
    production_contract_hash,
    reject_v1_v2_mix,
    resolve_rng_route,
)
from engine.tensor_world.c6_contract import (
    TensorWorldC6RouteRequestV1,
    resolve_tensor_world_c6_route_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError
from engine.tw_rngv2.status import Status


def _plan() -> tuple:
    genes = (0.1, -0.2, 10.0, 1.0, 0.4, 0.02, 0.5, 0.1)
    return ((7, 3, genes, genes, 0.2, 0.2, 0),)


def test_default_route_remains_v1() -> None:
    resolved = resolve_rng_route()
    assert DEFAULT_RNG_SEMANTICS == RNG_SEMANTICS_V1
    assert resolved["route_id"] == V1_ROUTE_ID
    assert resolved["default"] is True
    request = TensorWorldC6RouteRequestV1(
        batch=2,
        grid_size=32,
        initial_population=8,
        slot_capacity=16,
        policy_family="fixed_mlp",
        mechanics=("population",),
        strict=False,
    )
    assert "rng_semantics_version" not in request.to_dict()
    resolution = resolve_tensor_world_c6_route_v1(request)
    assert resolution.route_id == V1_ROUTE_ID


def test_v2_route_is_explicit_and_default_off() -> None:
    resolved = resolve_rng_route(rng_semantics_version=2)
    assert resolved["route_id"] == V2_ROUTE_ID
    assert resolved["default"] is False
    assert resolved["host_tapes_allowed"] is False
    assert resolved["contract_hash"] == production_contract_hash()
    request = TensorWorldC6RouteRequestV1(
        batch=2,
        grid_size=32,
        initial_population=8,
        slot_capacity=16,
        policy_family="fixed_mlp",
        mechanics=("population",),
        strict=False,
        rng_semantics_version=2,
    )
    assert request.to_dict()["rng_semantics_version"] == 2
    resolution = resolve_tensor_world_c6_route_v1(request)
    assert resolution.route_id == V2_ROUTE_ID


def test_no_v1_v2_mix_and_no_hidden_tapes() -> None:
    empty = make_c4_cpython_random_state_v1(batch=2, tape_capacity=0)
    assert_v2_no_host_tapes(empty)
    reject_v1_v2_mix(rng_semantics_version=RNG_SEMANTICS_V2, tape_capacity=0)
    with pytest.raises(TensorWorldContractError, match="tapes"):
        reject_v1_v2_mix(rng_semantics_version=RNG_SEMANTICS_V2, tape_capacity=2048)
    with pytest.raises(TensorWorldContractError, match="zero-tape"):
        reject_v1_v2_mix(rng_semantics_version=RNG_SEMANTICS_V1, tape_capacity=0)
    tapes = make_c4_cpython_random_state_v1(batch=2, tape_capacity=8)
    with pytest.raises(TensorWorldContractError, match="forbids"):
        assert_v2_no_host_tapes(tapes)


def test_slot_reuse_does_not_change_semantic_addresses() -> None:
    first = mutate_planned_births_v2(world_uid=0, absolute_tick=4, plans=_plan())
    reused_slot_same_ids = mutate_planned_births_v2(
        world_uid=0, absolute_tick=4, plans=_plan()
    )
    assert first[0].child_genes == reused_slot_same_ids[0].child_genes
    genes = _plan()[0][2]
    together = (
        (7, 3, genes, genes, 0.2, 0.2, 0),
        (7, 9, genes, genes, 0.2, 0.2, 0),
    )
    ranked = mutate_planned_births_v2(world_uid=0, absolute_tick=4, plans=together)
    assert ranked[0].child_genes != ranked[1].child_genes
    directed = (
        (2, 5, genes, genes, 0.2, 0.2, 0),
        (5, 2, genes, genes, 0.2, 0.2, 0),
    )
    directed_out = mutate_planned_births_v2(world_uid=0, absolute_tick=4, plans=directed)
    assert directed_out[0].child_genes != directed_out[1].child_genes
    assert first[0].status is Status.RNG_OK


def test_checkpoint_roundtrip_and_late_failure() -> None:
    snap = capture_v2_checkpoint(
        world_uids=(0, 1),
        absolute_tick=5,
        next_entity_uid=(8, 8),
    )
    assert snap.host_tape_bytes == 0
    restored = restore_v2_checkpoint(snap.to_dict())
    assert restored.digest() == snap.digest()
    before = mutate_planned_births_v2(world_uid=0, absolute_tick=5, plans=_plan())
    after = mutate_planned_births_v2(world_uid=0, absolute_tick=5, plans=_plan())
    assert before[0].child_genes == after[0].child_genes
    with pytest.raises(TensorWorldContractError, match="schema"):
        restore_v2_checkpoint({**snap.to_dict(), "schema_version": "v1"})
    with pytest.raises(TensorWorldContractError, match="tape"):
        restore_v2_checkpoint({**snap.to_dict(), "host_tape_bytes": 8})
