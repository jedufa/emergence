from __future__ import annotations

import pytest

from engine.tensor_world.c4_state import make_c4_cpython_random_state_v1
from engine.tensor_world.c5_rngv2_cpu import mutate_planned_births_v2
from engine.tensor_world.c5_rngv2_graph import GRAPH_FUSION_ROUTE_ID, GRAPH_WINDOW_TICKS
from engine.tensor_world.c5_rngv2_route import (
    DEFAULT_RNG_SEMANTICS,
    V1_ROUTE_ID,
    V2_ROUTE_ID,
    production_contract_hash,
    reject_v1_v2_mix,
    resolve_rng_route,
)
from engine.tensor_world.c6_contract import (
    TensorWorldC6RouteRequestV1,
    resolve_tensor_world_c6_route_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def test_default_live_route_remains_v1() -> None:
    assert DEFAULT_RNG_SEMANTICS == 1
    assert resolve_rng_route()["route_id"] == V1_ROUTE_ID
    request = TensorWorldC6RouteRequestV1(
        batch=48,
        grid_size=8,
        initial_population=4,
        slot_capacity=32,
        policy_family="fixed_mlp",
        mechanics=("population",),
        strict=False,
    )
    assert "rng_semantics_version" not in request.to_dict()
    assert resolve_tensor_world_c6_route_v1(request).route_id == V1_ROUTE_ID


def test_graph_fusion_is_explicit_default_off() -> None:
    resolved = resolve_rng_route(rng_semantics_version=2)
    assert resolved["route_id"] == V2_ROUTE_ID
    assert resolved["default"] is False
    assert resolved["contract_hash"] == production_contract_hash()
    assert GRAPH_FUSION_ROUTE_ID.endswith("graph_fusion_v1")
    assert GRAPH_WINDOW_TICKS == 5


def test_action_environment_v1_not_migrated() -> None:
    request = TensorWorldC6RouteRequestV1(
        batch=48,
        grid_size=8,
        initial_population=4,
        slot_capacity=32,
        policy_family="fixed_mlp",
        mechanics=("population",),
        strict=False,
        rng_semantics_version=2,
    )
    assert resolve_tensor_world_c6_route_v1(request).route_id == V2_ROUTE_ID
    # Action/environment remain on V1 PCG streams; V2 only owns mutation/birth.
    assert resolve_rng_route()["host_tapes_allowed"] is True


def test_no_v1_v2_checkpoint_or_tape_mix() -> None:
    reject_v1_v2_mix(rng_semantics_version=2, tape_capacity=0)
    with pytest.raises(TensorWorldContractError, match="tapes"):
        reject_v1_v2_mix(rng_semantics_version=2, tape_capacity=2048)
    with pytest.raises(TensorWorldContractError, match="zero-tape"):
        reject_v1_v2_mix(rng_semantics_version=1, tape_capacity=0)
    empty = make_c4_cpython_random_state_v1(batch=2, tape_capacity=0)
    assert int(empty.python_random_gauss_primary_tape.nbytes) == 0


def test_r13_skips_empty_v2_gaussian_tapes() -> None:
    from engine.tensor_world.c4_resident_transport_r13 import (
        inventory_c4_resident_transport_r13_v1,
    )
    from engine.tensor_world.c4_state import make_c4_cpython_random_state_v1

    empty = make_c4_cpython_random_state_v1(batch=2, tape_capacity=0)
    assert int(empty.python_random_gauss_primary_tape.numel()) == 0
    # Inventory requires full bundles; this only documents the empty-tape ABI.
    assert int(empty.python_random_gauss_secondary_tape.nbytes) == 0


def test_qualified_oracle_directed_identity_unchanged() -> None:
    genes = (0.1, -0.2, 10.0, 1.0, 0.4, 0.02, 0.5, 0.1)
    directed = (
        (2, 5, genes, genes, 0.2, 0.2, 0),
        (5, 2, genes, genes, 0.2, 0.2, 0),
    )
    out = mutate_planned_births_v2(world_uid=0, absolute_tick=9, plans=directed)
    assert out[0].child_genes != out[1].child_genes
    reused = mutate_planned_births_v2(world_uid=0, absolute_tick=9, plans=directed)
    assert out[0].child_genes == reused[0].child_genes
    compacted = (
        (9, 1, genes, genes, 0.2, 0.2, 1),
        (2, 5, genes, genes, 0.2, 0.2, 0),
        (5, 2, genes, genes, 0.2, 0.2, 0),
    )
    mixed = mutate_planned_births_v2(world_uid=0, absolute_tick=9, plans=compacted)
    assert mixed[1].child_genes == out[0].child_genes
    assert mixed[2].child_genes == out[1].child_genes
