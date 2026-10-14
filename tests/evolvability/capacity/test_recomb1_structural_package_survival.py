from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.inheritance_registry_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
    LEGACY_INHERITANCE_PROFILE,
)
from evolution.recombination_c2 import (
    C2_STRUCTURAL_LINKAGE_RULE,
    _linked_component_plan,
    recombine_lineage_payloads,
)
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
)


class SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self._values))


def _identity(name: str, index: int) -> dict[str, Any]:
    return new_native_identity(
        context={
            "campaign_id": "recomb1-structural-package-tests",
            "branch_id": "inheritance",
            "generation": 0,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": f"recomb1-{name}",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=name,
    )


def _development(prerequisites: list[str]) -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": [1.0, 1.0, 1.0],
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": False,
    }


def _module(
    local_id: str,
    *,
    identity: dict[str, Any],
    marker: float,
    prerequisites: list[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": float(marker),
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": 1.0,
    }
    if prerequisites is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(prerequisites)
    return {
        "module_id": local_id,
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **deepcopy(identity),
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(src: str, dst: str, marker: float, *, connection_id: str = "c000") -> dict[str, Any]:
    return {
        "connection_id": connection_id,
        "src_module_id": src,
        "src_slot": "out",
        "dst_module_id": dst,
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "generic",
        "weight": float(marker),
        "enabled": True,
    }


def _payload(
    marker: float,
    *,
    dependencies: tuple[list[str] | None, ...] | None = None,
    identities: tuple[dict[str, Any], ...] | None = None,
    connections: tuple[tuple[int, int], ...] = ((0, 1),),
) -> dict[str, Any]:
    identities = identities or (_identity("shared-0", 0), _identity("shared-1", 1))
    dependencies = dependencies or tuple(None for _ in identities)
    modules = [
        _module(
            f"m{index:03d}",
            identity=identity,
            marker=marker,
            prerequisites=dependencies[index],
        )
        for index, identity in enumerate(identities)
    ]
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": modules,
        "connections": [
            _connection(f"m{src:03d}", f"m{dst:03d}", marker, connection_id=f"c{index:03d}")
            for index, (src, dst) in enumerate(connections)
        ],
    }


def _package_markers(payload: dict[str, Any]) -> list[float]:
    values = [float(module["params"]["expression_scale"]) for module in payload["modules"]]
    values.extend(float(connection["weight"]) for connection in payload["connections"])
    return values


def _repack(payload: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(payload)
    old_ids = [str(module["module_id"]) for module in out["modules"]]
    new_ids = [f"slot-{index}" for index in reversed(range(len(old_ids)))]
    remap = dict(zip(old_ids, new_ids))
    for module in out["modules"]:
        module["module_id"] = remap[str(module["module_id"])]
    out["modules"].reverse()
    for connection in out["connections"]:
        connection["src_module_id"] = remap[str(connection["src_module_id"])]
        connection["dst_module_id"] = remap[str(connection["dst_module_id"])]
    out["connections"].reverse()
    return out


def test_legacy_default_is_byte_exact_and_keeps_old_draw_formula() -> None:
    parent_a = _payload(1.0)
    parent_b = _payload(-1.0)
    default_rng = SequenceRng([0.1, 0.9, 0.1])
    explicit_rng = SequenceRng([0.1, 0.9, 0.1])

    default_child, default_summary = recombine_lineage_payloads(
        parent_a_payload=deepcopy(parent_a),
        parent_b_payload=deepcopy(parent_b),
        rng=default_rng,
    )
    explicit_child, explicit_summary = recombine_lineage_payloads(
        parent_a_payload=deepcopy(parent_a),
        parent_b_payload=deepcopy(parent_b),
        rng=explicit_rng,
        inheritance_profile=LEGACY_INHERITANCE_PROFILE,
    )

    assert json.dumps(default_child, sort_keys=True) == json.dumps(explicit_child, sort_keys=True)
    assert default_summary == explicit_summary
    assert default_rng.draw_count == explicit_rng.draw_count == 3
    assert "c2_inheritance_profile" not in default_child["genes"]


def test_linked_profile_preserves_motif_and_uses_one_draw_per_component() -> None:
    parent_a = _payload(1.0)
    parent_b = _payload(-1.0)
    rng = SequenceRng([0.1])

    child, summary = recombine_lineage_payloads(
        parent_a_payload=parent_a,
        parent_b_payload=parent_b,
        rng=rng,
        inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
    )

    assert len(set(_package_markers(child))) == 1
    assert rng.draw_count == summary.budget_draws == 1
    assert child["genes"]["c2_linked_component_count"] == 1
    assert child["genes"]["c2_structural_linkage_rule"] == C2_STRUCTURAL_LINKAGE_RULE


def test_linked_profile_is_parent_order_and_local_repack_invariant() -> None:
    parent_a = _payload(1.0)
    parent_b = _payload(-1.0)

    child_ab, _ = recombine_lineage_payloads(
        parent_a_payload=deepcopy(parent_a),
        parent_b_payload=deepcopy(parent_b),
        rng=SequenceRng([0.9]),
        inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
    )
    child_ba, _ = recombine_lineage_payloads(
        parent_a_payload=_repack(parent_b),
        parent_b_payload=_repack(parent_a),
        rng=SequenceRng([0.9]),
        inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
    )

    assert child_ab == child_ba


def test_dependency_union_prevents_opposite_parent_hybrid_cycle() -> None:
    identities = (_identity("shared-0", 0), _identity("shared-1", 1))
    h0, h1 = (str(identity["homology_id"]) for identity in identities)
    parent_a = _payload(1.0, identities=identities, dependencies=([h1], []))
    parent_b = _payload(-1.0, identities=identities, dependencies=([], [h0]))
    rng = SequenceRng([0.1])

    child, summary = recombine_lineage_payloads(
        parent_a_payload=parent_a,
        parent_b_payload=parent_b,
        rng=rng,
        inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
    )

    assert len(set(_package_markers(child))) == 1
    assert rng.draw_count == summary.budget_draws == 1


def test_static_invalid_parent_dependency_rejects_before_draws() -> None:
    identities = (_identity("shared-0", 0), _identity("shared-1", 1))
    h0, h1 = (str(identity["homology_id"]) for identity in identities)
    invalid = _payload(1.0, identities=identities, dependencies=([h1], [h0]))
    valid = _payload(-1.0, identities=identities, dependencies=([], []))
    rng = SequenceRng([0.1])

    with pytest.raises((ValueError, RuntimeError), match="ESCAP6_DEPENDENCY_CYCLE"):
        recombine_lineage_payloads(
            parent_a_payload=invalid,
            parent_b_payload=valid,
            rng=rng,
            inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
        )
    assert rng.draw_count == 0


def test_unassignable_matched_connection_is_a_predraw_static_failure() -> None:
    native_a = {"h0": {"params": {}}, "h1": {"params": {}}}
    native_b = deepcopy(native_a)
    connection = {"src_module_id": "m0", "dst_module_id": "m1"}

    with pytest.raises(RuntimeError, match="DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION"):
        _linked_component_plan(
            matched_homology=["h0", "h1"],
            native_a=native_a,
            native_b=native_b,
            matched_conn_keys=["matched-edge"],
            conn_a={"matched-edge": connection},
            conn_b={"matched-edge": connection},
            endpoint_a={"m0": "native:h0", "m1": "native:h1"},
            endpoint_b={"m0": "native:h0"},
        )


def test_draw_dependent_hybrid_conflict_consumes_all_components_without_redraw() -> None:
    shared = (_identity("shared-0", 0), _identity("shared-1", 1))
    unique_a = _identity("unique-a", 2)
    unique_b = _identity("unique-b", 3)
    h0, h1 = (str(identity["homology_id"]) for identity in shared)
    ha = str(unique_a["homology_id"])
    hb = str(unique_b["homology_id"])

    # Parent A: h0 -> ha -> h1.  Parent B: h1 -> hb -> h0.  The unique
    # modules are retained by the unmatched union.  Opposite donors for the
    # two isolated matched components close a child-only cycle.
    parent_a = _payload(
        1.0,
        identities=(shared[0], shared[1], unique_a),
        dependencies=([], [ha], [h0]),
        connections=(),
    )
    parent_b = _payload(
        -1.0,
        identities=(shared[0], shared[1], unique_b),
        dependencies=([hb], [], [h1]),
        connections=(),
    )
    rng = SequenceRng([0.1, 0.9])

    with pytest.raises((ValueError, RuntimeError), match="ESCAP6_DEPENDENCY_CYCLE"):
        recombine_lineage_payloads(
            parent_a_payload=parent_a,
            parent_b_payload=parent_b,
            rng=rng,
            inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
        )
    assert rng.draw_count == 2
