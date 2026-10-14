from __future__ import annotations

from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
    compare_execution_routes_v1,
)


def _route(
    *,
    cache: str = "cache-v1",
    checkpoint: str = "checkpoint-v1",
    controller: str = "controller-v1",
) -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="provider-v1",
        profile_id="profile-v1",
        components=(
            {
                "component_id": "controller",
                "contract_version": controller,
            },
            {
                "component_id": "policy_cache_refill",
                "contract_version": cache,
            },
            {
                "component_id": "checkpoint_restore",
                "contract_version": checkpoint,
            },
        ),
        required_obligations=(
            "cold_start",
            "generation_boundary",
            "post_boundary_policy_cache_refill",
            "checkpoint_export",
            "fresh_owner_restore",
        ),
    )


def test_unchanged_route_reuses_route_and_mechanic_qualification() -> None:
    receipt = compare_execution_routes_v1(_route(), _route())

    assert receipt["reuses_route_qualification"] is True
    assert receipt["mechanic_qualification_reusable"] is True
    assert receipt["route_canary_required"] is False
    assert receipt["changed_components"] == []


def test_cache_change_invalidates_only_cache_dependent_obligation() -> None:
    receipt = compare_execution_routes_v1(
        _route(), _route(cache="cache-v2")
    )

    assert receipt["reuses_route_qualification"] is False
    assert receipt["mechanic_qualification_reusable"] is True
    assert receipt["route_canary_required"] is True
    assert [row["component_id"] for row in receipt["changed_components"]] == [
        "policy_cache_refill"
    ]
    assert receipt["affected_obligations"] == [
        "post_boundary_policy_cache_refill"
    ]


def test_controller_change_conservatively_invalidates_the_complete_route() -> None:
    old = _route()
    new = _route(controller="controller-v2")
    receipt = compare_execution_routes_v1(old, new)

    assert receipt["affected_obligations"] == sorted(
        old.required_obligations
    )
    assert receipt["invalidation_policy"] == (
        "dependent_route_only_conservative_source_fallback"
    )
