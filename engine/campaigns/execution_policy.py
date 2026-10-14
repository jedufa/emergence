"""Fail-closed production execution policy for declarative campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .contracts import CampaignPlanV1, RuntimeProfileV1


MULTIWORLD_CPU_PRODUCTION_FORBIDDEN_V1: Final[str] = (
    "multiworld_cpu_production_forbidden"
)


def _logical_world_count(plan: CampaignPlanV1 | Mapping[str, Any]) -> int | None:
    cells = plan.get("cells") if isinstance(plan, Mapping) else plan.cells
    if (
        isinstance(cells, (str, bytes))
        or not isinstance(cells, Sequence)
        or not cells
    ):
        return None
    return len(cells)


def _is_route_qualification_case(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> bool:
    if not isinstance(plan, Mapping):
        return False
    case_id = plan.get("route_qualification_case_id")
    return isinstance(case_id, str) and bool(case_id)


def _uses_only_serial_b1_batches(plan: CampaignPlanV1 | Mapping[str, Any]) -> bool:
    if isinstance(plan, Mapping):
        batches = plan.get("batches")
        if batches is None:
            batches = plan.get("physical_batches")
    else:
        batches = plan.batches
    if (
        isinstance(batches, (str, bytes))
        or not isinstance(batches, Sequence)
        or not batches
    ):
        return False
    for batch in batches:
        cell_ids = batch.get("cell_ids") if isinstance(batch, Mapping) else batch.cell_ids
        if (
            isinstance(cell_ids, (str, bytes))
            or not isinstance(cell_ids, Sequence)
            or len(cell_ids) != 1
        ):
            return False
    return True


def production_execution_policy_reasons_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
    profile: RuntimeProfileV1,
) -> tuple[dict[str, str], ...]:
    """Withhold simultaneous multi-world production authority from CPU routes."""

    backend = profile.backend.strip().lower()
    if not backend.startswith("cpu") or _is_route_qualification_case(plan):
        return ()
    world_count = _logical_world_count(plan)
    if world_count == 1 or _uses_only_serial_b1_batches(plan):
        return ()
    if world_count is None:
        message = (
            "CPU production authority requires an exact single logical world; "
            "the compiled world count is unavailable"
        )
    else:
        message = (
            f"CPU production authority is limited to one logical world, not "
            f"{world_count}; multi-world production requires a qualified GPU route"
        )
    return (
        {
            "code": MULTIWORLD_CPU_PRODUCTION_FORBIDDEN_V1,
            "path": "resolved_profile.backend",
            "message": message,
        },
    )


__all__ = [
    "MULTIWORLD_CPU_PRODUCTION_FORBIDDEN_V1",
    "production_execution_policy_reasons_v1",
]
