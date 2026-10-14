"""Deterministic physical-B extent over the frozen B48 treatment matrix.

Factory 200 is the leading 48-cell source. Larger B appends new ecology-seed
treatments; it does not remint or reorder those 48 rows.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from engine.perf.cuda_architecture_b_hapb_v1 import hapb_physical_cell_count_v1


CANONICAL_B48_V1 = 48
FOUNDERS_V1 = ("a_high", "b_high", "symmetric_mix")
TASK_SEEDS_V1 = (1001, 2002, 3003, 4004)
CELLS_PER_EXTRA_ECOLOGY_V1 = len(FOUNDERS_V1) * len(TASK_SEEDS_V1)
EXTRA_ECOLOGY_SEEDS_V1 = tuple(505 + 101 * index for index in range(16))


def extra_ecology_seeds_for_physical_cells_v1(cells: int) -> tuple[int, ...]:
    count = hapb_physical_cell_count_v1(cells)
    if count <= CANONICAL_B48_V1:
        return ()
    extra = count - CANONICAL_B48_V1
    seeds_needed = (extra + CELLS_PER_EXTRA_ECOLOGY_V1 - 1) // CELLS_PER_EXTRA_ECOLOGY_V1
    if seeds_needed > len(EXTRA_ECOLOGY_SEEDS_V1):
        raise ValueError(
            f"physical B={count} needs {seeds_needed} extra ecology seeds; "
            f"only {len(EXTRA_ECOLOGY_SEEDS_V1)} are declared"
        )
    return EXTRA_ECOLOGY_SEEDS_V1[:seeds_needed]


def expand_campaign_spec_physical_cells_v1(
    spec: Mapping[str, Any], cells: int
) -> dict[str, Any]:
    """Return a spec whose compiled matrix covers at least ``cells`` identities."""

    count = hapb_physical_cell_count_v1(cells)
    expanded = deepcopy(dict(spec))
    extra_seeds = extra_ecology_seeds_for_physical_cells_v1(count)
    if extra_seeds:
        matrix = dict(expanded["matrix"])
        dimensions = []
        found_ecology = False
        for dimension in matrix["dimensions"]:
            row = dict(dimension)
            if row.get("id") == "ecology_seed":
                found_ecology = True
                values = list(row["values"])
                for seed in extra_seeds:
                    if seed not in values:
                        values.append(seed)
                row["values"] = values
            dimensions.append(row)
        if not found_ecology:
            raise ValueError("campaign spec lacks ecology_seed for physical-B expansion")
        compiled_count = CANONICAL_B48_V1 + CELLS_PER_EXTRA_ECOLOGY_V1 * len(extra_seeds)
        matrix["dimensions"] = dimensions
        matrix["expected_cells"] = compiled_count
        matrix["max_cells"] = compiled_count
        expanded["matrix"] = matrix
        execution = dict(expanded["execution"])
        batch = dict(execution.get("batch") or {})
        batch["maximum"] = max(int(batch.get("maximum") or 0), compiled_count)
        execution["batch"] = batch
        expanded["execution"] = execution
    return expanded


def flatten_plan_cell_ids_v1(plan: Mapping[str, Any]) -> tuple[str, ...]:
    batches = plan.get("physical_batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise ValueError("compiled plan has no physical batches")
    cell_ids: list[str] = []
    for batch in batches:
        if not isinstance(batch, Mapping):
            raise ValueError("compiled physical batch is malformed")
        rows = batch.get("cell_ids")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("compiled physical batch has no cell_ids")
        cell_ids.extend(str(cell_id) for cell_id in rows)
    if not cell_ids:
        raise ValueError("compiled plan has no cell identities")
    return tuple(cell_ids)


def leading_source_extent_cell_ids_v1(
    *,
    canonical_b48_cell_ids: Sequence[str],
    expanded_cell_ids: Sequence[str],
    cells: int,
) -> tuple[str, ...]:
    """Keep factory-200 order as the leading extent; append new identities."""

    count = hapb_physical_cell_count_v1(cells)
    leading = tuple(canonical_b48_cell_ids)
    if len(leading) != CANONICAL_B48_V1:
        raise ValueError("canonical source extent is not exact B48")
    if count == CANONICAL_B48_V1:
        return leading
    leading_set = set(leading)
    extras = tuple(
        cell_id for cell_id in expanded_cell_ids if cell_id not in leading_set
    )
    needed = count - CANONICAL_B48_V1
    if len(extras) < needed:
        raise ValueError(
            f"expanded matrix supplies {len(extras)} new cells; {needed} required"
        )
    return leading + extras[:needed]
