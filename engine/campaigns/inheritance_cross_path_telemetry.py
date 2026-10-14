"""Contract-driven telemetry for inheritance cross-path tensor campaigns."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

from engine.perf.inheritance_cross_path_tensor import TensorCrossPathStateV1


TELEMETRY_SCHEMA_VERSION = "inheritance_cross_path_generation_metrics_v1"


class InheritanceCrossPathTelemetryError(ValueError):
    """A tensor boundary cannot support complete cross-path telemetry."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _tensor_descriptor_bytes(name: str, value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return descriptor + b"\0" + tensor.view(torch.uint8).numpy().tobytes()


def cell_state_sha256(state: TensorCrossPathStateV1, cell: int) -> str:
    digest = hashlib.sha256()
    digest.update(state.contract.sha256.encode("ascii"))
    for name in ("loci", "alive", "agent_ids", "ancestry"):
        digest.update(
            _tensor_descriptor_bytes(name, getattr(state, name)[cell])
        )
    for name in ("next_id", "generation"):
        digest.update(
            _tensor_descriptor_bytes(name, getattr(state, name)[cell : cell + 1])
        )
    digest.update(
        json.dumps(
            {
                "seed": state.seeds[cell],
                "profile_id": state.profile_ids[cell],
                "path_id": state.path_ids[cell],
                "pattern_id": state.pattern_ids[cell],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return digest.hexdigest()


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise InheritanceCrossPathTelemetryError("quantile requires values")
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    alpha = position - lower
    return float(
        sorted_values[lower]
        + alpha * (sorted_values[upper] - sorted_values[lower])
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise InheritanceCrossPathTelemetryError(
            "correlation vectors are misaligned"
        )
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    delta_left = [value - mean_left for value in left]
    delta_right = [value - mean_right for value in right]
    numerator = sum(
        a * b for a, b in zip(delta_left, delta_right, strict=True)
    )
    denominator = math.sqrt(
        sum(value * value for value in delta_left)
        * sum(value * value for value in delta_right)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _locus_statistics(
    values: Sequence[float],
    *,
    lower: float,
    upper: float,
    neutral: float,
) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    width = upper - lower
    edge_tolerance = width * 0.05
    neutral_tolerance = width * 0.05
    signs = {-1 if value < 0.0 else 1 if value > 0.0 else 0 for value in ordered}
    return {
        "mean": mean,
        "variance": variance,
        "standard_deviation": math.sqrt(variance),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "quartiles": {
            "q1": _quantile(ordered, 0.25),
            "median": _quantile(ordered, 0.5),
            "q3": _quantile(ordered, 0.75),
        },
        "unique_values": len(set(ordered)),
        "fraction_near_lower_extreme": sum(
            value <= lower + edge_tolerance for value in ordered
        )
        / len(ordered),
        "fraction_near_upper_extreme": sum(
            value >= upper - edge_tolerance for value in ordered
        )
        / len(ordered),
        "fraction_near_both_extremes": sum(
            value <= lower + edge_tolerance
            or value >= upper - edge_tolerance
            for value in ordered
        )
        / len(ordered),
        "fraction_near_neutral": sum(
            abs(value - neutral) <= neutral_tolerance for value in ordered
        )
        / len(ordered),
        "negative_fraction": sum(value < 0.0 for value in ordered) / len(ordered),
        "positive_fraction": sum(value > 0.0 for value in ordered) / len(ordered),
        "zero_fraction": sum(value == 0.0 for value in ordered) / len(ordered),
        "signs_present": sorted(signs),
    }


def _package_statistics(
    rows: Sequence[Sequence[float]],
    founder_packages: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    package_rows = {
        str(group_id): tuple(float(value) for value in package)
        for group_id, package in founder_packages.items()
    }
    intact_counts = {group_id: 0 for group_id in package_rows}
    recombinant = 0
    neither = 0
    correlations: dict[str, float] = {}
    allowed_by_locus = tuple(
        {package[index] for package in package_rows.values()}
        for index in range(len(next(iter(package_rows.values()))))
    )
    for row in rows:
        intact_group = next(
            (
                group_id
                for group_id, package in package_rows.items()
                if tuple(row) == package
            ),
            None,
        )
        if intact_group is not None:
            intact_counts[intact_group] += 1
        elif all(
            value in allowed
            for value, allowed in zip(row, allowed_by_locus, strict=True)
        ):
            recombinant += 1
        else:
            neither += 1
    intact_total = sum(intact_counts.values())
    return {
        "intact_package_counts": intact_counts,
        "intact_package_count": intact_total,
        "intact_package_retention": intact_total / len(rows),
        "recombinant_count": recombinant,
        "neither_package_count": neither,
        "correlations": correlations,
    }


def _event_rows(
    output: Mapping[str, torch.Tensor] | None,
    *,
    cell: int,
    step: int,
    linked_block_ids: Sequence[str],
    locus_names: Sequence[str],
    founder_packages: Mapping[str, Sequence[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if output is None:
        return [], {
            "same_group_pairings": 0,
            "cross_group_pairings": 0,
            "selected_parent_count": 0,
            "parent_contribution_distribution": {},
            "parent_contribution_hhi": 0.0,
            "top_parent_contribution_share": 0.0,
            "linked_package_integrity": True,
            "inheritance_draw_count": 0,
            "mutation_events": 0,
        }
    valid = output["event_valid"][cell, step].detach().cpu().tolist()
    tensors = {
        name: value[cell, step].detach().cpu()
        for name, value in output.items()
        if value.ndim >= 3 and value.shape[:2] == output["event_valid"].shape[:2]
    }
    events: list[dict[str, Any]] = []
    contributions: Counter[int] = Counter()
    same = 0
    cross = 0
    integrity = True
    draw_count = 0
    mutation_events = 0

    def group(values: Sequence[float]) -> str:
        distances = {
            str(group_id): sum(
                (value - target) ** 2
                for value, target in zip(values, package, strict=True)
            )
            for group_id, package in founder_packages.items()
        }
        return min(distances, key=lambda group_id: (distances[group_id], group_id))

    for event, is_valid in enumerate(valid):
        if not is_valid:
            continue
        parent_loci = [
            float(value) for value in tensors["parent_loci"][event].tolist()
        ]
        mate_loci = [
            float(value) for value in tensors["mate_loci"][event].tolist()
        ]
        child_loci = [
            float(value) for value in tensors["child_loci"][event].tolist()
        ]
        parent_id = int(tensors["parent_agent_ids"][event].item())
        mate_id = int(tensors["mate_agent_ids"][event].item())
        child_id = int(tensors["child_agent_ids"][event].item())
        donors = [
            int(value)
            for value in tensors["linkage_donor_ordinals"][event].tolist()
        ]
        event_integrity = bool(
            tensors["linked_package_integrity"][event].item()
        )
        event_success = bool(tensors["event_success"][event].item())
        failure_code = int(tensors["failure_reason_code"][event].item())
        event_draws = int(
            tensors["inheritance_draw_count"][event].item()
        )
        event_mutations = int(tensors["mutation_events"][event].item())
        parent_group = group(parent_loci)
        mate_group = group(mate_loci)
        same += int(parent_group == mate_group)
        cross += int(parent_group != mate_group)
        contributions.update((parent_id, mate_id))
        integrity = integrity and event_integrity
        draw_count += event_draws
        mutation_events += event_mutations
        events.append(
            {
                "event_ordinal": event,
                "parent_agent_id": parent_id,
                "mate_agent_id": mate_id,
                "child_agent_id": child_id,
                "success": event_success,
                "failure_reason": (
                    None if failure_code == 0 else "capacity_full"
                ),
                "parent_group": parent_group,
                "mate_group": mate_group,
                "parent_loci": dict(zip(locus_names, parent_loci, strict=True)),
                "mate_loci": dict(zip(locus_names, mate_loci, strict=True)),
                "child_loci": dict(zip(locus_names, child_loci, strict=True)),
                "linkage_donor_by_block": dict(
                    zip(linked_block_ids, donors, strict=True)
                ),
                "linked_package_integrity": event_integrity,
                "inheritance_draw_count": event_draws,
                "mutation_events": event_mutations,
                "sign_flip_loci": [
                    locus_names[index]
                    for index, child_value in enumerate(child_loci)
                    if event_success
                    and (
                        -1
                        if child_value < 0.0
                        else 1
                        if child_value > 0.0
                        else 0
                    )
                    not in {
                        (
                            -1
                            if parent_loci[index] < 0.0
                            else 1
                            if parent_loci[index] > 0.0
                            else 0
                        ),
                        (
                            -1
                            if mate_loci[index] < 0.0
                            else 1
                            if mate_loci[index] > 0.0
                            else 0
                        ),
                    }
                ],
            }
        )
    total = sum(contributions.values())
    return events, {
        "same_group_pairings": same,
        "cross_group_pairings": cross,
        "selected_parent_count": len(contributions),
        "parent_contribution_distribution": {
            str(parent): count for parent, count in sorted(contributions.items())
        },
        "parent_contribution_hhi": (
            sum((count / total) ** 2 for count in contributions.values())
            if total
            else 0.0
        ),
        "top_parent_contribution_share": (
            max(contributions.values()) / total if total else 0.0
        ),
        "linked_package_integrity": integrity,
        "inheritance_draw_count": draw_count,
        "mutation_events": mutation_events,
    }


def generation_row(
    state: TensorCrossPathStateV1,
    *,
    cell: int,
    cell_id: str,
    generation: int,
    founder_packages: Mapping[str, Sequence[float]],
    neutral_package: Sequence[float],
    output: Mapping[str, torch.Tensor] | None,
    output_step: int = 0,
) -> dict[str, Any]:
    """Project one explicit generation boundary onto canonical CPU telemetry."""

    contract = state.contract
    population = contract.population_size
    founder_groups = contract.resolved_founder_groups()
    group_ids = tuple(group.group_id for group in founder_groups)
    packages = {
        str(group_id): tuple(float(value) for value in package)
        for group_id, package in founder_packages.items()
    }
    if set(packages) != set(group_ids):
        raise InheritanceCrossPathTelemetryError(
            "founder packages must exactly match declared founder groups"
        )
    if any(len(package) != len(contract.locus_names) for package in packages.values()):
        raise InheritanceCrossPathTelemetryError(
            "founder package width must match the locus registry"
        )
    if len(neutral_package) != len(contract.locus_names):
        raise InheritanceCrossPathTelemetryError(
            "neutral package width must match the locus registry"
        )
    values = state.loci[cell, :population].detach().cpu().tolist()
    ancestry = state.ancestry[cell, :population].detach().cpu().tolist()
    locus_rows = {
        name: _locus_statistics(
            [float(row[index]) for row in values],
            lower=contract.lower_bounds[index],
            upper=contract.upper_bounds[index],
            neutral=float(neutral_package[index]),
        )
        for index, name in enumerate(contract.locus_names)
    }
    package = _package_statistics(values, packages)
    columns = {
        name: [float(row[index]) for row in values]
        for index, name in enumerate(contract.locus_names)
    }
    package["correlations"] = {
        f"{left}__{right}": _correlation(columns[left], columns[right])
        for left_index, left in enumerate(contract.locus_names)
        for right in contract.locus_names[left_index + 1 :]
    }
    event_rows, reproduction = _event_rows(
        output,
        cell=cell,
        step=output_step,
        linked_block_ids=contract.linkage_block_ids,
        locus_names=contract.locus_names,
        founder_packages=packages,
    )
    if output is None:
        opportunities = successes = failures = refill = ordinary = 0
    else:
        opportunities = int(
            output["birth_opportunities"][cell, output_step].detach().cpu().item()
        )
        successes = int(
            output["successful_births"][cell, output_step].detach().cpu().item()
        )
        failures = int(
            output["failed_birth_attempts"][cell, output_step]
            .detach()
            .cpu()
            .item()
        )
        refill = int(
            output["refill_children"][cell, output_step].detach().cpu().item()
        )
        ordinary = int(
            output["ordinary_births"][cell, output_step].detach().cpu().item()
        )
    ancestry_mean = [
        sum(row[index] for row in ancestry) / population
        for index in range(population)
    ]
    ancestry_square_sum = sum(value * value for value in ancestry_mean)
    group_shares: dict[str, float] = {}
    founder_offset = 0
    for founder_group in founder_groups:
        next_offset = founder_offset + founder_group.count
        group_shares[founder_group.group_id] = sum(
            ancestry_mean[founder_offset:next_offset]
        )
        founder_offset = next_offset
    row = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "cell_id": cell_id,
        "generation": generation,
        "identity": {
            "mode": state.profile_ids[cell],
            "pattern": state.pattern_ids[cell],
            "reproduction_path": state.path_ids[cell],
            "paired_seed": state.seeds[cell],
        },
        "population": {
            "alive": int(
                state.alive[cell, :population].detach().cpu().sum().item()
            ),
            "capacity": contract.capacity,
        },
        "loci": locus_rows,
        "packages": package,
        "reproduction": {
            **reproduction,
            "birth_opportunities": opportunities,
            "successful_births": successes,
            "failed_birth_attempts": failures,
            "refill_child_count": refill,
            "ordinary_birth_count": ordinary,
            "births_per_eligible_opportunity": (
                successes / opportunities if opportunities else 0.0
            ),
            "events": event_rows,
        },
        "lineage": {
            "effective_founder_ancestry_count": (
                1.0 / ancestry_square_sum if ancestry_square_sum else 0.0
            ),
            "dominant_founder_ancestry_share": max(ancestry_mean),
            "founder_group_ancestry_shares": group_shares,
            "descendant_branch_diversity": (
                1.0 / ancestry_square_sum if ancestry_square_sum else 0.0
            ),
        },
        "mutation_events": reproduction["mutation_events"],
        "linked_package_integrity": reproduction[
            "linked_package_integrity"
        ],
        "state_sha256": cell_state_sha256(state, cell),
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def summaries_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise InheritanceCrossPathTelemetryError("generation rows are empty")
    reproduction = {
        "birth_opportunities": sum(
            int(row["reproduction"]["birth_opportunities"]) for row in rows
        ),
        "successful_births": sum(
            int(row["reproduction"]["successful_births"]) for row in rows
        ),
        "failed_birth_attempts": sum(
            int(row["reproduction"]["failed_birth_attempts"]) for row in rows
        ),
        "refill_child_count": sum(
            int(row["reproduction"]["refill_child_count"]) for row in rows
        ),
        "ordinary_birth_count": sum(
            int(row["reproduction"]["ordinary_birth_count"]) for row in rows
        ),
        "same_group_pairings": sum(
            int(row["reproduction"]["same_group_pairings"]) for row in rows
        ),
        "cross_group_pairings": sum(
            int(row["reproduction"]["cross_group_pairings"]) for row in rows
        ),
    }
    terminal = rows[-1]
    return {
        "reproduction_summary": reproduction,
        "inheritance_summary": {
            "terminal_loci": terminal["loci"],
            "terminal_packages": terminal["packages"],
            "mutation_events": sum(int(row["mutation_events"]) for row in rows),
            "linked_package_integrity": all(
                bool(row["linked_package_integrity"]) for row in rows
            ),
        },
        "lineage_summary": dict(terminal["lineage"]),
    }


__all__ = [
    "InheritanceCrossPathTelemetryError",
    "TELEMETRY_SCHEMA_VERSION",
    "canonical_sha256",
    "cell_state_sha256",
    "generation_row",
    "summaries_from_rows",
]
