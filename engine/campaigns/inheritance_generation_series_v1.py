"""Fail-closed projection of resident inheritance generation evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Final, Mapping, Sequence

from .canonical import canonical_sha256


SCHEMA_VERSION: Final[str] = "eiel_inheritance_generation_series_v1"


class InheritanceGenerationSeriesError(ValueError):
    """Resident evidence cannot support an exact generation series."""


def _fail(message: str) -> InheritanceGenerationSeriesError:
    return InheritanceGenerationSeriesError(message)


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identity(cell: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = cell.get("dimensions")
    parameters = cell.get("parameters")
    if not isinstance(dimensions, Mapping) or not isinstance(parameters, Mapping):
        raise _fail("compiled cell dimensions or parameters are absent")
    identity = {
        name: dimensions.get(name)
        for name in (
            "inheritance_mode",
            "founder_pattern",
            "reproduction_path",
            "paired_seed",
        )
    }
    if (
        any(value is None for value in identity.values())
        or parameters.get("inheritance_cross_path.mutation_enabled") is not False
    ):
        raise _fail("compiled cell identity or zero-mutation authority is invalid")
    return identity


def _population_contract(cell: Mapping[str, Any]) -> tuple[int, int]:
    parameters = cell["parameters"]
    initial = _integer(
        parameters.get("patch.initial_population"),
        field="patch.initial_population",
        minimum=1,
    )
    capacity = _integer(
        parameters.get("patch.capacity"),
        field="patch.capacity",
        minimum=initial,
    )
    return initial, capacity


def _checkpoint_boundaries(
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    cell_ids: tuple[str, ...],
    terminal_generation: int,
) -> tuple[int, ...]:
    generations: list[int] = []
    for key, checkpoint in catalog.items():
        try:
            generation = int(key)
        except (TypeError, ValueError) as exc:
            raise _fail("checkpoint catalog key is not a generation") from exc
        if str(generation) != key or not isinstance(checkpoint, Mapping):
            raise _fail("checkpoint catalog is malformed")
        if checkpoint.get("generation") != generation:
            raise _fail("checkpoint generation contradicts its catalog key")
        bundles = checkpoint.get("cell_bundles")
        digests = checkpoint.get("terminal_cell_digests")
        if (
            not isinstance(bundles, list)
            or [row.get("cell_id") for row in bundles if isinstance(row, Mapping)]
            != list(cell_ids)
            or len(bundles) != len(cell_ids)
            or not isinstance(digests, Mapping)
            or list(digests) != list(cell_ids)
        ):
            raise _fail("checkpoint cell order or digest authority is incomplete")
        for cell_id in cell_ids:
            _sha256(digests[cell_id], field=f"checkpoint digest {cell_id}")
        generations.append(generation)
    boundaries = tuple(sorted(generations))
    if (
        len(boundaries) != len(set(boundaries))
        or not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != terminal_generation
    ):
        raise _fail("checkpoint catalog lacks exact initial and terminal boundaries")
    return boundaries


def _generic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_ids: tuple[str, ...],
    terminal_generation: int,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    indexed = {cell_id: {} for cell_id in cell_ids}
    allowed_fields = ("alive", "births", "failed_births")
    for row in rows:
        if not isinstance(row, Mapping):
            raise _fail("generic Patch telemetry row is malformed")
        cell_id = row.get("cell_id")
        generation = row.get("generation")
        if (
            cell_id not in indexed
            or type(generation) is not int
            or generation in indexed[cell_id]
        ):
            raise _fail("generic Patch telemetry identity is unknown or duplicated")
        for field in allowed_fields:
            _integer(
                row.get(field),
                field=f"generic row {cell_id}/G{generation}/{field}",
            )
        indexed[cell_id][generation] = row
    expected = set(range(terminal_generation))
    if any(set(by_generation) != expected for by_generation in indexed.values()):
        raise _fail("generic Patch telemetry is not the exact transition prefix")
    return indexed


def _segment_cells(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    cell_ids: tuple[str, ...],
    boundaries: tuple[int, ...],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    by_cell: dict[str, dict[int, Mapping[str, Any]]] = {
        cell_id: {} for cell_id in cell_ids
    }
    expected_segments = len(boundaries) - 1
    cumulative = {cell_id: (0, 0, 0) for cell_id in cell_ids}
    cell_positions = {
        cell_id: position for position, cell_id in enumerate(cell_ids)
    }
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise _fail("native exposure segment artifact is malformed")
        segment = artifact.get("segment_index")
        cell_id = artifact.get("cell_id")
        cell = artifact.get("cell")
        if (
            type(segment) is not int
            or not 0 <= segment < expected_segments
            or cell_id not in by_cell
            or segment in by_cell[cell_id]
            or not isinstance(cell, Mapping)
            or cell.get("cell_index") != cell_positions.get(cell_id)
        ):
            raise _fail("native exposure segment identity is invalid or duplicated")
        supplied = _sha256(
            cell.get("segment_sha256"),
            field=f"segment {segment}/{cell_id} digest",
        )
        unsigned = dict(cell)
        unsigned.pop("segment_sha256", None)
        unsigned.pop("previous_chain_sha256", None)
        unsigned.pop("chain_sha256_after_segment", None)
        expected_digest = canonical_sha256(
            {
                "schema_version": artifact.get("schema_version"),
                "segment_index": segment,
                "cell": unsigned,
            }
        )
        if supplied != expected_digest:
            raise _fail("native exposure cell segment digest does not verify")
        attempts = cell.get("attempts")
        if not isinstance(attempts, list):
            raise _fail("native exposure attempts are absent")
        start, end = boundaries[segment], boundaries[segment + 1]
        seen_attempts: set[int] = set()
        success_count = 0
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise _fail("native exposure attempt is malformed")
            ordinal = _integer(
                attempt.get("attempt_index"), field="attempt_index"
            )
            generation = _integer(
                attempt.get("generation"), field="attempt generation"
            )
            if ordinal in seen_attempts or not start <= generation < end:
                raise _fail("native exposure attempt is duplicated or outside its segment")
            seen_attempts.add(ordinal)
            if type(attempt.get("success")) is not bool:
                raise _fail("native exposure attempt success is not boolean")
            success_count += int(attempt["success"])
        failures = len(attempts) - success_count
        if (
            cell.get("attempt_count") != len(attempts)
            or cell.get("success_count") != success_count
            or cell.get("failure_count") != failures
        ):
            raise _fail("native exposure segment counters contradict its attempts")
        prior_attempts, prior_successes, prior_failures = cumulative[cell_id]
        after = cell.get("cumulative_counts_after_segment")
        if (
            not isinstance(after, Mapping)
            or after.get("attempts") != prior_attempts + len(attempts)
            or after.get("successes") != prior_successes + success_count
            or after.get("failures") != prior_failures + failures
        ):
            raise _fail("native exposure cumulative counters are discontinuous")
        cumulative[cell_id] = (
            prior_attempts + len(attempts),
            prior_successes + success_count,
            prior_failures + failures,
        )
        by_cell[cell_id][segment] = cell
    expected = set(range(expected_segments))
    if any(set(rows) != expected for rows in by_cell.values()):
        raise _fail("native exposure segments do not exactly cover every cell")
    return by_cell


def project_inheritance_generation_series_v1(
    *,
    cells: Sequence[Mapping[str, Any]],
    generic_patch_rows: Sequence[Mapping[str, Any]],
    native_exposure_segment_artifacts: Sequence[Mapping[str, Any]],
    checkpoint_catalog: Mapping[str, Mapping[str, Any]],
    terminal_generation: int,
) -> list[dict[str, Any]]:
    """Return exact cell-major G0..G rows from bound resident evidence.

    Native generation ``n`` describes the transition ending at boundary
    ``n + 1``.  G0 is the compiled initial population with zero reproduction.
    """

    terminal = _integer(
        terminal_generation, field="terminal_generation", minimum=1
    )
    cell_rows = tuple(cells)
    cell_ids = tuple(
        cell.get("cell_id") if isinstance(cell, Mapping) else None
        for cell in cell_rows
    )
    if (
        not cell_ids
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
    ):
        raise _fail("compiled cell identities are empty or duplicated")
    typed_ids = tuple(str(cell_id) for cell_id in cell_ids)
    boundaries = _checkpoint_boundaries(
        checkpoint_catalog,
        cell_ids=typed_ids,
        terminal_generation=terminal,
    )
    generic = _generic_rows(
        generic_patch_rows,
        cell_ids=typed_ids,
        terminal_generation=terminal,
    )
    segments = _segment_cells(
        native_exposure_segment_artifacts,
        cell_ids=typed_ids,
        boundaries=boundaries,
    )

    output: list[dict[str, Any]] = []
    for cell in cell_rows:
        cell_id = str(cell["cell_id"])
        identity = _identity(cell)
        initial, capacity = _population_contract(cell)
        attempts_by_generation: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for segment in range(len(boundaries) - 1):
            for attempt in segments[cell_id][segment]["attempts"]:
                attempts_by_generation[int(attempt["generation"])].append(attempt)
        for boundary_generation in range(terminal + 1):
            if boundary_generation == 0:
                alive = initial
                attempts: Sequence[Mapping[str, Any]] = ()
            else:
                source_generation = boundary_generation - 1
                generic_row = generic[cell_id][source_generation]
                alive = int(generic_row["alive"])
                attempts = attempts_by_generation.get(source_generation, ())
                successes = sum(int(row["success"]) for row in attempts)
                failures = len(attempts) - successes
                if (
                    generic_row["births"] != successes
                    or generic_row["failed_births"] != failures
                ):
                    raise _fail(
                        "generic Patch and native exposure reproduction counters differ"
                    )
            successes = sum(int(row["success"]) for row in attempts)
            failures = len(attempts) - successes
            path = identity["reproduction_path"]
            reproduction = {
                "birth_opportunities": len(attempts),
                "successful_births": successes,
                "failed_birth_attempts": failures,
                "refill_child_count": (
                    successes if path == "generation_boundary_refill" else 0
                ),
                "ordinary_birth_count": (
                    successes if path == "ordinary_live_birth" else 0
                ),
            }
            if path not in {
                "generation_boundary_refill",
                "ordinary_live_birth",
            }:
                raise _fail("compiled reproduction path is unsupported")
            row = {
                "schema_version": SCHEMA_VERSION,
                "cell_id": cell_id,
                "generation": boundary_generation,
                "identity": dict(identity),
                "mutation_events": 0,
                "population": {"alive": alive, "capacity": capacity},
                "reproduction": reproduction,
            }
            row["row_sha256"] = canonical_sha256(row)
            output.append(row)
    return output


__all__ = [
    "InheritanceGenerationSeriesError",
    "SCHEMA_VERSION",
    "project_inheritance_generation_series_v1",
]
