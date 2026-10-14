"""Observer-only RECOMB-72 telemetry for native-resident GPU boundaries."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from engine.campaigns.canonical import canonical_bytes, canonical_sha256
from engine.evolution.recomb72_inheritance import (
    AUDITED_LOCI,
    FOUNDER_PATTERNS,
    INHERITANCE_MODES,
    LOCUS_BOUNDS,
    MUTATION_OFF_CONTRACT,
    NEUTRAL_PACKAGE,
    founder_packages,
    mode_sha256,
    registry_sha256,
)
from engine.evolution.recomb72_metrics import (
    GENERATION_METRICS_SCHEMA_VERSION,
    ancestry_concentration,
    locus_statistics,
    package_statistics,
    reproduction_concentration,
)
from engine.perf.recomb72_gpu_native import (
    AUDITED_LOCI as NATIVE_AUDITED_LOCI,
    SCHEMA_VERSION as NATIVE_CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
)
from simulation.chronicle.recorder import ScientificChronicleRecorder
from simulation.chronicle.schema import event_order_key, make_event
from simulation.chronicle.storage import read_json, sha256_file


BOUNDARY_SCHEMA_VERSION = "recomb72_gpu_generation_boundary_projection_v1"
TELEMETRY_SCHEMA_VERSION = "recomb72_gpu_telemetry_v1"
CHECKPOINT_CATALOG_SCHEMA_VERSION = "recomb72_gpu_checkpoint_catalog_v1"
CHECKPOINT_SCHEMA_VERSION = "recomb72_gpu_checkpoint_v1"
INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION = (
    "recomb72_gpu_inheritance_event_summary_v1"
)
CHRONICLE_RECEIPT_SCHEMA_VERSION = "recomb72_gpu_chronicle_receipt_v1"
SOURCE_COMPLETENESS = "native_resident_generation_boundary_complete"
EXPECTED_GENERATIONS = tuple(range(31))
CHECKPOINT_GENERATIONS = (0, 5, 10, 20, 30)
POPULATION_SIZE = 64
FOUNDER_COUNT = 64
# Native ancestry is stored as float32. A normalized row may therefore differ
# from one by one unit-scale float32 epsilon after repeated convex updates.
ANCESTRY_SUM_ABS_TOLERANCE = 2.0**-23
CHECKPOINT_TENSOR_NAMES = (
    "loci",
    "alive",
    "agent_ids",
    "ancestry",
    "next_id",
    "generation",
    "seeds",
    "modes",
    "patterns",
    "status",
)


class Recomb72GpuTelemetryError(ValueError):
    """A native GPU boundary cannot support truthful RECOMB-72 telemetry."""


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Recomb72GpuTelemetryError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Recomb72GpuTelemetryError(f"{name} must be an array")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Recomb72GpuTelemetryError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise Recomb72GpuTelemetryError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Recomb72GpuTelemetryError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise Recomb72GpuTelemetryError(f"{name} must be finite")
    return result


def _float32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).item())


def _digest(value: Any, *, name: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise Recomb72GpuTelemetryError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return result


def checkpoint_tensor_sha256(payload: Mapping[str, Any]) -> dict[str, str]:
    """Hash the exact tensors emitted by ``Recomb72GpuStateV1.checkpoint_payload``."""

    if payload.get("schema_version") != NATIVE_CHECKPOINT_PAYLOAD_SCHEMA_VERSION:
        raise Recomb72GpuTelemetryError("RECOMB-72 GPU checkpoint payload schema mismatch")
    if tuple(payload.get("locus_names") or ()) != tuple(NATIVE_AUDITED_LOCI):
        raise Recomb72GpuTelemetryError("RECOMB-72 GPU checkpoint locus schema mismatch")
    tensors = _object(payload.get("tensors"), name="checkpoint_payload.tensors")
    if set(tensors) != set(CHECKPOINT_TENSOR_NAMES):
        raise Recomb72GpuTelemetryError(
            "RECOMB-72 GPU checkpoint must contain exactly the native state tensors"
        )
    digests: dict[str, str] = {}
    for name in CHECKPOINT_TENSOR_NAMES:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise Recomb72GpuTelemetryError(
                f"checkpoint_payload.tensors.{name} must be a tensor"
            )
        canonical = tensor.detach().cpu().contiguous()
        descriptor = canonical_bytes(
            {
                "name": name,
                "dtype": str(canonical.dtype),
                "shape": list(canonical.shape),
            }
        )
        raw = canonical.view(torch.uint8).numpy().tobytes()
        digests[name] = hashlib.sha256(descriptor + b"\0" + raw).hexdigest()
    return digests


def checkpoint_state_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the complete native checkpoint state from its per-tensor digests."""

    return canonical_sha256(
        {
            "schema_version": NATIVE_CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
            "locus_names": list(NATIVE_AUDITED_LOCI),
            "tensor_sha256": checkpoint_tensor_sha256(payload),
        }
    )


def _identity(
    boundaries: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, int]:
    if not boundaries or len(boundaries) > len(EXPECTED_GENERATIONS):
        raise Recomb72GpuTelemetryError(
            "GPU telemetry requires a nonempty G0-G30 prefix"
        )
    ordered = sorted(boundaries, key=lambda row: row.get("generation", -1))
    generations = [row.get("generation") for row in ordered]
    if generations != list(range(len(ordered))):
        raise Recomb72GpuTelemetryError(
            "GPU telemetry must be a contiguous G0 prefix; complete telemetry "
            "is exactly G0 through G30"
        )
    identities = {
        (
            str(row.get("cell_id") or ""),
            str(row.get("mode") or ""),
            str(row.get("pattern") or ""),
            row.get("paired_seed"),
        )
        for row in ordered
    }
    if len(identities) != 1:
        raise Recomb72GpuTelemetryError(
            "GPU generation boundaries do not share one cell identity"
        )
    cell_id, mode, pattern, paired_seed = identities.pop()
    if not cell_id or Path(cell_id).name != cell_id:
        raise Recomb72GpuTelemetryError("GPU telemetry cell_id is unsafe")
    if mode not in INHERITANCE_MODES:
        raise Recomb72GpuTelemetryError("GPU telemetry inheritance mode is invalid")
    if pattern not in FOUNDER_PATTERNS:
        raise Recomb72GpuTelemetryError("GPU telemetry founder pattern is invalid")
    paired_seed = _integer(paired_seed, name="paired_seed")
    return cell_id, mode, pattern, paired_seed


def _canonical_state(
    boundary: Mapping[str, Any],
    *,
    generation: int,
) -> tuple[
    list[int],
    list[dict[str, float]],
    list[str],
    dict[int, dict[str, float]],
]:
    state = _object(boundary.get("state"), name=f"G{generation}.state")
    raw_agent_ids = _sequence(
        state.get("alive_agent_ids"), name=f"G{generation}.state.alive_agent_ids"
    )
    agent_ids = [
        _integer(value, name=f"G{generation}.alive_agent_ids[{index}]")
        for index, value in enumerate(raw_agent_ids)
    ]
    if len(agent_ids) != POPULATION_SIZE or len(set(agent_ids)) != POPULATION_SIZE:
        raise Recomb72GpuTelemetryError(
            f"G{generation} must expose exactly 64 unique living agents"
        )

    raw_loci = _object(state.get("locus_values"), name=f"G{generation}.locus_values")
    if set(raw_loci) != set(AUDITED_LOCI):
        raise Recomb72GpuTelemetryError(
            f"G{generation} must expose exactly the six audited loci"
        )
    columns: dict[str, list[float]] = {}
    for locus in AUDITED_LOCI:
        raw_values = _sequence(
            raw_loci[locus], name=f"G{generation}.locus_values.{locus}"
        )
        if len(raw_values) != POPULATION_SIZE:
            raise Recomb72GpuTelemetryError(
                f"G{generation}.{locus} must contain 64 values"
            )
        lower, upper = (_float32(value) for value in LOCUS_BOUNDS[locus])
        values = [
            _float32(_finite(value, name=f"G{generation}.{locus}[{index}]"))
            for index, value in enumerate(raw_values)
        ]
        if any(value < lower or value > upper for value in values):
            raise Recomb72GpuTelemetryError(
                f"G{generation}.{locus} contains an out-of-bounds value"
            )
        columns[locus] = values
    packages = [
        {locus: columns[locus][index] for locus in AUDITED_LOCI}
        for index in range(POPULATION_SIZE)
    ]

    raw_founders = _sequence(
        state.get("founder_ids"), name=f"G{generation}.state.founder_ids"
    )
    founder_ids = [str(value) for value in raw_founders]
    if (
        len(founder_ids) != FOUNDER_COUNT
        or len(set(founder_ids)) != FOUNDER_COUNT
        or any(not value for value in founder_ids)
    ):
        raise Recomb72GpuTelemetryError(
            f"G{generation} must expose one stable 64-founder basis"
        )
    raw_weights = _sequence(
        state.get("founder_ancestry_weights"),
        name=f"G{generation}.state.founder_ancestry_weights",
    )
    if len(raw_weights) != POPULATION_SIZE:
        raise Recomb72GpuTelemetryError(
            f"G{generation} must expose ancestry for all 64 living agents"
        )
    ancestry: dict[int, dict[str, float]] = {}
    for agent_index, (agent_id, raw_row) in enumerate(
        zip(agent_ids, raw_weights, strict=True)
    ):
        row = _sequence(
            raw_row, name=f"G{generation}.ancestry[{agent_index}]"
        )
        if len(row) != FOUNDER_COUNT:
            raise Recomb72GpuTelemetryError(
                f"G{generation} ancestry rows must use the complete 64-founder basis"
            )
        values = [
            _finite(value, name=f"G{generation}.ancestry[{agent_index}][{index}]")
            for index, value in enumerate(row)
        ]
        if any(value < 0.0 for value in values) or not math.isclose(
            math.fsum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=ANCESTRY_SUM_ABS_TOLERANCE,
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation} ancestry weights must be non-negative and sum to one"
            )
        ancestry[agent_id] = dict(zip(founder_ids, values, strict=True))
    return agent_ids, packages, founder_ids, ancestry


def _not_applicable_packages() -> dict[str, Any]:
    return {
        "population_count": POPULATION_SIZE,
        "intact_package_count": 0,
        "intact_package_fraction": 0.0,
        "recombinant_count": 0,
        "recombinant_fraction": 0.0,
        "neither_package_count": POPULATION_SIZE,
        "neither_package_fraction": 1.0,
        "not_applicable": True,
    }


def _float32_package(package: Mapping[str, float]) -> dict[str, float]:
    """Return the exact scalar encodings stored by the native float32 state."""

    return {
        locus: _float32(package[locus])
        for locus in AUDITED_LOCI
    }


def _ancestry_metrics(
    agent_ids: Sequence[int],
    packages: Sequence[Mapping[str, float]],
    founder_ids: Sequence[str],
    ancestry: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    concentration = ancestry_concentration(agent_ids, ancestry)
    first_group = set(founder_ids[: FOUNDER_COUNT // 2])
    grouped: dict[str, list[int]] = {"a": [], "b": [], "mixed": []}
    for index, agent_id in enumerate(agent_ids):
        weights = ancestry[agent_id]
        a_mass = sum(weight for founder, weight in weights.items() if founder in first_group)
        b_mass = 1.0 - a_mass
        group = "a" if a_mass > b_mass else "b" if b_mass > a_mass else "mixed"
        grouped[group].append(index)
    centroids: dict[str, dict[str, float]] = {}
    for group in ("a", "b"):
        if grouped[group]:
            centroids[group] = {
                locus: sum(packages[index][locus] for index in grouped[group])
                / len(grouped[group])
                for locus in AUDITED_LOCI
            }
    distance = 0.0
    if set(centroids) == {"a", "b"}:
        distance = math.sqrt(
            sum(
                (
                    (centroids["a"][locus] - centroids["b"][locus])
                    / (LOCUS_BOUNDS[locus][1] - LOCUS_BOUNDS[locus][0])
                )
                ** 2
                for locus in AUDITED_LOCI
            )
            / len(AUDITED_LOCI)
        )
    genealogy_persists = bool(grouped["a"] and grouped["b"])
    return {
        **concentration,
        "descendant_branch_counts": {
            group: len(grouped[group]) for group in ("a", "b", "mixed")
        },
        "descendant_branch_diversity": sum(bool(values) for values in grouped.values()),
        "effective_alive_lineages": concentration[
            "effective_founder_ancestry_count"
        ],
        "trait_distance_between_branches": float(distance),
        "package_distance_between_branches": float(distance),
        "genealogy_persists": genealogy_persists,
        "genetic_distinction_lost_threshold": 0.10,
        "genealogy_persists_while_genetic_distinction_disappears": (
            genealogy_persists and distance < 0.10
        ),
    }


def _population_contract(
    boundary: Mapping[str, Any], *, generation: int
) -> dict[str, int]:
    raw = _object(boundary.get("population"), name=f"G{generation}.population")
    required = (
        "before_cull",
        "eligible_parents",
        "selected_parent_events",
        "pairings",
        "children",
        "after_refill",
        "failed_refill_events",
        "ordinary_births",
        "mutation_events",
    )
    result = {
        name: _integer(raw.get(name), name=f"G{generation}.population.{name}")
        for name in required
    }
    expected = (
        {
            "before_cull": 64,
            "eligible_parents": 64,
            "selected_parent_events": 0,
            "pairings": 0,
            "children": 0,
            "after_refill": 64,
            "failed_refill_events": 0,
            "ordinary_births": 0,
            "mutation_events": 0,
        }
        if generation == 0
        else {
            "before_cull": 64,
            "eligible_parents": 32,
            "selected_parent_events": 64,
            "pairings": 32,
            "children": 32,
            "after_refill": 64,
            "failed_refill_events": 0,
            "ordinary_births": 0,
            "mutation_events": 0,
        }
    )
    if result != expected:
        raise Recomb72GpuTelemetryError(
            f"G{generation} population boundary violates the neutral cull/refill contract"
        )
    return result


def _integrity_contract(
    boundary: Mapping[str, Any],
    *,
    generation: int,
    mode: str,
) -> dict[str, Any]:
    raw = _object(boundary.get("integrity"), name=f"G{generation}.integrity")
    expected = {
        "genotype_blind_cull": True,
        "all_survivors_eligible": True,
        "actual_ticks_per_generation": 80 if generation > 0 else 0,
        "no_fallback": True,
        "native_gpu_owner": True,
    }
    if any(raw.get(name) != value for name, value in expected.items()):
        raise Recomb72GpuTelemetryError(
            f"G{generation} GPU integrity contract is incomplete"
        )
    result = {
        **expected,
        "registry_sha256": registry_sha256(),
        "mode_sha256": mode_sha256(mode),
        "mutation_off_contract": copy.deepcopy(MUTATION_OFF_CONTRACT),
        "source_completeness": SOURCE_COMPLETENESS,
    }
    if generation == 0:
        result["founder_split"] = {"a": 32, "b": 32}
    return result


def _inheritance_event_summary(
    boundary: Mapping[str, Any],
    *,
    generation: int,
) -> dict[str, Any]:
    raw = _object(
        boundary.get("inheritance_events"),
        name=f"G{generation}.inheritance_events",
    )
    if (
        raw.get("schema_version") != INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION
        or raw.get("source") != "native_gpu_event_tensors"
    ):
        raise Recomb72GpuTelemetryError(
            f"G{generation} inheritance events lack GPU-derived authority"
        )
    if generation == 0:
        expected = {
            "schema_version": INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION,
            "source": "native_gpu_event_tensors",
            "event_count": 0,
            "loci": {},
            "full_package_preservation_count": 0,
            "full_package_preservation_fraction": 0.0,
            "not_applicable": True,
        }
        if dict(raw) != expected:
            raise Recomb72GpuTelemetryError(
                "G0 inheritance events must be exactly not applicable"
            )
        return expected

    event_count = _integer(
        raw.get("event_count"),
        name=f"G{generation}.inheritance_events.event_count",
    )
    if event_count != 32 or raw.get("not_applicable", False) is not False:
        raise Recomb72GpuTelemetryError(
            f"G{generation} inheritance events must cover exactly 32 children"
        )
    raw_loci = _object(
        raw.get("loci"), name=f"G{generation}.inheritance_events.loci"
    )
    if set(raw_loci) != set(AUDITED_LOCI):
        raise Recomb72GpuTelemetryError(
            f"G{generation} inheritance events must cover all six loci"
        )
    distance_names = (
        "parent_pair_distance_mean",
        "child_to_parent_a_distance_mean",
        "child_to_parent_b_distance_mean",
        "child_to_parental_midpoint_distance_mean",
    )
    count_names = ("exact_parent_count", "strictly_between_count")
    loci: dict[str, dict[str, Any]] = {}
    for locus in AUDITED_LOCI:
        item = _object(
            raw_loci[locus],
            name=f"G{generation}.inheritance_events.loci.{locus}",
        )
        if set(item) != set(distance_names + count_names):
            raise Recomb72GpuTelemetryError(
                f"G{generation}.{locus} inheritance summary fields are incomplete"
            )
        distances = {
            name: _finite(
                item.get(name),
                name=f"G{generation}.{locus}.{name}",
            )
            for name in distance_names
        }
        if any(value < 0.0 or value > 1.0 for value in distances.values()):
            raise Recomb72GpuTelemetryError(
                f"G{generation}.{locus} normalized inheritance distance is out of range"
            )
        counts = {
            name: _integer(
                item.get(name),
                name=f"G{generation}.{locus}.{name}",
            )
            for name in count_names
        }
        if (
            any(value > event_count for value in counts.values())
            or sum(counts.values()) > event_count
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation}.{locus} inheritance counts exceed 32 events"
            )
        loci[locus] = {**distances, **counts}
    preservation_count = _integer(
        raw.get("full_package_preservation_count"),
        name=f"G{generation}.full_package_preservation_count",
    )
    preservation_fraction = _finite(
        raw.get("full_package_preservation_fraction"),
        name=f"G{generation}.full_package_preservation_fraction",
    )
    if (
        preservation_count > event_count
        or preservation_fraction < 0.0
        or preservation_fraction > 1.0
        or not math.isclose(
            preservation_fraction,
            preservation_count / event_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise Recomb72GpuTelemetryError(
            f"G{generation} full-package preservation is inconsistent"
        )
    return {
        "schema_version": INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION,
        "source": "native_gpu_event_tensors",
        "event_count": event_count,
        "loci": loci,
        "full_package_preservation_count": preservation_count,
        "full_package_preservation_fraction": preservation_fraction,
    }


def mechanics_digest_chain(
    boundaries: Sequence[Mapping[str, Any]],
) -> str:
    """Hash only provider-owned mechanics digests, independent of reporting."""

    ordered = sorted(
        [_object(row, name="boundary") for row in boundaries],
        key=lambda row: row.get("generation", -1),
    )
    _identity(ordered)
    values = [
        {
            "generation": row["generation"],
            "mechanics_digest_sha256": _digest(
                row.get("mechanics_digest_sha256"),
                name=f"G{row['generation']}.mechanics_digest_sha256",
            ),
        }
        for row in ordered
    ]
    return canonical_sha256(values)


def build_generation_rows(
    boundaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project immutable GPU generation boundaries into frozen scientific rows."""

    canonical_boundaries = [
        _object(row, name=f"boundaries[{index}]")
        for index, row in enumerate(boundaries)
    ]
    cell_id, mode, pattern, paired_seed = _identity(canonical_boundaries)
    ordered = sorted(canonical_boundaries, key=lambda row: row["generation"])
    first_founders: list[str] | None = None
    initial_variance: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    frozen_package_a, frozen_package_b = founder_packages(pattern)
    package_a = _float32_package(frozen_package_a)
    package_b = _float32_package(frozen_package_b)
    float32_bounds = {
        locus: tuple(_float32(value) for value in LOCUS_BOUNDS[locus])
        for locus in AUDITED_LOCI
    }
    float32_neutral = _float32_package(NEUTRAL_PACKAGE)
    for boundary in ordered:
        generation = _integer(boundary.get("generation"), name="generation")
        if boundary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
            raise Recomb72GpuTelemetryError(
                f"G{generation} GPU boundary schema is invalid"
            )
        mechanics_digest = _digest(
            boundary.get("mechanics_digest_sha256"),
            name=f"G{generation}.mechanics_digest_sha256",
        )
        agent_ids, packages, founder_ids, ancestry = _canonical_state(
            boundary, generation=generation
        )
        if first_founders is None:
            first_founders = founder_ids
        elif founder_ids != first_founders:
            raise Recomb72GpuTelemetryError(
                f"G{generation} changed the ordered 64-founder basis"
            )
        loci: dict[str, dict[str, Any]] = {}
        for locus in AUDITED_LOCI:
            values = [package[locus] for package in packages]
            stats = locus_statistics(
                values,
                lower_bound=float32_bounds[locus][0],
                upper_bound=float32_bounds[locus][1],
                neutral_value=float32_neutral[locus],
                generation_zero_variance=(
                    None if generation == 0 else initial_variance[locus]
                ),
            )
            if generation == 0:
                initial_variance[locus] = float(stats["variance_population"])
            loci[locus] = stats
        packages_row = (
            package_statistics(
                packages,
                package_a=package_a,
                package_b=package_b,
                bounds=float32_bounds,
            )
            if pattern == "correlated_multi_trait_packages"
            else _not_applicable_packages()
        )
        contributions = _object(
            boundary.get("parent_contribution_counts", {}),
            name=f"G{generation}.parent_contribution_counts",
        )
        reproduction = reproduction_concentration(contributions)
        population = _population_contract(boundary, generation=generation)
        if (
            reproduction["parent_contribution_total"]
            != population["selected_parent_events"]
            or reproduction["contributing_parent_count"]
            > population["eligible_parents"]
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation} parent contributions contradict the population boundary"
            )
        integrity = _integrity_contract(
            boundary, generation=generation, mode=mode
        )
        inheritance_events = _inheritance_event_summary(
            boundary, generation=generation
        )
        if generation == 0:
            expected_packages = [package_a] * 32 + [package_b] * 32
            if packages != expected_packages:
                raise Recomb72GpuTelemetryError(
                    "G0 founder packages are not the exact frozen 32/32 baseline"
                )
            one_hot = [
                [
                    ancestry[agent_id][founder]
                    for founder in founder_ids
                ]
                for agent_id in agent_ids
            ]
            if any(
                sum(value == 1.0 for value in weights) != 1
                or any(value not in {0.0, 1.0} for value in weights)
                for weights in one_hot
            ):
                raise Recomb72GpuTelemetryError(
                    "G0 must expose exact one-hot ancestry for all 64 founders"
                )
            founder_uses = [
                sum(weights[index] == 1.0 for weights in one_hot)
                for index in range(FOUNDER_COUNT)
            ]
            if founder_uses != [1] * FOUNDER_COUNT:
                raise Recomb72GpuTelemetryError(
                    "G0 must assign every founder to exactly one living agent"
                )
        row = {
            "schema_version": GENERATION_METRICS_SCHEMA_VERSION,
            "authority": "native_resident_gpu_generation_boundary_projection",
            "cell_id": cell_id,
            "generation": generation,
            "mode": mode,
            "pattern": pattern,
            "paired_seed": paired_seed,
            "population": population,
            "reproduction": reproduction,
            "ancestry": _ancestry_metrics(
                agent_ids, packages, founder_ids, ancestry
            ),
            "loci": loci,
            "packages": packages_row,
            "inheritance_events": inheritance_events,
            "integrity": integrity,
            "mechanics_digest_sha256": mechanics_digest,
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
    return rows


def validate_generation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless rows are an exact, self-consistent G0-G30 series."""

    if len(rows) != len(EXPECTED_GENERATIONS):
        raise Recomb72GpuTelemetryError("generation rows must cover exactly G0-G30")
    identities: set[tuple[Any, ...]] = set()
    for expected_generation, raw in zip(EXPECTED_GENERATIONS, rows, strict=True):
        row = _object(raw, name=f"rows[{expected_generation}]")
        if (
            row.get("schema_version") != GENERATION_METRICS_SCHEMA_VERSION
            or row.get("generation") != expected_generation
        ):
            raise Recomb72GpuTelemetryError(
                "generation rows are not the exact ordered G0-G30 series"
            )
        recorded = _digest(
            row.get("row_sha256"), name=f"G{expected_generation}.row_sha256"
        )
        payload = dict(row)
        del payload["row_sha256"]
        if canonical_sha256(payload) != recorded:
            raise Recomb72GpuTelemetryError(
                f"G{expected_generation} generation row digest mismatch"
            )
        if set(_object(row.get("loci"), name="loci")) != set(AUDITED_LOCI):
            raise Recomb72GpuTelemetryError(
                f"G{expected_generation} lacks six-locus telemetry"
            )
        _inheritance_event_summary(
            {"inheritance_events": row.get("inheritance_events")},
            generation=expected_generation,
        )
        ancestry = _object(row.get("ancestry"), name="ancestry")
        if len(_object(ancestry.get("founder_ancestry_shares"), name="shares")) > 64:
            raise Recomb72GpuTelemetryError("ancestry exceeds the 64-founder basis")
        identities.add(
            (
                row.get("cell_id"),
                row.get("mode"),
                row.get("pattern"),
                row.get("paired_seed"),
            )
        )
    if len(identities) != 1:
        raise Recomb72GpuTelemetryError("generation row identity changed")


def build_checkpoint_catalog(
    boundaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and verify the exact G0/G5/G10/G20/G30 RECOMB checkpoint catalog."""

    canonical_boundaries = [
        _object(row, name=f"boundaries[{index}]")
        for index, row in enumerate(boundaries)
    ]
    cell_id, _mode, _pattern, _seed = _identity(canonical_boundaries)
    by_generation = {int(row["generation"]): row for row in canonical_boundaries}
    checkpoints: dict[str, dict[str, Any]] = {}
    for generation in CHECKPOINT_GENERATIONS:
        boundary = by_generation[generation]
        checkpoint = _object(
            boundary.get("checkpoint"), name=f"G{generation}.checkpoint"
        )
        path = Path(str(checkpoint.get("checkpoint_path") or "")).expanduser()
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("payload_schema_version")
            != NATIVE_CHECKPOINT_PAYLOAD_SCHEMA_VERSION
            or checkpoint.get("generation") != generation
            or not path.is_absolute()
            or checkpoint.get("fresh_owner_cuda_restore") is not True
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint is not a RECOMB-72 GPU checkpoint v1"
            )
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint payload does not exist"
            )
        try:
            payload = torch.load(
                resolved_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint payload cannot be loaded"
            ) from exc
        payload = _object(payload, name=f"G{generation}.checkpoint_payload")
        tensor_sha256 = checkpoint_tensor_sha256(payload)
        declared_tensor_sha256 = _object(
            checkpoint.get("tensor_sha256"),
            name=f"G{generation}.checkpoint.tensor_sha256",
        )
        if set(declared_tensor_sha256) != set(CHECKPOINT_TENSOR_NAMES) or any(
            _digest(
                declared_tensor_sha256.get(name),
                name=f"G{generation}.checkpoint.tensor_sha256.{name}",
            )
            != tensor_sha256[name]
            for name in CHECKPOINT_TENSOR_NAMES
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint tensor digest mismatch"
            )
        state_sha256 = checkpoint_state_sha256(payload)
        if (
            _digest(
                checkpoint.get("state_sha256"),
                name=f"G{generation}.checkpoint.state_sha256",
            )
            != state_sha256
        ):
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint state digest mismatch"
            )
        generation_tensor = payload["tensors"]["generation"].reshape(-1)
        if generation_tensor.numel() != 1 or int(generation_tensor.item()) != generation:
            raise Recomb72GpuTelemetryError(
                f"G{generation} checkpoint payload generation mismatch"
            )
        checkpoints[str(generation)] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "payload_schema_version": NATIVE_CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
            "generation": generation,
            "cell_id": cell_id,
            "checkpoint_path": str(resolved_path),
            "tensor_sha256": tensor_sha256,
            "state_sha256": state_sha256,
            "source_projection_sha256": _digest(
                checkpoint.get("source_projection_sha256"),
                name=f"G{generation}.source_projection_sha256",
            ),
            "fresh_owner_cuda_restore": True,
            "mechanics_digest_sha256": _digest(
                boundary.get("mechanics_digest_sha256"),
                name=f"G{generation}.mechanics_digest_sha256",
            ),
        }
    return {
        "schema_version": CHECKPOINT_CATALOG_SCHEMA_VERSION,
        "cell_id": cell_id,
        "generations": list(CHECKPOINT_GENERATIONS),
        "checkpoints": checkpoints,
        "catalog_sha256": canonical_sha256(checkpoints),
    }


def build_gpu_telemetry(
    boundaries: Sequence[Mapping[str, Any]],
    *,
    reporting_enabled: bool,
) -> dict[str, Any]:
    """Return telemetry while keeping mechanics identity independent of reporting."""

    if type(reporting_enabled) is not bool:
        raise Recomb72GpuTelemetryError("reporting_enabled must be boolean")
    mechanics_chain = mechanics_digest_chain(boundaries)
    computed_rows = build_generation_rows(boundaries)
    validate_generation_rows(computed_rows)
    rows = computed_rows if reporting_enabled else []
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "reporting_enabled": reporting_enabled,
        "mechanics_digest_chain_sha256": mechanics_chain,
        "generation_metrics_rows": rows,
        "checkpoint_catalog": build_checkpoint_catalog(boundaries),
    }


def materialize_scientific_chronicle(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_root: Path,
    campaign_id: str,
    cell_id: str,
    mechanics_digest_chain_sha256: str,
) -> dict[str, Any]:
    """Materialize one terminal-formatted Chronicle from exact native boundaries."""

    validate_generation_rows(rows)
    if any(row.get("cell_id") != cell_id for row in rows):
        raise Recomb72GpuTelemetryError("Chronicle rows do not match cell_id")
    mechanics_chain = _digest(
        mechanics_digest_chain_sha256, name="mechanics_digest_chain_sha256"
    )
    root = Path(run_root).expanduser().resolve()
    sim_cfg = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile="scientific_full",
        allow_large_output_profile=True,
        generations=30,
        campaign_id=str(campaign_id),
        chronicle_run_id=cell_id,
    )
    recorder = ScientificChronicleRecorder(
        SimpleNamespace(sim_cfg=sim_cfg, run_dir=root)
    )
    summaries = []
    for row in rows:
        population = row["population"]
        ancestry = row["ancestry"]
        summaries.append(
            {
                "generation": row["generation"],
                "alive": population["after_refill"],
                "births": population["children"],
                "deaths": (
                    population["before_cull"] - population["eligible_parents"]
                ),
                "failed_births": population["failed_refill_events"],
                "patch_metrics": {
                    "lineage_effective_alive_lineages": ancestry[
                        "effective_alive_lineages"
                    ],
                    "recomb72_top1_founder_ancestry_share": ancestry[
                        "top1_founder_ancestry_share"
                    ],
                },
                "chronicle_source_component": (
                    "engine.campaigns.recomb72_gpu_telemetry"
                ),
                "chronicle_source_record_ref": (
                    f"generation_metrics_rows:G{row['generation']}:"
                    f"{row['row_sha256']}"
                ),
            }
        )
    recorder.observe_native_terminal_summaries(summaries)
    recorder_state = recorder.export_state()
    chronicle_events = [
        dict(event) for event in recorder_state.get("events", [])
    ]
    seen_event_ids = {str(event["event_id"]) for event in chronicle_events}
    for row in rows:
        event = make_event(
            run_id=cell_id,
            campaign_id=campaign_id,
            generation=int(row["generation"]),
            step=-1,
            phase_order=900,
            intra_step_sequence=0,
            event_type="recomb72_native_generation_boundary",
            event_category="inheritance",
            subject_type="population",
            subject_id=cell_id,
            scientific_importance=5,
            historical_importance=4,
            confidence_class="measured",
            causality_class="direct_mechanism",
            source_component="engine.campaigns.recomb72_gpu_telemetry",
            source_record_refs=(
                f"generation_metrics_rows:G{row['generation']}:{row['row_sha256']}",
            ),
            measurements={
                "generation_metrics": row,
                "mechanics_digest_chain_sha256": mechanics_chain,
            },
            tags=("recomb72", "native_gpu", "full_profile"),
        )
        if event["event_id"] in seen_event_ids:
            raise Recomb72GpuTelemetryError("duplicate Chronicle event identity")
        chronicle_events.append(event)
        seen_event_ids.add(event["event_id"])
    chronicle_events.sort(key=event_order_key)
    recorder_state["events"] = chronicle_events
    recorder_state["source_completeness"] = SOURCE_COMPLETENESS
    recorder.import_state(recorder_state)
    recorder.finalize(status="complete")
    chronicle_root = root / "chronicle"
    manifest_path = chronicle_root / "chronicle_manifest.json"
    summary_path = chronicle_root / "scientific_chronicle_summary.json"
    if not recorder.finalized or not manifest_path.is_file() or not summary_path.is_file():
        raise Recomb72GpuTelemetryError("GPU Chronicle did not finalize")
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    required = {
        "run_id": cell_id,
        "campaign_id": str(campaign_id),
        "profile": "scientific_full",
        "finalized": True,
        "source_completeness": SOURCE_COMPLETENESS,
    }
    if any(manifest.get(name) != value for name, value in required.items()) or any(
        summary.get(name) != value for name, value in required.items()
    ):
        raise Recomb72GpuTelemetryError("GPU Chronicle identity or authority mismatch")
    if summary.get("last_generation") != 30 or summary.get("write_failures") != 0:
        raise Recomb72GpuTelemetryError("GPU Chronicle is incomplete")
    return {
        "schema_version": CHRONICLE_RECEIPT_SCHEMA_VERSION,
        "cell_id": cell_id,
        "campaign_id": str(campaign_id),
        "chronicle_root": str(chronicle_root),
        "chronicle_manifest_sha256": sha256_file(manifest_path),
        "mechanics_digest_chain_sha256": mechanics_chain,
        "profile": "scientific_full",
        "source_completeness": SOURCE_COMPLETENESS,
        "finalized": True,
        "generation_count": len(rows),
        "write_failures": 0,
    }


__all__ = [
    "BOUNDARY_SCHEMA_VERSION",
    "CHECKPOINT_CATALOG_SCHEMA_VERSION",
    "CHECKPOINT_GENERATIONS",
    "CHRONICLE_RECEIPT_SCHEMA_VERSION",
    "EXPECTED_GENERATIONS",
    "Recomb72GpuTelemetryError",
    "SOURCE_COMPLETENESS",
    "TELEMETRY_SCHEMA_VERSION",
    "build_checkpoint_catalog",
    "build_generation_rows",
    "build_gpu_telemetry",
    "materialize_scientific_chronicle",
    "mechanics_digest_chain",
    "validate_generation_rows",
]
