"""Fail-closed materialization of full-stack WRC science requests."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from math import prod
from itertools import product
from pathlib import Path
from typing import Any

from .canonical import canonical_sha256
from .contracts import RuntimeProfileV1
from .wrc_variability import (
    WRCParameterVariabilityEnvelopeV1,
    WRCVariabilityEnvelopeError,
    build_wrc_parameter_variability_envelope_v1,
)


FULL_STACK_OVERLAYS_V1 = ("eed_resource_impact", "jungle", "wdl_delayed_hazard")
RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1 = (
    (
        public_runtime_path('evidence_raw/'),
        public_runtime_path('knobs_discovery_evidence_raw/'),
    ),
)


class WRCScienceRequestError(ValueError):
    """The requested WRC dial positions are outside the qualified capability."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relocate_verified_runtime_artifacts_v1(spec: dict[str, Any]) -> None:
    """Rewrite only missing runtime paths whose relocated bytes retain the pin.

    The artifact SHA-256 remains the authority.  This permits a WRC request to
    survive the runtime evidence-root relocation without accepting a new file
    or changing any experiment-variable value.
    """

    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    relocated = dict(artifacts)
    for artifact_id, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            continue
        declared = Path(str(raw.get("path") or ""))
        digest = raw.get("sha256")
        if (
            not declared.is_absolute()
            or declared.is_symlink()
            or declared.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            continue
        rendered = str(declared)
        for source_root, destination_root in RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1:
            if not rendered.startswith(source_root):
                continue
            candidate = Path(destination_root + rendered[len(source_root) :])
            if (
                not candidate.is_symlink()
                and candidate.is_file()
                and _file_sha256(candidate) == digest
            ):
                relocated[str(artifact_id)] = {**dict(raw), "path": str(candidate)}
            break
    spec["artifacts"] = relocated


def _relocate_runtime_output_root_v1(spec: dict[str, Any]) -> None:
    """Keep new WRC evidence writes beside a verified relocated package."""

    output = spec.get("output")
    if not isinstance(output, Mapping):
        return
    declared = Path(str(output.get("runtime_root") or ""))
    if not declared.is_absolute() or declared.is_symlink() or declared.exists():
        return
    rendered = str(declared)
    for source_root, destination_root in RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1:
        if not rendered.startswith(source_root):
            continue
        candidate = Path(destination_root + rendered[len(source_root) :])
        if candidate.parent.is_dir() and not candidate.parent.is_symlink():
            spec["output"] = {**dict(output), "runtime_root": str(candidate)}
        return


def materialize_full_stack_wrc_science_spec_v1(
    base_spec: Mapping[str, Any],
    *,
    profile: RuntimeProfileV1,
    axes: Mapping[str, Sequence[Any]],
    configuration_rows: Sequence[Mapping[str, Any]] | None = None,
    seeds: Mapping[str, Sequence[int]] | None = None,
    fixed_context_axes: Mapping[str, Sequence[Any]] | None = None,
    generations: int | None = None,
) -> tuple[dict[str, Any], WRCParameterVariabilityEnvelopeV1]:
    """Materialize approved Cartesian dial positions without selecting a receipt.

    This is deliberately a planning boundary.  The resulting spec still must
    pass compiler validation, activation preflight, route-receipt matching and
    current-device admission before it can run.
    """

    requested_rows = () if configuration_rows is None else tuple(configuration_rows)
    if bool(axes) == bool(requested_rows):
        raise WRCScienceRequestError(
            "SCIENCE_DESIGN_INVALID: supply exactly one of axes or configuration_rows"
        )
    normalized_rows: tuple[dict[str, Any], ...] = ()
    if requested_rows:
        rows: list[dict[str, Any]] = []
        row_ids: set[str] = set()
        control_sets: set[tuple[str, ...]] = set()
        vector_digests: set[str] = set()
        for index, raw_row in enumerate(requested_rows):
            if not isinstance(raw_row, Mapping):
                raise WRCScienceRequestError(
                    f"SCIENCE_DESIGN_INVALID: configuration_rows[{index}] must be an object"
                )
            unknown = sorted(set(raw_row) - {"id", "controls"})
            row_id = raw_row.get("id")
            controls = raw_row.get("controls")
            if (
                unknown
                or not isinstance(row_id, str)
                or not row_id
                or row_id in row_ids
                or not isinstance(controls, Mapping)
                or not controls
                or any(not isinstance(name, str) or not name for name in controls)
            ):
                raise WRCScienceRequestError(
                    f"SCIENCE_DESIGN_INVALID: configuration_rows[{index}] is malformed"
                )
            canonical_controls = dict(sorted(controls.items()))
            control_sets.add(tuple(canonical_controls))
            vector_digest = canonical_sha256(canonical_controls)
            if vector_digest in vector_digests:
                raise WRCScienceRequestError(
                    "SCIENCE_DESIGN_INVALID: configuration rows must be distinct"
                )
            row_ids.add(row_id)
            vector_digests.add(vector_digest)
            rows.append({"id": row_id, "controls": canonical_controls})
        if len(control_sets) != 1:
            raise WRCScienceRequestError(
                "SCIENCE_DESIGN_INVALID: configuration rows must declare the same controls"
            )
        normalized_rows = tuple(rows)
        axes = {
            name: tuple(
                dict.fromkeys(row["controls"][name] for row in normalized_rows)
            )
            for name in control_sets.pop()
        }
    spec = deepcopy(dict(base_spec))
    _relocate_verified_runtime_artifacts_v1(spec)
    _relocate_runtime_output_root_v1(spec)
    world = spec.get("world")
    if not isinstance(world, Mapping) or world.get("base") != "patch" or tuple(
        world.get("overlays", ())
    ) != FULL_STACK_OVERLAYS_V1:
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: not the full-stack route")
    names = set(axes)
    for name, values in axes.items():
        contract = profile.parameter_contracts.get(name)
        if (
            contract is None
            or contract.variability_semantics != "experiment_variable"
            or not contract.per_cell
            or contract.qualification_affecting
        ):
            raise WRCScienceRequestError(f"PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: {name}")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise WRCScienceRequestError(f"PARAMETER_VALUE_OUT_OF_RANGE: {name}")
        if any(not contract.accepts(value) for value in values):
            raise WRCScienceRequestError(f"PARAMETER_VALUE_OUT_OF_RANGE: {name}")
    try:
        envelope = build_wrc_parameter_variability_envelope_v1(
            profile,
            world_base="patch",
            world_overlays=FULL_STACK_OVERLAYS_V1,
            requested_parameter_names=names,
        )
    except WRCVariabilityEnvelopeError as exc:
        raise WRCScienceRequestError(f"PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: {exc}") from exc
    if envelope is None:
        raise WRCScienceRequestError("PARAMETER_NOT_IN_QUALIFIED_ENVELOPE")
    matrix = spec.get("matrix")
    execution = spec.get("execution")
    if not isinstance(matrix, Mapping) or not isinstance(execution, Mapping):
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: malformed base spec")
    dimensions = [dict(row) for row in matrix.get("dimensions", ())]
    parameter_bindings = dict(matrix.get("parameter_bindings", {}))
    seed_bindings = dict(matrix.get("seed_bindings", {}))
    existing_dimensions = {
        str(parameter): dimension for dimension, parameter in parameter_bindings.items()
    }
    used_dimension_ids = {
        str(row.get("id"))
        for row in dimensions
        if isinstance(row.get("id"), str)
    }
    next_axis_index = 0
    for name, values in sorted(axes.items()):
        dimension_id = existing_dimensions.get(name)
        if dimension_id is None:
            while f"wrc_axis_{next_axis_index:02d}" in used_dimension_ids:
                next_axis_index += 1
            dimension_id = f"wrc_axis_{next_axis_index:02d}"
            next_axis_index += 1
        used_dimension_ids.add(dimension_id)
        replacement = {"id": dimension_id, "values": list(values)}
        matching = [
            position
            for position, row in enumerate(dimensions)
            if row.get("id") == dimension_id
        ]
        if matching:
            dimensions[matching[0]] = replacement
        else:
            dimensions.append(replacement)
        parameter_bindings[dimension_id] = name
    requested_fixed_context = (
        {} if fixed_context_axes is None else dict(fixed_context_axes)
    )
    unsupported_context = sorted(
        set(requested_fixed_context) - set(existing_dimensions)
    )
    if unsupported_context:
        raise WRCScienceRequestError(
            "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: fixed context controls must "
            f"already be base dimensions: {unsupported_context!r}"
        )
    for name, values in sorted(requested_fixed_context.items()):
        contract = profile.parameter_contracts.get(name)
        if (
            contract is None
            or not contract.per_cell
            or contract.qualification_affecting
            or not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(not contract.accepts(value) for value in values)
        ):
            raise WRCScienceRequestError(
                f"PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: fixed context {name}"
            )
        dimension_id = existing_dimensions[name]
        for position, row in enumerate(dimensions):
            if row.get("id") == dimension_id:
                dimensions[position] = {"id": dimension_id, "values": list(values)}
                break
    seed_dimensions = {str(owner): dimension for dimension, owner in seed_bindings.items()}
    requested_seeds = {} if seeds is None else dict(seeds)
    unknown_owners = sorted(set(requested_seeds) - set(seed_dimensions))
    if unknown_owners:
        raise WRCScienceRequestError(
            f"PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: seed owners {unknown_owners!r}"
        )
    for owner, values in sorted(requested_seeds.items()):
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise WRCScienceRequestError(f"PARAMETER_VALUE_OUT_OF_RANGE: seed {owner}")
        dimension_id = seed_dimensions[owner]
        for position, row in enumerate(dimensions):
            if row.get("id") == dimension_id:
                dimensions[position] = {"id": dimension_id, "values": list(values)}
                break
    include: list[dict[str, Any]] = []
    if normalized_rows:
        row_controls = set(normalized_rows[0]["controls"])
        # Existing base axes are compatibility defaults unless the science
        # request names them.  Keeping their first sealed value prevents an
        # accidental Cartesian expansion that was not present in the request.
        for position, row in enumerate(dimensions):
            dimension_id = str(row.get("id"))
            parameter = parameter_bindings.get(dimension_id)
            owner = seed_bindings.get(dimension_id)
            if parameter not in row_controls and parameter not in requested_fixed_context:
                if parameter is not None:
                    dimensions[position] = {
                        "id": dimension_id,
                        "values": [list(row["values"])[0]],
                    }
            if owner is not None and owner not in requested_seeds:
                dimensions[position] = {
                    "id": dimension_id,
                    "values": [list(row["values"])[0]],
                }
        seed_value_sets = [
            (dimension_id, tuple(row["values"]))
            for row in dimensions
            for dimension_id in (str(row["id"]),)
            if dimension_id in seed_bindings
        ]
        for configuration in normalized_rows:
            for seed_values in product(*(values for _name, values in seed_value_sets)):
                selector: dict[str, Any] = {}
                seed_selector = dict(
                    zip(
                        (name for name, _values in seed_value_sets),
                        seed_values,
                        strict=True,
                    )
                )
                for row in dimensions:
                    dimension_id = str(row["id"])
                    parameter = parameter_bindings.get(dimension_id)
                    if parameter in configuration["controls"]:
                        selector[dimension_id] = configuration["controls"][parameter]
                    elif dimension_id in seed_selector:
                        selector[dimension_id] = seed_selector[dimension_id]
                    else:
                        values = list(row["values"])
                        if len(values) != 1:
                            raise WRCScienceRequestError(
                                "SCIENCE_DESIGN_INVALID: unrequested base dimension remains variable"
                            )
                        selector[dimension_id] = values[0]
                include.append(selector)
        if len({canonical_sha256(row) for row in include}) != len(include):
            raise WRCScienceRequestError(
                "SCIENCE_DESIGN_INVALID: row and seed expansion is not unique"
            )
    base_cell_limit = matrix.get("max_cells")
    if type(base_cell_limit) is not int or base_cell_limit < 1:
        raise WRCScienceRequestError(
            "ROUTE_CAPABILITY_CHANGED: base matrix max_cells is invalid"
        )
    cells = prod(len(row["values"]) for row in dimensions)
    if normalized_rows:
        cells = len(include)
    if cells > base_cell_limit:
        raise WRCScienceRequestError(
            "RESOURCE_NOT_ADMITTED: request exceeds base matrix max_cells "
            f"{base_cell_limit}"
        )
    matrix = dict(matrix)
    matrix["dimensions"] = dimensions
    matrix["parameter_bindings"] = parameter_bindings
    matrix["include"] = include
    matrix["exclude"] = []
    matrix["expected_cells"] = cells
    matrix["max_cells"] = cells
    spec["matrix"] = matrix
    execution = dict(execution)
    tick_native = (
        execution.get("execution_route_identity")
        == "tick_native_execution_route_v1"
        or "total_ticks" in execution
    )
    if generations is not None and tick_native:
        raise WRCScienceRequestError(
            "ROUTE_CAPABILITY_CHANGED: tick-native WRC requests forbid generation authority"
        )
    if generations is not None:
        if type(generations) is not int or not 3 <= generations <= 128:
            raise WRCScienceRequestError(
                "PARAMETER_VALUE_OUT_OF_RANGE: generations must be an integer in [3, 128]"
            )
        execution["generations"] = generations
        checkpoints = dict(spec.get("checkpoints", {}))
        checkpoints["milestones"] = [generations]
        spec["checkpoints"] = checkpoints
    if tick_native:
        total_ticks = execution.get("total_ticks")
        if type(total_ticks) is not int or total_ticks < 1:
            raise WRCScienceRequestError(
                "ROUTE_CAPABILITY_CHANGED: tick-native WRC base lacks total_ticks"
            )
        forbidden_time_fields = sorted(
            {"generation", "generations"} & set(execution)
        )
        if forbidden_time_fields:
            raise WRCScienceRequestError(
                "ROUTE_CAPABILITY_CHANGED: tick-native WRC base contains generation authority "
                f"{forbidden_time_fields!r}"
            )
    execution["batch"] = {"mode": "automatic", "maximum": cells}
    spec["execution"] = execution
    request_identity: dict[str, Any]
    if tick_native:
        request_identity = {
            "schema_version": "eiel_wrc_science_request_v2",
            "time_authority": "absolute_tick_v1",
            "total_ticks": execution["total_ticks"],
            "checkpoint_ticks": list(execution.get("checkpoint_ticks", ())),
            "base_campaign_id": spec.get("campaign_id"),
            "axes": {name: list(values) for name, values in sorted(axes.items())},
            "configuration_rows": [dict(row) for row in normalized_rows],
            "seeds": {
                owner: list(values)
                for owner, values in sorted(requested_seeds.items())
            },
            "fixed_context": {
                name: list(values)
                for name, values in sorted(requested_fixed_context.items())
            },
        }
    else:
        request_identity = {
            "schema_version": "eiel_wrc_science_request_v1",
            "base_campaign_id": spec.get("campaign_id"),
            "axes": {name: list(values) for name, values in sorted(axes.items())},
            "configuration_rows": [dict(row) for row in normalized_rows],
            "seeds": {owner: list(values) for owner, values in sorted(requested_seeds.items())},
            "fixed_context": {
                name: list(values)
                for name, values in sorted(requested_fixed_context.items())
            },
            "generations": execution["generations"],
        }
    request_sha256 = canonical_sha256(request_identity)
    campaign_id = spec.get("campaign_id")
    output = spec.get("output")
    if not isinstance(campaign_id, str) or not campaign_id or not isinstance(output, Mapping):
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: malformed base identity")
    runtime_root = output.get("runtime_root")
    if not isinstance(runtime_root, str) or not runtime_root:
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: malformed runtime root")
    spec["campaign_id"] = f"{campaign_id}-wrc-{request_sha256[:12]}"
    spec["output"] = {
        **dict(output),
        "runtime_root": f"{runtime_root.rstrip('/')}/wrc_science_v1/{request_sha256}",
    }
    return spec, envelope


__all__ = [
    "FULL_STACK_OVERLAYS_V1",
    "WRCScienceRequestError",
    "materialize_full_stack_wrc_science_spec_v1",
]
