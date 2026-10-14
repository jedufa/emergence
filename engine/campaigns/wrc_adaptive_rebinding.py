"""Fail-closed validation for generation-boundary WRC adaptive rebinding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_sha256
from .contracts import RuntimeProfileV1
from .wrc_variability import (
    INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    validate_wrc_parameter_variability_envelope_v1,
)


SCHEMA_VERSION: Final[str] = "wrc_checkpoint_adaptive_rebinding_v1"
_FIXED_MLP_WDL_SHARED_PARAMETERS: Final[tuple[str, ...]] = (
    "wdl.delay_ticks",
    "wdl.cue_duration_ticks",
    "wdl.active_duration_ticks",
    "env_cfg.wdl_delayed_hazard_event_period_steps",
    "wdl.intensity",
)
_INHERITANCE_SIDECAR_FIELDS_V1: Final[tuple[str, ...]] = (
    "diploid_checkpoint_projection",
    "genome_upgrade_v2_checkpoint_projection",
)


class WRCCheckpointAdaptiveRebindingError(ValueError):
    """An adaptive restore request exceeds its qualified WRC envelope."""


def _fail(code: str, message: str) -> WRCCheckpointAdaptiveRebindingError:
    return WRCCheckpointAdaptiveRebindingError(f"{code}: {message}")


def materialize_wrc_adaptive_restore_plan_v1(
    source_plan: Mapping[str, Any],
    *,
    restore_generation: int,
    parameter_arrays: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Create a deterministic continuation plan without regenerating cell identity."""

    source = _verified_plan_payload(source_plan, label="source")
    execution = source.get("execution")
    cells = source.get("cells")
    if not isinstance(execution, Mapping) or not isinstance(cells, list) or not cells:
        raise _fail("PLAN_MALFORMED", "source execution or cells are absent")
    source_generation = execution.get("generations")
    if (
        type(source_generation) is not int
        or type(restore_generation) is not int
        or not source_generation < restore_generation <= 128
    ):
        raise _fail(
            "GENERATION_CONTINUATION_INVALID",
            "restore generation must increase inside G3-G128",
        )
    if not parameter_arrays:
        raise _fail("NO_ADAPTIVE_CHANGE", "no parameter arrays were supplied")
    normalized: dict[str, list[Any]] = {}
    for name, values in sorted(parameter_arrays.items()):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != len(cells)
        ):
            raise _fail("PARAMETER_ARRAYS_MALFORMED", str(name))
        normalized[name] = list(values)
    restored = deepcopy(source)
    restored.pop("plan_sha256", None)
    restored_execution = dict(restored["execution"])
    restored_execution["generations"] = restore_generation
    prior_checkpoints = restored_execution.get("checkpoints", [])
    if not isinstance(prior_checkpoints, list) or any(
        type(value) is not int for value in prior_checkpoints
    ):
        raise _fail("PLAN_MALFORMED", "source checkpoints are malformed")
    restored_execution["checkpoints"] = sorted(
        {value for value in prior_checkpoints if 0 <= value <= restore_generation}
        | {restore_generation}
    )
    tick_schedule = restored_execution.get("tick_schedule")
    if tick_schedule is not None:
        if not isinstance(tick_schedule, Mapping):
            raise _fail("PLAN_MALFORMED", "tick schedule is malformed")
        restored_tick_schedule = dict(tick_schedule)
        legacy_projection = restored_tick_schedule.get("legacy_projection")
        if (
            not isinstance(legacy_projection, Mapping)
            or legacy_projection.get("ticks_per_generation") != 160
        ):
            raise _fail("PLAN_MALFORMED", "tick schedule legacy projection differs")
        restored_tick_schedule["total_ticks"] = restore_generation * 160
        restored_tick_schedule["legacy_projection"] = {
            **dict(legacy_projection),
            "generations": restore_generation,
        }
        restored_execution["tick_schedule"] = restored_tick_schedule
    restored["execution"] = restored_execution
    restored_cells: list[dict[str, Any]] = []
    for ordinal, raw_cell in enumerate(restored["cells"]):
        if not isinstance(raw_cell, Mapping):
            raise _fail("PLAN_MALFORMED", "source cell is malformed")
        cell = dict(raw_cell)
        parameters = cell.get("parameters")
        if not isinstance(parameters, Mapping):
            raise _fail("PLAN_MALFORMED", "source cell parameters are absent")
        updated = dict(parameters)
        unknown = sorted(set(normalized) - set(updated))
        if unknown:
            raise _fail(
                "PARAMETER_SET_CHANGED",
                f"parameter names were added: {unknown!r}",
            )
        for name, values in normalized.items():
            updated[name] = values[ordinal]
        cell["parameters"] = updated
        restored_cells.append(cell)
    restored["cells"] = restored_cells
    restored["plan_sha256"] = canonical_sha256(restored)
    return restored


def _verified_plan_payload(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    payload = dict(value)
    declared = payload.pop("plan_sha256", None)
    if not isinstance(declared, str) or canonical_sha256(payload) != declared:
        raise _fail("PLAN_IDENTITY_MISMATCH", f"{label} plan hash is invalid")
    payload["plan_sha256"] = declared
    return payload


def _request_payload(value: Any) -> dict[str, Any]:
    payload = value.payload() if hasattr(value, "payload") else value
    if not isinstance(payload, Mapping):
        raise _fail("REQUEST_MALFORMED", "resident request must be an object")
    return dict(payload)


def _stable_plan_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(plan)
    projected.pop("plan_sha256", None)
    execution = dict(projected.get("execution") or {})
    execution.pop("generations", None)
    execution.pop("checkpoints", None)
    tick_schedule = execution.get("tick_schedule")
    if isinstance(tick_schedule, Mapping):
        stable_tick_schedule = dict(tick_schedule)
        stable_tick_schedule.pop("total_ticks", None)
        legacy_projection = stable_tick_schedule.get("legacy_projection")
        if isinstance(legacy_projection, Mapping):
            stable_legacy = dict(legacy_projection)
            stable_legacy.pop("generations", None)
            stable_tick_schedule["legacy_projection"] = stable_legacy
        execution["tick_schedule"] = stable_tick_schedule
    projected["execution"] = execution
    cells = projected.get("cells")
    if not isinstance(cells, list):
        raise _fail("PLAN_MALFORMED", "compiled plan cells are absent")
    projected["cells"] = [
        {key: item for key, item in dict(cell).items() if key != "parameters"}
        if isinstance(cell, Mapping)
        else cell
        for cell in cells
    ]
    return projected


def _cells_by_request(
    plan: Mapping[str, Any], request: Mapping[str, Any], *, label: str
) -> list[dict[str, Any]]:
    raw_cells = plan.get("cells")
    cell_ids = request.get("cell_ids")
    if not isinstance(raw_cells, list) or not isinstance(cell_ids, list):
        raise _fail("CELL_MAPPING_INVALID", f"{label} cells are malformed")
    by_id = {
        str(cell.get("cell_id")): dict(cell)
        for cell in raw_cells
        if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
    }
    try:
        cells = [by_id[str(cell_id)] for cell_id in cell_ids]
    except KeyError as exc:
        raise _fail(
            "CELL_MAPPING_INVALID", f"{label} request references an unknown cell"
        ) from exc
    if [cell["cell_id"] for cell in cells] != cell_ids:
        raise _fail("CELL_MAPPING_INVALID", f"{label} cell order is not deterministic")
    return cells


def _validate_request_arrays(
    request: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, list[Any]]:
    raw_arrays = request.get("parameter_arrays")
    if not isinstance(raw_arrays, Mapping):
        raise _fail("PARAMETER_ARRAYS_MALFORMED", f"{label} arrays are absent")
    arrays = {str(name): list(values) for name, values in raw_arrays.items()}
    if any(len(values) != len(cells) for values in arrays.values()):
        raise _fail("PARAMETER_ARRAYS_MALFORMED", f"{label} arrays are not cell aligned")
    for ordinal, cell in enumerate(cells):
        parameters = cell.get("parameters")
        if not isinstance(parameters, Mapping) or set(parameters) != set(arrays):
            raise _fail(
                "PARAMETER_ARRAYS_MALFORMED",
                f"{label} cell parameters differ from request arrays",
            )
        if any(parameters[name] != values[ordinal] for name, values in arrays.items()):
            raise _fail(
                "PARAMETER_ARRAYS_MALFORMED",
                f"{label} cell values differ from request arrays",
            )
    return arrays


def _effective_control(
    name: str,
    *,
    parameters: Mapping[str, Any],
    fixed_invariants: Mapping[str, Any],
) -> Any:
    if name in parameters:
        return parameters[name]
    if name == "env_cfg.patch_world_enable":
        return fixed_invariants.get("runtime.world_base") == "patch"
    raise _fail(
        "PARAMETER_CONTEXT_UNRESOLVED",
        f"activation dependency {name!r} has no effective value",
    )


def _validate_activation_context(
    name: str,
    *,
    contract: Any,
    parameters: Mapping[str, Any],
    fixed_invariants: Mapping[str, Any],
) -> None:
    required = contract.activation_contract.get("required_effective_controls", {})
    if not isinstance(required, Mapping):
        raise _fail("PARAMETER_CONTEXT_INVALID", f"{name} contract is malformed")
    for control_name, expected in sorted(required.items()):
        actual = _effective_control(
            str(control_name),
            parameters=parameters,
            fixed_invariants=fixed_invariants,
        )
        if actual != expected:
            raise _fail(
                "PARAMETER_CONTEXT_INACTIVE",
                f"{name} requires {control_name}={expected!r}, got {actual!r}",
            )


def _validate_parameter_dependencies(parameters: Mapping[str, Any]) -> None:
    """Validate material cross-parameter contexts after a rebound."""

    names = (
        "wdl.delay_ticks",
        "wdl.cue_duration_ticks",
        "wdl.active_duration_ticks",
        "env_cfg.wdl_delayed_hazard_event_period_steps",
    )
    present = {name: parameters.get(name) for name in names if name in parameters}
    if not present:
        return
    if set(present) != set(names) or any(
        type(value) is not int or value < 1 for value in present.values()
    ):
        raise _fail(
            "PARAMETER_CONTEXT_UNRESOLVED",
            "complete positive-integer WDL cadence context is required",
        )
    delay = present["wdl.delay_ticks"]
    cue_duration = present["wdl.cue_duration_ticks"]
    active_duration = present["wdl.active_duration_ticks"]
    event_period = present["env_cfg.wdl_delayed_hazard_event_period_steps"]
    if cue_duration > delay:
        raise _fail(
            "PARAMETER_CONTEXT_INVALID",
            "wdl.cue_duration_ticks must not exceed wdl.delay_ticks",
        )
    if event_period < delay + active_duration + 1:
        raise _fail(
            "PARAMETER_CONTEXT_INVALID",
            "WDL event period must cover delay plus active duration plus one",
        )


def _validate_batch_representability(
    parameter_arrays: Mapping[str, Sequence[Any]],
    *,
    fixed_invariants: Mapping[str, Any],
) -> None:
    """Reject qualified scalar values that the exact batch ABI cannot mix."""

    if fixed_invariants.get("runtime.policy_family") != "fixed_mlp":
        return
    for name in _FIXED_MLP_WDL_SHARED_PARAMETERS:
        values = parameter_arrays.get(name)
        if values is not None and any(value != values[0] for value in values[1:]):
            raise _fail(
                "PARAMETER_CONTEXT_UNREPRESENTABLE",
                f"{name} must remain batch-wide for the fixed-MLP WDL route",
            )


def validate_wrc_checkpoint_adaptive_rebinding_v1(
    *,
    source_plan: Mapping[str, Any],
    restore_plan: Mapping[str, Any],
    source_request: Any,
    restore_request: Any,
    checkpoint_generation: int,
    profile: RuntimeProfileV1,
    generation_min: int,
    generation_max: int,
) -> dict[str, Any]:
    """Validate one adaptive continuation and return its canonical receipt."""

    source = _verified_plan_payload(source_plan, label="source")
    restore = _verified_plan_payload(restore_plan, label="restore")
    source_req = _request_payload(source_request)
    restore_req = _request_payload(restore_request)
    if _stable_plan_projection(source) != _stable_plan_projection(restore):
        raise _fail(
            "NON_ADAPTIVE_PLAN_CHANGE",
            "plans differ outside generation coordinates and cell parameters",
        )
    source_q = source.get("qualification_key")
    restore_q = restore.get("qualification_key")
    if (
        not isinstance(source_q, Mapping)
        or source_q != restore_q
        or source.get("qualification_key_sha256")
        != restore.get("qualification_key_sha256")
    ):
        raise _fail("QUALIFICATION_BINDING_CHANGED", "qualification key changed")
    envelope_payload = source_q.get("variability_envelope")
    if not isinstance(envelope_payload, Mapping):
        raise _fail("WRC_ENVELOPE_ABSENT", "qualification key has no WRC envelope")
    try:
        envelope = validate_wrc_parameter_variability_envelope_v1(envelope_payload)
    except WRCVariabilityEnvelopeError as exc:
        raise _fail("WRC_ENVELOPE_INVALID", str(exc)) from exc
    if envelope.profile_id != profile.profile_id:
        raise _fail("WRC_ENVELOPE_INVALID", "profile identity differs")
    fixed_invariants = source_q.get("fixed_invariants")
    if (
        not isinstance(fixed_invariants, Mapping)
        or fixed_invariants.get(INVARIANT_NAME) != envelope.envelope_sha256
    ):
        raise _fail("WRC_ENVELOPE_INVALID", "qualification invariant differs")

    exact_request_fields = (
        "profile_id",
        "cell_ids",
        "ticks_per_generation",
        "seed_arrays",
        "checkpoint_storage_mode",
        "reporting_enabled",
        "execution_route_sha256",
        "genome_v2_checkpoint_authority_binding",
    )
    for field in exact_request_fields:
        if source_req.get(field) != restore_req.get(field):
            raise _fail("PRESERVED_REQUEST_FIELD_CHANGED", field)
    if source_req.get("profile_id") != profile.profile_id:
        raise _fail("PROFILE_IDENTITY_CHANGED", "request profile differs")
    if source_req.get("ticks_per_generation") != 160:
        raise _fail("TICKS_CONTRACT_CHANGED", "T160 is required")
    source_generations = source_req.get("generations")
    restore_generations = restore_req.get("generations")
    if (
        type(checkpoint_generation) is not int
        or type(source_generations) is not int
        or type(restore_generations) is not int
        or source_generations != checkpoint_generation
        or not generation_min <= source_generations < restore_generations <= generation_max
    ):
        raise _fail(
            "GENERATION_CONTINUATION_INVALID",
            "adaptive restore must continue a terminal qualified boundary",
        )
    restore_milestones = restore_req.get("checkpoint_milestones")
    if (
        not isinstance(restore_milestones, list)
        or restore_generations not in restore_milestones
        or any(type(value) is not int or value < 0 or value > restore_generations for value in restore_milestones)
    ):
        raise _fail("GENERATION_CONTINUATION_INVALID", "restore milestones are invalid")
    allocation_end = restore_req.get("agent_id_allocation_generation_count")
    if allocation_end is not None and (
        type(allocation_end) is not int or allocation_end < restore_generations
    ):
        raise _fail("GENERATION_CONTINUATION_INVALID", "agent ID extent is too short")

    source_cells = _cells_by_request(source, source_req, label="source")
    restore_cells = _cells_by_request(restore, restore_req, label="restore")
    source_arrays = _validate_request_arrays(source_req, source_cells, label="source")
    restore_arrays = _validate_request_arrays(restore_req, restore_cells, label="restore")
    if set(source_arrays) != set(restore_arrays):
        raise _fail("PARAMETER_SET_CHANGED", "parameter names were added or removed")
    _validate_batch_representability(
        restore_arrays,
        fixed_invariants=fixed_invariants,
    )
    envelope_rows = {
        str(row.get("name")): dict(row) for row in envelope.parameter_contracts
    }
    changed_names = sorted(
        name for name in source_arrays if source_arrays[name] != restore_arrays[name]
    )
    if not changed_names:
        raise _fail("NO_ADAPTIVE_CHANGE", "no parameter array changed")
    changed_cells: list[dict[str, Any]] = []
    for name in changed_names:
        contract = profile.parameter_contracts.get(name)
        envelope_row = envelope_rows.get(name)
        if (
            contract is None
            or envelope_row is None
            or contract.variability_semantics != "experiment_variable"
            or not contract.per_cell
            or contract.qualification_affecting
            or envelope_row != contract.variability_contract_payload_v1()
        ):
            raise _fail(
                "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE",
                name,
            )
        for value in restore_arrays[name]:
            if not contract.accepts(value):
                raise _fail("PARAMETER_VALUE_OUT_OF_RANGE", f"{name}={value!r}")
    for ordinal, (source_cell, restore_cell) in enumerate(
        zip(source_cells, restore_cells, strict=True)
    ):
        source_parameters = dict(source_cell["parameters"])
        restore_parameters = dict(restore_cell["parameters"])
        # A changed control may deactivate an unchanged dependent variable.
        # Validate the complete post-rebind context, not only changed names.
        for name in sorted(restore_parameters):
            contract = profile.parameter_contracts.get(name)
            if contract is not None and contract.activation_contract:
                _validate_activation_context(
                    name,
                    contract=contract,
                    parameters=restore_parameters,
                    fixed_invariants=fixed_invariants,
                )
        _validate_parameter_dependencies(restore_parameters)
        cell_changes: dict[str, dict[str, Any]] = {}
        for name in changed_names:
            if source_parameters[name] == restore_parameters[name]:
                continue
            cell_changes[name] = {
                "source": source_parameters[name],
                "restore": restore_parameters[name],
            }
        if cell_changes:
            changed_cells.append(
                {
                    "cell_id": restore_req["cell_ids"][ordinal],
                    "cell_ordinal": ordinal,
                    "changes": cell_changes,
                }
            )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "capability_id": "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1",
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "qualification_key_sha256": source["qualification_key_sha256"],
        "execution_route_sha256": source_req["execution_route_sha256"],
        "variability_envelope_sha256": envelope.envelope_sha256,
        "source_generation": source_generations,
        "restore_generation": restore_generations,
        "ticks_per_generation": 160,
        "cell_ids": list(source_req["cell_ids"]),
        "changed_parameter_names": changed_names,
        "changed_cells": changed_cells,
        "preserved": {
            "cell_identity_sha256": canonical_sha256(
                [
                    {
                        key: cell.get(key)
                        for key in (
                            "ordinal",
                            "cell_id",
                            "branch_id",
                            "rng_lineage_id",
                            "dimensions",
                            "seeds",
                        )
                    }
                    for cell in source_cells
                ]
            ),
            "seed_arrays_sha256": canonical_sha256(source_req["seed_arrays"]),
            "artifacts_sha256": canonical_sha256(source["artifacts"]),
            "stable_plan_projection_sha256": canonical_sha256(
                _stable_plan_projection(source)
            ),
        },
    }
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


def validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(value)
    declared = receipt.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("capability_id") != "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1"
        or not isinstance(declared, str)
        or canonical_sha256(receipt) != declared
    ):
        raise _fail("ADAPTIVE_REBINDING_RECEIPT_INVALID", "receipt hash differs")
    return {**receipt, "receipt_sha256": declared}


def materialize_wrc_adaptive_runtime_projections_v1(
    *,
    adaptive_rebinding: Mapping[str, Any],
    source_projections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Overlay only declared rebound controls on loaded checkpoint metadata.

    The durable checkpoint projections and their tensors remain untouched. The
    returned projections are ephemeral source-import views for the new owner.
    """

    adaptive = validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
        adaptive_rebinding
    )
    cell_ids = adaptive.get("cell_ids")
    changed_cells = adaptive.get("changed_cells")
    changed_parameter_names = adaptive.get("changed_parameter_names")
    if (
        not isinstance(cell_ids, list)
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
        or not isinstance(changed_cells, list)
        or not isinstance(changed_parameter_names, list)
        or any(
            not isinstance(name, str) or not name
            for name in changed_parameter_names
        )
        or len(set(changed_parameter_names)) != len(changed_parameter_names)
        or len(source_projections) != len(cell_ids)
    ):
        raise _fail(
            "ADAPTIVE_RUNTIME_REBINDING_INCOMPLETE",
            "projection and cell mappings differ",
        )
    changes_by_cell: dict[str, Mapping[str, Any]] = {}
    observed_names: set[str] = set()
    for row in changed_cells:
        if not isinstance(row, Mapping):
            continue
        cell_id = row.get("cell_id")
        changes = row.get("changes")
        ordinal = row.get("cell_ordinal")
        if (
            not isinstance(cell_id, str)
            or cell_id not in cell_ids
            or cell_id in changes_by_cell
            or ordinal != cell_ids.index(cell_id)
            or not isinstance(changes, Mapping)
            or not changes
        ):
            continue
        valid = True
        for name, change in changes.items():
            if (
                name not in changed_parameter_names
                or not isinstance(change, Mapping)
                or set(change) != {"source", "restore"}
                or type(change["restore"]) not in {bool, int, float, str}
                or change["source"] == change["restore"]
                or (
                    isinstance(change["restore"], float)
                    and not math.isfinite(change["restore"])
                )
            ):
                valid = False
                break
            observed_names.add(str(name))
        if valid:
            changes_by_cell[cell_id] = changes
    if (
        len(changes_by_cell) != len(changed_cells)
        or observed_names != set(changed_parameter_names)
    ):
        raise _fail(
            "ADAPTIVE_RUNTIME_REBINDING_INCOMPLETE",
            "changed-cell mappings are malformed or duplicated",
        )

    from engine.campaigns.production_control_path_v1 import runtime_field_name_v1
    from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
    from engine.perf.inheritance_sidecar_identity_v1 import (
        INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        inheritance_sidecar_payload_sha256_v1,
    )

    rebound: list[dict[str, Any]] = []
    for cell_id, projection in zip(cell_ids, source_projections, strict=True):
        if not isinstance(projection, Mapping):
            raise _fail(
                "ADAPTIVE_RUNTIME_REBINDING_INCOMPLETE",
                f"source projection is malformed: {cell_id}",
            )
        metadata = projection.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _fail(
                "ADAPTIVE_RUNTIME_REBINDING_INCOMPLETE",
                f"source metadata is absent: {cell_id}",
            )
        rebound_projection = dict(projection)
        rebound_metadata = dict(metadata)
        changes = changes_by_cell.get(cell_id, {})
        for parameter_name, change in sorted(changes.items()):
            runtime_field = runtime_field_name_v1(str(parameter_name))
            restore_value = (
                change.get("restore") if isinstance(change, Mapping) else None
            )
            if runtime_field is None or runtime_field not in rebound_metadata:
                raise _fail(
                    "ADAPTIVE_RUNTIME_PARAMETER_UNREPRESENTABLE",
                    f"{cell_id}:{parameter_name}",
                )
            rebound_metadata[runtime_field] = restore_value
        rebound_projection["metadata"] = rebound_metadata
        rebound_metadata["import_digest"] = projection_digest(
            rebound_projection
        )
        source_base_projection = dict(projection)
        rebound_base_projection = dict(rebound_projection)
        for field in _INHERITANCE_SIDECAR_FIELDS_V1:
            source_base_projection.pop(field, None)
            rebound_base_projection.pop(field, None)
        source_base_sha256 = projection_digest(source_base_projection)
        rebound_base_sha256 = projection_digest(rebound_base_projection)
        for field in _INHERITANCE_SIDECAR_FIELDS_V1:
            sidecar = projection.get(field)
            if sidecar is None:
                continue
            if not isinstance(sidecar, Mapping):
                raise _fail(
                    "ADAPTIVE_RUNTIME_INHERITANCE_IDENTITY_INVALID",
                    f"{cell_id}:{field}: sidecar is malformed",
                )
            identity = sidecar.get("identity")
            if not isinstance(identity, Mapping):
                raise _fail(
                    "ADAPTIVE_RUNTIME_INHERITANCE_IDENTITY_INVALID",
                    f"{cell_id}:{field}: identity is absent",
                )
            payload = dict(sidecar)
            payload.pop("identity", None)
            expected_identity = {
                "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
                "cell_id": cell_id,
                "seed": metadata.get("seed"),
                "generation": metadata.get("generation"),
                "tick": metadata.get("tick"),
                "base_projection_sha256": source_base_sha256,
                "payload_sha256": inheritance_sidecar_payload_sha256_v1(payload),
            }
            if dict(identity) != expected_identity:
                raise _fail(
                    "ADAPTIVE_RUNTIME_INHERITANCE_IDENTITY_INVALID",
                    f"{cell_id}:{field}: source binding differs",
                )
            rebound_sidecar = dict(sidecar)
            rebound_sidecar["identity"] = {
                **expected_identity,
                "base_projection_sha256": rebound_base_sha256,
            }
            rebound_projection[field] = rebound_sidecar
        rebound.append(rebound_projection)
    return tuple(rebound)


def validate_wrc_adaptive_runtime_readback_v1(
    *,
    adaptive_rebinding: Mapping[str, Any],
    production_control_readbacks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that the first continued segment applied every rebound value."""

    adaptive = validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
        adaptive_rebinding
    )
    source_generation = adaptive.get("source_generation")
    candidates = [
        dict(row)
        for row in production_control_readbacks
        if isinstance(row, Mapping)
        and row.get("start_generation") == source_generation
    ]
    if len(candidates) != 1:
        raise _fail(
            "ADAPTIVE_RUNTIME_READBACK_INCOMPLETE",
            "exactly one first continued-segment readback is required",
        )
    descriptor = candidates[0]
    path_value = descriptor.get("path")
    declared_file_sha256 = descriptor.get("receipt_sha256")
    if not isinstance(path_value, str) or not isinstance(declared_file_sha256, str):
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", "descriptor is malformed")
    path = Path(path_value)
    try:
        raw = path.read_bytes()
        readback = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", str(exc)) from exc
    if not isinstance(readback, Mapping):
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INVALID", "receipt is malformed")
    unsigned_readback = dict(readback)
    embedded_sha256 = unsigned_readback.pop("receipt_sha256", None)
    if (
        embedded_sha256 != declared_file_sha256
        or canonical_sha256(unsigned_readback) != declared_file_sha256
    ):
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INVALID", "receipt digest differs")
    rows = readback.get("cells")
    if not isinstance(rows, list):
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", "cell rows are absent")
    by_cell = {
        str(row.get("cell_id")): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }
    verified_cells: list[dict[str, Any]] = []
    changed_cells = adaptive.get("changed_cells")
    if not isinstance(changed_cells, list):
        raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", "changes are absent")
    for changed_cell in changed_cells:
        if not isinstance(changed_cell, Mapping):
            raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", "change row is malformed")
        cell_id = str(changed_cell.get("cell_id") or "")
        readback_cell = by_cell.get(cell_id)
        if not isinstance(readback_cell, Mapping):
            raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", f"missing {cell_id}")
        controls = readback_cell.get("controls")
        if not isinstance(controls, list):
            raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", f"missing controls for {cell_id}")
        controls_by_name = {
            str(row.get("control")): row
            for row in controls
            if isinstance(row, Mapping) and isinstance(row.get("control"), str)
        }
        changes = changed_cell.get("changes")
        if not isinstance(changes, Mapping):
            raise _fail("ADAPTIVE_RUNTIME_READBACK_INCOMPLETE", f"missing changes for {cell_id}")
        verified_controls: list[str] = []
        for name, change in sorted(changes.items()):
            row = controls_by_name.get(str(name))
            expected = change.get("restore") if isinstance(change, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or row.get("status") != "READBACK_COMPLETE"
                or any(row.get(field) != expected for field in ("requested", "compiled", "applied", "readback"))
            ):
                raise _fail(
                    "ADAPTIVE_RUNTIME_READBACK_MISMATCH",
                    f"{cell_id}:{name}",
                )
            verified_controls.append(str(name))
        verified_cells.append(
            {"cell_id": cell_id, "verified_controls": verified_controls}
        )
    receipt = {
        "schema_version": "wrc_checkpoint_adaptive_runtime_readback_v1",
        "adaptive_rebinding_receipt_sha256": adaptive["receipt_sha256"],
        "segment_control_readback_sha256": declared_file_sha256,
        "start_generation": descriptor.get("start_generation"),
        "end_generation": descriptor.get("end_generation"),
        "verified_cells": verified_cells,
        "status": "READBACK_COMPLETE",
    }
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


__all__ = [
    "SCHEMA_VERSION",
    "WRCCheckpointAdaptiveRebindingError",
    "materialize_wrc_adaptive_restore_plan_v1",
    "materialize_wrc_adaptive_runtime_projections_v1",
    "validate_wrc_checkpoint_adaptive_rebinding_receipt_v1",
    "validate_wrc_checkpoint_adaptive_rebinding_v1",
    "validate_wrc_adaptive_runtime_readback_v1",
]
