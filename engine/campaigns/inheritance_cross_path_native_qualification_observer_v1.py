"""Qualification-only observations from real cross-path resident dispatches."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Final, Mapping

import torch

from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
    FAILURE_AUTHORITATIVE_COMMIT_ABORTED,
    FAILURE_PENDING_COMMIT,
    LIVE_BIRTH_FAILURE_ID_BY_CODE_V1,
    PATH_GENERATION_BOUNDARY_REFILL,
    PATH_ORDINARY_LIVE_BIRTH,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_qualification_controls_v1 import (
    ResidentCapacityFaultControlV1,
    ResidentCapacityFaultObservationV1,
    ResidentFixedActionControlV1,
    ResidentFixedActionObservationV1,
)


FAULT_DIRECTIVE_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_attempt_capacity_fault_directive_v1"
)
DIRECT_PROJECTION_CAPTURE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_native_direct_projection_capture_v2"
)
DIRECT_PROJECTION_CELL_CAPTURE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_native_direct_projection_cell_capture_v2"
)
DIRECT_PROJECTION_PATH_INSTRUMENTATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_native_path_instrumentation_v1"
)
REGISTERED_PATH_ID_BY_CODE_V1: Final[Mapping[int, str]] = {
    PATH_GENERATION_BOUNDARY_REFILL: "generation_boundary_refill",
    PATH_ORDINARY_LIVE_BIRTH: "ordinary_live_birth",
}
class InheritanceCrossPathNativeQualificationObserverError(RuntimeError):
    """Resident qualification evidence was incomplete or contradictory."""


def _fail(message: str) -> InheritanceCrossPathNativeQualificationObserverError:
    return InheritanceCrossPathNativeQualificationObserverError(message)


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ResidentAttemptCapacityFaultDirectiveV1:
    """Declarative request from which runtime-bound typed receipts are made."""

    target_occurrence_id: str
    target_occurrence_authority_sha256: str
    target_cell_index: int
    target_cell_authority_sha256: str
    effective_bound: int = 0
    expected_status_code: int = CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
    schema_version: str = FAULT_DIRECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != FAULT_DIRECTIVE_SCHEMA_VERSION
            or not isinstance(self.target_occurrence_id, str)
            or not self.target_occurrence_id
            or type(self.target_cell_index) is not int
            or self.target_cell_index < 0
            or type(self.effective_bound) is not int
            or self.effective_bound != 0
            or self.expected_status_code
            != CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
        ):
            raise _fail("attempt-capacity fault directive is invalid")
        _digest(
            self.target_occurrence_authority_sha256,
            name="target_occurrence_authority_sha256",
        )
        _digest(
            self.target_cell_authority_sha256,
            name="target_cell_authority_sha256",
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_occurrence_id": self.target_occurrence_id,
            "target_occurrence_authority_sha256": (
                self.target_occurrence_authority_sha256
            ),
            "target_cell_index": self.target_cell_index,
            "target_cell_authority_sha256": self.target_cell_authority_sha256,
            "effective_bound": self.effective_bound,
            "expected_status_code": self.expected_status_code,
        }

    @classmethod
    def from_payload(cls, value: Any) -> "ResidentAttemptCapacityFaultDirectiveV1":
        fields = {
            "schema_version",
            "target_occurrence_id",
            "target_occurrence_authority_sha256",
            "target_cell_index",
            "target_cell_authority_sha256",
            "effective_bound",
            "expected_status_code",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise _fail("attempt-capacity fault directive field set is invalid")
        return cls(**{field: value[field] for field in fields})


def _tensor_manifest(
    tensor: torch.Tensor,
    *,
    cell: int | None,
    batch_cells: int,
) -> dict[str, Any]:
    selected = tensor
    if cell is not None:
        if (
            tensor.ndim == 0
            or tensor.shape[0] != batch_cells
            or cell < 0
            or cell >= batch_cells
        ):
            raise _fail(
                "cell-local authority requires an explicit leading cell axis"
            )
        selected = tensor[cell]
    host = selected.detach().to(device="cpu").contiguous()
    raw = host.numpy().tobytes(order="C")
    return {
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _authority_identity(value: Any, *, name: str) -> Any:
    """Normalize an embedded authority without discarding tensor identity."""

    if isinstance(value, torch.Tensor):
        return {
            "tensor_manifest": _tensor_manifest(
                value,
                cell=None,
                batch_cells=-1,
            )
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) for key in value):
            raise _fail(f"{name} contains a non-canonical mapping key")
        return {
            str(key): _authority_identity(child, name=f"{name}.{key}")
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [
            _authority_identity(child, name=f"{name}.{index}")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise _fail(f"{name} contains an unsupported authority value")


def _founder_source_authority_sha256(projection: Any, *, cell_id: str) -> str:
    from .inheritance_cross_path_source_materializer_v1 import (
        SOURCE_AUTHORITY_FIELD,
    )

    if not isinstance(projection, Mapping):
        raise _fail(f"{cell_id} source projection is not an object")
    authority = projection.get(SOURCE_AUTHORITY_FIELD)
    if not isinstance(authority, Mapping):
        raise _fail(f"{cell_id} source projection lacks founder authority")
    if authority.get("cell_id") != cell_id:
        raise _fail(f"{cell_id} founder authority differs from physical order")
    return canonical_sha256(
        _authority_identity(authority, name=f"{cell_id}.founder_authority")
    )


def _mechanics_sha256(prepared: Any, cell: int) -> str:
    source = prepared.patch.patch.source
    batch_cells = prepared.batch_cells
    manifests: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            manifests[name] = _tensor_manifest(
                value, cell=cell, batch_cells=batch_cells
            )
        elif isinstance(value, Mapping):
            for child_name, child in sorted(value.items()):
                add(f"{name}.{child_name}", child)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                add(f"{name}.{index}", child)

    add("active_agent_ids", source.active_agent_ids)
    add("active_count_state", source.active_count_state)
    add("live_state", source.live_state)
    add("full_tick_state", source.full_tick_state_tensors)
    add("ordinary_birth_state", source.ordinary_birth_state_tensors)
    linkage = prepared.linkage
    tensor_map = getattr(linkage, "tensor_map", None)
    if callable(tensor_map):
        linkage_tensors = tensor_map()
    elif is_dataclass(linkage):
        linkage_tensors = {
            field.name: value
            for field in fields(linkage)
            if isinstance((value := getattr(linkage, field.name)), torch.Tensor)
        }
    else:
        raise _fail("prepared linkage owner exposes no typed tensor fields")
    add("linkage", linkage_tensors)
    if not manifests:
        raise _fail("prepared resident source has no mechanics tensors")
    return canonical_sha256(manifests)


def _fault_transaction_sha256(prepared: Any, cell: int) -> str:
    """Hash only state owned by the failing cross-path transaction.

    A reproduction attempt can occur after ordinary world evolution within a
    tick, so the complete cell state is not expected to equal its
    pre-dispatch state.  Fail-closed qualification instead binds every
    linkage and exposure tensor the transaction may commit, excluding only
    the explicit qualification controls and the expected failure-status
    scalar itself.
    """

    linkage = prepared.linkage
    linkage_map = getattr(linkage, "_tensor_map", None)
    if not callable(linkage_map):
        linkage_map = getattr(linkage, "tensor_map", None)
    exposure_map = getattr(prepared.exposure, "tensor_map", None)
    if not callable(linkage_map) or not callable(exposure_map):
        raise _fail("fault target exposes no typed transaction tensor maps")
    manifests: dict[str, Any] = {}
    for name, value in sorted(linkage_map().items()):
        if name == "transaction_status":
            continue
        if not isinstance(value, torch.Tensor):
            raise _fail("linkage transaction state contains a non-tensor")
        manifests[f"linkage.{name}"] = _tensor_manifest(
            value,
            cell=cell,
            batch_cells=prepared.batch_cells,
        )
    excluded_exposure = {
        "cell_status",
        "qualification_attempt_capacity_enabled",
        "qualification_effective_attempt_capacity",
    }
    for name, value in sorted(exposure_map().items()):
        if name in excluded_exposure:
            continue
        if not isinstance(value, torch.Tensor):
            raise _fail("exposure transaction state contains a non-tensor")
        manifests[f"exposure.{name}"] = _tensor_manifest(
            value,
            cell=cell,
            batch_cells=prepared.batch_cells,
        )
    if not manifests:
        raise _fail("fault target exposes no transaction-owned tensors")
    return canonical_sha256(manifests)


def _peer_fault_plane_sha256(prepared: Any, target: int) -> str:
    exposure = prepared.exposure
    rows = []
    for cell in range(prepared.batch_cells):
        if cell == target:
            continue
        rows.append(
            {
                "cell_index": cell,
                "enabled": bool(
                    exposure.qualification_attempt_capacity_enabled[cell].item()
                ),
                "effective_capacity": int(
                    exposure.qualification_effective_attempt_capacity[cell].item()
                ),
                "status": int(exposure.cell_status[cell].item()),
            }
        )
    if not rows:
        raise _fail("capacity-fault qualification requires at least one peer")
    return canonical_sha256(rows)


def _action_policy_semantic_draw_count(
    prepared: Any,
    mask: tuple[bool, ...],
) -> int:
    """Count only stochastic action-policy draws for the selected cells.

    The resident ``policy_np_semantic_draws`` state also owns environment,
    chemistry, and reproduction-planning draws.  It therefore cannot prove
    that the fixed-action bypass skipped action-policy work.  The native
    action-policy ABI records exact refilled rows, and each stochastic refill
    consumes ``cache_width * action_count`` Gumbel draws.
    """

    source = prepared.patch.patch.source
    configuration = source.configuration
    cache_width = getattr(configuration, "cache_width", None)
    action_count = getattr(configuration, "action_count", None)
    if (
        type(cache_width) is not int
        or cache_width < 1
        or type(action_count) is not int
        or action_count < 1
    ):
        raise _fail("resident action-policy draw geometry is invalid")
    cells = [index for index, enabled in enumerate(mask) if enabled]
    refill_rows = int(source.outputs["policy_refill_rows"][cells].sum().item())
    if refill_rows < 0:
        raise _fail("resident action-policy refill rows are negative")
    if bool(source.deterministic):
        return 0
    return refill_rows * cache_width * action_count


def _rng_sha256(prepared: Any, cell: int) -> str:
    state = prepared.patch.patch.source.full_tick_state_tensors
    if len(state) != 29:
        raise _fail("resident RNG ABI does not contain 29 tensors")
    return canonical_sha256(
        {
            f"state.{index}": _tensor_manifest(
                state[index],
                cell=cell,
                batch_cells=prepared.batch_cells,
            )
            for index in (16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28)
        }
    )


def _policy_evaluation_count(prepared: Any, mask: tuple[bool, ...]) -> int:
    outputs = prepared.patch.patch.source.outputs
    cells = [index for index, enabled in enumerate(mask) if enabled]
    return int(
        (
            outputs["policy_refill_rows"][cells]
            + outputs["policy_cache_hit_rows"][cells]
        )
        .sum()
        .item()
    )


def _rest_actions(prepared: Any, mask: tuple[bool, ...]) -> list[int]:
    outputs = prepared.patch.patch.source.outputs
    actions = outputs["row_actions"]
    active = outputs["active_count"]
    rows: list[int] = []
    for cell, selected in enumerate(mask):
        if not selected:
            continue
        for tick in range(active.shape[1]):
            count = int(active[cell, tick].item())
            rows.extend(int(value) for value in actions[cell, tick, :count].tolist())
    return rows


class CrossPathNativeQualificationObserverV1:
    """Tombstone for the retired standalone native qualification observer."""

    def __init__(
        self,
        *,
        fault: ResidentAttemptCapacityFaultDirectiveV1 | None,
        fixed_action_required: bool,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path native qualification observation is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        self._fault = fault
        self._fixed_action_required = fixed_action_required
        self._fault_control: ResidentCapacityFaultControlV1 | None = None
        self._fault_mechanics_before: str | None = None
        self._fault_mechanics_after: str | None = None
        self._fault_peer_before: str | None = None
        self._fault_peer_after: str | None = None
        self._fault_status_before: int | None = None
        self._fault_status_after: int | None = None
        self._fault_status_after_drain: int | None = None
        self._batch_cells: int | None = None
        self._fixed_control: ResidentFixedActionControlV1 | None = None
        self._fixed_mask: tuple[bool, ...] | None = None
        self._fixed_evaluations_before: int | None = None
        self._fixed_evaluations_after = 0
        self._fixed_draws_before: int | None = None
        self._fixed_draws_after: int | None = None
        self._fixed_actions: list[int] = []
        self._resident_observations: list[dict[str, Any]] = []
        self._direct_capture_cells: list[dict[str, Any]] | None = None
        self._direct_capture_unavailable_reason: str | None = None
        self._direct_capture_started_at_utc: str | None = None

    def __call__(
        self,
        *,
        phase: str,
        prepared: Any,
        direct_result: Any,
        **_metadata: Any,
    ) -> None:
        del direct_result
        if phase == "prepared":
            self._prepared(prepared)
        elif phase == "executed":
            self._executed(prepared)
        elif phase == "settled":
            self._settled(prepared)
        else:
            raise _fail("unknown resident qualification observation phase")

    def _prepared(self, prepared: Any) -> None:
        if self._batch_cells is None:
            self._batch_cells = prepared.batch_cells
        elif self._batch_cells != prepared.batch_cells:
            raise _fail("physical batch size changed across resident owners")
        self._capture_direct_projection_prepared(prepared)
        if self._fault is not None:
            target = self._fault.target_cell_index
            if target >= prepared.batch_cells:
                raise _fail("capacity-fault target is outside the physical batch")
            exposure = prepared.exposure
            if self._fault_control is None:
                status = int(exposure.cell_status[target].item())
                mechanics = _fault_transaction_sha256(prepared, target)
                peer = _peer_fault_plane_sha256(prepared, target)
                pre_fault = canonical_sha256(
                    {
                        "target_cell_index": target,
                        "target_cell_authority_sha256": (
                            self._fault.target_cell_authority_sha256
                        ),
                        "mechanics_sha256": mechanics,
                        "exposure": {
                            "status": status,
                            "attempt_cursor": int(
                                exposure.attempt_cursors[target].item()
                            ),
                            "shape": exposure.shape.payload(),
                        },
                    }
                )
                self._fault_control = ResidentCapacityFaultControlV1(
                    target_occurrence_id=self._fault.target_occurrence_id,
                    target_occurrence_authority_sha256=(
                        self._fault.target_occurrence_authority_sha256
                    ),
                    target_cell_index=target,
                    target_cell_authority_sha256=(
                        self._fault.target_cell_authority_sha256
                    ),
                    capacity_kind="attempt",
                    nominal_bound=exposure.shape.attempt_capacity,
                    effective_bound=self._fault.effective_bound,
                    expected_status_code=self._fault.expected_status_code,
                    pre_fault_authority_sha256=pre_fault,
                )
                self._fault_status_before = status
                self._fault_mechanics_before = mechanics
                self._fault_peer_before = peer
                exposure.arm_qualification_attempt_capacity_v1(
                    target_cell_index=target,
                    effective_capacity=self._fault.effective_bound,
                )

        if self._fixed_action_required:
            neutral_mask = tuple(
                bool(value)
                for value in prepared.controls.neutral_fixture_enabled.tolist()
            )
            mask = tuple(
                enabled
                and (
                    self._fault is None
                    or index != self._fault.target_cell_index
                )
                for index, enabled in enumerate(neutral_mask)
            )
            if self._fixed_mask is None:
                if not any(mask):
                    raise _fail("fixed-action qualification selected no resident cells")
                self._fixed_mask = mask
                self._fixed_control = ResidentFixedActionControlV1(
                    cell_mask_sha256=canonical_sha256(list(mask)),
                    selected_cell_count=sum(mask),
                )
                self._fixed_evaluations_before = _policy_evaluation_count(
                    prepared, mask
                )
                self._fixed_draws_before = (
                    _action_policy_semantic_draw_count(prepared, mask)
                )
            elif mask != self._fixed_mask:
                raise _fail("fixed-action mask changed across resident owners")

    def _executed(self, prepared: Any) -> None:
        self._capture_direct_projection_executed(prepared)
        if self._fault is not None:
            target = self._fault.target_cell_index
            self._fault_status_after = int(
                prepared.exposure.cell_status[target].item()
            )
            self._fault_mechanics_after = _fault_transaction_sha256(
                prepared, target
            )
            self._fault_peer_after = _peer_fault_plane_sha256(prepared, target)
        if self._fixed_mask is not None:
            self._fixed_evaluations_after += _policy_evaluation_count(
                prepared, self._fixed_mask
            )
            self._fixed_draws_after = (
                int(self._fixed_draws_after or 0)
                + _action_policy_semantic_draw_count(
                    prepared,
                    self._fixed_mask,
                )
            )
            self._fixed_actions.extend(_rest_actions(prepared, self._fixed_mask))
        self._resident_observations.append(
            {
                "dispatch_index": len(self._resident_observations),
                "cell_state_sha256": [
                    _mechanics_sha256(prepared, cell)
                    for cell in range(prepared.batch_cells)
                ],
                "cell_rng_sha256": [
                    _rng_sha256(prepared, cell)
                    for cell in range(prepared.batch_cells)
                ],
            }
        )

    def _settled(self, prepared: Any) -> None:
        if self._fault is not None:
            self._fault_status_after_drain = int(
                prepared.exposure.cell_status[
                    self._fault.target_cell_index
                ].item()
            )

    def _capture_direct_projection_prepared(self, prepared: Any) -> None:
        """Capture first-owner identities while source projections are live."""

        if self._direct_capture_cells is not None:
            return
        self._direct_capture_started_at_utc = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        controls = getattr(prepared, "controls", None)
        source_controls = getattr(controls, "source_controls", None)
        cell_ids = tuple(getattr(prepared, "cell_ids", ()))
        source_projections = tuple(getattr(prepared, "source_projections", ()))
        path_codes = getattr(controls, "reproduction_path_codes", None)
        if (
            not isinstance(source_controls, (tuple, list))
            or len(source_controls) != prepared.batch_cells
            or len(cell_ids) != prepared.batch_cells
            or len(source_projections) != prepared.batch_cells
            or not isinstance(path_codes, torch.Tensor)
            or path_codes.ndim != 1
            or path_codes.shape[0] != prepared.batch_cells
        ):
            self._direct_capture_unavailable_reason = (
                "prepared owner lacks ordered source controls, source "
                "projections, or native reproduction path codes"
            )
            return
        rows: list[dict[str, Any]] = []
        for index, (cell_id, control, projection) in enumerate(
            zip(cell_ids, source_controls, source_projections, strict=True)
        ):
            if (
                not isinstance(cell_id, str)
                or not cell_id
                or getattr(control, "cell_id", None) != cell_id
                or not isinstance(getattr(control, "reproduction_path", None), str)
                or not callable(getattr(control, "payload", None))
            ):
                raise _fail("direct projection controls differ from physical order")
            pre_dispatch = _mechanics_sha256(prepared, index)
            founder = _founder_source_authority_sha256(
                projection, cell_id=cell_id
            )
            path_id = control.reproduction_path
            path_code = int(path_codes[index].item())
            control_payload = control.payload()
            cell_authority = canonical_sha256(
                {
                    "occurrence_id": cell_id,
                    "control": control_payload,
                    "pre_dispatch_state_sha256": pre_dispatch,
                    "founder_source_authority_sha256": founder,
                }
            )
            rows.append(
                {
                    "schema_version": (
                        DIRECT_PROJECTION_CELL_CAPTURE_SCHEMA_VERSION
                    ),
                    "occurrence_id": cell_id,
                    "reproduction_path": path_id,
                    "native_path_code": path_code,
                    "pre_dispatch_state_sha256": pre_dispatch,
                    "founder_source_authority_sha256": founder,
                    "cell_authority_sha256": cell_authority,
                    "path_observation": {
                        "authoritative_attempt_rows": 0,
                        "successful_commit_rows": 0,
                        "failed_attempt_rows": 0,
                        "eligibility_snapshot_rows": 0,
                        "donor_draw_rows": 0,
                        "sidecar_attempt_rows": 0,
                    },
                    "path_instrumentation": {},
                }
            )
        if len({row["reproduction_path"] for row in rows}) != len(
            {row["native_path_code"] for row in rows}
        ):
            raise _fail("native path codes do not uniquely bind path IDs")
        self._direct_capture_cells = rows
        self._direct_capture_unavailable_reason = None

    def _capture_direct_projection_executed(self, prepared: Any) -> None:
        """Count only rows directly resident in the exact exposure sidecar."""

        rows = self._direct_capture_cells
        if rows is None:
            return
        exposure = prepared.exposure
        required = (
            "attempt_cursors",
            "snapshot_cursors",
            "attempt_paths",
            "attempt_success",
            "attempt_failure_codes",
            "attempt_donor_draw_counts",
            "registered_path_codes",
            "path_rng_counter_states",
            "path_counter_write_counts",
            "path_event_row_write_counts",
            "path_rng_state_write_counts",
        )
        if any(not isinstance(getattr(exposure, name, None), torch.Tensor) for name in required):
            raise _fail("resident exposure lacks exact direct-projection tensors")
        for cell, row in enumerate(rows):
            attempts = int(exposure.attempt_cursors[cell].item())
            snapshots = int(exposure.snapshot_cursors[cell].item())
            path_code = row["native_path_code"]
            attempt_paths = exposure.attempt_paths[cell, :attempts]
            if attempts and bool((attempt_paths != path_code).any().item()):
                raise _fail("exposure attempt path differs from captured control")
            success = exposure.attempt_success[cell, :attempts]
            failures = exposure.attempt_failure_codes[cell, :attempts]
            if bool((success & (failures != 0)).any().item()) or bool(
                ((~success) & (failures <= 0)).any().item()
            ):
                raise _fail("exposure success/failure rows are contradictory")
            observed = row["path_observation"]
            observed["authoritative_attempt_rows"] += attempts
            observed["successful_commit_rows"] += int(success.sum().item())
            observed["failed_attempt_rows"] += attempts - int(
                success.sum().item()
            )
            observed["eligibility_snapshot_rows"] += snapshots
            observed["donor_draw_rows"] += int(
                exposure.attempt_donor_draw_counts[cell, :attempts].sum().item()
            )
            observed["sidecar_attempt_rows"] += attempts
            registered_codes = tuple(
                int(value)
                for value in exposure.registered_path_codes.detach()
                .to(device="cpu")
                .tolist()
            )
            if (
                registered_codes != tuple(REGISTERED_PATH_ID_BY_CODE_V1)
                or len(set(registered_codes)) != len(registered_codes)
            ):
                raise _fail(
                    "resident exposure registered path axis differs from the "
                    "observer path registry"
                )
            instrumentation: dict[str, dict[str, Any]] = {}
            for path_index, registered_code in enumerate(registered_codes):
                path_id = REGISTERED_PATH_ID_BY_CODE_V1[registered_code]
                instrumentation[path_id] = {
                    "schema_version": (
                        DIRECT_PROJECTION_PATH_INSTRUMENTATION_SCHEMA_VERSION
                    ),
                    "path_id": path_id,
                    "native_path_code": registered_code,
                    "rng_counter_state": int(
                        exposure.path_rng_counter_states[cell, path_index].item()
                    ),
                    "counter_write_count": int(
                        exposure.path_counter_write_counts[cell, path_index].item()
                    ),
                    "event_row_write_count": int(
                        exposure.path_event_row_write_counts[cell, path_index].item()
                    ),
                    "rng_state_write_count": int(
                        exposure.path_rng_state_write_counts[cell, path_index].item()
                    ),
                }
            row["path_instrumentation"] = instrumentation

    def direct_projection_capture_payload(self) -> dict[str, Any]:
        """Return exact path-local counters and write planes from the native ABI."""

        if self._direct_capture_cells is None:
            reason = self._direct_capture_unavailable_reason or (
                "observer has not received a prepared resident owner"
            )
            raise _fail(f"direct projection capture is unavailable: {reason}")
        path_code_by_id = {
            path_id: path_code
            for path_code, path_id in REGISTERED_PATH_ID_BY_CODE_V1.items()
        }
        for row in self._direct_capture_cells:
            path_id = row["reproduction_path"]
            path_code = row["native_path_code"]
            prior = path_code_by_id.setdefault(path_id, path_code)
            if prior != path_code:
                raise _fail("one path ID maps to multiple native path codes")
            instrumentation = row.get("path_instrumentation")
            if (
                not isinstance(instrumentation, Mapping)
                or tuple(instrumentation) != tuple(path_code_by_id)
            ):
                raise _fail("direct projection path instrumentation is incomplete")
        cells = [
            {
                **row,
                "path_observation": dict(row["path_observation"]),
                "path_instrumentation": {
                    path_id: dict(value)
                    for path_id, value in row["path_instrumentation"].items()
                },
            }
            for row in self._direct_capture_cells
        ]
        return {
            "schema_version": DIRECT_PROJECTION_CAPTURE_SCHEMA_VERSION,
            "captured_at_utc": self._direct_capture_started_at_utc,
            "path_code_by_id": path_code_by_id,
            "failure_code_by_value": {
                **{
                    str(code): failure_id
                    for code, failure_id in LIVE_BIRTH_FAILURE_ID_BY_CODE_V1.items()
                },
                str(FAILURE_PENDING_COMMIT): "pending_authoritative_commit",
                str(FAILURE_AUTHORITATIVE_COMMIT_ABORTED): (
                    "authoritative_commit_aborted"
                ),
            },
            "cells": cells,
            "instrumentation_owner": (
                "cuda_resident_reproduction_exposure_checkpoint_v1"
            ),
        }

    def payload(self) -> dict[str, Any]:
        fault_observation = None
        if self._fault_control is not None:
            if None in {
                self._fault_status_before,
                self._fault_status_after,
                self._fault_status_after_drain,
                self._fault_mechanics_before,
                self._fault_mechanics_after,
                self._fault_peer_before,
                self._fault_peer_after,
            }:
                raise _fail("capacity-fault observation is incomplete")
            fault_observation_value = ResidentCapacityFaultObservationV1(
                control_authority_sha256=(
                    self._fault_control.control_authority_sha256
                ),
                target_occurrence_authority_sha256=(
                    self._fault_control.target_occurrence_authority_sha256
                ),
                target_cell_index=self._fault_control.target_cell_index,
                target_cell_authority_sha256=(
                    self._fault_control.target_cell_authority_sha256
                ),
                pre_fault_authority_sha256=(
                    self._fault_control.pre_fault_authority_sha256
                ),
                status_before_code=int(self._fault_status_before),
                status_after_code=int(self._fault_status_after),
                status_after_drain_code=int(self._fault_status_after_drain),
                mechanics_before_sha256=str(self._fault_mechanics_before),
                mechanics_after_sha256=str(self._fault_mechanics_after),
                peer_count=len(self._fixed_mask or ())
                - 1
                if self._fixed_mask is not None
                else int(self._batch_cells) - 1,
                peer_set_before_sha256=str(self._fault_peer_before),
                peer_set_after_sha256=str(self._fault_peer_after),
            )
            fault_observation_value.validate_against(self._fault_control)
            fault_observation = fault_observation_value.payload()

        fixed_observation = None
        if self._fixed_control is not None:
            if (
                self._fixed_evaluations_before is None
                or self._fixed_draws_before is None
                or self._fixed_draws_after is None
                or not self._fixed_actions
            ):
                raise _fail("fixed-action observation is incomplete")
            fixed_value = ResidentFixedActionObservationV1(
                control_authority_sha256=(
                    self._fixed_control.control_authority_sha256
                ),
                cell_mask_sha256=self._fixed_control.cell_mask_sha256,
                action_id=self._fixed_control.action_id,
                observed_action_count=len(self._fixed_actions),
                all_rest_actions_sha256=canonical_sha256(self._fixed_actions),
                policy_evaluation_count_before=self._fixed_evaluations_before,
                policy_evaluation_count_after=self._fixed_evaluations_after,
                policy_evaluation_count_delta=(
                    self._fixed_evaluations_after
                    - self._fixed_evaluations_before
                ),
                semantic_draw_count_before=self._fixed_draws_before,
                semantic_draw_count_after=self._fixed_draws_after,
                semantic_draw_count_delta=(
                    self._fixed_draws_after - self._fixed_draws_before
                ),
            )
            fixed_value.validate_against(self._fixed_control)
            fixed_observation = fixed_value.payload()

        return {
            "capacity_fault_control": (
                None
                if self._fault_control is None
                else self._fault_control.payload()
            ),
            "capacity_fault_observation": fault_observation,
            "fixed_action_control": (
                None
                if self._fixed_control is None
                else self._fixed_control.payload()
            ),
            "fixed_action_observation": fixed_observation,
            "resident_dispatch_observations": self._resident_observations,
        }


__all__ = [
    "CrossPathNativeQualificationObserverV1",
    "DIRECT_PROJECTION_CAPTURE_SCHEMA_VERSION",
    "DIRECT_PROJECTION_CELL_CAPTURE_SCHEMA_VERSION",
    "DIRECT_PROJECTION_PATH_INSTRUMENTATION_SCHEMA_VERSION",
    "FAULT_DIRECTIVE_SCHEMA_VERSION",
    "InheritanceCrossPathNativeQualificationObserverError",
    "ResidentAttemptCapacityFaultDirectiveV1",
]
