"""Runtime-backend composite state authority for Patch/WDL/Jungle/EED releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Final

import torch

from engine.perf.cxxbackbone10_resident_state_owner import projection_digest

COMPOSITE_RUNTIME_STATE_SCHEMA_VERSION: Final[str] = (
    "eiel_patch_wdl_jungle_eed_composite_runtime_state_v1"
)


class CompositeRuntimeStateError(ValueError):
    """Integrated state is incomplete, contradictory, or multiply owned."""


def _tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    try:
        return bool(left == right)
    except (RuntimeError, TypeError, ValueError):
        return False


def _clone_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def _sidecar_identity_rows(
    value: Any, *, path: str = "projection"
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_path = f"{path}.{name}"
            if (
                isinstance(child, Mapping)
                and "cell_id" in child
                and (
                    "sidecar" in str(name)
                    or "checkpoint_projection" in str(name)
                    or "sidecar" in str(child.get("schema_version", ""))
                )
            ):
                rows.append((child_path, child))
            rows.extend(_sidecar_identity_rows(child, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            rows.extend(_sidecar_identity_rows(child, path=f"{path}[{index}]"))
    return tuple(rows)


def _validate_capability_markers(
    capability_mask: Mapping[str, Any], execution_markers: Mapping[str, Any]
) -> None:
    marker_by_capability = {
        "wdl": "wdl_jungle_native_owner",
        "jungle": "wdl_jungle_native_owner",
        "eed": "eed_native_owner",
    }
    for capability, marker in marker_by_capability.items():
        admitted = capability_mask.get(capability)
        if type(admitted) is not bool:
            continue
        executed = execution_markers.get(marker)
        if admitted is True and executed is not True:
            raise CompositeRuntimeStateError(
                f"capability {capability!r} lacks execution marker {marker!r}"
            )
        if admitted is False and executed is True:
            raise CompositeRuntimeStateError(
                f"execution marker {marker!r} contradicts capability {capability!r}"
            )


@dataclass(frozen=True, slots=True)
class CompositeRuntimeStateV1:
    """One final projection plus a non-wire ownership and identity proof."""

    projection: Mapping[str, Any]
    tensor_owners: Mapping[str, tuple[str, ...]]
    metadata_owner: str
    component_ids: tuple[str, ...]
    projection_sha256: str
    projection_digest_fn: Callable[[Mapping[str, Any]], str] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.metadata_owner or not self.component_ids:
            raise CompositeRuntimeStateError("composite state ownership is incomplete")
        tensors = self.projection.get("tensors")
        metadata = self.projection.get("metadata")
        if not isinstance(tensors, Mapping) or not isinstance(metadata, Mapping):
            raise CompositeRuntimeStateError(
                "composite projection requires metadata and tensors mappings"
            )
        if set(tensors) != set(self.tensor_owners):
            raise CompositeRuntimeStateError(
                "every final tensor must have one declared ownership row"
            )
        if any(
            not owners or len(owners) != len(set(owners))
            for owners in self.tensor_owners.values()
        ):
            raise CompositeRuntimeStateError("tensor ownership rows must be non-empty")
        if self.projection_digest_fn(self.projection) != self.projection_sha256:
            raise CompositeRuntimeStateError(
                "composite digest was not computed from the final projection"
            )

    def wire_projection(self) -> dict[str, Any]:
        """Return the unchanged checkpoint wire shape."""

        return dict(self.projection)


def compose_composite_runtime_state_v1(
    base_projection: Mapping[str, Any],
    *,
    base_component_id: str,
    component_tensors: Mapping[str, Mapping[str, Any]],
    shared_tensor_owners: Mapping[str, Sequence[str]] | None = None,
    expected_existing_tensors: Mapping[str, Any] | None = None,
    execution_markers: Mapping[str, Any] | None = None,
    expected_cell_id: str | None = None,
    projection_digest_fn: Callable[[Mapping[str, Any]], str],
) -> CompositeRuntimeStateV1:
    """Compose final state once while preserving the established wire payload."""

    if not isinstance(base_projection, Mapping):
        raise CompositeRuntimeStateError("base projection must be a mapping")
    if not base_component_id:
        raise CompositeRuntimeStateError("base component identity must be non-empty")
    projection = dict(base_projection)
    raw_tensors = projection.get("tensors")
    metadata = projection.get("metadata")
    if not isinstance(raw_tensors, Mapping) or not isinstance(metadata, Mapping):
        raise CompositeRuntimeStateError(
            "base projection requires metadata and tensors mappings"
        )
    tensors = {str(name): value for name, value in raw_tensors.items()}
    owners: dict[str, tuple[str, ...]] = {
        name: (base_component_id,) for name in tensors
    }
    components = [base_component_id]
    for component_id, contribution in component_tensors.items():
        if not component_id or component_id in components:
            raise CompositeRuntimeStateError(
                "component identities must be non-empty and unique"
            )
        if not isinstance(contribution, Mapping):
            raise CompositeRuntimeStateError(
                f"component {component_id!r} tensor contribution is malformed"
            )
        components.append(component_id)
        for raw_name, value in contribution.items():
            name = str(raw_name)
            if name in tensors and not _tensor_equal(tensors[name], value):
                raise CompositeRuntimeStateError(
                    f"component tensor {name!r} contradicts the base projection"
                )
            tensors[name] = _clone_tensor(value)
            owners[name] = (component_id,)
    for name, expected in dict(expected_existing_tensors or {}).items():
        if name not in tensors or not _tensor_equal(tensors[name], expected):
            raise CompositeRuntimeStateError(
                f"canonical shared tensor {name!r} differs from its resident owner"
            )
    for name, raw_owners in dict(shared_tensor_owners or {}).items():
        declared = tuple(str(owner) for owner in raw_owners)
        if (
            name not in tensors
            or not declared
            or any(owner not in components for owner in declared)
        ):
            raise CompositeRuntimeStateError(
                f"shared tensor ownership for {name!r} is invalid"
            )
        owners[name] = declared
    capability_mask = metadata.get("capability_mask")
    if execution_markers:
        if not isinstance(capability_mask, Mapping):
            raise CompositeRuntimeStateError(
                "terminal composite state lacks a capability mask"
            )
        _validate_capability_markers(capability_mask, execution_markers)
    if expected_cell_id is not None:
        if not expected_cell_id:
            raise CompositeRuntimeStateError("expected cell identity is empty")
        metadata_cell_id = metadata.get("cell_id")
        if metadata_cell_id is not None and metadata_cell_id != expected_cell_id:
            raise CompositeRuntimeStateError(
                "projection metadata cell identity differs from its batch owner"
            )
        for path, row in _sidecar_identity_rows(projection):
            if row.get("cell_id") != expected_cell_id:
                raise CompositeRuntimeStateError(
                    f"sidecar identity differs from its cell owner at {path}"
                )
    projection["metadata"] = dict(metadata)
    projection["tensors"] = tensors
    digest = projection_digest_fn(projection)
    return CompositeRuntimeStateV1(
        projection=projection,
        tensor_owners=dict(sorted(owners.items())),
        metadata_owner=base_component_id,
        component_ids=tuple(components),
        projection_sha256=digest,
        projection_digest_fn=projection_digest_fn,
    )


def validate_composite_state_exact_equality_v1(
    source: CompositeRuntimeStateV1,
    restored: CompositeRuntimeStateV1,
) -> None:
    """Fail closed unless checkpoint reload reconstructs exact canonical state."""

    if source.component_ids != restored.component_ids:
        raise CompositeRuntimeStateError(
            "restored composite component identity differs"
        )
    if dict(source.tensor_owners) != dict(restored.tensor_owners):
        raise CompositeRuntimeStateError("restored tensor ownership differs")
    if source.projection_sha256 != restored.projection_sha256:
        raise CompositeRuntimeStateError("restored composite projection differs")


__all__ = [
    "COMPOSITE_RUNTIME_STATE_SCHEMA_VERSION",
    "CompositeRuntimeStateError",
    "CompositeRuntimeStateV1",
    "compose_composite_runtime_state_v1",
    "projection_digest",
    "validate_composite_state_exact_equality_v1",
]
