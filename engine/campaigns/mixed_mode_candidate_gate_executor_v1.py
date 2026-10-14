"""Generic factory and executor for a preregistered mixed-mode candidate gate.

This module owns orchestration only.  It does not compose or alter a gate plan,
candidate binding, campaign plan, callback authority, or simulation mechanic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inheritance_cross_path_candidate_isolation_bridge_v1 import (
    InheritanceCrossPathCandidateIsolationBridgeV1,
)
from .inheritance_cross_path_candidate_plan_v1 import (
    InheritanceCrossPathCandidatePlanV1,
    bind_inheritance_cross_path_candidate_plan_v1,
)
from .inheritance_cross_path_mechanic_gate_adapter_v1 import (
    InheritanceCrossPathMechanicGateCallbackV1,
)
from .mixed_mode_resident_isolation_plan_composer_v1 import (
    verify_composed_plan_artifact_v1,
)
from .mixed_mode_resident_isolation_runner import (
    MixedModeIsolationRunnerReceiptV1,
    MixedModeResidentIsolationRunnerPlanV1,
    execute_mixed_mode_resident_isolation_gate_v1,
)


class MixedModeCandidateGateExecutorError(ValueError):
    """The immutable candidate-gate inputs do not bind one execution."""


def _fail(message: str) -> MixedModeCandidateGateExecutorError:
    return MixedModeCandidateGateExecutorError(message)


def _path_ids(
    runner_plan: MixedModeResidentIsolationRunnerPlanV1,
) -> tuple[str, ...]:
    reference_id = runner_plan.gate_plan.mixed_reference_run_id
    reference = next(
        (run for run in runner_plan.runs if run.run_id == reference_id),
        None,
    )
    if reference is None:
        raise _fail("mixed reference run is absent from the runner plan")
    cells = tuple(sorted(reference.cells, key=lambda cell: cell.batch_position))
    paths = tuple(cell.config.reproduction_path for cell in cells)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise _fail("mixed reference paths must be unique and contain at least two")
    return paths


def _artifact_root(value: Path) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise _fail("artifact_root must be absolute")
    if root.exists() or root.is_symlink():
        raise _fail("artifact_root must be a new non-symlink path")
    for parent in root.parents:
        if parent.exists() and parent.is_symlink():
            raise _fail("artifact_root must not traverse a symlink")
    return root


@dataclass(frozen=True, slots=True)
class PreparedMixedModeCandidateGateExecutionV1:
    """Fully validated execution binding, before any evidence directory exists."""

    runner_plan: MixedModeResidentIsolationRunnerPlanV1
    candidate_plan: InheritanceCrossPathCandidatePlanV1
    bridge: InheritanceCrossPathCandidateIsolationBridgeV1
    callback: InheritanceCrossPathMechanicGateCallbackV1
    path_ids: tuple[str, ...]

    def execute(self, artifact_root: Path) -> MixedModeIsolationRunnerReceiptV1:
        root = _artifact_root(artifact_root)
        if (
            self.callback.execution_authority_sha256
            != self.runner_plan.execution_authority_sha256
        ):
            raise _fail(
                "prepared callback authority differs from runner preregistration"
            )
        return execute_mixed_mode_resident_isolation_gate_v1(
            self.runner_plan,
            self.callback,
            root,
        )


def prepare_mixed_mode_candidate_gate_execution_v1(
    *,
    composed_plan_artifact: Mapping[str, Any],
    candidate_binding_manifest: Path,
    compiled_source_campaign_plan: Mapping[str, Any],
    logical_template_cell_ids: Mapping[str, str],
    direct_segment_owner_id: str,
    direct_segment_owner_sha256: str,
) -> PreparedMixedModeCandidateGateExecutionV1:
    """Reject preparation of the retired mixed-mode candidate gate."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "mixed-mode candidate-gate preparation is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    verified = verify_composed_plan_artifact_v1(composed_plan_artifact)
    runner_plan = MixedModeResidentIsolationRunnerPlanV1.from_payload(
        verified["runner_plan"]
    )
    if runner_plan.digest_sha256 != verified["runner_plan_sha256"]:
        raise _fail("typed runner plan differs from the composed artifact identity")

    candidate_plan = bind_inheritance_cross_path_candidate_plan_v1(
        compiled_source_campaign_plan,
        Path(candidate_binding_manifest),
    )
    bridge = InheritanceCrossPathCandidateIsolationBridgeV1(
        runner_plan=runner_plan,
        candidate_plan=candidate_plan,
        logical_template_cell_ids=logical_template_cell_ids,
    )
    paths = _path_ids(runner_plan)
    callback = bridge.build_callback(
        direct_segment_owner_id=direct_segment_owner_id,
        direct_segment_owner_sha256=direct_segment_owner_sha256,
        path_ids=paths,
    )
    if callback.execution_authority_sha256 != runner_plan.execution_authority_sha256:
        raise _fail("callback authority differs from runner preregistration")
    return PreparedMixedModeCandidateGateExecutionV1(
        runner_plan=runner_plan,
        candidate_plan=candidate_plan,
        bridge=bridge,
        callback=callback,
        path_ids=paths,
    )


def execute_mixed_mode_candidate_gate_v1(
    *,
    composed_plan_artifact: Mapping[str, Any],
    candidate_binding_manifest: Path,
    compiled_source_campaign_plan: Mapping[str, Any],
    logical_template_cell_ids: Mapping[str, str],
    artifact_root: Path,
    direct_segment_owner_id: str,
    direct_segment_owner_sha256: str,
) -> MixedModeIsolationRunnerReceiptV1:
    """Reject execution of the retired mixed-mode candidate gate."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "mixed-mode candidate-gate execution is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    prepared = prepare_mixed_mode_candidate_gate_execution_v1(
        composed_plan_artifact=composed_plan_artifact,
        candidate_binding_manifest=candidate_binding_manifest,
        compiled_source_campaign_plan=compiled_source_campaign_plan,
        logical_template_cell_ids=logical_template_cell_ids,
        direct_segment_owner_id=direct_segment_owner_id,
        direct_segment_owner_sha256=direct_segment_owner_sha256,
    )
    return prepared.execute(artifact_root)


__all__ = [
    "MixedModeCandidateGateExecutorError",
    "PreparedMixedModeCandidateGateExecutionV1",
    "execute_mixed_mode_candidate_gate_v1",
    "prepare_mixed_mode_candidate_gate_execution_v1",
]
