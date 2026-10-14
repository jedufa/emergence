"""Fail-closed Phase F v2 forced-combination battery orchestration.

The battery has three deliberately separate stages: deterministic planning,
CPU expectation materialization, and authorized CUDA execution.  Importing this
module never loads a native extension or touches a GPU.  Every accelerated item
is a fresh, one-cell generation-owner invocation and is persisted before the
next item starts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Final

import torch

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import validate_phase_f_v2_matrix
from evolution.phase_f_v2.receipts import validate_matrix_registry_receipt
from evolution.phase_f_v2.storage import atomic_write_canonical_json
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    modular_storage_from_projection,
)
from perf.cuda_architecture_b_phase_e_gpu_gate import (
    PhaseEGpuAuthorization,
    PhaseEGpuGateBindings,
    require_phase_e_gpu_authorization,
    validate_phase_e_gpu_final_receipt,
)
from perf.eiel_phase_f_boundary_actions_v2 import (
    build_checkpoint_schedule_descriptor,
    capture_modular_checkpoint,
    restore_fresh_modular_owner,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    validate_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_execution_source_v2 import (
    PreparedPhaseFV2ExecutionSource,
    prepare_phase_f_execution_source_v2,
    validate_prepared_phase_f_execution_source_v2,
)
from perf.eiel_phase_f_forced_combination_gate_v2 import (
    EXPECTED_ACCELERATED_INVOCATIONS,
    EXPECTED_CPU_EVALUATIONS,
    EXPECTED_LOGICAL_CASES,
    EXPECTED_RESUMED_INVOCATIONS,
    EXPECTED_UNINTERRUPTED_INVOCATIONS,
    compare_phase_f_invocation_v2,
)
from perf.eiel_phase_f_forced_combination_oracle_v2 import (
    build_phase_f_cpu_expectation_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    PANEL_AGENT_COUNT,
    PhaseFV2SourceState,
)


PLAN_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_battery_plan_v1"
ENTRY_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_execution_entry_receipt_v1"
CPU_MANIFEST_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_cpu_manifest_v1"
LEDGER_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_resume_ledger_v1"
OWNER_WITNESS_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_owner_witness_v1"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "status",
        "source_commit",
        "matrix_sha256",
        "matrix_receipt_sha256",
        "plan_sha256",
        "fixture_registry_sha256",
        "source_manifest_sha256",
        "cpu_manifest_sha256",
        "phase_e_final_receipt_sha256",
        "phase_f_authorized",
        "smoke_required_before_full",
        "logical_case_count",
        "cpu_expectation_count",
        "accelerated_invocation_count",
        "receipt_sha256",
    }
)

# These are required battery invocations, not extra work.  They are ordered at
# the front and must pass before the remaining 264 items may start.
SMOKE_KEYS: Final[tuple[tuple[str, str, int], ...]] = (
    ("pf2_f01_b12_r0", "uninterrupted", 0),  # whole-invocation disabled
    ("pf2_f01_b12_r1", "uninterrupted", 0),  # enabled, empty bank
    ("pf2_f06_b16_r0", "uninterrupted", 0),  # structured composite
    ("pf2_f12_b16_r4", "uninterrupted", 0),  # boundary-family source
)


class PhaseFV2BatteryError(RuntimeError):
    """The battery is incomplete, unauthorized, dirty, or mismatched."""


def _sha(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFV2BatteryError(f"{field} must be a lowercase SHA-256")
    return value


def _commit(value: Any, *, field: str = "source_commit") -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise PhaseFV2BatteryError(f"{field} must be a full lowercase commit")
    return value


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in matrix["cases"]}


def _fixture_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in registry["cases"]}


def _source_seed(case: Mapping[str, Any]) -> int:
    return int(case["fixture_blueprint"]["construction_seed"])


def build_phase_f_battery_plan_v2(
    *, matrix: Any, fixture_registry: Any
) -> dict[str, Any]:
    """Build the exact 120/134/268 plan; no execution is performed."""

    checked_matrix = validate_phase_f_v2_matrix(matrix)
    fixtures = validate_phase_f_fixture_registry_v2(
        fixture_registry, matrix=checked_matrix
    )
    fixture_by_id = _fixture_map(fixtures)
    cpu: list[dict[str, Any]] = []
    accelerated: list[dict[str, Any]] = []
    for case in checked_matrix["cases"]:
        case_id = str(case["case_id"])
        fixture = fixture_by_id[case_id]
        common = {
            "case_id": case_id,
            "case_sha256": case["case_sha256"],
            "fixture_blueprint_sha256": case["fixture_blueprint"][
                "fixture_blueprint_sha256"
            ],
            "fixture_sha256": fixture["fixture_sha256"],
            "profile_id": case["profile_id"],
            "source_seed": _source_seed(case),
            "checkpoint_kind": case["checkpoint_schedule"]["kind"],
        }
        cpu.append({**common, "expectation_kind": "uninterrupted"})
        if common["checkpoint_kind"] != "none":
            cpu.append({**common, "expectation_kind": "resumed"})
        for repeat in range(2):
            accelerated.append(
                {
                    **common,
                    "invocation_kind": "uninterrupted",
                    "repeat_index": repeat,
                    "boundary_index": 0,
                    "execution_id": f"{case_id}.uninterrupted.{repeat}",
                }
            )
        if common["checkpoint_kind"] != "none":
            for repeat in range(2):
                accelerated.append(
                    {
                        **common,
                        "invocation_kind": "resumed",
                        "repeat_index": repeat,
                        "boundary_index": 1,
                        "execution_id": f"{case_id}.resumed.{repeat}",
                    }
                )

    smoke_lookup = {
        (item["case_id"], item["invocation_kind"], item["repeat_index"]): item
        for item in accelerated
    }
    smoke = [smoke_lookup[key] for key in SMOKE_KEYS]
    smoke_ids = {item["execution_id"] for item in smoke}
    ordered_accelerated = [
        *smoke,
        *(x for x in accelerated if x["execution_id"] not in smoke_ids),
    ]
    if (
        len(checked_matrix["cases"]) != EXPECTED_LOGICAL_CASES
        or len(cpu) != EXPECTED_CPU_EVALUATIONS
        or sum(x["invocation_kind"] == "uninterrupted" for x in accelerated)
        != EXPECTED_UNINTERRUPTED_INVOCATIONS
        or sum(x["invocation_kind"] == "resumed" for x in accelerated)
        != EXPECTED_RESUMED_INVOCATIONS
        or len(accelerated) != EXPECTED_ACCELERATED_INVOCATIONS
    ):
        raise PhaseFV2BatteryError("battery accounting is not exactly 120/134/268")
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "fixture_registry_sha256": fixtures["registry_sha256"],
        "logical_case_count": EXPECTED_LOGICAL_CASES,
        "cpu_expectation_count": EXPECTED_CPU_EVALUATIONS,
        "uninterrupted_invocation_count": EXPECTED_UNINTERRUPTED_INVOCATIONS,
        "resumed_invocation_count": EXPECTED_RESUMED_INVOCATIONS,
        "accelerated_invocation_count": EXPECTED_ACCELERATED_INVOCATIONS,
        "smoke_execution_ids": [item["execution_id"] for item in smoke],
        "cpu_expectations": cpu,
        "accelerated_invocations": ordered_accelerated,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_clean_frozen_checkout(*, repo_root: str | Path, source_commit: str) -> None:
    """Require exact HEAD and no tracked or untracked worktree changes."""

    expected = _commit(source_commit)
    root = Path(repo_root).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if head != expected:
        raise PhaseFV2BatteryError(
            f"checkout HEAD {head} is not frozen commit {expected}"
        )
    if status:
        raise PhaseFV2BatteryError("frozen execution checkout is not clean")


def validate_phase_e_final_for_phase_f_v2(
    *,
    final_receipt: Mapping[str, Any],
    qualification_receipt: Mapping[str, Any],
    bindings: PhaseEGpuGateBindings,
    source_commit: str,
) -> PhaseEGpuAuthorization:
    """Mint and require the official final Phase E Phase-F authorization."""

    authorization = validate_phase_e_gpu_final_receipt(
        final_receipt, bindings=bindings, qualification_receipt=qualification_receipt
    )
    if authorization.source_commit != _commit(source_commit):
        raise PhaseFV2BatteryError("Phase E receipt names a different frozen commit")
    if authorization.phase_f_authorized is not True:
        raise PhaseFV2BatteryError("Phase E final receipt does not authorize Phase F")
    if set(final_receipt.get("logical_profiles", ())) != {
        "b2-12x32-v1",
        "b2-16x48-v1",
    }:
        raise PhaseFV2BatteryError(
            "Phase E final receipt profile envelope is incomplete"
        )
    for profile in authorization.logical_profiles:
        require_phase_e_gpu_authorization(authorization, final=True, profile_id=profile)
    return authorization


def build_phase_f_entry_receipt_v2(
    *,
    plan: Mapping[str, Any],
    matrix_receipt: Mapping[str, Any],
    source_commit: str,
    phase_e_final_receipt: Mapping[str, Any],
    phase_e_authorization: PhaseEGpuAuthorization,
    cpu_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Cross-bind all immutable authorities before the first CUDA launch."""

    checked_matrix_receipt = validate_matrix_registry_receipt(matrix_receipt)
    commit = _commit(source_commit)
    verify_clean_frozen_checkout(repo_root=repo_root, source_commit=commit)
    if checked_matrix_receipt["source_commit"] != commit:
        raise PhaseFV2BatteryError("matrix receipt source commit mismatch")
    if plan.get("plan_sha256") != canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise PhaseFV2BatteryError("battery plan digest mismatch")
    if cpu_manifest.get("manifest_sha256") != canonical_sha256(
        {key: value for key, value in cpu_manifest.items() if key != "manifest_sha256"}
    ):
        raise PhaseFV2BatteryError("CPU manifest digest mismatch")
    if cpu_manifest.get("expectation_count") != EXPECTED_CPU_EVALUATIONS:
        raise PhaseFV2BatteryError("CPU manifest is incomplete")
    bound_source_manifest = _sha(
        cpu_manifest.get("source_manifest_sha256"),
        field="cpu_manifest.source_manifest_sha256",
    )
    if source_manifest_sha256 != bound_source_manifest:
        raise PhaseFV2BatteryError(
            "source manifest does not match the CPU-derived source inventory"
        )
    final_sha = canonical_sha256(dict(phase_e_final_receipt))
    if phase_e_final_receipt.get("receipt_sha256") != final_sha:
        # Gate receipts self-hash with receipt_sha256 omitted.
        unsigned = dict(phase_e_final_receipt)
        unsigned.pop("receipt_sha256", None)
        final_sha = canonical_sha256(unsigned)
        if phase_e_final_receipt.get("receipt_sha256") != final_sha:
            raise PhaseFV2BatteryError("Phase E final receipt digest mismatch")
    expected_matrix_bindings = {
        "phase_e_authorization_sha256": final_sha,
        "matrix_spec_sha256": plan["matrix_sha256"],
        "fixture_manifest_sha256": plan["fixture_registry_sha256"],
        "cell_registry_sha256": plan["plan_sha256"],
    }
    for field, expected_value in expected_matrix_bindings.items():
        if checked_matrix_receipt[field] != expected_value:
            raise PhaseFV2BatteryError(f"matrix receipt {field} binding mismatch")
    if (
        not isinstance(phase_e_authorization, PhaseEGpuAuthorization)
        or phase_e_authorization.source_commit != commit
        or phase_e_authorization.receipt_sha256 != final_sha
        or phase_e_authorization.phase_f_authorized is not True
    ):
        raise PhaseFV2BatteryError("Phase E authorization binding mismatch")
    for profile_id in phase_e_authorization.logical_profiles:
        require_phase_e_gpu_authorization(
            phase_e_authorization, final=True, profile_id=profile_id
        )
    receipt: dict[str, Any] = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "status": "execution_entry_authorized",
        "source_commit": commit,
        "matrix_sha256": _sha(plan["matrix_sha256"], field="matrix_sha256"),
        "matrix_receipt_sha256": checked_matrix_receipt["receipt_sha256"],
        "plan_sha256": _sha(plan["plan_sha256"], field="plan_sha256"),
        "fixture_registry_sha256": _sha(
            plan["fixture_registry_sha256"], field="fixture_registry_sha256"
        ),
        "source_manifest_sha256": bound_source_manifest,
        "cpu_manifest_sha256": cpu_manifest["manifest_sha256"],
        "phase_e_final_receipt_sha256": final_sha,
        "phase_f_authorized": True,
        "smoke_required_before_full": True,
        "logical_case_count": EXPECTED_LOGICAL_CASES,
        "cpu_expectation_count": EXPECTED_CPU_EVALUATIONS,
        "accelerated_invocation_count": EXPECTED_ACCELERATED_INVOCATIONS,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _fresh_checkpoint_source(
    *,
    matrix: Mapping[str, Any],
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_registry: Mapping[str, Any],
    source: PhaseFV2SourceState,
    source_preparation_receipt: Mapping[str, Any],
    prerequisite_receipt: Mapping[str, Any] | None,
) -> tuple[PhaseFV2SourceState, dict[str, Any]]:
    kind = str(case["checkpoint_schedule"]["kind"])
    if kind == "none":
        raise PhaseFV2BatteryError("non-checkpoint case requested a restored owner")
    authority = modular_storage_from_projection(
        source.owner_metadata, source.generation_tensors
    )
    if authority is None:
        raise PhaseFV2BatteryError("source modular owner projection is missing")
    try:
        schedule = build_checkpoint_schedule_descriptor(
            case_id=str(case["case_id"]),
            case_sha256=str(case["case_sha256"]),
            matrix_sha256=str(matrix["matrix_sha256"]),
            profile_id=str(case["profile_id"]),
            schedule_kind=kind,
            prerequisite_receipt=prerequisite_receipt,
        )
        checkpoint = capture_modular_checkpoint(authority, schedule)
        restored = restore_fresh_modular_owner(checkpoint)
    except Exception as exc:
        raise PhaseFV2BatteryError(
            f"checkpoint preparation failed closed for {case['case_id']}: {exc}"
        ) from exc
    if restored.owner.packed_sha256 != authority.packed_sha256:
        raise PhaseFV2BatteryError("fresh modular owner changed packed authority")
    restored_tensors = {
        **{
            name: value.detach().clone().contiguous()
            for name, value in source.generation_tensors.items()
            if not name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
        },
        **{
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}": value.detach()
            .clone()
            .contiguous()
            for name, value in restored.owner.tensors.items()
        },
    }
    restored_formal_inputs = tuple(
        replace(
            formal,
            planes={
                name: value[slot : slot + 1].detach().clone().contiguous()
                for name, value in restored.owner.tensors.items()
            },
        )
        for slot, formal in enumerate(source.formal_oracle_inputs)
    )
    fresh = replace(
        source,
        generation_tensors=restored_tensors,
        formal_oracle_inputs=restored_formal_inputs,
    )
    if any(
        fresh.generation_tensors[name].data_ptr()
        == source.generation_tensors[name].data_ptr()
        for name in fresh.generation_tensors
    ):
        raise PhaseFV2BatteryError("fresh restored source aliases original tensors")
    # Re-materialization inside the validator recomputes the canonical source
    # digest.  Equal restored bytes retain the same digest; changed bytes fail
    # closed instead of silently borrowing the pre-checkpoint binding.
    validate_prepared_phase_f_execution_source_v2(
        fresh,
        matrix=matrix,
        fixture_registry=fixture_registry,
        case_id=str(case["case_id"]),
        preparation_receipt=source_preparation_receipt,
    )
    evidence = {
        "checkpoint_scope": "fresh_modular_storage_owner_only",
        "full_runtime_resume_claimed": False,
        "schedule_receipt": schedule,
        "capture_receipt": checkpoint.receipt,
        "restore_receipt": restored.receipt,
        "restored_source_sha256": fresh.source_sha256,
    }
    return fresh, evidence


def materialize_phase_f_cpu_manifest_v2(
    *,
    matrix: Any,
    fixture_registry: Any,
    plan: Mapping[str, Any],
    output_directory: str | Path,
    checkpoint_prerequisites: Mapping[str, Mapping[str, Any]] | None = None,
    migration_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute and atomically persist all 134 independent CPU expectations."""

    checked_matrix = validate_phase_f_v2_matrix(matrix)
    registry = validate_phase_f_fixture_registry_v2(
        fixture_registry, matrix=checked_matrix
    )
    if plan.get("plan_sha256") != canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise PhaseFV2BatteryError("CPU plan digest mismatch")
    if (
        plan.get("matrix_sha256") != checked_matrix["matrix_sha256"]
        or plan.get("fixture_registry_sha256") != registry["registry_sha256"]
        or plan.get("cpu_expectation_count") != EXPECTED_CPU_EVALUATIONS
        or len(plan.get("cpu_expectations", ())) != EXPECTED_CPU_EVALUATIONS
    ):
        raise PhaseFV2BatteryError("CPU plan binding or accounting mismatch")
    cases = _case_map(checked_matrix)
    fixtures = _fixture_map(registry)
    prerequisites = {} if checkpoint_prerequisites is None else checkpoint_prerequisites
    contexts = {} if migration_contexts is None else migration_contexts
    root = Path(output_directory)
    records: list[dict[str, Any]] = []
    prepared_by_case: dict[str, PreparedPhaseFV2ExecutionSource] = {}
    for item in plan["cpu_expectations"]:
        case_id = str(item["case_id"])
        case = cases[case_id]
        fixture = fixtures[case_id]
        prepared = prepared_by_case.get(case_id)
        if prepared is None:
            prepared = prepare_phase_f_execution_source_v2(
                matrix=checked_matrix,
                fixture_registry=registry,
                case_id=case_id,
                migration_context=contexts.get(case_id),
            )
            prepared_by_case[case_id] = prepared
        source = prepared.source
        prerequisite = prepared.checkpoint_prerequisite_receipt
        external_prerequisite = prerequisites.get(case_id)
        if external_prerequisite is not None:
            if prerequisite is not None and canonical_sha256(
                external_prerequisite
            ) != canonical_sha256(prerequisite):
                raise PhaseFV2BatteryError(
                    f"external checkpoint prerequisite contradicts {case_id}"
                )
            prerequisite = dict(external_prerequisite)
        restored: PhaseFV2SourceState | None = None
        checkpoint_evidence: dict[str, Any] | None = None
        if item["expectation_kind"] == "resumed":
            restored, checkpoint_evidence = _fresh_checkpoint_source(
                matrix=checked_matrix,
                case=case,
                fixture=fixture,
                fixture_registry=registry,
                source=source,
                source_preparation_receipt=prepared.preparation_receipt,
                prerequisite_receipt=prerequisite,
            )
        expectation = build_phase_f_cpu_expectation_v2(
            matrix=checked_matrix,
            fixture_case=fixture,
            source_state=source,
            expectation_kind=str(item["expectation_kind"]),
            restored_source_state=restored,
            fixture_registry=registry,
            source_preparation_receipt=prepared.preparation_receipt,
        )
        path = root / "cpu_expectations" / case_id / f"{item['expectation_kind']}.json"
        atomic_write_canonical_json(path, expectation)
        records.append(
            {
                "case_id": case_id,
                "expectation_kind": item["expectation_kind"],
                "fixture_sha256": fixture["fixture_sha256"],
                "source_sha256": expectation["execution_source_sha256"],
                "expectation_sha256": expectation["expectation_sha256"],
                "receipt_sha256": expectation["receipt_sha256"],
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "checkpoint_evidence": checkpoint_evidence,
                "source_preparation_receipt": prepared.preparation_receipt,
                "boundary_receipts": list(prepared.boundary_receipts),
            }
        )
    if len(records) != EXPECTED_CPU_EVALUATIONS:
        raise PhaseFV2BatteryError("CPU expectation manifest is incomplete")
    source_inventory = [
        {
            "case_id": case_id,
            "fixture_sha256": fixtures[case_id]["fixture_sha256"],
            "source_sha256": next(
                row["source_sha256"]
                for row in records
                if row["case_id"] == case_id
                and row["expectation_kind"] == "uninterrupted"
            ),
        }
        for case_id in (str(case["case_id"]) for case in checked_matrix["cases"])
    ]
    manifest: dict[str, Any] = {
        "schema_version": CPU_MANIFEST_SCHEMA_VERSION,
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "fixture_registry_sha256": registry["registry_sha256"],
        "expectation_count": len(records),
        "source_manifest_sha256": canonical_sha256(source_inventory),
        "records": records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_canonical_json(root / "cpu_expectation_manifest.json", manifest)
    return manifest


def _load_expectation(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    if _file_sha256(path) != record["file_sha256"]:
        raise PhaseFV2BatteryError("CPU expectation file digest mismatch")
    value = json.loads(path.read_text(encoding="ascii"))
    if value.get("receipt_sha256") != record["receipt_sha256"]:
        raise PhaseFV2BatteryError("CPU expectation receipt binding mismatch")
    return value


def _cpu_record_map(
    cpu_manifest: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["case_id"]), str(row["expectation_kind"])): dict(row)
        for row in cpu_manifest["records"]
    }


def _disabled_modular_audit_v2(
    modular_tensors: Mapping[str, torch.Tensor], *, action_count: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build the exact zero-contribution audit for a disabled invocation."""

    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    module_state = modular_tensors[f"{prefix}module_state"][..., 0]
    cost_components = modular_tensors[f"{prefix}structural_cost_components"]
    cost_total = modular_tensors[f"{prefix}structural_cost_total"]
    module_shape = tuple(module_state.shape)
    audit = {
        "development_dependency_ready": torch.zeros(
            module_shape, dtype=torch.bool, device=module_state.device
        ),
        "development_effective_gain": torch.zeros_like(module_state),
        "module_output": torch.zeros_like(module_state),
        "previous_tick_connection_input": torch.zeros_like(module_state),
        "routed_observation": torch.zeros_like(module_state),
        "state_before": torch.zeros_like(module_state),
        "state_after": torch.zeros_like(module_state),
        "structural_cost_components": torch.zeros_like(cost_components),
        "structural_cost_total": torch.zeros_like(cost_total),
    }
    residuals = torch.zeros(
        (module_shape[0], int(action_count)),
        dtype=torch.float32,
        device=module_state.device,
    )
    return audit, residuals


def _initial_ledger(entry_receipt: Mapping[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "entry_receipt_sha256": entry_receipt["receipt_sha256"],
        "status": "ready",
        "smoke_status": "pending",
        "completed_count": 0,
        "first_difference": None,
        "completed": [],
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    ledger.pop("ledger_sha256", None)
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    atomic_write_canonical_json(path, ledger)


def execute_phase_f_battery_v2(
    *,
    matrix: Any,
    fixture_registry: Any,
    plan: Mapping[str, Any],
    cpu_manifest: Mapping[str, Any],
    entry_receipt: Mapping[str, Any],
    output_directory: str | Path,
    source_commit: str,
    matrix_receipt_sha256: str,
    owner_executor: Callable[
        [PhaseFV2SourceState, Mapping[str, Any], bool], Mapping[str, Any]
    ],
    checkpoint_prerequisites: Mapping[str, Mapping[str, Any]] | None = None,
    migration_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the exact ordered battery with atomic resume and first-difference stop.

    ``owner_executor`` is the narrow execution boundary.  Production callers
    pass :func:`execute_phase_f_cuda_owner_observation_v2`; tests may pass a
    CPU fake.  It is invoked exactly once per pending accelerated item.
    """

    checked_matrix = validate_phase_f_v2_matrix(matrix)
    registry = validate_phase_f_fixture_registry_v2(
        fixture_registry, matrix=checked_matrix
    )
    if plan.get("plan_sha256") != canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise PhaseFV2BatteryError("battery plan digest mismatch at execution entry")
    if (
        plan.get("matrix_sha256") != checked_matrix["matrix_sha256"]
        or plan.get("fixture_registry_sha256") != registry["registry_sha256"]
    ):
        raise PhaseFV2BatteryError("battery plan source binding mismatch")
    if (
        plan.get("cpu_expectation_count") != EXPECTED_CPU_EVALUATIONS
        or plan.get("accelerated_invocation_count") != EXPECTED_ACCELERATED_INVOCATIONS
        or len(plan.get("accelerated_invocations", ()))
        != EXPECTED_ACCELERATED_INVOCATIONS
    ):
        raise PhaseFV2BatteryError("battery plan accounting mismatch")
    if cpu_manifest.get("manifest_sha256") != canonical_sha256(
        {key: value for key, value in cpu_manifest.items() if key != "manifest_sha256"}
    ):
        raise PhaseFV2BatteryError("CPU manifest digest mismatch at execution entry")
    if (
        cpu_manifest.get("expectation_count") != EXPECTED_CPU_EVALUATIONS
        or len(cpu_manifest.get("records", ())) != EXPECTED_CPU_EVALUATIONS
    ):
        raise PhaseFV2BatteryError("CPU manifest accounting mismatch")
    _sha(
        cpu_manifest.get("source_manifest_sha256"),
        field="cpu_manifest.source_manifest_sha256",
    )
    if set(entry_receipt) != _ENTRY_FIELDS:
        raise PhaseFV2BatteryError("execution entry receipt fields are not exact")
    if (
        entry_receipt.get("schema_version") != ENTRY_SCHEMA_VERSION
        or entry_receipt.get("status") != "execution_entry_authorized"
    ):
        raise PhaseFV2BatteryError("execution entry receipt is missing or unsupported")
    if entry_receipt.get("receipt_sha256") != canonical_sha256(
        {key: value for key, value in entry_receipt.items() if key != "receipt_sha256"}
    ):
        raise PhaseFV2BatteryError("execution entry receipt digest mismatch")
    if entry_receipt.get("source_commit") != source_commit:
        raise PhaseFV2BatteryError("entry receipt source commit mismatch")
    if entry_receipt.get("matrix_sha256") != checked_matrix["matrix_sha256"]:
        raise PhaseFV2BatteryError("entry receipt matrix mismatch")
    if entry_receipt.get("fixture_registry_sha256") != registry["registry_sha256"]:
        raise PhaseFV2BatteryError("entry receipt fixture registry mismatch")
    if entry_receipt.get("matrix_receipt_sha256") != matrix_receipt_sha256:
        raise PhaseFV2BatteryError("entry receipt matrix receipt mismatch")
    if entry_receipt.get("plan_sha256") != plan["plan_sha256"]:
        raise PhaseFV2BatteryError("entry receipt plan mismatch")
    if entry_receipt.get("cpu_manifest_sha256") != cpu_manifest["manifest_sha256"]:
        raise PhaseFV2BatteryError("entry receipt CPU manifest mismatch")
    if (
        entry_receipt.get("source_manifest_sha256")
        != cpu_manifest["source_manifest_sha256"]
    ):
        raise PhaseFV2BatteryError("entry receipt source manifest mismatch")
    if entry_receipt.get("phase_f_authorized") is not True:
        raise PhaseFV2BatteryError("execution entry does not authorize Phase F")
    if entry_receipt.get("smoke_required_before_full") is not True:
        raise PhaseFV2BatteryError("execution entry does not require smoke first")
    if (
        entry_receipt.get("logical_case_count") != EXPECTED_LOGICAL_CASES
        or entry_receipt.get("cpu_expectation_count") != EXPECTED_CPU_EVALUATIONS
        or entry_receipt.get("accelerated_invocation_count")
        != EXPECTED_ACCELERATED_INVOCATIONS
    ):
        raise PhaseFV2BatteryError("execution entry accounting mismatch")
    cases, fixtures = _case_map(checked_matrix), _fixture_map(registry)
    cpu_records = _cpu_record_map(cpu_manifest)
    prerequisites = {} if checkpoint_prerequisites is None else checkpoint_prerequisites
    contexts = {} if migration_contexts is None else migration_contexts
    root = Path(output_directory)
    ledger_path = root / "resume_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="ascii"))
        observed_digest = ledger.pop("ledger_sha256", None)
        if observed_digest != canonical_sha256(ledger):
            raise PhaseFV2BatteryError("resume ledger digest mismatch")
        ledger["ledger_sha256"] = observed_digest
        if ledger["entry_receipt_sha256"] != entry_receipt["receipt_sha256"]:
            raise PhaseFV2BatteryError("resume ledger belongs to a different entry")
        if ledger["status"] == "failed":
            raise PhaseFV2BatteryError("battery is stopped at its first difference")
    else:
        ledger = _initial_ledger(entry_receipt)
        _write_ledger(ledger_path, ledger)
    completed_ids = {row["execution_id"] for row in ledger["completed"]}
    smoke_ids = set(plan["smoke_execution_ids"])
    prepared_by_case: dict[str, PreparedPhaseFV2ExecutionSource] = {}
    for index, item in enumerate(plan["accelerated_invocations"]):
        execution_id = str(item["execution_id"])
        if execution_id in completed_ids:
            continue
        if index >= len(smoke_ids) and not smoke_ids.issubset(completed_ids):
            raise PhaseFV2BatteryError("full battery cannot start before smoke passes")
        case_id = str(item["case_id"])
        case, fixture = cases[case_id], fixtures[case_id]
        prepared = prepared_by_case.get(case_id)
        if prepared is None:
            prepared = prepare_phase_f_execution_source_v2(
                matrix=checked_matrix,
                fixture_registry=registry,
                case_id=case_id,
                migration_context=contexts.get(case_id),
            )
            prepared_by_case[case_id] = prepared
        source = prepared.source
        prerequisite = prepared.checkpoint_prerequisite_receipt
        external_prerequisite = prerequisites.get(case_id)
        if external_prerequisite is not None:
            if prerequisite is not None and canonical_sha256(
                external_prerequisite
            ) != canonical_sha256(prerequisite):
                raise PhaseFV2BatteryError(
                    f"external checkpoint prerequisite contradicts {case_id}"
                )
            prerequisite = dict(external_prerequisite)
        checkpoint_evidence = None
        if item["invocation_kind"] == "resumed":
            source, checkpoint_evidence = _fresh_checkpoint_source(
                matrix=checked_matrix,
                case=case,
                fixture=fixture,
                fixture_registry=registry,
                source=source,
                source_preparation_receipt=prepared.preparation_receipt,
                prerequisite_receipt=prerequisite,
            )
        modular_enabled = bool(fixture["execution_recipe"]["modular_enabled"])
        observed = dict(owner_executor(source, item, modular_enabled))
        observation_path = root / "observations" / f"{execution_id}.json"
        atomic_write_canonical_json(observation_path, observed)
        cpu_record = cpu_records[(case_id, str(item["invocation_kind"]))]
        expectation = _load_expectation(cpu_record)
        bindings = {
            "case_id": case_id,
            "case_sha256": case["case_sha256"],
            "execution_id": execution_id,
            "source_commit": source_commit,
            "matrix_sha256": checked_matrix["matrix_sha256"],
            "matrix_receipt_sha256": matrix_receipt_sha256,
            "entry_receipt_sha256": entry_receipt["receipt_sha256"],
            "fixture_blueprint_sha256": case["fixture_blueprint"][
                "fixture_blueprint_sha256"
            ],
            "fixture_sha256": fixture["fixture_sha256"],
            "source_sha256": source.source_sha256,
            "owner_id": checked_matrix["owner_contract"]["accelerated_owner"],
            "profile_id": case["profile_id"],
            "lane_id": checked_matrix["owner_contract"]["modular_tick_owner"],
            "invocation_kind": item["invocation_kind"],
            "repeat_index": item["repeat_index"],
            "boundary_index": item["boundary_index"],
            "checkpoint_kind": case["checkpoint_schedule"]["kind"],
            "protocol": deepcopy(checked_matrix["protocol"]),
            "deterministic_action_binding": True,
            "hidden_fallback_count": 0,
            "modular_rng_draw_count": 0,
        }
        receipt = compare_phase_f_invocation_v2(
            matrix=checked_matrix,
            bindings=bindings,
            expected=expectation["expectation"],
            observed=observed,
            frozen_source_commit=source_commit,
            matrix_receipt_sha256=matrix_receipt_sha256,
            entry_receipt_sha256=entry_receipt["receipt_sha256"],
            fixture_sha256=fixture["fixture_sha256"],
            source_sha256=source.source_sha256,
        )
        receipt_path = root / "invocations" / f"{execution_id}.json"
        atomic_write_canonical_json(receipt_path, receipt)
        completed = {
            "execution_id": execution_id,
            "case_id": case_id,
            "invocation_kind": item["invocation_kind"],
            "receipt_sha256": receipt["receipt_sha256"],
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": _file_sha256(receipt_path),
            "observed_sha256": canonical_sha256(observed),
            "observation_path": str(observation_path),
            "observation_file_sha256": _file_sha256(observation_path),
            "checkpoint_evidence": checkpoint_evidence,
            "source_preparation_receipt": prepared.preparation_receipt,
            "boundary_receipts": list(prepared.boundary_receipts),
        }
        ledger["completed"].append(completed)
        completed_ids.add(execution_id)
        ledger["completed_count"] = len(ledger["completed"])
        if receipt["outcome"] != "no_mismatch":
            ledger["status"] = "failed"
            ledger["smoke_status"] = "failed" if execution_id in smoke_ids else "passed"
            comparison = receipt["comparison_receipt"]
            ledger["first_difference"] = comparison.get(
                "first_difference", comparison
            )
            _write_ledger(ledger_path, ledger)
            return ledger
        if smoke_ids.issubset(completed_ids):
            ledger["smoke_status"] = "passed"
        ledger["status"] = (
            "passed"
            if len(completed_ids) == EXPECTED_ACCELERATED_INVOCATIONS
            else "running"
        )
        _write_ledger(ledger_path, ledger)
    return ledger


def execute_phase_f_cuda_owner_observation_v2(
    source: PhaseFV2SourceState,
    invocation: Mapping[str, Any],
    modular_enabled: bool,
    *,
    build_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    """Invoke one fresh CUDA generation owner.

    The generation result does not expose every named modular audit tensor.
    The adapter therefore runs the public native modular tick on a separate
    fresh witness bank, then the single-cell generation owner once.  The
    witness and owner share no mutable tensors.  Their resulting modular plane
    banks must be byte-exact before the witness audit is admitted.
    """

    if not torch.cuda.is_available():
        raise PhaseFV2BatteryError("CUDA is unavailable")
    from perf.cuda_architecture_b_generation import execute_cuda_modular_generation
    from perf.cuda_architecture_b_modular_tick_native import execute_modular_tick_native

    owner_tensors = {
        name: value.detach().clone().to("cuda").contiguous()
        for name, value in source.generation_tensors.items()
    }
    witness_tensors = {
        name: value.detach().clone().to("cuda").contiguous()
        for name, value in source.generation_tensors.items()
        if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    active_observations = torch.cat(
        [item.modular_observations for item in source.formal_oracle_inputs], dim=0
    ).to("cuda")
    observations = torch.zeros(
        (
            int(owner_tensors["alive_mask"].numel()),
            int(active_observations.shape[1]),
        ),
        dtype=active_observations.dtype,
        device="cuda",
    )
    observations[:PANEL_AGENT_COUNT] = active_observations
    fatigue = owner_tensors["drives"][:, 2].contiguous()
    formal_stages = torch.cat(
        [item.stage_indices for item in source.formal_oracle_inputs], dim=0
    ).to("cuda")
    witness_before = {name: value.clone() for name, value in witness_tensors.items()}
    survival = owner_tensors["survival_steps"].to(torch.int64)
    derived_stages = torch.where(
        survival
        >= int(source.generation_config.structural_development_late_start_step),
        torch.full_like(survival, 2),
        torch.where(
            survival
            >= int(source.generation_config.structural_development_mature_start_step),
            torch.ones_like(survival),
            torch.zeros_like(survival),
        ),
    )
    if not torch.equal(formal_stages, derived_stages[:PANEL_AGENT_COUNT]):
        raise PhaseFV2BatteryError(
            "formal stage indices disagree with owner survival_steps"
        )
    witness = None
    if modular_enabled:
        witness = execute_modular_tick_native(
            witness_tensors,
            observations,
            fatigue,
            derived_stages,
            build_directory=build_directory,
        )
    result = execute_cuda_modular_generation(
        owner_tensors,
        source.generation_config,
        enabled=modular_enabled,
        modular_tensors=owner_tensors,
        profile_id=source.profile_id,
        deterministic=True,
        policy_family="fixed_mlp",
        build_directory=build_directory,
    )
    for name, witness_value in witness_tensors.items():
        if not torch.equal(owner_tensors[name], witness_value):
            raise PhaseFV2BatteryError(
                f"generation owner and public modular witness disagree: {name}"
            )
    if int(result.slots_by_alive_order.numel()) != PANEL_AGENT_COUNT:
        raise PhaseFV2BatteryError(
            "generation owner did not return exactly 16 live rows"
        )
    if result.slots_by_alive_order.detach().cpu().tolist() != list(
        range(PANEL_AGENT_COUNT)
    ):
        raise PhaseFV2BatteryError("generation owner live-row order changed")
    if witness is not None and (
        witness.receipt.hidden_fallback_count != 0 or witness.receipt.rng_draws != 0
    ):
        raise PhaseFV2BatteryError("modular witness used fallback or RNG")

    prefix = MODULAR_STORAGE_PROJECTION_PREFIX

    def plane(mapping: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        return mapping[f"{prefix}{name}"]

    def tensor_json(value: torch.Tensor) -> Any:
        host = value.detach().cpu().contiguous()
        if host.is_floating_point() and not bool(torch.all(torch.isfinite(host))):
            raise PhaseFV2BatteryError("owner observation contains a nonfinite tensor")
        return host.tolist()

    def tensor_manifest(value: torch.Tensor) -> dict[str, Any]:
        host = value.detach().cpu().contiguous()
        return {
            "dtype": str(host.dtype),
            "shape": list(host.shape),
            "sha256": sha256(host.numpy().tobytes(order="C")).hexdigest(),
        }

    plane_names = sorted(name.removeprefix(prefix) for name in witness_before)

    def per_agent_values(value: torch.Tensor) -> list[Any]:
        return [tensor_json(value[row]) for row in range(PANEL_AGENT_COUNT)]

    def per_agent_plane(name: str, *, after: bool) -> list[Any]:
        mapping = owner_tensors if after else witness_before
        value = plane(mapping, name)
        return [tensor_json(value[row]) for row in range(PANEL_AGENT_COUNT)]

    def plane_manifests(*, after: bool) -> list[dict[str, Any]]:
        mapping = owner_tensors if after else witness_before
        return [
            {
                name: tensor_manifest(plane(mapping, name)[row : row + 1])
                for name in plane_names
            }
            for row in range(PANEL_AGENT_COUNT)
        ]

    if witness is None:
        audit_tensors, action_residuals = _disabled_modular_audit_v2(
            witness_before,
            action_count=source.generation_config.action_count,
        )
        fallback_count = 0
        rng_draw_count = 0
    else:
        audit_tensors = witness.audit_tensors
        action_residuals = witness.action_residuals
        fallback_count = witness.receipt.hidden_fallback_count
        rng_draw_count = witness.receipt.rng_draws

    owner = {
        "owner_id": "single_cell_native_generation_phase_e_v1",
        "lane_id": "native_aten_device_resident_modular_tick_v1",
        "profile_id": source.profile_id,
        "case_id": source.case_id,
        "modular_enabled": modular_enabled,
    }
    actions = tensor_json(result.actions_by_tick_slot[0, :PANEL_AGENT_COUNT])
    logits = result.logits[:PANEL_AGENT_COUNT]
    all_true = [[True] * int(source.generation_config.action_count)] * PANEL_AGENT_COUNT
    observation: dict[str, Any] = {
        "actions": actions,
        "checkpoint_projection": {
            "expectation_kind": str(invocation["invocation_kind"]),
            "source_sha256": source.source_sha256,
            "execution_source_sha256": source.source_sha256,
            "planes_before": plane_manifests(after=False),
            "planes_after": plane_manifests(after=True),
        },
        "dependency_readiness": per_agent_values(
            audit_tensors["development_dependency_ready"]
        ),
        "developmental_stage": tensor_json(formal_stages),
        "dormancy": per_agent_plane("module_dormant", after=False),
        "fallback_count": 0,
        "free_slot_maps": [
            {
                "module_free": [
                    index
                    for index, occupied in enumerate(
                        tensor_json(plane(owner_tensors, "module_occupied")[row])
                    )
                    if not occupied
                ],
                "connection_free": [
                    index
                    for index, occupied in enumerate(
                        tensor_json(plane(owner_tensors, "connection_occupied")[row])
                    )
                    if not occupied
                ],
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
        "homology": per_agent_plane("module_homology_id_index", after=True),
        "identities": [
            {
                "agent_id": int(formal.agent_id),
                "agent_slot": int(formal.agent_slot),
                "agent_slot_token": int(formal.agent_slot_token),
                "module_instance_id_index": tensor_json(
                    plane(owner_tensors, "module_instance_id_index")[row]
                ),
                "identity_origin_code": tensor_json(
                    plane(owner_tensors, "identity_origin_code")[row]
                ),
            }
            for row, formal in enumerate(source.formal_oracle_inputs)
        ],
        "modular_rng_draw_count": 0,
        "module_and_connection_presence": [
            {
                name: tensor_json(plane(owner_tensors, name)[row])
                for name in (
                    "module_occupied",
                    "connection_occupied",
                    "module_state_present",
                    "module_output_present",
                )
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
        "profile_and_owner_ids": owner,
        "semantic_ids": [
            {
                name: tensor_json(plane(owner_tensors, name)[row])
                for name in (
                    "input_semantic_code",
                    "input_semantic_id_index",
                    "output_semantic_code",
                    "output_semantic_id_index",
                    "output_action_index",
                )
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
        "slot_maps": [
            {
                name: tensor_json(plane(owner_tensors, name)[row])
                for name in (
                    "module_execution_order",
                    "dependency_topological_order",
                    "connection_execution_order",
                    "connection_src_module_slot",
                    "connection_dst_module_slot",
                    "connection_src_slot_index",
                    "connection_dst_slot_index",
                )
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
        "validity_masks": [
            {
                "validity_before": row,
                "validity_after": row,
                "authority_before": row,
                "authority_after": row,
                "effective": row,
            }
            for row in all_true
        ],
        "action_logits": tensor_json(logits),
        "developmental_effective_gain": per_agent_values(
            audit_tensors["development_effective_gain"]
        ),
        "module_output": per_agent_values(audit_tensors["module_output"]),
        "module_state": per_agent_plane("module_state", after=True),
        "previous_output": per_agent_plane("module_previous_output", after=False),
        "previous_tick_connection_input": per_agent_values(
            audit_tensors["previous_tick_connection_input"]
        ),
        "routed_action_residual": per_agent_values(action_residuals),
        "routed_observation": per_agent_values(audit_tensors["routed_observation"]),
        "state_before_and_after": [
            {
                "before": tensor_json(audit_tensors["state_before"][row]),
                "after": tensor_json(audit_tensors["state_after"][row]),
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
        "structural_cost_components_and_total": [
            {
                "components": tensor_json(
                    audit_tensors["structural_cost_components"][row]
                ),
                "total": tensor_json(audit_tensors["structural_cost_total"][row]),
            }
            for row in range(PANEL_AGENT_COUNT)
        ],
    }
    witness_receipt = {
        "schema_version": OWNER_WITNESS_SCHEMA_VERSION,
        "case_id": source.case_id,
        "source_sha256": source.source_sha256,
        "modular_enabled": modular_enabled,
        "owner_result_fields": sorted(vars(result)),
        "witness_audit_fields": sorted(audit_tensors),
        "hidden_fallback_count": fallback_count,
        "modular_rng_draw_count": rng_draw_count,
        "single_cell_owner_invocations": 1,
        "observation_sha256": canonical_sha256(observation),
    }
    canonical_sha256(witness_receipt)
    return observation


__all__ = [
    "CPU_MANIFEST_SCHEMA_VERSION",
    "ENTRY_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "SMOKE_KEYS",
    "PhaseFV2BatteryError",
    "build_phase_f_battery_plan_v2",
    "build_phase_f_entry_receipt_v2",
    "execute_phase_f_battery_v2",
    "execute_phase_f_cuda_owner_observation_v2",
    "materialize_phase_f_cpu_manifest_v2",
    "validate_phase_e_final_for_phase_f_v2",
    "verify_clean_frozen_checkout",
]
