#!/usr/bin/env python3
"""Run the fail-closed CUDA D0 gate for the fixed-MLP M7/M8 campaign.

The public ``run`` mode is an orchestrator.  Every CUDA owner is constructed in
an isolated worker process.  In particular, the checkpoint producer exits
before either restore worker starts.  The route supports durable restore only
at a generation boundary, so this gate records the unsupported T80 boundary
explicitly and proves the strongest faithful B48 G3 -> G6 continuation.

This is a QualPrep proof harness, not a benchmark or production admission.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNTIME_SOURCE_ROOT = Path(
    os.environ.get("EIEL_GPU_TICK_RUNTIME_SOURCE_ROOT", str(ROOT))
).resolve()
ENGINE_ROOT = RUNTIME_SOURCE_ROOT / "engine"
for import_root in (ENGINE_ROOT, RUNTIME_SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SCHEMA_VERSION = "eiel_fixed_mlp_policy_d0_gate_v1"
AUTHORITY = "fixed_mlp_policy_cuda_d0_qualprep_not_admission_v1"
ARMS = ("legacy", "m7_physical_refill", "m8_suffix_reuse")
EXPECTED_CELLS = 48
TICKS_PER_GENERATION = 160
CHECKPOINT_GENERATION = 3
TERMINAL_GENERATION = 6
WORKER_RECEIPT = "worker_receipt.json"
HEX64 = re.compile(r"[0-9a-f]{64}")


class D0GateError(RuntimeError):
    """Raised whenever evidence is absent, malformed, or contradictory."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a local immutable receipt without pre-importing checkout runtime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D0GateError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise D0GateError(f"{label} is not a mapping: {path}")
    return dict(value)


def _validated_artifact_set(
    bindings: Mapping[str, tuple[Path, Path]],
) -> dict[str, dict[str, Any]]:
    """Validate all three immutable binary/manifest identities and one source."""

    from tools.scripts.eiel_b48_fixed_mlp_policy_birth_refill_profile_v1 import (
        _validated_manifest_identity,
    )

    if tuple(bindings) != ARMS:
        raise D0GateError("artifact bindings must contain ordered legacy/M7/M8 arms")
    identities: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        artifact, manifest_path = bindings[arm]
        artifact = artifact.expanduser().resolve()
        manifest_path = manifest_path.expanduser().resolve()
        if not artifact.is_file() or not manifest_path.is_file():
            raise D0GateError(f"{arm} artifact or manifest is absent")
        manifest = _read_mapping(manifest_path, label=f"{arm} manifest")
        try:
            identity = _validated_manifest_identity(
                manifest, artifact=artifact, expected_arm=arm
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise D0GateError(f"{arm} artifact identity is invalid: {exc}") from exc
        identities[arm] = {
            **identity,
            "artifact_path": str(artifact),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        }
    source_commits = {row["artifact_source_commit"] for row in identities.values()}
    snapshots = {row["compiled_source_snapshot_sha256"] for row in identities.values()}
    if len(source_commits) != 1 or len(snapshots) != 1:
        raise D0GateError("three artifact arms do not share one frozen source snapshot")
    return identities


def _tensor_digests(tensors: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        result[name] = digest.hexdigest()
    return result


def _component_worker(
    *, arm: str, artifact: Path, manifest_path: Path, output_root: Path
) -> dict[str, Any]:
    """Exercise exact B48 reuse and forced-refill component surfaces on CUDA."""

    if arm not in ARMS:
        raise D0GateError(f"unknown component arm: {arm}")
    output_root.mkdir(parents=True, exist_ok=False)
    from tools.scripts.eiel_b48_fixed_mlp_policy_birth_refill_profile_v1 import (
        EPISODES,
        REQUIRED_FUTURE_EPISODES,
        _call,
        _call_reuse,
        _fixture,
        _hardware_identity,
        _load,
        _output_digest,
        _reset,
        _validated_manifest_identity,
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise D0GateError(
            "component D0 worker requires the single admitted CUDA device"
        )
    manifest = _read_mapping(manifest_path, label=f"{arm} manifest")
    identity = _validated_manifest_identity(
        manifest, artifact=artifact, expected_arm=arm
    )
    module = _load(artifact, str(manifest["module_name"]))
    device = torch.device("cuda:0")
    fixture = _fixture(device)
    _reset(fixture)

    # One cell is exactly one row short.  All arms therefore prove the same
    # physical-refill output from the same RNG/cursor input.
    forced_cursor = torch.zeros((EXPECTED_CELLS,), dtype=torch.int64, device=device)
    forced_cursor[-1] = EPISODES - REQUIRED_FUTURE_EPISODES + 1
    fixture["absolute_cursor"].copy_(forced_cursor)
    fixture["window_base"].zero_()
    before_preserved = {
        name: fixture[name].clone()
        for name in ("source_state", "absolute_cursor", "semantic_counter")
    }
    if arm == "m8_suffix_reuse":
        physical_receipt = dict(_call_reuse(module, fixture))
    else:
        physical_receipt = dict(_call(module, fixture))
    torch.cuda.synchronize(device)
    direct_physical = physical_receipt.get("operation") == (
        "refill_fixed_mlp_policy_window_v1"
    )
    m8_physical = (
        physical_receipt.get("physical_refill_performed") is True
        and physical_receipt.get("resident_window_reused") is False
    )
    if not (m8_physical if arm == "m8_suffix_reuse" else direct_physical) or any(
        not torch.equal(fixture[name], expected)
        for name, expected in before_preserved.items()
    ):
        raise D0GateError(f"{arm} did not prove a preserved-state physical refill")
    if (
        arm != "legacy"
        and physical_receipt.get("option_d_m7_batched_cpu_policy_refill_used")
        is not True
    ):
        raise D0GateError(f"{arm} physical refill did not use M7")
    if arm == "m8_suffix_reuse" and (
        physical_receipt.get("option_d_m8_resident_policy_suffix_reuse_used")
        is not False
        or physical_receipt.get("refill_trigger_cell_count") != 1
    ):
        raise D0GateError("M8 forced refill did not prove one-cell B48 skew fallback")
    physical_tensor_digests = _tensor_digests(fixture)
    physical_output_digest = _output_digest(fixture)

    reuse_proof: dict[str, Any] | None = None
    if arm == "m8_suffix_reuse":
        _reset(fixture)
        prefill = dict(_call(module, fixture))
        if prefill.get("option_d_m7_batched_cpu_policy_refill_used") is not True:
            raise D0GateError(
                "M8 initial physical fill did not use its exact M7 fallback"
            )
        heterogeneous = [
            (cell * 37) % (EPISODES - REQUIRED_FUTURE_EPISODES + 1)
            for cell in range(EXPECTED_CELLS)
        ]
        heterogeneous[-1] = EPISODES - REQUIRED_FUTURE_EPISODES
        fixture["absolute_cursor"].copy_(
            torch.tensor(heterogeneous, dtype=torch.int64, device=device)
        )
        fixture["window_base"].zero_()
        before = {name: value.clone() for name, value in fixture.items()}
        before_digest = _output_digest(fixture)
        reuse_receipt = dict(_call_reuse(module, fixture))
        torch.cuda.synchronize(device)
        changed = [
            name
            for name, value in fixture.items()
            if not torch.equal(value, before[name])
        ]
        if (
            changed
            or before_digest != _output_digest(fixture)
            or reuse_receipt.get("resident_window_reused") is not True
            or reuse_receipt.get("physical_refill_performed") is not False
            or reuse_receipt.get("option_d_m8_resident_policy_suffix_reuse_used")
            is not True
            or reuse_receipt.get("option_d_m7_batched_cpu_policy_refill_used")
            is not False
            or reuse_receipt.get("minimum_remaining_episodes")
            != REQUIRED_FUTURE_EPISODES
            or reuse_receipt.get("host_to_device_bytes_lower_bound") != 0
        ):
            raise D0GateError(
                "M8 heterogeneous B48 suffix reuse changed state or missed its exact boundary"
            )
        reuse_proof = {
            "heterogeneous_absolute_cursor": heterogeneous,
            "minimum_remaining_episodes": reuse_receipt["minimum_remaining_episodes"],
            "remaining_episodes_by_cell": reuse_receipt["remaining_episodes_by_cell"],
            "pre_and_post_tensor_digest_sha256": before_digest,
            "changed_tensor_names": changed,
            "receipt": reuse_receipt,
        }

    receipt = {
        "schema_version": f"{SCHEMA_VERSION}_component_worker_v1",
        "authority": AUTHORITY,
        "status": "passed",
        "worker_kind": "component",
        "arm": arm,
        "pid": os.getpid(),
        "artifact_identity": identity,
        "artifact_sha256": _sha256(artifact),
        "manifest_sha256": _sha256(manifest_path),
        "hardware_identity": _hardware_identity(device),
        "physical_refill": {
            "input_cursor": forced_cursor.detach().cpu().tolist(),
            "output_digest_sha256": physical_output_digest,
            "tensor_digest_sha256s": physical_tensor_digests,
            "rng_state_digest_sha256": physical_tensor_digests["source_state"],
            "absolute_cursor_digest_sha256": physical_tensor_digests["absolute_cursor"],
            "semantic_counter_digest_sha256": physical_tensor_digests[
                "semantic_counter"
            ],
            "packed_crossover_digest_sha256": physical_tensor_digests["crossover"],
            "packed_post_state_digest_sha256": physical_tensor_digests["post_state"],
            "receipt": physical_receipt,
        },
        "heterogeneous_suffix_reuse": reuse_proof,
        "qualification_claim": False,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(output_root / WORKER_RECEIPT, receipt)
    return receipt


def _projection_digests(
    projections: Sequence[Mapping[str, Any]], cell_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    from engine.native_runtime.checkpoint_digests import (
        DigestLayer,
        checkpoint_digest_manifest,
    )

    if len(projections) != EXPECTED_CELLS or len(cell_ids) != EXPECTED_CELLS:
        raise D0GateError("checkpoint projection does not cover exact B48")
    result: dict[str, dict[str, str]] = {}
    for cell_id, projection in zip(cell_ids, projections, strict=True):
        trajectory = checkpoint_digest_manifest(
            projection, layer=DigestLayer.TRAJECTORY_STATE
        )["digest"]
        canonical = checkpoint_digest_manifest(
            projection, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE
        )["digest"]
        result[str(cell_id)] = {
            "trajectory_state_sha256": str(trajectory),
            "canonical_checkpoint_storage_sha256": str(canonical),
        }
    return result


def _trajectory_execution_shape(action: str) -> tuple[int, int, int | None]:
    """Return an executor-admitted start, end, and optional exact tick span."""

    shapes = {
        "terminal": (0, 1, TICKS_PER_GENERATION),
        "prefix": (0, CHECKPOINT_GENERATION, None),
        "resume": (CHECKPOINT_GENERATION, TERMINAL_GENERATION, None),
        "uninterrupted": (0, TERMINAL_GENERATION, None),
    }
    try:
        return shapes[action]
    except KeyError as exc:
        raise D0GateError(f"unknown trajectory worker action: {action}") from exc


def _checkpoint_derivative_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Declare the exact absolute horizon used by checkpoint/replay workers."""

    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise D0GateError("checkpoint derivative lacks an execution contract")
    result = dict(plan)
    result["execution"] = {
        **execution,
        "generations": TERMINAL_GENERATION,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "checkpoints": [CHECKPOINT_GENERATION, TERMINAL_GENERATION],
    }
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _native_owner_contract(*, generations: int, tick_span: bool) -> dict[str, Any]:
    required: dict[str, Any] = {
        "owner": "native_extension",
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
        "completed_generations": generations,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
    }
    if tick_span:
        required.update(
            {
                "native_tick_iterations": generations * TICKS_PER_GENERATION,
                "terminal_generation_boundary": True,
            }
        )
    else:
        required["logical_generation_count"] = generations
    return required


def _policy_runtime_summary(
    direct_result: Mapping[str, Any],
    *,
    arm: str,
    generations: int,
    require_all_reuse: bool,
    generation_action: str | None = None,
    start_generation: int = 0,
) -> dict[str, Any]:
    from tools.scripts.eiel_gpu_tick_anatomy_v1 import (
        _m7_runtime_identity,
        _m8_runtime_identity,
    )

    expected_m7 = arm != "legacy"
    expected_m8 = arm == "m8_suffix_reuse"
    required = _native_owner_contract(
        generations=generations, tick_span=require_all_reuse
    )
    mismatches = {
        name: {"expected": expected, "observed": direct_result.get(name)}
        for name, expected in required.items()
        if direct_result.get(name) != expected
    }
    if mismatches:
        raise D0GateError(
            "native owner/control receipt mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if generation_action is None:
        if start_generation != 0:
            raise D0GateError("native tick proof must start at generation zero")
        receipts = direct_result.get("policy_birth_segment_refill_receipts")
        if not isinstance(receipts, list) or not receipts:
            raise D0GateError("native tick owner omitted policy callback receipts")
        child_receipts = [receipts]
        outer_refills: list[Mapping[str, Any]] = []
        expected_windows = 1
        turnover_sources = [direct_result]
    else:
        if (
            generation_action not in {"prefix", "resume", "uninterrupted"}
            or type(start_generation) is not int
            or start_generation < 0
            or (generation_action == "resume") != (start_generation > 0)
        ):
            raise D0GateError("generation action contradicts its start generation")
        physical_window_generations = direct_result.get("physical_window_generations")
        if type(physical_window_generations) is not int or (
            physical_window_generations < 1 or physical_window_generations > 2
        ):
            raise D0GateError(
                "native generation controller has an invalid physical-window lattice"
            )
        first_window_generations = min(
            physical_window_generations
            - start_generation % physical_window_generations,
            generations,
        )
        remaining = generations - first_window_generations
        expected_windows = (
            1
            + (remaining + physical_window_generations - 1)
            // physical_window_generations
        )
        expected_child_generations = [first_window_generations]
        while sum(expected_child_generations) < generations:
            expected_child_generations.append(
                min(
                    physical_window_generations,
                    generations - sum(expected_child_generations),
                )
            )
        controller_contract = {
            "schema_version": (
                "eiel_cuda_architecture_b_native_resident_window_controller_v1"
            ),
            "physical_window_generations": physical_window_generations,
            "logical_generation_count": generations,
            "window_count": expected_windows,
            "first_window_generations": first_window_generations,
            "settled_window_count": expected_windows,
            "completed_generations": generations,
            "physical_window_alignment": "absolute_generation_lattice_v1",
        }
        controller_mismatches = {
            name: {"expected": expected, "observed": direct_result.get(name)}
            for name, expected in controller_contract.items()
            if direct_result.get(name) != expected
        }
        windows = direct_result.get("window_receipts")
        outer = direct_result.get("policy_refill_receipts")
        if controller_mismatches:
            raise D0GateError(
                "native generation-controller contract mismatch: "
                + json.dumps(controller_mismatches, sort_keys=True)
            )
        if not isinstance(windows, list) or len(windows) != expected_windows:
            raise D0GateError(
                "native generation controller emitted the wrong settled window count"
            )
        child_receipts = []
        for window_index, (window, child_generations) in enumerate(
            zip(windows, expected_child_generations, strict=True)
        ):
            if not isinstance(window, Mapping):
                raise D0GateError(
                    f"native generation child window {window_index} is not a mapping"
                )
            rows = window.get("policy_birth_segment_refill_receipts")
            expected_child_callbacks = child_generations * (TICKS_PER_GENERATION - 1)
            if (
                window.get("requested_generations") != child_generations
                or window.get("completed_generations") != child_generations
                or not isinstance(rows, list)
                or len(rows) != expected_child_callbacks
            ):
                raise D0GateError(
                    f"native generation child window {window_index} did not emit "
                    "its exact generation-bound callback count"
                )
            child_receipts.append(rows)
        turnover_sources = windows
        if not isinstance(outer, list) or len(outer) != expected_windows - 1:
            raise D0GateError(
                "native generation controller emitted the wrong outer policy-refill count"
            )
        outer_refills = []
        for index, value in enumerate(outer):
            if not isinstance(value, Mapping):
                raise D0GateError(f"outer policy refill {index} is not a mapping")
            _validate_exact_m7_physical_refill(value, arm=arm)
            outer_refills.append(value)
        reconstructed_suffix = (
            start_generation > 0 and start_generation % physical_window_generations != 0
        )
        expected_resume_boundary = start_generation > 0 and not reconstructed_suffix
        if (
            direct_result.get("checkpointed_gaussian_policy_suffix_preserved")
            is not reconstructed_suffix
        ):
            raise D0GateError(
                "generation-controller reconstructed-suffix status differs from lattice"
            )
        resume_boundary = direct_result.get("resume_boundary_refill")
        resume_receipt = direct_result.get("resume_policy_refill_receipt")
        if expected_resume_boundary:
            if (
                resume_boundary is not True
                or not isinstance(resume_receipt, Mapping)
                or not resume_receipt
            ):
                raise D0GateError(
                    "fresh generation resume omitted its exact boundary refill"
                )
            _validate_exact_m7_physical_refill(resume_receipt, arm=arm)
        elif resume_boundary is not False or not (
            isinstance(resume_receipt, Mapping) and not resume_receipt
        ):
            raise D0GateError(
                "generation owner claimed an impossible resume-boundary refill"
            )
        receipts = [row for child in child_receipts for row in child]
    expected_callback_count = generations * (TICKS_PER_GENERATION - 1)
    reused = 0
    physical = 0
    boundary_receipts = 0
    for index, value in enumerate(receipts):
        if not isinstance(value, Mapping):
            raise D0GateError(f"policy callback receipt {index} is not a mapping")
        m8 = _m8_runtime_identity(value, expected_enabled=expected_m8)
        is_physical = bool(m8.get("physical_refill_performed")) if expected_m8 else True
        _m7_runtime_identity(
            value,
            expected_enabled=expected_m7,
            expected_used=(is_physical if expected_m8 else expected_m7),
        )
        physical += int(is_physical)
        reused += int(bool(m8.get("resident_window_reused")))
        boundary_receipts += int(value.get("turnover_boundary") is True)
    _validate_callback_counts(
        expected_m8=expected_m8,
        expected_callback_count=expected_callback_count,
        reused=reused,
        physical=physical,
        require_all_reuse=require_all_reuse,
    )
    turnover_fields = {
        name: sum(source.get(name, -1) for source in turnover_sources)
        for name in (
            "turnover_shadow_plan_launches",
            "turnover_cull_refill_plan_launches",
            "turnover_commit_launches",
            "turnover_publish_launches",
            "compact_lifecycle_digest_launches",
        )
    }
    if any(
        type(source.get(name)) is not int or source.get(name) < 0
        for source in turnover_sources
        for name in turnover_fields
    ):
        raise D0GateError("native turnover/control receipt fields are incomplete")
    active_turnover_fields = {
        name: value
        for name, value in turnover_fields.items()
        if name != "compact_lifecycle_digest_launches" and value != 0
    }
    if boundary_receipts != 0 or active_turnover_fields:
        raise D0GateError(
            "hybrid owner unexpectedly activated native terminal turnover"
        )
    return {
        "callback_receipt_count": len(receipts),
        "settled_window_count": expected_windows,
        "outer_physical_refill_count": len(outer_refills),
        "resume_boundary_refill": (
            direct_result.get("resume_boundary_refill") is True
            if generation_action is not None
            else False
        ),
        "resident_suffix_reuse_count": reused,
        "physical_refill_count": physical,
        "turnover_boundary_callback_count": boundary_receipts,
        "turnover_launches": turnover_fields,
        "control_contract": required,
    }


def _validate_exact_m7_physical_refill(receipt: Mapping[str, Any], *, arm: str) -> None:
    """Validate one controller-boundary refill without inventing M8 reuse."""

    from tools.scripts.eiel_gpu_tick_anatomy_v1 import _m7_runtime_identity

    expected_m7 = arm != "legacy"
    if (
        receipt.get("operation") != "refill_fixed_mlp_policy_window_v1"
        or receipt.get("absolute_cursor_preserved") is not True
        or receipt.get("semantic_draws_preserved") is not True
        or receipt.get("source_rng_state_preserved") is not True
    ):
        raise D0GateError("generation-controller boundary refill is not exact")
    try:
        _m7_runtime_identity(
            receipt,
            expected_enabled=expected_m7,
            expected_used=expected_m7,
        )
    except Exception as exc:
        raise D0GateError(
            "generation-controller boundary refill has the wrong M7 identity"
        ) from exc


def _validate_callback_counts(
    *,
    expected_m8: bool,
    expected_callback_count: int,
    reused: int,
    physical: int,
    require_all_reuse: bool,
) -> None:
    if expected_m8 and reused + physical != expected_callback_count:
        raise D0GateError("M8 callback accounting is incomplete")
    if (
        expected_m8
        and require_all_reuse
        and (reused != expected_callback_count or physical != 0)
    ):
        raise D0GateError(
            "B48/T160 M8 proof did not reuse every resident suffix callback"
        )
    if not expected_m8 and physical != expected_callback_count:
        raise D0GateError("physical-refill callback accounting is incomplete")


def _trajectory_worker(
    *,
    arm: str,
    action: str,
    plan_path: Path,
    artifact: Path,
    manifest_path: Path,
    output_root: Path,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Execute one real B48 native owner and optionally serialize its checkpoint."""

    import torch

    from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
        _wdl_source_capabilities_v2,
    )
    from tools.scripts.eiel_gpu_tick_anatomy_v1 import (
        AllocationDiagnosticError,
        AllocationOnlyProvider,
        _bind_artifacts,
        _diagnostic_cells_plan,
        _diagnostic_descriptor,
        _diagnostic_plan,
        _load_exact_module,
        _translation_compatible_plan,
        _validated_plan,
    )

    start, end, absolute_tick_span = _trajectory_execution_shape(action)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise D0GateError(
            "trajectory D0 worker requires the single admitted CUDA device"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    expected_m7 = arm != "legacy"
    expected_m8 = arm == "m8_suffix_reuse"
    source_plan = _validated_plan(plan_path)
    bound_plan = _bind_artifacts(
        source_plan,
        artifact=artifact,
        manifest=manifest_path,
        allow_candidate_artifact_rebind=True,
        expected_m7_enabled=expected_m7,
        expected_m8_enabled=expected_m8,
    )
    bound_plan = _translation_compatible_plan(
        _diagnostic_cells_plan(bound_plan, cells=EXPECTED_CELLS)
    )
    if action != "terminal":
        bound_plan = _checkpoint_derivative_plan(bound_plan)
    qualification, reporting = _diagnostic_descriptor(
        output_root,
        artifact_sha256=_sha256(artifact),
        manifest_sha256=_sha256(manifest_path),
        plan_sha256=str(bound_plan["plan_sha256"]),
    )
    plan = _diagnostic_plan(
        bound_plan, qualification=qualification, reporting=reporting
    )
    provider: Any = None
    module: Any = None
    prepared: Any = None
    sources: tuple[Mapping[str, Any], ...] | None = None
    try:
        module = _load_exact_module(artifact, eed_overlay=True)
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        batch = plan["physical_batches"][0]
        request = provider.translate_batch(plan, str(batch["batch_id"]))
        if len(request.cell_ids) != EXPECTED_CELLS:
            raise D0GateError("trajectory worker request is not exact B48")
        if action != "terminal" and (
            request.generations != TERMINAL_GENERATION
            or request.ticks_per_generation != TICKS_PER_GENERATION
            or request.checkpoint_milestones
            != (CHECKPOINT_GENERATION, TERMINAL_GENERATION)
            or request.agent_id_allocation_generation_count != TERMINAL_GENERATION
        ):
            raise D0GateError("checkpoint derivative request binding is invalid")
        adapters = dict(provider._resolved_execution_adapters(plan))
        selected_cells = provider._batch_cells(plan, request)
        context = SimpleNamespace(
            run_root=output_root, batch_root=output_root, progress_events=[]
        )
        component_checkpoint: Mapping[str, Any] | None = None
        source_checkpoint_sha256: str | None = None
        source_checkpoint_projection_digest_set_sha256: str | None = None
        source_owner_pid: int | None = None
        if action == "resume":
            if checkpoint_path is None or not checkpoint_path.is_file():
                raise D0GateError("resume worker lacks a serialized prefix checkpoint")
            source_checkpoint_sha256 = _sha256(checkpoint_path)
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version")
                != f"{SCHEMA_VERSION}_serialized_checkpoint_v1"
                or payload.get("arm") != arm
                or payload.get("artifact_sha256") != _sha256(artifact)
                or payload.get("manifest_sha256") != _sha256(manifest_path)
                or payload.get("absolute_generation") != CHECKPOINT_GENERATION
                or payload.get("cell_ids") != list(request.cell_ids)
                or not isinstance(payload.get("checkpoint_projections"), tuple)
                or not isinstance(payload.get("diagnostic_plan_sha256"), str)
                or not HEX64.fullmatch(payload["diagnostic_plan_sha256"])
                or not isinstance(payload.get("request_payload_sha256"), str)
                or not HEX64.fullmatch(payload["request_payload_sha256"])
            ):
                raise D0GateError("serialized prefix checkpoint binding is invalid")
            sources = tuple(payload["checkpoint_projections"])
            expected_checkpoint_projection_digest = payload.get(
                "checkpoint_projection_digest_set_sha256"
            )
            source_checkpoint_projection_digest_set_sha256 = canonical_sha256(
                _projection_digests(sources, request.cell_ids)
            )
            if (
                expected_checkpoint_projection_digest
                != source_checkpoint_projection_digest_set_sha256
            ):
                raise D0GateError(
                    "serialized checkpoint projection digest binding is invalid"
                )
            component_checkpoint = payload.get("component_checkpoint")
            source_owner_pid = payload.get("producer_pid")
            source_restorer = adapters.get("component_checkpoint_source_restorer")
            if component_checkpoint is not None:
                if not callable(source_restorer):
                    raise D0GateError(
                        "serialized component checkpoint lacks its source restorer"
                    )
                sources = tuple(
                    source_restorer(
                        sources, component_checkpoint, cell_ids=request.cell_ids
                    )
                )
        else:
            source_preparer = (
                adapters.get("source_preparer")
                or provider._prepare_generation_zero_sources_v2
            )
            sources = tuple(
                source_preparer(
                    plan,
                    request,
                    context,
                    runner_builder=adapters["runner_builder"],
                    cells=selected_cells,
                    artifact=str(artifact),
                    qualification=str(qualification),
                    reporting=str(reporting),
                    segment_generations=end,
                    source_capability_mask=_wdl_source_capabilities_v2(
                        {"genome_v2": True}, jungle_overlay=True, eed_overlay=True
                    ),
                )
            )
        allocation_generations = provider._agent_id_allocation_generation_count(
            request, start_generation=start
        )
        _, prepared_requests, prepared = provider._prepare_resident_segment_owner_v1(
            plan=plan,
            runner_builder=adapters["runner_builder"],
            request_builder=adapters["request_builder"],
            preparer=adapters["preparer"],
            cells=selected_cells,
            cell_ids=request.cell_ids,
            batch_root=output_root,
            start_generation=start,
            end_generation=end,
            ticks_per_generation=TICKS_PER_GENERATION,
            artifact=str(artifact),
            qualification=str(qualification),
            reporting=str(reporting),
            module=module,
            source_projections=sources,
            allocation_generations=allocation_generations,
            genome_v2_checkpoint_authority_binding=dict(
                request.genome_v2_checkpoint_authority_binding
            ),
            allow_short_technical_gate_generation_count=True,
            physical_window_generations=1,
        )
        sources = None
        if component_checkpoint is not None:
            restorer = adapters.get("component_checkpoint_restorer")
            if not callable(restorer):
                raise D0GateError("component checkpoint lacks its owner restorer")
            restored_generation = restorer(
                prepared.physical_window_arguments,
                component_checkpoint,
                absolute_generation=start,
                cell_ids=request.cell_ids,
            )
            if restored_generation != start:
                raise D0GateError("component checkpoint restored the wrong generation")
        torch.cuda.synchronize()
        execution_kwargs = {
            "physical_window_arguments": prepared.physical_window_arguments,
            "logical_generation_count": end - start,
        }
        if absolute_tick_span is not None:
            execution_kwargs["absolute_tick_span"] = absolute_tick_span
        direct_result = adapters["executor"](module, **execution_kwargs)
        torch.cuda.synchronize()
        if not isinstance(direct_result, Mapping):
            raise D0GateError("native trajectory worker returned no mapping")
        policy_summary = _policy_runtime_summary(
            direct_result,
            arm=arm,
            generations=end - start,
            require_all_reuse=action == "terminal",
            generation_action=(None if action == "terminal" else action),
            start_generation=start,
        )
        settlement = adapters["projector"](
            prepared,
            direct_result=direct_result,
            capability=prepared_requests[0].cuda_capability,
            reporting_enabled=False,
        )
        projections = tuple(settlement.checkpoint_projections)
        projection_digests = _projection_digests(projections, request.cell_ids)
        serialized_checkpoint_sha256: str | None = None
        if action == "prefix" and checkpoint_path is not None:
            if checkpoint_path.exists():
                raise D0GateError("prefix checkpoint destination is invalid")
            exporter = adapters.get("component_checkpoint_exporter")
            exported_component = (
                exporter(
                    prepared.physical_window_arguments,
                    absolute_generation=end,
                    cell_ids=request.cell_ids,
                )
                if callable(exporter)
                else None
            )
            checkpoint_payload = {
                "schema_version": f"{SCHEMA_VERSION}_serialized_checkpoint_v1",
                "authority": AUTHORITY,
                "arm": arm,
                "artifact_sha256": _sha256(artifact),
                "manifest_sha256": _sha256(manifest_path),
                "absolute_generation": end,
                "absolute_tick": 0,
                "cell_ids": list(request.cell_ids),
                "checkpoint_projections": projections,
                "checkpoint_projection_digest_set_sha256": canonical_sha256(
                    projection_digests
                ),
                "component_checkpoint": exported_component,
                "producer_pid": os.getpid(),
                "diagnostic_plan_sha256": str(plan["plan_sha256"]),
                "request_payload_sha256": canonical_sha256(request.payload()),
            }
            torch.save(checkpoint_payload, checkpoint_path)
            serialized_checkpoint_sha256 = _sha256(checkpoint_path)
        receipt = {
            "schema_version": f"{SCHEMA_VERSION}_trajectory_worker_v1",
            "authority": AUTHORITY,
            "status": "passed",
            "worker_kind": "trajectory",
            "action": action,
            "arm": arm,
            "pid": os.getpid(),
            "source_owner_pid": source_owner_pid,
            "start_generation": start,
            "terminal_generation": end,
            "terminal_tick": 0,
            "cells": EXPECTED_CELLS,
            "ticks_per_generation": TICKS_PER_GENERATION,
            "artifact_sha256": _sha256(artifact),
            "manifest_sha256": _sha256(manifest_path),
            "diagnostic_plan_sha256": str(plan["plan_sha256"]),
            "request_payload_sha256": canonical_sha256(request.payload()),
            "request_generations": request.generations,
            "request_checkpoint_milestones": list(request.checkpoint_milestones),
            "request_agent_id_allocation_terminal_generation": (
                request.agent_id_allocation_generation_count
            ),
            "segment_agent_id_allocation_generations": allocation_generations,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_checkpoint_projection_digest_set_sha256": (
                source_checkpoint_projection_digest_set_sha256
            ),
            "serialized_checkpoint_sha256": serialized_checkpoint_sha256,
            "serialized_checkpoint_projection_digest_set_sha256": (
                canonical_sha256(projection_digests) if action == "prefix" else None
            ),
            "projection_digests": projection_digests,
            "policy_runtime_summary": policy_summary,
            "fresh_owner_restore": action == "resume",
            "future_native_consumption": action == "resume",
            "turnover_and_control_receipts_valid": True,
            "qualification_claim": False,
            "production_admission": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(output_root / WORKER_RECEIPT, receipt)
        return receipt
    except AllocationDiagnosticError as exc:
        raise D0GateError(str(exc)) from exc
    finally:
        prepared = None
        sources = None
        module = None
        provider = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_worker_receipt(
    value: Any, *, kind: str, arm: str, action: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise D0GateError("worker receipt is not a mapping")
    receipt = dict(value)
    observed_sha = receipt.pop("receipt_sha256", None)
    expected_schema = f"{SCHEMA_VERSION}_{kind}_worker_v1"
    if (
        observed_sha != canonical_sha256(receipt)
        or receipt.get("schema_version") != expected_schema
        or receipt.get("authority") != AUTHORITY
        or receipt.get("status") != "passed"
        or receipt.get("worker_kind") != kind
        or receipt.get("arm") != arm
        or (action is not None and receipt.get("action") != action)
        or receipt.get("qualification_claim") is not False
        or receipt.get("production_admission") is not False
    ):
        raise D0GateError(f"{kind}/{arm}/{action or '-'} worker receipt is invalid")
    receipt["receipt_sha256"] = observed_sha
    return receipt


def _reduce_receipts(
    *,
    artifact_identities: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    arm_terminals: Mapping[str, Mapping[str, Any]],
    prefix: Mapping[str, Any],
    uninterrupted: Mapping[str, Any],
    restores: Sequence[Mapping[str, Any]],
    producer_exit_confirmed: bool,
) -> dict[str, Any]:
    """Reduce only exact, hash-bound worker receipts; never infer missing proof."""

    if tuple(artifact_identities) != ARMS or tuple(components) != ARMS:
        raise D0GateError("D0 reducer lacks the exact three-arm input set")
    checked_components = {
        arm: _validate_worker_receipt(components[arm], kind="component", arm=arm)
        for arm in ARMS
    }
    checked_terminals = {
        arm: _validate_worker_receipt(
            arm_terminals[arm], kind="trajectory", arm=arm, action="terminal"
        )
        for arm in ARMS
    }
    checked_prefix = _validate_worker_receipt(
        prefix, kind="trajectory", arm="m8_suffix_reuse", action="prefix"
    )
    checked_uninterrupted = _validate_worker_receipt(
        uninterrupted,
        kind="trajectory",
        arm="m8_suffix_reuse",
        action="uninterrupted",
    )
    if len(restores) != 2:
        raise D0GateError("checkpoint replay proof requires exactly two fresh restores")
    checked_restores = [
        _validate_worker_receipt(
            value, kind="trajectory", arm="m8_suffix_reuse", action="resume"
        )
        for value in restores
    ]
    for arm in ARMS:
        expected = artifact_identities[arm]
        for receipt in (checked_components[arm], checked_terminals[arm]):
            if receipt.get("artifact_sha256") != expected.get(
                "artifact_sha256"
            ) or receipt.get("manifest_sha256") != expected.get("manifest_sha256"):
                raise D0GateError(f"{arm} worker receipt changed artifact identity")
    expected_m8 = artifact_identities["m8_suffix_reuse"]
    for label, receipt in (
        ("prefix", checked_prefix),
        ("uninterrupted", checked_uninterrupted),
        *(
            (f"restore_{index}", receipt)
            for index, receipt in enumerate(checked_restores, start=1)
        ),
    ):
        if receipt.get("artifact_sha256") != expected_m8.get(
            "artifact_sha256"
        ) or receipt.get("manifest_sha256") != expected_m8.get("manifest_sha256"):
            raise D0GateError(f"{label} changed the frozen M8 artifact identity")
    expected_coordinates = {
        "prefix": (0, CHECKPOINT_GENERATION),
        "uninterrupted": (0, TERMINAL_GENERATION),
        "resume": (CHECKPOINT_GENERATION, TERMINAL_GENERATION),
    }
    for receipt in (checked_prefix, checked_uninterrupted, *checked_restores):
        expected_start, expected_end = expected_coordinates[str(receipt["action"])]
        expected_allocation_generations = min(TERMINAL_GENERATION - expected_start, 128)
        if (
            receipt.get("start_generation") != expected_start
            or receipt.get("terminal_generation") != expected_end
            or receipt.get("terminal_tick") != 0
            or receipt.get("request_generations") != TERMINAL_GENERATION
            or receipt.get("request_checkpoint_milestones")
            != [CHECKPOINT_GENERATION, TERMINAL_GENERATION]
            or receipt.get("request_agent_id_allocation_terminal_generation")
            != TERMINAL_GENERATION
            or receipt.get("segment_agent_id_allocation_generations")
            != expected_allocation_generations
            or not isinstance(receipt.get("diagnostic_plan_sha256"), str)
            or not HEX64.fullmatch(receipt["diagnostic_plan_sha256"])
            or not isinstance(receipt.get("request_payload_sha256"), str)
            or not HEX64.fullmatch(receipt["request_payload_sha256"])
        ):
            raise D0GateError("checkpoint worker changed its admitted request binding")
    physical_digests = {
        arm: checked_components[arm]["physical_refill"]["output_digest_sha256"]
        for arm in ARMS
    }
    physical_tensor_digests = {
        arm: checked_components[arm]["physical_refill"]["tensor_digest_sha256s"]
        for arm in ARMS
    }
    if (
        len(set(physical_digests.values())) != 1
        or len({canonical_sha256(value) for value in physical_tensor_digests.values()})
        != 1
    ):
        raise D0GateError("legacy/M7/M8 physical-refill tensors are not exact")
    reuse = checked_components["m8_suffix_reuse"].get("heterogeneous_suffix_reuse")
    if (
        not isinstance(reuse, Mapping)
        or reuse.get("changed_tensor_names") != []
        or not isinstance(reuse.get("pre_and_post_tensor_digest_sha256"), str)
    ):
        raise D0GateError("M8 heterogeneous suffix-reuse proof is absent")
    arm_projection_digests = {
        arm: checked_terminals[arm].get("projection_digests") for arm in ARMS
    }
    if len({canonical_sha256(value) for value in arm_projection_digests.values()}) != 1:
        raise D0GateError("terminal B48 trajectories differ across A/B/C")
    checkpoint_sha = checked_prefix.get("serialized_checkpoint_sha256")
    if (
        not producer_exit_confirmed
        or not isinstance(checkpoint_sha, str)
        or not HEX64.fullmatch(checkpoint_sha)
    ):
        raise D0GateError("checkpoint producer process release was not proven")
    checkpoint_projection_digest = canonical_sha256(
        checked_prefix.get("projection_digests")
    )
    if (
        checked_prefix.get("serialized_checkpoint_projection_digest_set_sha256")
        != checkpoint_projection_digest
    ):
        raise D0GateError("serialized checkpoint differs from prefix projection")
    expected_terminal = checked_uninterrupted.get("projection_digests")
    for index, restored in enumerate(checked_restores):
        if (
            restored.get("source_checkpoint_sha256") != checkpoint_sha
            or restored.get("source_checkpoint_projection_digest_set_sha256")
            != checkpoint_projection_digest
            or restored.get("projection_digests") != expected_terminal
            or restored.get("fresh_owner_restore") is not True
            or restored.get("future_native_consumption") is not True
        ):
            raise D0GateError(f"fresh restore {index + 1} differs from uninterrupted")
    if (
        checked_restores[0]["projection_digests"]
        != checked_restores[1]["projection_digests"]
    ):
        raise D0GateError("second checkpoint replay differs from first restore")
    all_trajectory_receipts = [
        *checked_terminals.values(),
        checked_prefix,
        checked_uninterrupted,
        *checked_restores,
    ]
    if any(
        receipt.get("turnover_and_control_receipts_valid") is not True
        for receipt in all_trajectory_receipts
    ):
        raise D0GateError("turnover/control-path compatibility was not proved")
    return {
        "d0_exact": True,
        "rng_cursor_packed_byte_parity": True,
        "heterogeneous_b48_suffix_reuse_exact": True,
        "forced_m8_to_m7_refill_exact": True,
        "terminal_b48_trajectory_parity": True,
        "turnover_control_path_compatible": True,
        "checkpoint_serialized": True,
        "checkpoint_producer_process_released": True,
        "fresh_owner_future_native_consumption": True,
        "checkpoint_resume_parity": True,
        "second_replay_parity": True,
        "partial_t80_checkpoint_provable": False,
        "partial_t80_checkpoint_status": "UNSUPPORTED_BY_NATIVE_RESTORE_CONTRACT",
        "faithful_checkpoint_boundary": "generation:3:tick:0",
        "checkpoint_terminal_boundary": "generation:6:tick:0",
        "physical_refill_output_digest_sha256": next(iter(physical_digests.values())),
        "terminal_projection_digest_set_sha256": canonical_sha256(expected_terminal),
    }


def _worker_command(
    args: argparse.Namespace,
    *,
    kind: str,
    arm: str,
    action: str | None,
    output_root: Path,
    checkpoint_path: Path | None = None,
) -> list[str]:
    artifact = getattr(args, f"{arm.split('_')[0]}_artifact")
    manifest = getattr(args, f"{arm.split('_')[0]}_manifest")
    command = [
        sys.executable,
        "-m",
        "tools.scripts.eiel_fixed_mlp_policy_d0_gate_v1",
        "--worker-mode",
        kind,
        "--arm",
        arm,
        "--artifact",
        str(artifact),
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_root),
    ]
    if kind == "trajectory":
        command.extend(["--action", str(action), "--plan", str(args.plan)])
        if checkpoint_path is not None:
            command.extend(["--checkpoint", str(checkpoint_path)])
    return command


def _run_worker(
    command: Sequence[str], *, env: Mapping[str, str]
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    output_index = list(command).index("--output-root") + 1
    receipt_path = Path(command[output_index]) / WORKER_RECEIPT
    if completed.returncode != 0 or not receipt_path.is_file():
        raise D0GateError(
            "CUDA D0 worker failed: "
            + json.dumps(
                {
                    "command": list(command),
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    return _read_mapping(receipt_path, label="worker receipt"), completed


def _load_reusable_static_evidence(
    *,
    source_root: Path,
    expected_contract: Mapping[str, Any],
    artifact_identities: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    """Reuse only self-sealed component/T160 receipts under an exact contract."""

    source_root = source_root.expanduser().resolve()
    contract_path = source_root / "execution_contract.json"
    source_contract = _read_mapping(contract_path, label="static source contract")
    source_contract_sha = source_contract.get("contract_sha256")
    unsigned_source_contract = dict(source_contract)
    unsigned_source_contract.pop("contract_sha256", None)
    unsigned_expected_contract = dict(expected_contract)
    unsigned_expected_contract.pop("contract_sha256", None)
    if (
        source_contract_sha != canonical_sha256(unsigned_source_contract)
        or unsigned_source_contract != unsigned_expected_contract
    ):
        raise D0GateError("reusable static evidence contract is not exact")

    components: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    receipt_bindings: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for kind, action, relative, destination in (
            ("component", None, Path("component") / arm, components),
            (
                "trajectory",
                "terminal",
                Path("terminal_b48_g1") / arm,
                terminals,
            ),
        ):
            receipt_path = source_root / relative / WORKER_RECEIPT
            raw = _read_mapping(receipt_path, label=f"reusable {kind} receipt")
            checked = _validate_worker_receipt(raw, kind=kind, arm=arm, action=action)
            expected_identity = artifact_identities[arm]
            if (
                checked.get("artifact_sha256")
                != expected_identity.get("artifact_sha256")
                or checked.get("manifest_sha256")
                != expected_identity.get("manifest_sha256")
                or (
                    kind == "trajectory"
                    and (
                        checked.get("start_generation") != 0
                        or checked.get("terminal_generation") != 1
                        or checked.get("terminal_tick") != 0
                        or checked.get("cells") != EXPECTED_CELLS
                        or checked.get("ticks_per_generation") != TICKS_PER_GENERATION
                    )
                )
            ):
                raise D0GateError(f"reusable {kind}/{arm} identity is not exact")
            destination[arm] = checked
            receipt_bindings[f"{kind}:{arm}"] = {
                "path": str(receipt_path.resolve()),
                "file_sha256": _sha256(receipt_path),
                "receipt_sha256": checked["receipt_sha256"],
            }
    return (
        components,
        terminals,
        {
            "source_root": str(source_root),
            "source_execution_contract_path": str(contract_path.resolve()),
            "source_execution_contract_file_sha256": _sha256(contract_path),
            "source_execution_contract_sha256": source_contract_sha,
            "receipts": receipt_bindings,
        },
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise D0GateError(f"refusing to overwrite immutable D0 root: {output_root}")
    bindings = {
        "legacy": (args.legacy_artifact, args.legacy_manifest),
        "m7_physical_refill": (args.m7_artifact, args.m7_manifest),
        "m8_suffix_reuse": (args.m8_artifact, args.m8_manifest),
    }
    artifact_identities = _validated_artifact_set(bindings)
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    artifact_source_commits = {
        row["artifact_source_commit"] for row in artifact_identities.values()
    }
    if artifact_source_commits != {repository_head}:
        raise D0GateError("D0 artifacts were not built from the frozen repository HEAD")
    plan = args.plan.expanduser().resolve()
    release_root = args.release_root.expanduser().resolve()
    if not plan.is_file() or not release_root.is_dir():
        raise D0GateError("frozen execution plan or sealed release root is absent")
    output_root.mkdir(parents=True)
    env = dict(os.environ)
    env["EIEL_GPU_TICK_RUNTIME_SOURCE_ROOT"] = str(release_root / "payload/python")
    contract = {
        "schema_version": f"{SCHEMA_VERSION}_execution_contract_v1",
        "authority": AUTHORITY,
        "repository_head": repository_head,
        "plan_path": str(plan),
        "plan_sha256": _sha256(plan),
        "release_root": str(release_root),
        "artifact_identities": artifact_identities,
        "cells": EXPECTED_CELLS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "checkpoint_contract": {
            "requested_partial_boundary": "generation:0:tick:80",
            "partial_boundary_supported": False,
            "reason": "native restore requires generation:N:tick:0",
            "minimum_admitted_generation_span": 3,
            "executed_prefix": "generation:0:tick:0_to_generation:3:tick:0",
            "executed_future": "generation:3:tick:0_to_generation:6:tick:0",
            "producer_and_restore_processes_must_differ": True,
            "restore_repeat_count": 2,
        },
        "qualification_claim": False,
        "production_admission": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    reusable_static_binding: dict[str, Any] | None = None
    components: dict[str, Mapping[str, Any]]
    terminals: dict[str, Mapping[str, Any]]
    if args.reuse_static_evidence_root is not None:
        components, terminals, reusable_static_binding = _load_reusable_static_evidence(
            source_root=args.reuse_static_evidence_root,
            expected_contract=contract,
            artifact_identities=artifact_identities,
        )
        contract.pop("contract_sha256")
        contract["reused_static_evidence"] = reusable_static_binding
        contract["contract_sha256"] = canonical_sha256(contract)
    atomic_write_json(output_root / "execution_contract.json", contract)

    if reusable_static_binding is None:
        components = {}
        terminals = {}
        for arm in ARMS:
            components[arm], _ = _run_worker(
                _worker_command(
                    args,
                    kind="component",
                    arm=arm,
                    action=None,
                    output_root=output_root / "component" / arm,
                ),
                env=env,
            )
            terminals[arm], _ = _run_worker(
                _worker_command(
                    args,
                    kind="trajectory",
                    arm=arm,
                    action="terminal",
                    output_root=output_root / "terminal_b48_g1" / arm,
                ),
                env=env,
            )

    checkpoint_path = output_root / "checkpoint_replay" / "m8_g1_checkpoint.pt"
    prefix, _ = _run_worker(
        _worker_command(
            args,
            kind="trajectory",
            arm="m8_suffix_reuse",
            action="prefix",
            output_root=output_root / "checkpoint_replay" / "prefix",
            checkpoint_path=checkpoint_path,
        ),
        env=env,
    )
    producer_pid = prefix.get("pid")
    producer_exit_confirmed = (
        type(producer_pid) is int and not Path(f"/proc/{producer_pid}").exists()
    )
    if not producer_exit_confirmed:
        raise D0GateError("prefix CUDA owner process did not terminate before restore")
    uninterrupted, _ = _run_worker(
        _worker_command(
            args,
            kind="trajectory",
            arm="m8_suffix_reuse",
            action="uninterrupted",
            output_root=output_root / "checkpoint_replay" / "uninterrupted",
        ),
        env=env,
    )
    restores = []
    for index in (1, 2):
        restored, _ = _run_worker(
            _worker_command(
                args,
                kind="trajectory",
                arm="m8_suffix_reuse",
                action="resume",
                output_root=output_root / "checkpoint_replay" / f"restore_{index}",
                checkpoint_path=checkpoint_path,
            ),
            env=env,
        )
        if restored.get("pid") == producer_pid:
            raise D0GateError("checkpoint restore reused the producer process")
        restores.append(restored)
    decision = _reduce_receipts(
        artifact_identities=artifact_identities,
        components=components,
        arm_terminals=terminals,
        prefix=prefix,
        uninterrupted=uninterrupted,
        restores=restores,
        producer_exit_confirmed=producer_exit_confirmed,
    )
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}_receipt_v1",
        "authority": AUTHORITY,
        "status": "passed",
        "execution_contract_sha256": contract["contract_sha256"],
        "artifact_identities": artifact_identities,
        "component_receipt_sha256s": {
            arm: components[arm]["receipt_sha256"] for arm in ARMS
        },
        "terminal_b48_receipt_sha256s": {
            arm: terminals[arm]["receipt_sha256"] for arm in ARMS
        },
        "reused_static_evidence": reusable_static_binding,
        "checkpoint_receipts": {
            "prefix": prefix["receipt_sha256"],
            "uninterrupted": uninterrupted["receipt_sha256"],
            "restore_1": restores[0]["receipt_sha256"],
            "restore_2": restores[1]["receipt_sha256"],
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "decision": decision,
        "qualification_claim": False,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(output_root / "d0_gate_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--legacy-artifact", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--m7-artifact", type=Path)
    parser.add_argument("--m7-manifest", type=Path)
    parser.add_argument("--m8-artifact", type=Path)
    parser.add_argument("--m8-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-static-evidence-root", type=Path)
    parser.add_argument("--worker-mode", choices=("component", "trajectory"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--action", choices=("terminal", "prefix", "resume", "uninterrupted")
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise D0GateError(f"missing required argument: {label}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_mode == "component":
        receipt = _component_worker(
            arm=_required(args.arm, "--arm"),
            artifact=_required(args.artifact, "--artifact").resolve(),
            manifest_path=_required(args.manifest, "--manifest").resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.worker_mode == "trajectory":
        receipt = _trajectory_worker(
            arm=_required(args.arm, "--arm"),
            action=_required(args.action, "--action"),
            plan_path=_required(args.plan, "--plan").resolve(),
            artifact=_required(args.artifact, "--artifact").resolve(),
            manifest_path=_required(args.manifest, "--manifest").resolve(),
            output_root=args.output_root.resolve(),
            checkpoint_path=(args.checkpoint.resolve() if args.checkpoint else None),
        )
    else:
        for field in (
            "plan",
            "release_root",
            "legacy_artifact",
            "legacy_manifest",
            "m7_artifact",
            "m7_manifest",
            "m8_artifact",
            "m8_manifest",
        ):
            _required(getattr(args, field), f"--{field.replace('_', '-')}")
        receipt = run_gate(args)
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
