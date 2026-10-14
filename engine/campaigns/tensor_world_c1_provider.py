"""Canonical campaign provider for the existing Tensor World C1 runtime.

This adapter owns campaign translation, artifact-unit validation, and durable
terminal projection.  Tensor World mechanics remain owned by
``TensorWorldC1BackendV1``; there is deliberately no CPU fallback.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import torch

from engine.tensor_world.c1_backend import TensorWorldC1BackendV1
from engine.tensor_world.c1_fixture import make_c1_fixture_v1
from engine.tensor_world.contracts import TickRequestV1
from engine.tensor_world.native_loader import MANIFEST_SCHEMA

from .canonical import canonical_sha256
from .contracts import (
    CampaignPlanV1,
    ExecutionRouteContractV1,
    MetricFamilyV1,
    ParameterContractV1,
    RuntimeProfileV1,
    UnsupportedReasonV1,
)
from .orchestration import atomic_write_json
from .providers import (
    CampaignProviderError,
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
    default_execution_provider_registry,
)
from .qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from .registry import ProfileRegistryV1
from .route_sentinels import RouteTransitionSentinelV1

TENSOR_WORLD_C1_PROFILE_ID: Final[str] = "tensor-world-c1-runtime-v1"
TENSOR_WORLD_C1_PROVIDER_ID: Final[str] = "tensor-world-c1-provider-v1"
TENSOR_WORLD_C1_WORLD_BASE: Final[str] = "tensor_world_c1"
TENSOR_WORLD_C1_ARTIFACT_ABI: Final[str] = "tensor_world_c1_cuda_v1"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_SOURCES: Final[tuple[str, ...]] = (
    "engine/tensor_world/native/tensor_world_c1_binding.cpp",
    "engine/tensor_world/native/tensor_world_c1_kernels.cu",
)
_ROUTE_SOURCES: Final[tuple[str, ...]] = (
    "engine/campaigns/tensor_world_c1_provider.py",
    "engine/tensor_world/c1_backend.py",
    "engine/tensor_world/c1_fixture.py",
    "engine/tensor_world/contracts.py",
    "engine/tensor_world/native_loader.py",
    "engine/tensor_world/state.py",
    *_NATIVE_SOURCES,
)


BackendFactoryV1 = Callable[[str], Any]
FixtureFactoryV1 = Callable[..., tuple[Any, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("Tensor World C1 plan must be a mapping")
    return dict(plan)


def _source_projection_v1() -> dict[str, Any]:
    hashes = {
        relative: _sha256_file(_REPOSITORY_ROOT / relative)
        for relative in _ROUTE_SOURCES
    }
    return {
        "projection_mode": "explicit_tensor_world_c1_route_closure_v1",
        "source_hashes": hashes,
        "source_projection_sha256": canonical_sha256(hashes),
    }


def _artifact_row(
    plan: Mapping[str, Any], name: str, *, kind: str
) -> tuple[Path, dict[str, Any]]:
    artifacts = plan.get("artifacts")
    raw = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(raw, Mapping):
        raise CampaignProviderError(f"Tensor World C1 artifact is missing: {name}")
    row = dict(raw)
    path = Path(str(row.get("path", ""))).expanduser()
    if path.is_symlink() or not path.is_file():
        raise CampaignProviderError(
            f"Tensor World C1 artifact is not a regular non-symlink file: {name}"
        )
    path = path.resolve()
    expected = row.get("sha256")
    if expected != _sha256_file(path):
        raise CampaignProviderError(f"Tensor World C1 artifact hash mismatch: {name}")
    if row.get("kind") != kind:
        raise CampaignProviderError(f"Tensor World C1 artifact kind mismatch: {name}")
    if row.get("abi") != TENSOR_WORLD_C1_ARTIFACT_ABI:
        raise CampaignProviderError(f"Tensor World C1 artifact ABI mismatch: {name}")
    return path, row


def _artifact_unit_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    binary, binary_row = _artifact_row(
        plan, "cuda_extension", kind="cuda_extension"
    )
    manifest_path, manifest_row = _artifact_row(
        plan, "extension_manifest", kind="extension_manifest"
    )
    expected_manifest = binary.with_suffix(binary.suffix + ".manifest.json")
    if manifest_path != expected_manifest:
        raise CampaignProviderError(
            "Tensor World C1 binary and companion manifest are not one resolved unit"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            f"Tensor World C1 manifest cannot be read: {exc}"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise CampaignProviderError("Tensor World C1 manifest schema is unsupported")
    if manifest.get("binary_sha256") != binary_row["sha256"]:
        raise CampaignProviderError(
            "Tensor World C1 manifest does not bind the declared binary"
        )
    source_hashes = manifest.get("sources")
    if not isinstance(source_hashes, Mapping):
        raise CampaignProviderError(
            "Tensor World C1 manifest source hashes are missing"
        )
    expected_sources = {
        relative: _sha256_file(_REPOSITORY_ROOT / relative)
        for relative in _NATIVE_SOURCES
    }
    if dict(source_hashes) != expected_sources:
        raise CampaignProviderError(
            "Tensor World C1 native artifact is not bound to current build sources"
        )
    return {
        "binary_path": str(binary),
        "binary_sha256": binary_row["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_row["sha256"],
        "manifest_schema": MANIFEST_SCHEMA,
        "build_source_commit": manifest.get("source_commit"),
        "source_hashes": expected_sources,
    }


def _digest_update(hasher: Any, value: Any, path: str) -> None:
    hasher.update(path.encode("utf-8"))
    hasher.update(b"\0")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", copy=True).contiguous()
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _digest_update(hasher, getattr(value, field.name), f"{path}.{field.name}")
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _digest_update(hasher, value[key], f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _digest_update(hasher, item, f"{path}[{index}]")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        hasher.update(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
    else:
        raise CampaignProviderError(
            f"Tensor World C1 terminal projection cannot digest {type(value).__name__}"
        )
    hasher.update(b"\0")


def _terminal_digest_v1(bundle: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"tensor_world_c1_terminal_projection_v1\0")
    _digest_update(digest, bundle, "bundle")
    return digest.hexdigest()


def tensor_world_c1_profile_v1() -> RuntimeProfileV1:
    metrics = MetricFamilyV1(
        family_id="tensor_world_c1_population_v1",
        metrics=("population.alive",),
        required_state_components=("agents",),
        null_semantics="not_null_for_completed_c1_execution",
    )
    return RuntimeProfileV1(
        profile_id=TENSOR_WORLD_C1_PROFILE_ID,
        profile_version=1,
        backend="cuda_resident",
        world_base=TENSOR_WORLD_C1_WORLD_BASE,
        world_overlays=frozenset(),
        policy_family="fixed_mlp",
        selection_modes=frozenset({"standard"}),
        observation_contract="patch_radius1_observation_8x9_plus2_v1",
        state_components=(
            "world",
            "agents",
            "occupancy",
            "policy",
            "identity",
            "rng",
            "transactions",
            "hot_events",
        ),
        checkpoint_components=("terminal_projection",),
        parameter_contracts={
            "patch.grid_size": ParameterContractV1(
                "patch.grid_size", int, lambda value: value == 32, "C1 grid width"
            ),
            "patch.initial_population": ParameterContractV1(
                "patch.initial_population",
                int,
                lambda value: value == 256,
                "C1 active resident count",
            ),
            "patch.capacity": ParameterContractV1(
                "patch.capacity", int, lambda value: value == 512, "C1 slot capacity"
            ),
        },
        uniform_batch_parameters=(
            "patch.grid_size",
            "patch.initial_population",
            "patch.capacity",
        ),
        metric_families=(metrics,),
        execution_owner="tensor_world_c1_backend_v1",
        arbitrary_positive_generations=True,
        continuation_supported=False,
        cancellation_supported=False,
        batch_min=1,
        batch_max=1,
        qualified_batch_min=1,
        qualified_batch_max=1,
        generation_counter_bits=64,
        qualification_invariants={
            "device_owner": "cuda_only_no_cpu_fallback",
            "shape": "b1_w32_p256_c512",
            "runtime_scope": "bounded_c1_route_execution_only",
            "formal_qualification_authority": "none",
        },
        qualification_status="experimental",
        limitations=(
            "Execution qualification is bounded to the historical C1 mechanics and exact route receipt.",
            "This profile grants no C6/C7 or formal Qualification authority.",
            "Tensor World C1 is CUDA-only and has no CPU fallback.",
        ),
    )


def tensor_world_c1_profile_registry_v1() -> ProfileRegistryV1:
    registry = ProfileRegistryV1()
    registry.register(tensor_world_c1_profile_v1())
    return registry


def tensor_world_c1_provider_registry_v1() -> ExecutionProviderRegistryV1:
    registry = default_execution_provider_registry()
    registry.register(TensorWorldC1ExecutionProviderV1())
    return registry


def is_tensor_world_c1_spec_v1(spec: Mapping[str, Any]) -> bool:
    world = spec.get("world")
    return (
        isinstance(world, Mapping) and world.get("base") == TENSOR_WORLD_C1_WORLD_BASE
    )


class TensorWorldC1ExecutionProviderV1:
    """Translate canonical campaign batches into the existing C1 owner."""

    provider_id = TENSOR_WORLD_C1_PROVIDER_ID
    profile_id = TENSOR_WORLD_C1_PROFILE_ID
    resident_segment_generation_min = 1
    reporting_parity_supported = False

    def __init__(
        self,
        *,
        backend_factory: BackendFactoryV1 | None = None,
        fixture_factory: FixtureFactoryV1 = make_c1_fixture_v1,
    ) -> None:
        self._backend_factory = backend_factory or (
            lambda extension: TensorWorldC1BackendV1(extension_path=extension)
        )
        self._fixture_factory = fixture_factory

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        value = _payload(plan)
        batches = value.get("physical_batches")
        batch = next(
            (
                row
                for row in batches or ()
                if isinstance(row, Mapping) and row.get("batch_id") == batch_id
            ),
            None,
        )
        if not isinstance(batch, Mapping):
            raise CampaignProviderError(f"Tensor World C1 batch is absent: {batch_id}")
        cell_ids = tuple(str(item) for item in batch.get("cell_ids", ()))
        cells = {
            str(row.get("cell_id")): row
            for row in value.get("cells", ())
            if isinstance(row, Mapping)
        }
        if not cell_ids or any(cell_id not in cells for cell_id in cell_ids):
            raise CampaignProviderError("Tensor World C1 batch cell binding is invalid")
        execution = value.get("execution")
        if not isinstance(execution, Mapping):
            raise CampaignProviderError("Tensor World C1 plan execution is missing")
        parameter_names = sorted(
            {
                str(name)
                for cell_id in cell_ids
                for name in cells[cell_id].get("parameters", {})
            }
        )
        seed_names = sorted(
            {
                str(name)
                for cell_id in cell_ids
                for name in cells[cell_id].get("seeds", {})
            }
        )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=cell_ids,
            generations=int(execution["generations"]),
            ticks_per_generation=int(execution["ticks_per_generation"]),
            checkpoint_milestones=tuple(execution.get("checkpoints", ())),
            parameter_arrays=tuple(
                (
                    name,
                    tuple(cells[cell_id]["parameters"][name] for cell_id in cell_ids),
                )
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (name, tuple(cells[cell_id]["seeds"][name] for cell_id in cell_ids))
                for name in seed_names
            ),
            checkpoint_storage_mode=str(
                execution.get("checkpoint_storage_mode", "disk")
            ),
            reporting_enabled=bool(execution.get("reporting_enabled", True)),
            execution_route_sha256=self.execution_route_contract(
                value
            ).execution_route_sha256,
        )

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        value = _payload(plan)
        reasons: list[UnsupportedReasonV1] = []
        if request.profile_id != self.profile_id:
            reasons.append(
                UnsupportedReasonV1(
                    "tensor_world_c1_profile_mismatch",
                    "resolved_profile",
                    "Tensor World C1 request binds another profile",
                    self.profile_id,
                )
            )
        if len(request.cell_ids) != 1:
            reasons.append(
                UnsupportedReasonV1(
                    "tensor_world_c1_batch_out_of_range",
                    "execution.batch",
                    "Tensor World C1 Launchpad profile admits physical B1 only",
                    self.profile_id,
                )
            )
        parameters = dict(request.parameter_arrays)
        expected = {
            "patch.grid_size": 32,
            "patch.initial_population": 256,
            "patch.capacity": 512,
        }
        if any(
            parameters.get(name) != (required,) for name, required in expected.items()
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "tensor_world_c1_shape_mismatch",
                    "parameters",
                    "Tensor World C1 requires B1/W32/P256/C512",
                    self.profile_id,
                )
            )
        if any(
            milestone != request.generations
            for milestone in request.checkpoint_milestones
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "tensor_world_c1_checkpoint_unsupported",
                    "checkpoints.milestones",
                    "C1 exposes only a terminal projection and cannot resume",
                    self.profile_id,
                )
            )
        try:
            _artifact_unit_v1(value)
            self.execution_route_contract(value, request)
        except (CampaignProviderError, OSError, ValueError) as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "tensor_world_c1_artifact_unit_invalid",
                    "artifacts",
                    str(exc),
                    self.profile_id,
                )
            )
        return tuple(reasons)

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        value = _payload(plan)
        artifact = _artifact_unit_v1(value)
        source = _source_projection_v1()
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "tensor_world_c1_runtime_owner",
                    "backend_id": "tensor_world_c1_v1",
                    "shape": "b1_w32_p256_c512",
                    "source_projection_sha256": source["source_projection_sha256"],
                    "source_file_count": len(source["source_hashes"]),
                },
                {
                    "component_id": "tensor_world_c1_native_artifact_unit",
                    "abi": TENSOR_WORLD_C1_ARTIFACT_ABI,
                    "binary_sha256": artifact["binary_sha256"],
                    "manifest_sha256": artifact["manifest_sha256"],
                    "manifest_schema": artifact["manifest_schema"],
                },
                {
                    "component_id": "tensor_world_c1_terminal_projection",
                    "contract_version": "tensor_world_c1_terminal_projection_v1",
                    "checkpoint_replay": False,
                },
            ),
            required_obligations=("cold_start", "first_tick", "terminal_settlement"),
        )
        if (
            request is not None
            and request.execution_route_sha256 is not None
            and request.execution_route_sha256 != contract.execution_route_sha256
        ):
            raise CampaignProviderError(
                "Tensor World C1 request execution-route binding mismatch"
            )
        return contract

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1],
    ) -> dict[str, Any]:
        del plan
        if not requests or any(len(request.cell_ids) != 1 for request in requests):
            raise CampaignProviderError("Tensor World C1 resource shape is not B1")
        return {
            "resource_estimator_id": "tensor_world_c1_b1_conservative_v1",
            "vram_bytes_upper_bound": 1_610_612_736,
            "wall_seconds_upper_bound": max(
                120,
                sum(
                    request.generations * request.ticks_per_generation
                    for request in requests
                )
                * 10,
            ),
            "disk_bytes_upper_bound": 64 * 1024**2,
        }

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        value = _payload(plan)
        reasons = self.readiness_reasons(value, request)
        if reasons:
            raise CampaignProviderError(reasons[0].message)
        artifact = _artifact_unit_v1(value)
        parameters = dict(request.parameter_arrays)
        bundle, config = self._fixture_factory(
            batch=1,
            width=int(parameters["patch.grid_size"][0]),
            active=int(parameters["patch.initial_population"][0]),
            capacity=int(parameters["patch.capacity"][0]),
        )
        backend = self._backend_factory(str(artifact["binary_path"]))
        state = None
        tick_receipts: list[dict[str, Any]] = []
        exported: Mapping[str, Any] | None = None
        try:
            state = backend.initialize(
                {"bundle": bundle, "config": asdict(config), "fixture_mode": False}
            )
            for _ in range(request.generations * request.ticks_per_generation):
                receipt = backend.advance(state, TickRequestV1())
                to_dict = getattr(receipt, "to_dict", None)
                tick_receipts.append(
                    dict(to_dict()) if callable(to_dict) else dict(receipt)
                )
            exported = backend.export_state(state)
        finally:
            if state is not None:
                backend.close(state)
        if not isinstance(exported, Mapping) or "bundle" not in exported:
            raise CampaignProviderError(
                "Tensor World C1 backend did not export terminal canonical state"
            )
        terminal_bundle = exported["bundle"]
        terminal_digest = _terminal_digest_v1(terminal_bundle)
        cell_id = request.cell_ids[0]
        alive = int(terminal_bundle.agents.alive.sum().item())
        total_ticks = request.generations * request.ticks_per_generation
        sentinel = RouteTransitionSentinelV1(
            transition_name="terminal_settlement",
            absolute_generation=request.generations,
            absolute_tick=total_ticks,
            physical_window_position=total_ticks,
            alive_identity_digest=terminal_digest,
            world_digest=terminal_digest,
            profile_fields={
                "backend_id": "tensor_world_c1_v1",
                "cell_id": cell_id,
            },
        )
        terminal_path = Path(context.batch_root) / "tensor_world_c1_terminal.json"
        terminal = {
            "schema_version": "tensor_world_c1_terminal_projection_v1",
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "backend_id": "tensor_world_c1_v1",
            "batch_id": request.batch_id,
            "cell_id": cell_id,
            "terminal_generation": request.generations,
            "terminal_tick": total_ticks,
            "terminal_digest_sha256": terminal_digest,
            "population_alive": alive,
            "tick_receipt_count": len(tick_receipts),
            "tick_receipts_sha256": canonical_sha256(tick_receipts),
            "artifact_unit": artifact,
            "execution_route_sha256": request.execution_route_sha256,
        }
        atomic_write_json(terminal_path, terminal)
        terminal_file_sha256 = _sha256_file(terminal_path)
        checkpoint = {
            "generation": request.generations,
            "path": str(terminal_path),
            "checkpoint_file_sha256": terminal_file_sha256,
            "state_sha256": terminal_digest,
            "terminal_cell_digests": {cell_id: terminal_digest},
            "execution_route_sha256": request.execution_route_sha256,
            "resume_supported": False,
        }
        return {
            "status": "completed",
            "checkpoint": checkpoint,
            "metrics_rows": [
                {"cell_id": cell_id, "metrics": {"population.alive": alive}}
            ],
            "result": {
                "backend_id": "tensor_world_c1_v1",
                "terminal_generation": request.generations,
                "terminal_tick": total_ticks,
                "terminal_artifact_path": str(terminal_path),
                "terminal_artifact_sha256": terminal_file_sha256,
                "terminal_cell_digests": {cell_id: terminal_digest},
                "transition_sentinels": {cell_id: [sentinel.payload()]},
                "artifact_unit": artifact,
            },
        }

    def execute_route_qualification(
        self,
        *,
        campaign_plan: Mapping[str, Any],
        canary: Mapping[str, Any],
        run_root: Path,
    ) -> dict[str, Any]:
        from .route_qualification import (
            bounded_route_qualification_preflight_v1,
            execute_route_qualification_v1,
        )

        artifacts = campaign_plan.get("artifacts")
        receipt = (
            artifacts.get("qualification_receipt")
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(receipt, Mapping):
            raise CampaignProviderError(
                "Tensor World C1 qualification receipt artifact is missing"
            )
        qualification_receipt_path = Path(str(receipt.get("path", ""))).resolve()
        registry = ExecutionProviderRegistryV1()
        registry.register(self)
        return execute_route_qualification_v1(
            campaign_plan=campaign_plan,
            canary=canary,
            run_root=run_root,
            route_contract=self.execution_route_contract(campaign_plan),
            qualification_receipt_path=qualification_receipt_path,
            preflight_runner=lambda plan: bounded_route_qualification_preflight_v1(
                self, plan, profile=tensor_world_c1_profile_v1()
            ),
            provider_registry=registry,
        )


__all__ = [
    "TENSOR_WORLD_C1_ARTIFACT_ABI",
    "TENSOR_WORLD_C1_PROFILE_ID",
    "TENSOR_WORLD_C1_PROVIDER_ID",
    "TENSOR_WORLD_C1_WORLD_BASE",
    "TensorWorldC1ExecutionProviderV1",
    "is_tensor_world_c1_spec_v1",
    "tensor_world_c1_profile_registry_v1",
    "tensor_world_c1_profile_v1",
    "tensor_world_c1_provider_registry_v1",
]
