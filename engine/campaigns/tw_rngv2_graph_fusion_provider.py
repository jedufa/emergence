"""Launchpad provider for default-off V2 live C0–C7 graph-fusion QualPrep."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from pathlib import Path
from typing import Any, Final, Mapping, Sequence

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

PROFILE_ID: Final[str] = "tw-rngv2-graph-fusion-qualprep-v1"
PROVIDER_ID: Final[str] = "tw-rngv2-graph-fusion-qualprep-provider-v1"
WORLD_BASE: Final[str] = "tensor_world_c0_c7_rngv2_graph_fusion"


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("graph-fusion plan must be a mapping")
    return dict(plan)


def is_tw_rngv2_graph_fusion_spec_v1(spec: Mapping[str, Any]) -> bool:
    world = spec.get("world")
    return isinstance(world, Mapping) and world.get("base") == WORLD_BASE


def tw_rngv2_graph_fusion_profile_v1() -> RuntimeProfileV1:
    metrics = MetricFamilyV1(
        family_id="tw_rngv2_graph_fusion_residency_v1",
        metrics=("graph.ticks", "graph.refills"),
        required_state_components=("graph",),
        null_semantics="not_null_for_completed_qualprep_execution",
    )
    return RuntimeProfileV1(
        profile_id=PROFILE_ID,
        profile_version=1,
        backend="cuda_resident",
        world_base=WORLD_BASE,
        world_overlays=frozenset(),
        policy_family="fixed_mlp",
        selection_modes=frozenset({"standard"}),
        observation_contract="r11_resident_graph_v1",
        state_components=("graph", "rng_v2", "identity"),
        checkpoint_components=("semantic_frontier",),
        parameter_contracts={
            "fusion.capacity": ParameterContractV1(
                "fusion.capacity", int, lambda value: value in {32, 512}, "R11 admitted capacity"
            ),
            "fusion.windows": ParameterContractV1(
                "fusion.windows", int, lambda value: value >= 1, "Five-tick graph windows"
            ),
        },
        uniform_batch_parameters=("fusion.capacity", "fusion.windows"),
        metric_families=(metrics,),
        execution_owner="tw_rngv2_graph_fusion_qualprep_v1",
        arbitrary_positive_generations=True,
        continuation_supported=False,
        cancellation_supported=False,
        batch_min=1,
        batch_max=1,
        qualified_batch_min=1,
        qualified_batch_max=1,
        generation_counter_bits=64,
        qualification_invariants={
            "device_owner": "cuda_only",
            "live_graph_batch": 48,
            "formal_qualification_authority": "none",
            "default_activation": False,
        },
        qualification_status="experimental",
        limitations=(
            "QualPrep only; no formal Q6.",
            "Live R11 graph remains B48; V2 mutation/birth is default-off.",
            "Action/environment RNG stays on V1 PCG64.",
        ),
    )


def tw_rngv2_graph_fusion_profile_registry_v1() -> ProfileRegistryV1:
    registry = ProfileRegistryV1()
    registry.register(tw_rngv2_graph_fusion_profile_v1())
    return registry


def tw_rngv2_graph_fusion_provider_registry_v1() -> ExecutionProviderRegistryV1:
    registry = default_execution_provider_registry()
    registry.register(TwRngv2GraphFusionQualPrepProviderV1())
    return registry


class TwRngv2GraphFusionQualPrepProviderV1:
    """Run the fused V2 live-graph QualPrep harness through Launchpad."""

    provider_id = PROVIDER_ID
    profile_id = PROFILE_ID
    resident_segment_generation_min = 1
    reporting_parity_supported = False

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
            raise CampaignProviderError(f"graph-fusion batch is absent: {batch_id}")
        cell_ids = tuple(str(item) for item in batch.get("cell_ids", ()))
        cells = {
            str(row.get("cell_id")): row
            for row in value.get("cells", ())
            if isinstance(row, Mapping)
        }
        if not cell_ids or any(cell_id not in cells for cell_id in cell_ids):
            raise CampaignProviderError("graph-fusion batch cell binding is invalid")
        execution = value.get("execution")
        if not isinstance(execution, Mapping):
            raise CampaignProviderError("graph-fusion plan execution is missing")
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
        reasons: list[UnsupportedReasonV1] = []
        if request.profile_id != self.profile_id:
            reasons.append(
                UnsupportedReasonV1(
                    "tw_rngv2_graph_fusion_profile_mismatch",
                    "resolved_profile",
                    "graph-fusion request binds another profile",
                    self.profile_id,
                )
            )
        if len(request.cell_ids) != 1:
            reasons.append(
                UnsupportedReasonV1(
                    "tw_rngv2_graph_fusion_batch_out_of_range",
                    "execution.batch",
                    "graph-fusion QualPrep admits physical B1 only",
                    self.profile_id,
                )
            )
        return tuple(reasons)

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "tw_rngv2_live_r11_graph_fusion",
                    "backend_id": "tensor_world_c0_c7_rngv2_graph_fusion_v1",
                    "shape": "b48_r11_window5",
                    "default_activation": False,
                },
            ),
            required_obligations=("cold_start", "first_tick", "terminal_settlement"),
        )
        if (
            request is not None
            and request.execution_route_sha256 is not None
            and request.execution_route_sha256 != contract.execution_route_sha256
        ):
            raise CampaignProviderError("graph-fusion execution-route binding mismatch")
        return contract

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1] | tuple,
    ) -> dict[str, Any]:
        del plan
        windows = 2
        if requests:
            parameters = dict(requests[0].parameter_arrays)
            raw = parameters.get("fusion.windows")
            if raw:
                windows = int(raw[0])
        return {
            "resource_estimator_id": "tw_rngv2_graph_fusion_b48_c32_v1",
            "vram_bytes_upper_bound": 1_500_000_000,
            "wall_seconds_upper_bound": max(300, 120 + windows),
            "disk_bytes_upper_bound": 64 * 1024**2,
        }

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        reasons = self.readiness_reasons(plan, request)
        if reasons:
            raise CampaignProviderError(reasons[0].message)
        parameters = dict(request.parameter_arrays)
        capacity = int(parameters["fusion.capacity"][0])
        windows = int(parameters["fusion.windows"][0])
        import importlib.util
        import os

        os.environ.setdefault(
            "TENSOR_WORLD_C2_PREBUILT",
            public_runtime_path('evidence_raw/tensor_world/c7_current/20260817-c7-fullroute-source-1d2bd319b-05/bin/tensor_world_c2_ext.so'),
        )

        harness = Path(__file__).resolve().parents[2] / "_artifacts" / "tw_rngv2_graph_fusion" / "run_q3_q4.py"
        spec = importlib.util.spec_from_file_location("tw_rngv2_graph_fusion_run_q3_q4", harness)
        if spec is None or spec.loader is None:
            raise CampaignProviderError("graph-fusion QualPrep harness is missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        output = Path(context.batch_root)
        output.mkdir(parents=True, exist_ok=True)
        build = public_runtime_path('build_cache/tw_rngv2_graph_fusion')
        if windows <= 2 and capacity == 32:
            payload = module.run_q3(output, build=build)
        else:
            payload = module.run_q4(
                output,
                build=build,
                ticks=windows * 5,
                capacity=capacity,
            )
        cell_id = request.cell_ids[0]
        ticks = int(payload.get("run", {}).get("ticks") or payload.get("geometry", {}).get("ticks") or windows * 5)
        refills = int(payload.get("refills") or payload.get("run", {}).get("refills") or 0)
        digest = canonical_sha256(payload)
        terminal_path = output / "graph_fusion_terminal.json"
        atomic_write_json(terminal_path, payload)
        sentinel = RouteTransitionSentinelV1(
            transition_name="live_graph_window",
            absolute_generation=request.generations,
            absolute_tick=ticks,
            physical_window_position=ticks,
            alive_identity_digest=digest,
            world_digest=digest,
            profile_fields={"backend_id": WORLD_BASE, "cell_id": cell_id},
        )
        return {
            "status": "completed",
            "checkpoint": {
                "generation": request.generations,
                "path": str(terminal_path),
                "checkpoint_file_sha256": digest,
                "state_sha256": digest,
                "terminal_cell_digests": {cell_id: digest},
                "execution_route_sha256": request.execution_route_sha256,
                "resume_supported": False,
            },
            "metrics_rows": [
                {
                    "cell_id": cell_id,
                    "metrics": {"graph.ticks": ticks, "graph.refills": refills},
                }
            ],
            "result": {
                "backend_id": WORLD_BASE,
                "terminal_artifact_path": str(terminal_path),
                "transition_sentinels": {cell_id: [sentinel.payload()]},
                "payload": payload,
            },
        }
