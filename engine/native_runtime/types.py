from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ABI_VERSION = "eiel_native_runtime_abi_v1"
SCHEMA_VERSION = "eiel_native_runtime_v1"
EXTENSION_VERSION = "eiel_native_runtime_flat_v1"


@dataclass(frozen=True)
class NativeRuntimeCapability:
    schema_version: str = SCHEMA_VERSION
    abi_version: str = ABI_VERSION
    backend: str = "resident_owner_migration_backend"
    stable_package: str = "native_runtime"
    extension_version: str = ""
    extension_abi_version: str = ""
    active_world: str = ""
    policy_family: str = ""
    device: str = ""
    lifecycle_branch: str = ""
    checkpoint_mode: str = ""
    reporting_support: str = ""
    benchmark_qualification: str = ""
    extension_abi_compatible: bool = False
    accepted_default_native_envelope: bool = False
    cpu: bool = False
    cuda: bool = False
    persistent_engine: bool = False
    runtime_jit_in_backend: bool = False
    production_jit_free: bool = False
    supports_flat_generation: bool = False
    supports_patch_generation: bool = False
    supports_asexual_resident_birth_commit: bool = False
    supports_patch_churn: bool = False
    supports_exact_production_asexual_child_construction: bool = False
    supports_exact_production_spawn_rng_order: bool = False
    supports_exact_production_mate_selection: bool = False
    supports_exact_production_recombination: bool = False
    supports_exact_production_mutation_rng_budget: bool = False
    supports_exact_death_cull_slot_reuse_lineage: bool = False
    supports_exact_typed_event_tape: bool = False
    supports_exact_public_output_serialization: bool = False
    supports_cxx19_formal_benchmark_gate: bool = False
    supports_python_agent_sync_in_normal_lane: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NativeEngineCounters:
    schema_version: str = "eiel_native_runtime_counters_v1"
    engine_constructions: int = 0
    generation_calls: int = 0
    import_projection_calls: int = 0
    checkpoint_restore_calls: int = 0
    validation_export_calls: int = 0
    report_export_calls: int = 0
    normal_apply_owner_checkpoint_to_runner_calls: int = 0
    normal_python_agent_traversals: int = 0
    runtime_jit_build_requests: int = 0
    batch_generation_calls: int = 0
    batch_python_cell_loops: int = 0
    native_batch_generation_calls: int = 0
    batch_native_cell_loops: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NativeGenerationResult:
    schema_version: str
    decision: str
    executed: bool
    generation_index: int
    steps_per_generation: int
    world: str
    native_tick_loop_count: int
    committed_birth_rows: int
    counters: NativeEngineCounters
    capability: NativeRuntimeCapability
    compact_result: dict[str, Any]
    raw_result: dict[str, Any]

    def to_diagnostic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "executed": bool(self.executed),
            "generation_index": int(self.generation_index),
            "steps_per_generation": int(self.steps_per_generation),
            "world": self.world,
            "native_tick_loop_count": int(self.native_tick_loop_count),
            "committed_birth_rows": int(self.committed_birth_rows),
            "counters": self.counters.to_dict(),
            "capability": self.capability.to_dict(),
            "compact_result": dict(self.compact_result),
        }
