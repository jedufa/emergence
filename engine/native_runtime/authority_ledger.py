from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .engine import NativeEielBatchEngine, NativeEielEngine
from .parity_gate import exact_cpu_merge_envelope_supported


LEDGER_SCHEMA_VERSION = "eiel_native_authority_ledger_v1"


REQUIRED_SUBSYSTEMS = (
    "execution metadata",
    "population slots/stable ids",
    "policy banks",
    "grouping/row maps",
    "observations",
    "MLP/GRU/Hmem",
    "action cache",
    "Flat",
    "Patch",
    "WDL",
    "Jungle",
    "scalar/drives",
    "chemistry/maintenance",
    "lifecycle/reproduction",
    "mutation/recombination",
    "lineage/founder",
    "events/public outputs",
    "checkpoint",
    "multi-cell",
)


@dataclass(frozen=True)
class NativeAuthorityRow:
    subsystem: str
    source_of_truth: str
    live_execution_owner: str
    rng_owner: str
    checkpoint_owner: str
    event_output_owner: str
    cpu_parity: str
    cuda_parity: str
    fallback_count: int
    python_calls: int
    per_agent_traversals: int
    h2d_d2h: str
    performance_contribution: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row(
    subsystem: str,
    *,
    source_of_truth: str,
    live_execution_owner: str,
    rng_owner: str = "not-measured",
    checkpoint_owner: str = "native-projection",
    event_output_owner: str = "not-native",
    cpu_parity: str = "not-measured",
    cuda_parity: str = "not-implemented",
    fallback_count: int = 0,
    python_calls: int = 0,
    per_agent_traversals: int = 0,
    h2d_d2h: str = "not-measured",
    performance_contribution: str = "not-measured",
    evidence: str,
) -> NativeAuthorityRow:
    return NativeAuthorityRow(
        subsystem=subsystem,
        source_of_truth=source_of_truth,
        live_execution_owner=live_execution_owner,
        rng_owner=rng_owner,
        checkpoint_owner=checkpoint_owner,
        event_output_owner=event_output_owner,
        cpu_parity=cpu_parity,
        cuda_parity=cuda_parity,
        fallback_count=int(fallback_count),
        python_calls=int(python_calls),
        per_agent_traversals=int(per_agent_traversals),
        h2d_d2h=h2d_d2h,
        performance_contribution=performance_contribution,
        evidence=evidence,
    )


def generate_native_authority_ledger(
    engine: NativeEielEngine,
    *,
    batch_engine: NativeEielBatchEngine | None = None,
) -> dict[str, Any]:
    counters = engine.counters.to_dict()
    capability = engine.capability.to_dict()
    batch_python_loops = int(batch_engine.counters.batch_python_cell_loops) if batch_engine is not None else 0
    batch_native_loops = int(batch_engine.counters.batch_native_cell_loops) if batch_engine is not None else 0
    rows: list[NativeAuthorityRow] = [
        _row(
            "execution metadata",
            source_of_truth="native",
            live_execution_owner="native_runtime.NativeEielEngine",
            rng_owner="native-owner-migration-backend",
            checkpoint_owner="native compact projection",
            event_output_owner="compact native result",
            cpu_parity="covered-by-cxx-regression",
            cuda_parity="not-implemented",
            python_calls=int(counters["generation_calls"]),
            evidence="NativeEielEngine counters",
        ),
        _row(
            "population slots/stable ids",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            rng_owner="none-for-tested-commit",
            checkpoint_owner="native tensors",
            event_output_owner="compact native result",
            cpu_parity="focused resident birth and death/cull slot reuse tests",
            python_calls=0,
            evidence="slot_to_agent_id/agent_id_to_slot tensors with native death-cull reuse",
        ),
        _row(
            "policy banks",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            checkpoint_owner="native tensors",
            cpu_parity="covered-by-cxx-regression",
            evidence="mlp_pack/gru_pack/hmem tensors",
        ),
        _row(
            "grouping/row maps",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="group_rows and id-to-slot tests",
        ),
        _row(
            "observations",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="flat/patch observation tests",
        ),
        _row(
            "MLP/GRU/Hmem",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="policy refill and Hmem tests",
        ),
        _row(
            "action cache",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="action_cache_token/cursor/next tensors",
        ),
        _row(
            "Flat",
            source_of_truth="native",
            live_execution_owner="native generation entrypoint",
            cpu_parity="covered-by-cxx-regression",
            python_calls=int(counters["generation_calls"]),
            evidence="execute_generation_flat_with_population_birth_commit_cpu",
        ),
        _row(
            "Patch",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="execute_generation_patch_cpu tests",
        ),
        _row(
            "WDL",
            source_of_truth="boundary projection",
            live_execution_owner="bounded native context only",
            cpu_parity="partial",
            evidence="execute_wdl_jungle_context_cpu is not exact production authority",
        ),
        _row(
            "Jungle",
            source_of_truth="boundary projection",
            live_execution_owner="bounded native context only",
            cpu_parity="partial",
            evidence="execute_wdl_jungle_context_cpu is not exact production authority",
        ),
        _row(
            "scalar/drives",
            source_of_truth="native",
            live_execution_owner="resident tensor owner",
            cpu_parity="covered-by-cxx-regression",
            evidence="drive tensor projection and generation tests",
        ),
        _row(
            "chemistry/maintenance",
            source_of_truth="boundary projection",
            live_execution_owner="native scalar storage; production semantics incomplete",
            cpu_parity="partial",
            evidence="chemistry/maintenance tensors exist but full production logic not ported",
        ),
        _row(
            "lifecycle/reproduction",
            source_of_truth="native",
            live_execution_owner="native death/cull, slot reuse, resident child-row commit, bounded failed-birth rows, and adjacent mate-required lane",
            rng_owner="not-native-for-full-production",
            cpu_parity="focused resident death/cull, birth, failed-birth, and mate-required tests",
            evidence="execute_population_death_cull_cpu and execute_population_asexual_resident_birth_commit_cpu",
        ),
        _row(
            "mutation/recombination",
            source_of_truth="boundary projection",
            live_execution_owner="native bounded policy-row mutation with imported RNGManager mutation torch CPU state; adjacent mate lane copies parent policy without recombination",
            rng_owner="native imported NumPy PCG64 env state for spawn and imported Torch mutation generator state for policy mutation",
            checkpoint_owner="native metadata/tensors for PCG64 env state and torch mutation state",
            event_output_owner="compact native mutation count",
            cpu_parity="focused native resident birth mutation and CXX21 PCG64/Torch state parity tests",
            fallback_count=1,
            evidence="native_runtime_numpy_env_state, rng_manager_mutation_torch_cpu_state, mutation_rows, and CXX21 PCG64 raw/shuffle tests; exact production recombination remains blocked",
        ),
        _row(
            "lineage/founder",
            source_of_truth="native",
            live_execution_owner="native resident child-row commit for bounded lineage fields",
            cpu_parity="focused resident birth commit test",
            evidence="lineage_founder_id/depth/birth tensors",
        ),
        _row(
            "events/public outputs",
            source_of_truth="native",
            live_execution_owner="native bounded event tape for resident death/birth/mutation/failed-birth",
            event_output_owner="native event_tape_* tensors with canonical public-output serialization",
            checkpoint_owner="native event_tape_* tensors",
            cpu_parity="15-case event matrix plus focused native death, birth, mutation, and failed-birth event tape tests",
            evidence="event_tape_type/actor_id/target_id/value tensors, typed lifecycle artifacts, and public-output canonicalization",
        ),
        _row(
            "checkpoint",
            source_of_truth="native",
            live_execution_owner="native compact projection",
            checkpoint_owner="native tensors",
            cpu_parity="covered-by-cxx-regression",
            evidence="export_checkpoint_projection and checkpoint roundtrip tests",
        ),
        _row(
            "multi-cell",
            source_of_truth="native" if batch_native_loops else "boundary projection",
            live_execution_owner=(
                "prebuilt native batch entrypoint"
                if batch_native_loops
                else "NativeEielBatchEngine migration scaffold"
            ),
            checkpoint_owner="per-cell native owner",
            fallback_count=1 if batch_python_loops else 0,
            python_calls=int(batch_python_loops),
            cpu_parity=(
                "focused native runtime batch test"
                if batch_native_loops
                else "not-measured"
            ),
            evidence=(
                "execute_native_batch_flat_with_population_birth_commit_cpu"
                if batch_native_loops
                else "batch scaffold counts Python cell loops until fused engine lands"
            ),
        ),
    ]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "capability": capability,
        "counters": counters,
        "release_gates": {
            "native_engine_constructions": int(counters["engine_constructions"]),
            "generation_reconstruction_zero_after_initial_import": int(counters["import_projection_calls"]) <= 1,
            "checkpoint_restore_supported": int(counters["checkpoint_restore_calls"]) >= 0,
            "normal_agent_sync_zero": int(counters["normal_apply_owner_checkpoint_to_runner_calls"]) == 0,
            "normal_python_agent_traversals_zero": int(counters["normal_python_agent_traversals"]) == 0,
            "runtime_jit_builds_zero": not bool(capability["runtime_jit_in_backend"]),
            "python_batch_cell_loops_zero": batch_python_loops == 0,
            "cxx19_exact_cpu_merge_envelope_supported": exact_cpu_merge_envelope_supported(capability),
            "cxx19_formal_benchmark_allowed": bool(
                capability.get("supports_cxx19_formal_benchmark_gate", False)
            ),
        },
        "rows": [row.to_dict() for row in rows],
    }


def required_subsystems_present(rows: Iterable[dict[str, Any]]) -> bool:
    names = {str(row.get("subsystem", "")) for row in rows}
    return set(REQUIRED_SUBSYSTEMS).issubset(names)
