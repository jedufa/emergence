from __future__ import annotations

from typing import Any

from engine.authoritative_state.identity import canonical_sha256
from engine.campaigns.inheritance_cross_path_exposure_projection_contract_v1 import (
    ExposureProjectionEnvelopeV1,
    compose_inheritance_cross_path_exposure_projection_contract_v1,
)
from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    BUILD_PROVENANCE_SCHEMA_VERSION,
    COMMITTED_SOURCE_SNAPSHOT_SCHEMA_VERSION,
)


BUILD_SOURCE_COMMIT = "6" * 40
COMPILED_SOURCE_RELATIVE_PATH = "engine/perf/test-build-source.cpp"
COMPILED_SOURCE_SHA256 = "5" * 64


def synthetic_build_provenance_manifest_fields_v1() -> dict[str, Any]:
    unsigned_snapshot = {
        "schema_version": COMMITTED_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "build_source_commit": BUILD_SOURCE_COMMIT,
        "sources": [
            {
                "path": COMPILED_SOURCE_RELATIVE_PATH,
                "sha256": COMPILED_SOURCE_SHA256,
            }
        ],
    }
    snapshot_sha256 = canonical_sha256(unsigned_snapshot)
    return {
        "authoritative_state_build_provenance_v1": {
            "schema_version": BUILD_PROVENANCE_SCHEMA_VERSION,
            "build_source_commit": BUILD_SOURCE_COMMIT,
            "compiled_source_snapshot_sha256": snapshot_sha256,
        },
        "committed_source_snapshot": {
            **unsigned_snapshot,
            "compiled_source_snapshot_sha256": snapshot_sha256,
        },
    }


def synthetic_exposure_projection_contract_v1(
    storage_mode: str,
) -> dict[str, Any]:
    return compose_inheritance_cross_path_exposure_projection_contract_v1(
        ExposureProjectionEnvelopeV1(
            max_batch=2,
            segment_generations=2,
            storage_mode=storage_mode,
            attempt_capacity=8,
            snapshot_capacity=16 if storage_mode == "exact_full" else 1,
            eligible_capacity=4,
            aggregate_capacity=300,
            agent_id_current_capacity=4,
            population_next_id=4,
            max_births_per_tick=1,
            slot_capacity=4,
            allocation_generation_count=2,
            retained_agent_id_capacity=4,
        )
    )
