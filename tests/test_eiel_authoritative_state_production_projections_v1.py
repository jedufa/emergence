from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import load_schema_v1
from engine.campaigns.analysis import (
    CampaignAnalysisError,
    authoritative_analysis_projection_v1,
)
from engine.campaigns.provider_native_chronicle_v1 import (
    ProviderNativeChronicleConsumerV1,
    ProviderNativeChronicleError,
    authoritative_chronicle_event_v1,
    authoritative_provider_projection_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
)
from simulation.chronicle.schema import canonical_json
from simulation.chronicle.storage import ARTIFACT_NAMES


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity() -> dict[str, str]:
    schema = load_schema_v1()
    return {
        "authoritative_schema_hash": schema.schema_hash,
        "binary_schema_hash": "b" * 64,
        "checkpoint_schema_hash": schema.components[
            "checkpoint_schema.json"
        ]["contract_hash"],
    }


def _attempt() -> dict:
    return {
        "attempt_index": 0,
        "path": 0,
        "generation": 0,
        "tick": 4,
        "eligible_parent_counts": [3, 2],
        "selected_parent_ids": [11, 13],
        "success": True,
        "failure_code": 0,
        "commit_ordinal": 7,
        "authoritative_commit": {
            "child_agent_id": 17,
            "commit_authority_sha256": "c" * 64,
        },
        "eligible_snapshot_refs": [0, 1],
    }


def _artifact() -> dict:
    cell = {
        "cell_index": 0,
        "enabled": True,
        "status": 0,
        "attempt_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "snapshot_count": 2,
        "attempts": [_attempt()],
        "eligible_parent_snapshots": [
            {"snapshot_index": 0},
            {"snapshot_index": 1},
        ],
        "selected_parent_counts": [{"agent_id": 11, "count": 1}],
        "parent_contribution_counts": [{"agent_id": 11, "count": 1}],
        "primary_eligibility_opportunity_counts": [
            {"agent_id": 11, "count": 1}
        ],
        "mate_eligibility_opportunity_counts": [
            {"agent_id": 13, "count": 1}
        ],
        "cumulative_counts_after_segment": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
            "eligible_snapshots": 2,
        },
    }
    schema_version = "inheritance_cross_path_reproduction_exposure_segment_v1"
    cell["segment_sha256"] = _digest(
        {
            "schema_version": schema_version,
            "segment_index": 0,
            "cell": cell,
        }
    )
    cell["previous_chain_sha256"] = "0" * 64
    cell["chain_sha256_after_segment"] = _digest(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": 0,
            "previous_chain_sha256": "0" * 64,
            "segment_sha256": cell["segment_sha256"],
        }
    )
    return {
        "schema_version": schema_version,
        "segment_index": 0,
        "capacity_semantics": {"storage_mode": "exact_full"},
        "cell_id": "cell-a",
        "cell": cell,
        "segment_output_sha256": "1" * 64,
    }


def test_production_chronicle_persists_schema_bound_provider_projection(
    tmp_path: Path,
) -> None:
    consumer = ProviderNativeChronicleConsumerV1(
        batch_root=tmp_path,
        campaign_id="campaign-a",
        batch_id="batch-a",
        cell_ids=("cell-a",),
        profile="scientific_full",
        authoritative_identity_bindings=_identity(),
    )
    consumer.observe_segment(
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        cell_ids=("cell-a",),
        segment_artifacts=(_artifact(),),
        checkpoint_projections=({},),
    )

    event_path = (
        tmp_path
        / "cells"
        / "cell-a"
        / "chronicle"
        / ARTIFACT_NAMES["events"]
    )
    event = json.loads(event_path.read_text(encoding="utf-8").splitlines()[0])
    authority = event["authoritative_state_schema_v1"]
    provider = authority["provider_projection"]
    chronicle = authority["chronicle_event"]
    assert provider["values"] == {
        "provider.reproduction_attempt_index.v1": 0,
        "provider.eligible_parent_count.v1": [3, 2],
        "provider.selected_parent_id.v1": [11, 13],
        "provider.reproduction_success.v1": True,
    }
    assert chronicle["values"]["chronicle.generation.v1"] == 0
    assert chronicle["values"]["chronicle.step.v1"] == 4
    assert (
        chronicle["values"]["chronicle.intra_step_sequence.v1"] == 0
    )
    assert (
        chronicle["open_value_provenance"]["measurements"][
            "provider.selected_parent_id.v1"
        ]["source_field_id"]
        == "provider.selected_parent_id.v1"
    )


def test_provider_and_chronicle_reject_campaign_identity_mismatch() -> None:
    stale = _identity()
    stale["authoritative_schema_hash"] = "f" * 64
    with pytest.raises(
        ProviderNativeChronicleError, match="authoritative_schema_hash"
    ):
        authoritative_provider_projection_v1(
            attempt=_attempt(),
            source_record_sha256="a" * 64,
            identity_bindings=stale,
        )

    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=_identity(),
    )
    provider["bindings"]["binary_schema_hash"] = "f" * 64
    with pytest.raises(ValueError, match="binary_schema_hash"):
        authoritative_chronicle_event_v1(
            run_id="cell-a",
            generation=0,
            step=4,
            intra_step_sequence=0,
            subject_id="0:0",
            provider_projection=provider,
            identity_bindings=_identity(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attempt_index", True, "integer"),
        ("selected_parent_ids", [11, -2], "selected parent ID"),
        ("success", 1, "exact boolean"),
    ],
)
def test_provider_projection_rejects_value_domain_drift(
    field: str, value: object, message: str
) -> None:
    attempt = _attempt()
    attempt[field] = value
    with pytest.raises(ValueError, match=message):
        authoritative_provider_projection_v1(
            attempt=attempt,
            source_record_sha256="a" * 64,
            identity_bindings=_identity(),
        )


def test_analysis_projection_validates_provider_and_source_provenance() -> None:
    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=_identity(),
    )
    source_values = {
        "reporting.injury_start.v1": [0.25, 1.5],
        "reporting.injury_end.v1": [0.75, 1.0],
        "identity.lineage_founder_id.v1": [7, -1],
        "reporting.offspring_count.v1": [2, 0],
    }
    source_hashes = {
        field_id: f"{index + 1:064x}"
        for index, field_id in enumerate(source_values)
    }
    projection = authoritative_analysis_projection_v1(
        source_values=source_values,
        source_record_hashes=source_hashes,
        provider_projection=provider,
        identity_bindings=_identity(),
    )
    assert projection["values"] == {
        "analysis.injury_delta.v1": [0.5, -0.5],
        "analysis.lineage_id.v1": ["7", "-1"],
        "analysis.offspring_count.v1": [2, 0],
    }
    assert projection["provenance"]["analysis.injury_delta.v1"][
        "source_record_hashes"
    ] == [source_hashes["reporting.injury_start.v1"], source_hashes[
        "reporting.injury_end.v1"
    ]]

    stale_provider = deepcopy(provider)
    stale_provider["bindings"]["binary_schema_hash"] = "e" * 64
    with pytest.raises(CampaignAnalysisError, match="binary_schema_hash"):
        authoritative_analysis_projection_v1(
            source_values=source_values,
            source_record_hashes=source_hashes,
            provider_projection=stale_provider,
            identity_bindings=_identity(),
        )


def test_analysis_projection_rejects_wrong_source_field_identity() -> None:
    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=_identity(),
    )
    values = {
        "reporting.injury_start.v1": [0.0],
        "reporting.injury_end.v1": [1.0],
        "identity.lineage_founder_id.v1": [1],
        "reporting.offspring_total.v1": [2],
    }
    with pytest.raises(CampaignAnalysisError, match="source field IDs"):
        authoritative_analysis_projection_v1(
            source_values=values,
            source_record_hashes={
                field_id: "a" * 64 for field_id in values
            },
            provider_projection=provider,
            identity_bindings=_identity(),
        )


@pytest.mark.parametrize("value", [True, 2**63, -1])
def test_analysis_projection_rejects_non_signed_int64_offspring(
    value: object,
) -> None:
    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=_identity(),
    )
    values = {
        "reporting.injury_start.v1": [0.0],
        "reporting.injury_end.v1": [1.0],
        "identity.lineage_founder_id.v1": [1],
        "reporting.offspring_count.v1": [value],
    }
    with pytest.raises(CampaignAnalysisError, match="offspring_count"):
        authoritative_analysis_projection_v1(
            source_values=values,
            source_record_hashes={
                field_id: "a" * 64 for field_id in values
            },
            provider_projection=provider,
            identity_bindings=_identity(),
        )


def test_analysis_projection_accepts_signed_int64_maximum() -> None:
    provider = authoritative_provider_projection_v1(
        attempt=_attempt(),
        source_record_sha256="a" * 64,
        identity_bindings=_identity(),
    )
    values = {
        "reporting.injury_start.v1": [0.0],
        "reporting.injury_end.v1": [1.0],
        "identity.lineage_founder_id.v1": [1],
        "reporting.offspring_count.v1": [2**63 - 1],
    }
    projection = authoritative_analysis_projection_v1(
        source_values=values,
        source_record_hashes={
            field_id: "a" * 64 for field_id in values
        },
        provider_projection=provider,
        identity_bindings=_identity(),
    )
    assert projection["values"]["analysis.offspring_count.v1"] == [
        2**63 - 1
    ]
