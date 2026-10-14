from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    ScienceWorldRecordV1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
    ScienceWorldBundleV1,
    canonical_sha256,
)
from eiel_composition.science_instrument import (
    ScienceCRCASemanticBindingV1,
    compile_science_campaign_v1,
)
from eiel_composition.science_instrument_demo_v1 import (
    _control_descriptors,
    _instrument_manifest,
    _profile_descriptor,
    _reducer_descriptor,
    _request,
    _resource_envelope,
    _route_and_empty_envelope,
)
from eiel_composition.science_request_v2 import (
    normalize_rich_production_science_request_v2,
)
from eiel_composition.science_world_bundle_v1 import (
    ScienceWorldBundleError,
    admit_science_world_bundle_v1,
    demultiplex_world_results_v1,
    load_science_world_bundle_v1,
    materialize_science_world_bundle_v1,
    pack_admitted_science_worlds_v1,
    write_science_world_bundle_v1,
)


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _compiled_families():
    controls = _control_descriptors()
    reducer = _reducer_descriptor()
    profile = _profile_descriptor()
    manifest = _instrument_manifest(controls, reducer, profile)
    route, _ = _route_and_empty_envelope(
        manifest, _resource_envelope(), route_id="world-bundle-test-route"
    )
    crca = ScienceCRCASemanticBindingV1(
        semantic_run_digest=canonical_sha256({"run": "world-bundle-test"}),
        projection_sha256=canonical_sha256({"projection": "world-bundle-test"}),
        absolute_start_next_tick=0,
        absolute_target_next_tick=20,
        trajectory_checkpoint_ticks=(0, 10, 20),
    )
    compiled = []
    for family in ("MUTSUP", "G2B", "RETAIN"):
        request = deepcopy(_request(manifest, reducer))
        campaign_document = request["science_campaign"]
        campaign_document["campaign_id"] = f"bundle-{family.lower()}"
        ownership = campaign_document["question"]["science_ownership"]
        ownership["test_id"] = family
        ownership["study_id"] = f"{family}-BUNDLE-DEMO"
        campaign = normalize_rich_production_science_request_v2(request).campaign
        compilation = compile_science_campaign_v1(
            campaign,
            instrument_manifest=manifest,
            route_binding=route,
            control_descriptors=controls,
            reducer_descriptors=(reducer,),
            crca_semantic_binding=crca,
            physical_execution_core=_payload(
                str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
                {"route_id": route.route_id, "logical_world_partitions": [list(range(12))]},
            ),
            compiler_component_sha256=canonical_sha256(
                {"component": "compile_science_campaign_v1", "version": 1}
            ),
        )
        compiled.append((campaign, compilation))
    return tuple(compiled)


def _admit(bundle):
    return admit_science_world_bundle_v1(
        bundle,
        supported_observations=(
            "birth", "death", "genome_identity", "lineage_ancestry",
            "birth_count", "population_count", "causal_lifecycle",
        ),
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=5000,
    )


def test_compact_mixed_studies_materialize_to_explicit_canonical_worlds() -> None:
    bundle = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-MIXED-TEST"
    )
    assert bundle.schema == "EIEL_SCIENCE_WORLD_BUNDLE_V1"
    assert bundle.world_count == 36
    assert {world.science.to_value()["test_id"] for world in bundle.worlds} == {
        "MUTSUP", "G2B", "RETAIN"
    }
    assert all(world.semantic_inputs.to_value()["seed_bindings"] for world in bundle.worlds)
    assert all(world.acquisition.to_value()["event_observations"] for world in bundle.worlds)
    assert all(world.reducers.to_value() for world in bundle.worlds)
    assert len({world.acquisition_sha256.hex for world in bundle.worlds}) == bundle.world_count
    restored = ScienceWorldBundleV1.from_dict(bundle.to_dict())
    assert restored.bundle_content_sha256() == bundle.bundle_content_sha256()
    schema = json.loads(
        Path("_spec/science_instrument/v1/science_world_bundle.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(bundle.to_dict())


def test_materialized_bundle_file_is_canonical_immutable_and_content_addressed(
    tmp_path: Path,
) -> None:
    bundle = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-FILE-TEST"
    )
    target = tmp_path / "bundle.json"
    receipt = write_science_world_bundle_v1(bundle, target)
    assert receipt["schema_version"] == "eiel_science_world_bundle_file_v1"
    assert receipt["immutable"] is True
    assert receipt["world_count"] == bundle.world_count
    assert target.stat().st_mode & 0o222 == 0
    assert load_science_world_bundle_v1(target).bundle_content_sha256() == (
        bundle.bundle_content_sha256()
    )
    with pytest.raises(ScienceWorldBundleError, match="fresh regular path"):
        write_science_world_bundle_v1(bundle, target)

    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ScienceWorldBundleError, match="canonical JSON"):
        load_science_world_bundle_v1(tampered)


def test_world_semantics_exclude_physical_placement_and_support_1000_plus() -> None:
    bundle = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-PACKING-TEST"
    )
    admission = _admit(bundle)
    assert admission.outcome is ScienceAdmissionOutcomeV1.ADMITTED
    one_carrier = pack_admitted_science_worlds_v1(
        bundle, admission, carrier_id="future-b1088", carrier_capacity=1088
    )
    partitioned = pack_admitted_science_worlds_v1(
        bundle,
        admission,
        carrier_id="b136",
        carrier_capacity=13,
        order=reversed(admission.admitted_world_ids),
    )
    assert len(one_carrier.batches) == 1
    assert len(partitioned.batches) == 3
    semantic_before = {
        world.world_id: world.semantic_world_sha256 for world in bundle.worlds
    }
    assert semantic_before == {
        world.world_id: world.semantic_world_sha256 for world in bundle.worlds
    }
    assert one_carrier.packing_sha256() != partitioned.packing_sha256()


def test_admission_fails_before_execution_for_missing_observation() -> None:
    bundle = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-REJECTION-TEST"
    )
    admission = admit_science_world_bundle_v1(
        bundle,
        supported_observations=("birth",),
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=5000,
    )
    assert admission.outcome is ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY
    assert admission.admitted_world_ids == ()
    assert admission.findings[0].world_id.startswith("world-")
    with pytest.raises(ScienceWorldBundleError):
        pack_admitted_science_worlds_v1(
            bundle, admission, carrier_id="b136", carrier_capacity=136
        )


def test_mixed_results_demultiplex_without_physical_batch_fields() -> None:
    bundle = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-DEMUX-TEST"
    )
    rows = [
        {"world_id": world.world_id, "result": "technical-only"}
        for world in bundle.worlds
    ]
    result = demultiplex_world_results_v1(bundle, reversed(rows))
    assert len(result["groups"]) == bundle.world_count
    assert all("physical" not in group["group_key"] for group in result["groups"])
    assert {group["group_key"].split("/")[1] for group in result["groups"]} == {
        "MUTSUP", "G2B", "RETAIN"
    }


def _strategy_bundle(*, omit: str | None = None) -> ScienceWorldBundleV1:
    """Add the generic opt-in strategy contract to a fixture bundle."""

    base = materialize_science_world_bundle_v1(
        _compiled_families(), bundle_id="SCIENCE-BUNDLE-STRATEGY-CONTRACT-TEST"
    )
    worlds: list[ScienceWorldRecordV1] = []
    for world in base.worlds:
        rows = [dict(row) for row in world.reducers.to_value()]
        row = rows[0]
        row["configuration"] = {
            "strategy_attribution": {
                "schema_version": SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
                "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
                "identity_mode": "declared_policy_mapping_v1",
                "lineage_mode": "parent_ancestry_v1",
                "strategy_ids": ["alpha", "beta"],
                "policy_id_to_strategy_id": {
                    "sha256:" + "a" * 64: "alpha",
                    "sha256:" + "b" * 64: "beta",
                },
                "denominator_fields": list(
                    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
                ),
            }
        }
        capabilities = set(row["required_observations"])
        capabilities.update(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
        if omit is not None:
            capabilities.discard(omit)
        row["required_observations"] = sorted(capabilities)
        semantic = world.semantic_inputs.to_value()
        science = world.science.to_value()
        semantic_sha = canonical_sha256(
            {
                "schema_version": "eiel_science_world_record_v1",
                "science": science,
                "semantic_inputs": semantic,
            }
        )
        acquisition = world.acquisition.to_value()
        acquisition_sha = canonical_sha256(
            {
                "schema_version": "eiel_science_world_record_v1",
                "semantic_world_sha256": semantic_sha.to_dict(),
                "acquisition": acquisition,
                "reducers": rows,
            }
        )
        worlds.append(
            ScienceWorldRecordV1(
                world_id=world.world_id,
                science=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_science_ownership_v1"), science
                ),
                semantic_inputs=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_science_resolved_inputs_v1"), semantic
                ),
                acquisition=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_science_acquisition_contract_v1"), acquisition
                ),
                reducers=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_science_world_reducers_v1"), rows
                ),
                semantic_world_sha256=semantic_sha,
                acquisition_sha256=acquisition_sha,
            )
        )
    return ScienceWorldBundleV1(
        bundle_id=base.bundle_id,
        compact_source_specs=base.compact_source_specs,
        compiler_component_sha256=base.compiler_component_sha256,
        worlds=tuple(worlds),
    )


def test_bundle_admission_enforces_strategy_contract_per_world() -> None:
    supported = {
        "birth",
        "death",
        "genome_identity",
        "lineage_ancestry",
        "birth_count",
        "population_count",
        "causal_lifecycle",
        *SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    }
    admitted = admit_science_world_bundle_v1(
        _strategy_bundle(),
        supported_observations=supported,
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=5000,
    )
    assert admitted.outcome is ScienceAdmissionOutcomeV1.ADMITTED
    rejected = admit_science_world_bundle_v1(
        _strategy_bundle(omit="strategy.parent_attribution"),
        supported_observations=supported,
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=5000,
    )
    assert rejected.outcome is ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY
    assert any(
        "strategy.parent_attribution" in finding.detail
        for finding in rejected.findings
    )
