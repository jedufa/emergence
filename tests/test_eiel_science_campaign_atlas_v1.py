from __future__ import annotations

from copy import deepcopy

import pytest

from emergence_contracts import ContentDigest, ScienceWorldBundleV1, canonical_sha256
from emergence_evidence.science_campaign_atlas_v1 import (
    SCIENCE_CAMPAIGN_ATLAS_AGGREGATE_SCHEMA_V1,
    SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1,
    SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1,
    ScienceCampaignAtlasError,
    aggregate_science_campaign_atlas_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
    COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
    COMPACT_WORLD_ATLAS_REDUCER_ID,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
    strategy_attribution_contract_v1,
)
from tests.test_eiel_science_compact_world_atlas_v1 import _payload, _world


def _bundle(worlds: tuple[object, ...]) -> ScienceWorldBundleV1:
    return ScienceWorldBundleV1(
        bundle_id="CAMPAIGN-BUNDLE",
        compact_source_specs=_payload(
            "eiel_science_compact_source_specs_v1", [{"source": "aggregate-test"}]
        ),
        compiler_component_sha256=ContentDigest("sha256", "4" * 64),
        worlds=worlds,
    )


def _subset_bundle(
    campaign_bundle: ScienceWorldBundleV1,
    worlds: tuple[object, ...],
) -> ScienceWorldBundleV1:
    return ScienceWorldBundleV1(
        bundle_id="PHYSICAL-BATCH",
        compact_source_specs=campaign_bundle.compact_source_specs,
        compiler_component_sha256=campaign_bundle.compiler_component_sha256,
        worlds=worlds,
        instrument_contract=campaign_bundle.instrument_contract,
        schema=campaign_bundle.schema,
        schema_version=campaign_bundle.schema_version,
    )


def _rehash(value: dict[str, object], field: str = "result_sha256") -> None:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    ).to_dict()


def _world_result(world: object, *, marker: int) -> dict[str, object]:
    body = {
        "world_id": world.world_id,
        "science": world.science.to_value(),
        "dimension_bindings": world.semantic_inputs.to_value()["dimension_bindings"],
        "semantic_world_sha256": world.semantic_world_sha256.to_dict(),
        "acquisition_sha256": world.acquisition_sha256.to_dict(),
        # The campaign combiner treats reducer-owned observations as opaque.
        # Their canonical digest, rather than duplicated science logic here,
        # authenticates the complete payload.
        "observations": {"marker": marker},
    }
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


def _batch(
    campaign_bundle: ScienceWorldBundleV1,
    *,
    request_sha256: ContentDigest,
    physical_batch_index: int,
    worlds: tuple[object, ...],
    strategy_attribution: bool = False,
) -> dict[str, object]:
    batch_bundle = _subset_bundle(campaign_bundle, worlds)
    rows = [
        _world_result(world, marker=physical_batch_index * 100 + index)
        for index, world in enumerate(worlds)
    ]
    atlas_body = {
        "schema_version": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "bundle_content_sha256": batch_bundle.bundle_content_sha256().to_dict(),
        "capture_index_sha256": canonical_sha256(
            {
                "physical_batch_index": physical_batch_index,
                "world_ids": [world.world_id for world in worlds],
            }
        ).hex,
        "logical_world_count": len(rows),
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "world_results": rows,
    }
    atlas = {
        **atlas_body,
        "result_sha256": canonical_sha256(atlas_body).to_dict(),
    }
    result = {
        "physical_batch_index": physical_batch_index,
        "request_sha256": request_sha256.to_dict(),
        "atlas_result": atlas,
    }
    if strategy_attribution:
        strategy_rows = []
        for row, world in zip(rows, worlds, strict=True):
            contract = strategy_attribution_contract_v1(world)
            assert contract is not None
            strategy_rows.append(
                {
                    field: row[field]
                    for field in (
                        "world_id",
                        "science",
                        "dimension_bindings",
                        "semantic_world_sha256",
                        "acquisition_sha256",
                    )
                }
                | {
                    "strategy_contract": {
                        "schema_version": contract["schema_version"],
                        "strategy_ids": list(contract["strategy_ids"]),
                        "identity_mode": contract["identity_mode"],
                        "lineage_mode": contract["lineage_mode"],
                        "denominator_fields": list(contract["denominator_fields"]),
                        "policy_mapping_count": len(
                            contract["policy_id_to_strategy_id"]
                        ),
                    }
                }
            )
        strategy_body = {
            "schema_version": STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
            "bundle_content_sha256": atlas["bundle_content_sha256"],
            "capture_index_sha256": atlas["capture_index_sha256"],
            "logical_world_count": len(strategy_rows),
            "physical_batch_count": 1,
            "physical_composition_used": False,
            "early_life_tick": 10_000,
            "pressure_pairing_factor": "pressure_regime",
            "world_results": strategy_rows,
            "classification_counts": {
                "A": 0,
                "B": 0,
                "C": 0,
                "D": 0,
                "E": 0,
                "UNRESOLVED": len(strategy_rows),
            },
        }
        result["strategy_attribution_result"] = {
            **strategy_body,
            "result_sha256": canonical_sha256(strategy_body).to_dict(),
        }
    return result


def _valid_inputs() -> tuple[
    ContentDigest,
    ScienceWorldBundleV1,
    list[dict[str, object]],
]:
    worlds = tuple(
        _world(f"campaign-world-{index:04d}", seed_offset=index) for index in range(4)
    )
    bundle = _bundle(worlds)
    request_sha256 = canonical_sha256({"request": "CAMPAIGN"})
    batches = [
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=0,
            worlds=(worlds[0], worlds[2]),
        ),
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=1,
            worlds=(worlds[1], worlds[3]),
        ),
    ]
    return request_sha256, bundle, batches


def test_aggregate_authenticates_batches_and_canonicalizes_semantic_world_order() -> (
    None
):
    request_sha256, bundle, batches = _valid_inputs()
    untouched = deepcopy(batches)

    aggregate = aggregate_science_campaign_atlas_v1(
        request_sha256=request_sha256.hex,
        world_bundle=bundle.to_dict(),
        batch_results=list(reversed(batches)),
    )

    assert batches == untouched
    assert aggregate["schema_version"] == SCIENCE_CAMPAIGN_ATLAS_AGGREGATE_SCHEMA_V1
    result = aggregate["result"]
    receipt = aggregate["verification_receipt"]
    assert result["schema_version"] == SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1
    assert (
        receipt["schema_version"]
        == SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1
    )
    assert result["request_sha256"] == request_sha256.to_dict()
    assert result["bundle_content_sha256"] == bundle.bundle_content_sha256().to_dict()
    assert result["logical_world_count"] == bundle.world_count == 4
    assert result["physical_batch_count"] == 2
    assert result["physical_composition_used"] is True
    assert [row["world_id"] for row in result["world_results"]] == [
        world.world_id for world in bundle.worlds
    ]
    assert result["source_batch_result_sha256s"] == [
        batches[index]["atlas_result"]["result_sha256"] for index in range(2)
    ]
    assert receipt["physical_batch_count"] == 2
    assert [row["physical_batch_index"] for row in receipt["batch_results"]] == [
        0,
        1,
    ]
    assert receipt["verification"] == {
        "batch_result_digests_authenticated": True,
        "world_result_digests_authenticated": True,
        "complete_disjoint_world_coverage": True,
        "semantic_world_order_canonical": True,
    }
    assert receipt["trajectory_effects_performed"] is False
    assert receipt["authority_effect"] == "none"
    assert (
        result["result_sha256"]
        == canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        ).to_dict()
    )
    assert (
        receipt["receipt_sha256"]
        == canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        ).to_dict()
    )

    canonical_input_order = aggregate_science_campaign_atlas_v1(
        request_sha256=request_sha256,
        world_bundle=bundle,
        batch_results=batches,
    )
    assert canonical_input_order == aggregate


def test_aggregate_reports_one_actual_physical_batch_without_composition() -> None:
    request_sha256, bundle, _ = _valid_inputs()
    only_batch = _batch(
        bundle,
        request_sha256=request_sha256,
        physical_batch_index=0,
        worlds=bundle.worlds,
    )

    aggregate = aggregate_science_campaign_atlas_v1(
        request_sha256=request_sha256,
        world_bundle=bundle,
        batch_results=[only_batch],
    )

    assert aggregate["result"]["physical_batch_count"] == 1
    assert aggregate["result"]["physical_composition_used"] is False
    assert aggregate["verification_receipt"]["physical_batch_count"] == 1


def test_aggregate_carries_and_authenticates_strategy_attribution() -> None:
    worlds = tuple(
        _world(
            f"strategy-campaign-world-{index:04d}",
            seed_offset=index,
            strategy_attribution=True,
        )
        for index in range(2)
    )
    bundle = _bundle(worlds)
    request_sha256 = canonical_sha256({"request": "STRATEGY-CAMPAIGN"})
    batch = _batch(
        bundle,
        request_sha256=request_sha256,
        physical_batch_index=0,
        worlds=worlds,
        strategy_attribution=True,
    )

    aggregate = aggregate_science_campaign_atlas_v1(
        request_sha256=request_sha256,
        world_bundle=bundle,
        batch_results=[batch],
    )

    strategy = batch["strategy_attribution_result"]
    assert aggregate["result"]["strategy_attribution_result"] == strategy
    assert aggregate["verification_receipt"]["verification"] == {
        "batch_result_digests_authenticated": True,
        "world_result_digests_authenticated": True,
        "complete_disjoint_world_coverage": True,
        "semantic_world_order_canonical": True,
        "strategy_attribution_digests_authenticated": True,
        "strategy_attribution_world_coverage_authenticated": True,
        "strategy_attribution_result_present": True,
    }
    assert aggregate["verification_receipt"]["batch_results"][0][
        "strategy_result_sha256"
    ] == strategy["result_sha256"]


def test_aggregate_rejects_missing_or_tampered_required_strategy_attribution() -> None:
    worlds = tuple(
        _world(
            f"required-strategy-world-{index:04d}",
            seed_offset=index,
            strategy_attribution=True,
        )
        for index in range(2)
    )
    bundle = _bundle(worlds)
    request_sha256 = canonical_sha256({"request": "REQUIRED-STRATEGY"})
    batch = _batch(
        bundle,
        request_sha256=request_sha256,
        physical_batch_index=0,
        worlds=worlds,
        strategy_attribution=True,
    )

    missing = deepcopy(batch)
    del missing["strategy_attribution_result"]
    with pytest.raises(ScienceCampaignAtlasError, match="envelope fields are not exact"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=[missing],
        )

    tampered = deepcopy(batch)
    tampered["strategy_attribution_result"]["classification_counts"]["A"] = 1
    with pytest.raises(
        ScienceCampaignAtlasError,
        match="strategy attribution result digest differs",
    ):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=[tampered],
        )


def test_aggregate_rejects_unauthenticated_or_inconsistent_identities() -> None:
    request_sha256, bundle, batches = _valid_inputs()

    tampered_atlas = deepcopy(batches)
    tampered_atlas[0]["atlas_result"]["logical_world_count"] = 99
    with pytest.raises(ScienceCampaignAtlasError, match="atlas result digest differs"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=tampered_atlas,
        )

    tampered_row = deepcopy(batches)
    tampered_row[0]["atlas_result"]["world_results"][0]["observations"] = {
        "marker": 999
    }
    _rehash(tampered_row[0]["atlas_result"])
    with pytest.raises(
        ScienceCampaignAtlasError, match="world result 0 digest differs"
    ):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=tampered_row,
        )

    wrong_request = deepcopy(batches)
    wrong_request[1]["request_sha256"] = "f" * 64
    with pytest.raises(ScienceCampaignAtlasError, match="request identity differs"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=wrong_request,
        )

    wrong_reducer = deepcopy(batches)
    wrong_reducer[1]["atlas_result"]["reducer_id"] = "science.foreign"
    _rehash(wrong_reducer[1]["atlas_result"])
    with pytest.raises(ScienceCampaignAtlasError, match="reducer identity differs"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=wrong_reducer,
        )

    wrong_batch_bundle = deepcopy(batches)
    wrong_batch_bundle[0]["atlas_result"]["bundle_content_sha256"] = ContentDigest(
        "sha256", "f" * 64
    ).to_dict()
    _rehash(wrong_batch_bundle[0]["atlas_result"])
    with pytest.raises(ScienceCampaignAtlasError, match="bundle identity differs"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=wrong_batch_bundle,
        )

    wrong_world_identity = deepcopy(batches)
    row = wrong_world_identity[0]["atlas_result"]["world_results"][0]
    row["semantic_world_sha256"] = ContentDigest("sha256", "f" * 64).to_dict()
    _rehash(row)
    _rehash(wrong_world_identity[0]["atlas_result"])
    with pytest.raises(ScienceCampaignAtlasError, match="world .* identity differs"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=wrong_world_identity,
        )


def test_aggregate_requires_complete_disjoint_nonforeign_world_coverage() -> None:
    request_sha256, bundle, _ = _valid_inputs()
    worlds = bundle.worlds

    duplicate = [
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=0,
            worlds=(worlds[0], worlds[1]),
        ),
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=1,
            worlds=(worlds[1], worlds[2], worlds[3]),
        ),
    ]
    with pytest.raises(ScienceCampaignAtlasError, match="world .* is duplicated"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=duplicate,
        )

    missing = [
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=0,
            worlds=(worlds[0], worlds[1]),
        ),
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=1,
            worlds=(worlds[2],),
        ),
    ]
    with pytest.raises(ScienceCampaignAtlasError, match="missing worlds"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=missing,
        )

    foreign_world = _world("foreign-world", seed_offset=99)
    foreign = [
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=0,
            worlds=(worlds[0], worlds[1]),
        ),
        _batch(
            bundle,
            request_sha256=request_sha256,
            physical_batch_index=1,
            worlds=(worlds[2], worlds[3], foreign_world),
        ),
    ]
    with pytest.raises(ScienceCampaignAtlasError, match="foreign world"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=foreign,
        )


def test_aggregate_requires_unique_contiguous_physical_batch_identity() -> None:
    request_sha256, bundle, batches = _valid_inputs()

    duplicate_index = deepcopy(batches)
    duplicate_index[1]["physical_batch_index"] = 0
    with pytest.raises(ScienceCampaignAtlasError, match="complete range 0..N-1"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=duplicate_index,
        )

    missing_index = deepcopy(batches)
    missing_index[1]["physical_batch_index"] = 2
    with pytest.raises(ScienceCampaignAtlasError, match="complete range 0..N-1"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=missing_index,
        )

    duplicate_result = deepcopy(batches)
    duplicate_result[1]["atlas_result"] = deepcopy(duplicate_result[0]["atlas_result"])
    with pytest.raises(
        ScienceCampaignAtlasError, match="batch result digest is duplicated"
    ):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=duplicate_result,
        )


def test_aggregate_rejects_empty_or_nonexact_batch_inputs() -> None:
    request_sha256, bundle, batches = _valid_inputs()
    with pytest.raises(ScienceCampaignAtlasError, match="cannot be empty"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=[],
        )

    extra_field = deepcopy(batches)
    extra_field[0]["ignored"] = True
    with pytest.raises(ScienceCampaignAtlasError, match="fields are not exact"):
        aggregate_science_campaign_atlas_v1(
            request_sha256=request_sha256,
            world_bundle=bundle,
            batch_results=extra_field,
        )
