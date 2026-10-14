"""Pure campaign aggregation for authenticated compact-world-atlas batches.

The compact world-atlas reducer owns one physical batch at a time.  This
module composes those already-reduced results without reinterpreting their
scientific payload.  Every payload copied into the campaign result is first
authenticated by its canonical digest and checked against the bound semantic
world bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

from emergence_contracts import ContentDigest, ScienceWorldBundleV1, canonical_sha256

from .science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
    COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
    COMPACT_WORLD_ATLAS_REDUCER_ID,
    CompactWorldAtlasReducerError,
    compact_world_atlas_reducer_descriptor_v1,
    validate_hereditary_birth_record_artifact_v1,
)
from .science_strategy_attribution_v1 import (
    STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
    StrategyAttributionError,
    strategy_attribution_contract_v1,
    strategy_attribution_required_for_bundle_v1,
    validate_genotype_reproductive_attribution_v1,
)


SCIENCE_CAMPAIGN_ATLAS_AGGREGATE_SCHEMA_V1: Final[str] = (
    "eiel_science_campaign_atlas_aggregate_v1"
)
SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1: Final[str] = (
    "eiel_science_campaign_atlas_result_v1"
)
SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_science_campaign_atlas_verification_receipt_v1"
)

_BATCH_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {"physical_batch_index", "request_sha256", "atlas_result"}
)
_STRATEGY_BATCH_ENVELOPE_FIELD: Final[str] = "strategy_attribution_result"
_STRATEGY_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "bundle_content_sha256",
        "capture_index_sha256",
        "logical_world_count",
        "physical_batch_count",
        "physical_composition_used",
        "early_life_tick",
        "pressure_pairing_factor",
        "world_results",
        "classification_counts",
        "result_sha256",
    }
)
_ATLAS_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "reducer_id",
        "reducer_contract_version",
        "bundle_content_sha256",
        "capture_index_sha256",
        "logical_world_count",
        "physical_batch_count",
        "physical_composition_used",
        "world_results",
        "result_sha256",
    }
)
_WORLD_RESULT_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "world_id",
        "science",
        "dimension_bindings",
        "semantic_world_sha256",
        "acquisition_sha256",
        "result_sha256",
    }
)


class ScienceCampaignAtlasError(ValueError):
    """An atlas input is unauthenticated or violates campaign ownership."""


@dataclass(frozen=True, slots=True)
class _VerifiedBatchV1:
    physical_batch_index: int
    result_sha256: ContentDigest
    bundle_content_sha256: ContentDigest
    capture_index_sha256: ContentDigest
    world_ids: tuple[str, ...]
    world_results: tuple[dict[str, Any], ...]
    strategy_result: dict[str, Any] | None


def _digest(value: object, *, field: str) -> ContentDigest:
    try:
        if isinstance(value, ContentDigest):
            return value
        if isinstance(value, str):
            return ContentDigest("sha256", value)
        if isinstance(value, Mapping) and set(value) == {"algorithm", "hex"}:
            return ContentDigest(value["algorithm"], value["hex"])
    except (TypeError, ValueError) as error:
        raise ScienceCampaignAtlasError(
            f"{field} is not a valid SHA-256 digest"
        ) from error
    raise ScienceCampaignAtlasError(f"{field} is not a valid SHA-256 digest")


def _world_bundle(
    value: ScienceWorldBundleV1 | Mapping[str, Any],
) -> ScienceWorldBundleV1:
    if isinstance(value, ScienceWorldBundleV1):
        return value
    if not isinstance(value, Mapping):
        raise ScienceCampaignAtlasError("world_bundle is invalid")
    try:
        return ScienceWorldBundleV1.from_dict(dict(value))
    except (TypeError, ValueError) as error:
        raise ScienceCampaignAtlasError("world_bundle identity is invalid") from error


def _authenticated_mapping(
    value: object,
    *,
    digest_field: str,
    label: str,
) -> tuple[dict[str, Any], ContentDigest]:
    if not isinstance(value, Mapping):
        raise ScienceCampaignAtlasError(f"{label} must be an object")
    snapshot = deepcopy(dict(value))
    if digest_field not in snapshot:
        raise ScienceCampaignAtlasError(f"{label} lacks {digest_field}")
    claimed = _digest(snapshot[digest_field], field=f"{label}.{digest_field}")
    body = {key: item for key, item in snapshot.items() if key != digest_field}
    try:
        actual = canonical_sha256(body)
    except (TypeError, ValueError) as error:
        raise ScienceCampaignAtlasError(f"{label} is outside canonical JSON") from error
    if actual != claimed:
        raise ScienceCampaignAtlasError(f"{label} digest differs")
    return snapshot, claimed


def _validate_world_reducer_bindings(bundle: ScienceWorldBundleV1) -> None:
    expected_contract_version = str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION)
    for world in bundle.worlds:
        reducers = world.reducers.to_value()
        matches = [
            row
            for row in reducers
            if isinstance(row, Mapping)
            and row.get("reducer_id") == COMPACT_WORLD_ATLAS_REDUCER_ID
        ]
        if len(matches) != 1:
            raise ScienceCampaignAtlasError(
                f"world {world.world_id} has no unique compact atlas reducer binding"
            )
        binding = matches[0]
        if (
            binding.get("contract_version") != expected_contract_version
            or binding.get("output_contract_id")
            != COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID
        ):
            raise ScienceCampaignAtlasError(
                f"world {world.world_id} compact atlas reducer identity differs"
            )


def _strategy_attribution_required_v1(bundle: ScienceWorldBundleV1) -> bool:
    """Return whether every logical world requires a strategy result."""

    try:
        return strategy_attribution_required_for_bundle_v1(bundle)
    except StrategyAttributionError as error:
        raise ScienceCampaignAtlasError(
            f"strategy attribution bundle contract is invalid: {error}"
        ) from error


def _strategy_contract_projection_v1(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    projection = {
        "schema_version": contract.get("schema_version"),
        "strategy_ids": list(contract.get("strategy_ids", [])),
        "identity_mode": contract.get("identity_mode"),
        "lineage_mode": contract.get("lineage_mode"),
        "denominator_fields": list(contract.get("denominator_fields", [])),
        "policy_mapping_count": len(contract.get("policy_id_to_strategy_id", {})),
    }
    if contract.get("direct_genotype_attribution") is True:
        projection.update(
            {
                "direct_genotype_attribution": True,
                "genotype_identity_mode": contract.get("genotype_identity_mode"),
                "genotype_denominator_fields": list(
                    contract.get("genotype_denominator_fields", [])
                ),
            }
        )
    return projection


def _expected_batch_bundle_sha256(
    campaign_bundle: ScienceWorldBundleV1,
    *,
    world_ids: tuple[str, ...],
) -> ContentDigest:
    selected = set(world_ids)
    subset_worlds = tuple(
        world for world in campaign_bundle.worlds if world.world_id in selected
    )
    # bundle_id is deliberately not part of bundle_content_sha256.  Rebuilding
    # the subset therefore authenticates the exact semantic batch whether the
    # executor used the campaign bundle directly or a batch-local bundle ID.
    return ScienceWorldBundleV1(
        bundle_id=campaign_bundle.bundle_id,
        compact_source_specs=campaign_bundle.compact_source_specs,
        compiler_component_sha256=campaign_bundle.compiler_component_sha256,
        worlds=subset_worlds,
        instrument_contract=campaign_bundle.instrument_contract,
        schema=campaign_bundle.schema,
        schema_version=campaign_bundle.schema_version,
    ).bundle_content_sha256()


def _verify_batch_v1(
    envelope: Mapping[str, Any],
    *,
    request_sha256: ContentDigest,
    campaign_bundle: ScienceWorldBundleV1,
    worlds_by_id: Mapping[str, Any],
    strategy_required: bool,
) -> _VerifiedBatchV1:
    expected_envelope_fields = set(_BATCH_ENVELOPE_FIELDS)
    if strategy_required:
        expected_envelope_fields.add(_STRATEGY_BATCH_ENVELOPE_FIELD)
    if set(envelope) != expected_envelope_fields:
        raise ScienceCampaignAtlasError("batch result envelope fields are not exact")
    physical_batch_index = envelope["physical_batch_index"]
    if type(physical_batch_index) is not int or physical_batch_index < 0:
        raise ScienceCampaignAtlasError("physical_batch_index is invalid")
    if (
        _digest(
            envelope["request_sha256"],
            field=f"batch[{physical_batch_index}].request_sha256",
        )
        != request_sha256
    ):
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} request identity differs"
        )

    atlas, atlas_sha256 = _authenticated_mapping(
        envelope["atlas_result"],
        digest_field="result_sha256",
        label=f"batch {physical_batch_index} atlas result",
    )
    if set(atlas) != _ATLAS_RESULT_FIELDS:
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} atlas result fields are not exact"
        )
    if (
        atlas["schema_version"] != COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID
        or atlas["reducer_id"] != COMPACT_WORLD_ATLAS_REDUCER_ID
        or atlas["reducer_contract_version"]
        != str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION)
    ):
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} reducer identity differs"
        )
    if (
        type(atlas["physical_batch_count"]) is not int
        or atlas["physical_batch_count"] != 1
        or atlas["physical_composition_used"] is not False
    ):
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} is not one canonical physical batch"
        )
    raw_world_results = atlas["world_results"]
    if not isinstance(raw_world_results, list) or not raw_world_results:
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} world_results are invalid"
        )
    if type(atlas["logical_world_count"]) is not int or atlas[
        "logical_world_count"
    ] != len(raw_world_results):
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} logical world count differs"
        )

    verified_rows: list[dict[str, Any]] = []
    world_ids: list[str] = []
    validated_birth_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_row in enumerate(raw_world_results):
        label = f"batch {physical_batch_index} world result {row_index}"
        row, _ = _authenticated_mapping(
            raw_row,
            digest_field="result_sha256",
            label=label,
        )
        if not _WORLD_RESULT_IDENTITY_FIELDS.issubset(row):
            raise ScienceCampaignAtlasError(f"{label} identity fields are incomplete")
        world_id = row["world_id"]
        if not isinstance(world_id, str) or not world_id:
            raise ScienceCampaignAtlasError(f"{label} world_id is invalid")
        if world_id in world_ids:
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} duplicates world {world_id}"
            )
        world = worlds_by_id.get(world_id)
        if world is None:
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} contains foreign world {world_id}"
            )
        if (
            row["semantic_world_sha256"] != world.semantic_world_sha256.to_dict()
            or row["acquisition_sha256"] != world.acquisition_sha256.to_dict()
            or row["science"] != world.science.to_value()
            or row["dimension_bindings"]
            != world.semantic_inputs.to_value()["dimension_bindings"]
        ):
            raise ScienceCampaignAtlasError(f"world {world_id} identity differs")
        hereditary = row.get("hereditary_attribution")
        if isinstance(hereditary, Mapping):
            birth_artifact = hereditary.get("birth_parent_child_records")
            if isinstance(birth_artifact, Mapping):
                path = birth_artifact.get("path")
                file_sha256 = birth_artifact.get("file_sha256")
                if not isinstance(path, str) or not isinstance(file_sha256, str):
                    raise ScienceCampaignAtlasError(
                        f"{label} hereditary birth artifact identity is incomplete"
                    )
                artifact_key = (path, file_sha256)
                if artifact_key not in validated_birth_artifacts:
                    try:
                        validated_birth_artifacts[artifact_key] = (
                            validate_hereditary_birth_record_artifact_v1(
                                birth_artifact,
                                expected_world_id=world_id,
                            )
                        )
                    except (
                        CompactWorldAtlasReducerError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as error:
                        raise ScienceCampaignAtlasError(
                            f"{label} hereditary birth artifact is invalid"
                        ) from error
                elif birth_artifact.get("world_id") != world_id:
                    raise ScienceCampaignAtlasError(
                        f"{label} hereditary birth artifact world differs"
                    )
        world_ids.append(world_id)
        verified_rows.append(row)

    canonical_world_ids = tuple(
        world.world_id
        for world in campaign_bundle.worlds
        if world.world_id in world_ids
    )
    batch_bundle_sha256 = _digest(
        atlas["bundle_content_sha256"],
        field=f"batch[{physical_batch_index}].bundle_content_sha256",
    )
    expected_batch_bundle_sha256 = _expected_batch_bundle_sha256(
        campaign_bundle,
        world_ids=canonical_world_ids,
    )
    if batch_bundle_sha256 != expected_batch_bundle_sha256:
        raise ScienceCampaignAtlasError(
            f"batch {physical_batch_index} bundle identity differs"
        )
    capture_index_sha256 = _digest(
        atlas["capture_index_sha256"],
        field=f"batch[{physical_batch_index}].capture_index_sha256",
    )
    strategy_result: dict[str, Any] | None = None
    if strategy_required:
        strategy_result, _ = _authenticated_mapping(
            envelope[_STRATEGY_BATCH_ENVELOPE_FIELD],
            digest_field="result_sha256",
            label=f"batch {physical_batch_index} strategy attribution result",
        )
        if set(strategy_result) != _STRATEGY_RESULT_FIELDS:
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} strategy attribution result fields are not exact"
            )
        if (
            strategy_result["schema_version"] != STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1
            or type(strategy_result["logical_world_count"]) is not int
            or strategy_result["logical_world_count"] != len(verified_rows)
            or strategy_result["physical_batch_count"] != 1
            or strategy_result["physical_composition_used"] is not False
            or type(strategy_result["early_life_tick"]) is not int
            or strategy_result["early_life_tick"] < 1
            or not isinstance(strategy_result["pressure_pairing_factor"], str)
            or not strategy_result["pressure_pairing_factor"]
            or not isinstance(strategy_result["classification_counts"], Mapping)
        ):
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} strategy attribution result metadata differs"
            )
        if (
            _digest(
                strategy_result["bundle_content_sha256"],
                field=f"batch[{physical_batch_index}].strategy_attribution_result.bundle_content_sha256",
            )
            != batch_bundle_sha256
            or _digest(
                strategy_result["capture_index_sha256"],
                field=f"batch[{physical_batch_index}].strategy_attribution_result.capture_index_sha256",
            )
            != capture_index_sha256
        ):
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} strategy attribution ownership differs"
            )
        strategy_rows = strategy_result["world_results"]
        if (
            not isinstance(strategy_rows, list)
            or len(strategy_rows) != len(canonical_world_ids)
            or [
                row.get("world_id")
                for row in strategy_rows
                if isinstance(row, Mapping)
            ]
            != list(canonical_world_ids)
        ):
            raise ScienceCampaignAtlasError(
                f"batch {physical_batch_index} strategy attribution world order differs"
            )
        atlas_rows_by_id = {row["world_id"]: row for row in verified_rows}
        for row_index, strategy_row in enumerate(strategy_rows):
            if not isinstance(strategy_row, Mapping):
                raise ScienceCampaignAtlasError(
                    f"batch {physical_batch_index} strategy world result {row_index} is invalid"
                )
            world_id = strategy_row["world_id"]
            atlas_row = atlas_rows_by_id[world_id]
            for field in _WORLD_RESULT_IDENTITY_FIELDS - {"result_sha256"}:
                if strategy_row.get(field) != atlas_row.get(field):
                    raise ScienceCampaignAtlasError(
                        f"batch {physical_batch_index} strategy world {world_id} identity differs"
                    )
            try:
                contract = strategy_attribution_contract_v1(worlds_by_id[world_id])
            except StrategyAttributionError as error:
                raise ScienceCampaignAtlasError(
                    f"batch {physical_batch_index} strategy world {world_id} contract is invalid"
                ) from error
            if contract is None or strategy_row.get("strategy_contract") != _strategy_contract_projection_v1(contract):
                raise ScienceCampaignAtlasError(
                    f"batch {physical_batch_index} strategy world {world_id} contract differs"
                )
            if contract.get("direct_genotype_attribution") is True:
                genotype_result = strategy_row.get(
                    "genotype_reproductive_attribution"
                )
                try:
                    validate_genotype_reproductive_attribution_v1(genotype_result)
                except StrategyAttributionError as error:
                    raise ScienceCampaignAtlasError(
                        f"batch {physical_batch_index} strategy world {world_id} genotype attribution differs"
                    ) from error
    rows_by_id = {row["world_id"]: row for row in verified_rows}
    return _VerifiedBatchV1(
        physical_batch_index=physical_batch_index,
        result_sha256=atlas_sha256,
        bundle_content_sha256=batch_bundle_sha256,
        capture_index_sha256=capture_index_sha256,
        world_ids=canonical_world_ids,
        world_results=tuple(rows_by_id[world_id] for world_id in canonical_world_ids),
        strategy_result=strategy_result,
    )


def aggregate_science_campaign_atlas_v1(
    *,
    request_sha256: ContentDigest | str,
    world_bundle: ScienceWorldBundleV1 | Mapping[str, Any],
    batch_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate and compose batch atlases in bound semantic-world order.

    Each item in ``batch_results`` is the exact projection
    ``{physical_batch_index, request_sha256, atlas_result}``.  ``atlas_result``
    is the unmodified output of ``science.compact.world_atlas.v1``.  The
    function performs no I/O and grants no scientific or qualification
    authority.
    """

    request_digest = _digest(request_sha256, field="request_sha256")
    bundle = _world_bundle(world_bundle)
    _validate_world_reducer_bindings(bundle)
    strategy_required = _strategy_attribution_required_v1(bundle)
    if isinstance(batch_results, (str, bytes)) or not isinstance(
        batch_results, Sequence
    ):
        raise ScienceCampaignAtlasError("batch_results must be a sequence")
    if not batch_results:
        raise ScienceCampaignAtlasError("batch_results cannot be empty")

    worlds_by_id = {world.world_id: world for world in bundle.worlds}
    verified: list[_VerifiedBatchV1] = []
    for input_index, raw_batch in enumerate(batch_results):
        if not isinstance(raw_batch, Mapping):
            raise ScienceCampaignAtlasError(
                f"batch_results[{input_index}] must be an object"
            )
        verified.append(
            _verify_batch_v1(
                raw_batch,
                request_sha256=request_digest,
                campaign_bundle=bundle,
                worlds_by_id=worlds_by_id,
                strategy_required=strategy_required,
            )
        )

    verified.sort(key=lambda batch: batch.physical_batch_index)
    actual_indices = [batch.physical_batch_index for batch in verified]
    expected_indices = list(range(len(verified)))
    if actual_indices != expected_indices:
        raise ScienceCampaignAtlasError(
            "physical_batch_index values must be the complete range 0..N-1"
        )
    result_digests = [batch.result_sha256 for batch in verified]
    if len({digest.hex for digest in result_digests}) != len(result_digests):
        raise ScienceCampaignAtlasError("batch result digest is duplicated")
    capture_digests = [batch.capture_index_sha256 for batch in verified]
    if len({digest.hex for digest in capture_digests}) != len(capture_digests):
        raise ScienceCampaignAtlasError("batch capture index digest is duplicated")

    owner_by_world: dict[str, int] = {}
    rows_by_world: dict[str, dict[str, Any]] = {}
    for batch in verified:
        for world_id, row in zip(batch.world_ids, batch.world_results, strict=True):
            prior = owner_by_world.get(world_id)
            if prior is not None:
                raise ScienceCampaignAtlasError(
                    f"world {world_id} is duplicated by batches {prior} and "
                    f"{batch.physical_batch_index}"
                )
            owner_by_world[world_id] = batch.physical_batch_index
            rows_by_world[world_id] = row

    expected_world_ids = tuple(world.world_id for world in bundle.worlds)
    missing = [
        world_id for world_id in expected_world_ids if world_id not in rows_by_world
    ]
    if missing:
        raise ScienceCampaignAtlasError(
            "campaign atlas is missing worlds: " + ", ".join(missing)
        )
    if set(rows_by_world) != set(expected_world_ids):
        # Foreign worlds are rejected per batch.  Keep the equality assertion
        # fail-closed in case a future input normalizer changes that boundary.
        raise ScienceCampaignAtlasError("campaign atlas world coverage differs")

    canonical_rows = [rows_by_world[world_id] for world_id in expected_world_ids]
    descriptor = compact_world_atlas_reducer_descriptor_v1()
    descriptor_sha256 = descriptor.descriptor_sha256()
    bundle_sha256 = bundle.bundle_content_sha256()
    physical_batch_count = len(verified)
    batch_result_sha256s = [digest.to_dict() for digest in result_digests]
    result_body: dict[str, Any] = {
        "schema_version": SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1,
        "request_sha256": request_digest.to_dict(),
        "bundle_content_sha256": bundle_sha256.to_dict(),
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "reducer_descriptor_sha256": descriptor_sha256.to_dict(),
        "output_contract_id": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "logical_world_count": len(canonical_rows),
        "physical_batch_count": physical_batch_count,
        "physical_composition_used": physical_batch_count > 1,
        "source_batch_result_sha256s": batch_result_sha256s,
        "world_results": canonical_rows,
    }
    if strategy_required:
        strategy_results = [batch.strategy_result for batch in verified]
        if any(result is None for result in strategy_results):
            raise ScienceCampaignAtlasError(
                "strategy attribution is required for every verified batch"
            )
        if physical_batch_count == 1:
            result_body["strategy_attribution_result"] = dict(strategy_results[0])
        else:
            result_body["strategy_attribution_results"] = [
                dict(result) for result in strategy_results
            ]
    aggregate_result_sha256 = canonical_sha256(result_body)
    result = {
        **result_body,
        "result_sha256": aggregate_result_sha256.to_dict(),
    }

    batch_receipts = [
        {
            "physical_batch_index": batch.physical_batch_index,
            "batch_result_sha256": batch.result_sha256.to_dict(),
            "batch_bundle_content_sha256": batch.bundle_content_sha256.to_dict(),
            "capture_index_sha256": batch.capture_index_sha256.to_dict(),
            "logical_world_count": len(batch.world_ids),
            "world_ids": list(batch.world_ids),
            **(
                {
                    "strategy_result_sha256": batch.strategy_result["result_sha256"]
                }
                if batch.strategy_result is not None
                else {}
            ),
        }
        for batch in verified
    ]
    semantic_world_order_sha256 = canonical_sha256(
        [
            {
                "world_id": world.world_id,
                "semantic_world_sha256": world.semantic_world_sha256.to_dict(),
                "acquisition_sha256": world.acquisition_sha256.to_dict(),
            }
            for world in bundle.worlds
        ]
    )
    receipt_body: dict[str, Any] = {
        "schema_version": SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1,
        "request_sha256": request_digest.to_dict(),
        "bundle_content_sha256": bundle_sha256.to_dict(),
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "reducer_descriptor_sha256": descriptor_sha256.to_dict(),
        "output_contract_id": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "aggregate_result_sha256": aggregate_result_sha256.to_dict(),
        "semantic_world_order_sha256": semantic_world_order_sha256.to_dict(),
        "logical_world_count": len(canonical_rows),
        "physical_batch_count": physical_batch_count,
        "batch_results": batch_receipts,
        "verification": {
            "batch_result_digests_authenticated": True,
            "world_result_digests_authenticated": True,
            "complete_disjoint_world_coverage": True,
            "semantic_world_order_canonical": True,
        },
        "trajectory_effects_performed": False,
        "authority_effect": "none",
    }
    if strategy_required:
        receipt_body["verification"].update(
            {
                "strategy_attribution_digests_authenticated": True,
                "strategy_attribution_world_coverage_authenticated": True,
                "strategy_attribution_result_present": True,
            }
        )
    verification_receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body).to_dict(),
    }
    return {
        "schema_version": SCIENCE_CAMPAIGN_ATLAS_AGGREGATE_SCHEMA_V1,
        "result": result,
        "verification_receipt": verification_receipt,
    }


__all__ = (
    "SCIENCE_CAMPAIGN_ATLAS_AGGREGATE_SCHEMA_V1",
    "SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1",
    "SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1",
    "ScienceCampaignAtlasError",
    "aggregate_science_campaign_atlas_v1",
)
