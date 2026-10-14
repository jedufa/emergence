"""Strict route-neutral normalization for ``ProductionScienceRequestV2``.

The request may carry one inline campaign or one immutable content-addressed
reference.  Resolver location is provenance only: both forms normalize to the
same request payload and request identity.  This module performs no route
selection, QRR resolution, build, filesystem write, or execution effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Protocol

import yaml

from emergence_contracts import (
    ContentDigest,
    ContractError,
    ScienceCampaignV1,
    canonical_sha256,
)
from engine.campaigns.baseline_v1 import (
    BaselineDeclarationV1,
    BaselineError,
    validate_baseline_purpose_v1,
)

PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION = "eiel_production_science_request_v2"
SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION = "eiel_science_campaign_ref_v1"
SCIENCE_CAMPAIGN_RESOLUTION_RECEIPT_SCHEMA_VERSION = (
    "eiel_science_campaign_resolution_receipt_v1"
)

_COMMON_KEYS = frozenset({"schema_version", "request_id", "baseline", "request_sha256"})
_RICH_DISCRIMINANTS = frozenset({"science_campaign", "science_campaign_ref"})
_PATH_FREE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class ScienceRequestV2Error(ValueError):
    """A rich ProductionScienceRequestV2 cannot be normalized exactly."""


class ScienceCampaignResolverV1(Protocol):
    """Instrument-bound, read-only resolver for immutable campaign bytes."""

    def resolve(self, immutable_object_id: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class NormalizedRichScienceRequestV2:
    normalized_request: dict[str, Any]
    campaign: ScienceCampaignV1
    content_resolution_receipt: dict[str, Any]


def is_rich_production_science_request_v2(value: Mapping[str, Any]) -> bool:
    return bool(set(value) & _RICH_DISCRIMINANTS)


def _exact_text(value: object, *, field: str, path_free: bool = False) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScienceRequestV2Error(f"{field} must be a non-empty exact string")
    if path_free and _PATH_FREE_TOKEN.fullmatch(value) is None:
        raise ScienceRequestV2Error(f"{field} must be a path-free immutable token")
    return value


def _digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise ScienceRequestV2Error(f"{field} must be an exact content digest")
    try:
        return ContentDigest(value["algorithm"], value["hex"])
    except (TypeError, ValueError) as error:
        raise ScienceRequestV2Error(f"{field} is invalid") from error


def _campaign_from_document(value: object) -> ScienceCampaignV1:
    try:
        return ScienceCampaignV1.from_campaign_document(value)
    except (ContractError, TypeError, ValueError) as error:
        raise ScienceRequestV2Error(f"science_campaign is invalid: {error}") from error


def _decode_campaign_bytes(value: bytes) -> Mapping[str, Any]:
    try:
        decoded = value.decode("utf-8")
        document = yaml.safe_load(decoded)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ScienceRequestV2Error(
            "resolved science campaign is not valid UTF-8 JSON/YAML"
        ) from error
    if not isinstance(document, Mapping):
        raise ScienceRequestV2Error("resolved science campaign must be an object")
    return document


def _resolve_campaign(
    raw: Mapping[str, Any],
    *,
    resolvers: Mapping[str, ScienceCampaignResolverV1] | None,
) -> tuple[ScienceCampaignV1, dict[str, Any]]:
    if "science_campaign" in raw:
        document = raw["science_campaign"]
        if not isinstance(document, Mapping):
            raise ScienceRequestV2Error("science_campaign must be an object")
        campaign = _campaign_from_document(document)
        source_digest = canonical_sha256(dict(document))
        receipt_core: dict[str, Any] = {
            "schema_version": SCIENCE_CAMPAIGN_RESOLUTION_RECEIPT_SCHEMA_VERSION,
            "source_kind": "inline",
            "resolver_id": None,
            "immutable_object_id": None,
            "source_content_sha256": source_digest.to_dict(),
            "campaign_spec_sha256": campaign.campaign_spec_sha256().to_dict(),
            "normalized_campaign_sha256": canonical_sha256(
                campaign.to_campaign_document()
            ).to_dict(),
            "effects_performed": False,
        }
    else:
        reference = raw["science_campaign_ref"]
        if not isinstance(reference, Mapping) or set(reference) != {
            "schema_version",
            "resolver_id",
            "immutable_object_id",
            "content_sha256",
        }:
            raise ScienceRequestV2Error(
                "science_campaign_ref must contain exactly its frozen fields"
            )
        if reference["schema_version"] != SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION:
            raise ScienceRequestV2Error("science_campaign_ref schema is unsupported")
        resolver_id = _exact_text(
            reference["resolver_id"],
            field="science_campaign_ref.resolver_id",
            path_free=True,
        )
        immutable_object_id = _exact_text(
            reference["immutable_object_id"],
            field="science_campaign_ref.immutable_object_id",
            path_free=True,
        )
        expected_digest = _digest(
            reference["content_sha256"], field="science_campaign_ref.content_sha256"
        )
        resolver = None if resolvers is None else resolvers.get(resolver_id)
        if resolver is None:
            raise ScienceRequestV2Error(
                f"science campaign resolver is not registered: {resolver_id}"
            )
        try:
            content = resolver.resolve(immutable_object_id)
        except Exception as error:
            raise ScienceRequestV2Error("science campaign resolver failed") from error
        if not isinstance(content, bytes):
            raise ScienceRequestV2Error(
                "science campaign resolver must return exact bytes"
            )
        actual_digest = ContentDigest("sha256", hashlib.sha256(content).hexdigest())
        if actual_digest != expected_digest:
            raise ScienceRequestV2Error("science campaign content SHA-256 mismatch")
        campaign = _campaign_from_document(_decode_campaign_bytes(content))
        receipt_core = {
            "schema_version": SCIENCE_CAMPAIGN_RESOLUTION_RECEIPT_SCHEMA_VERSION,
            "source_kind": "immutable_reference",
            "resolver_id": resolver_id,
            "immutable_object_id": immutable_object_id,
            "source_content_sha256": actual_digest.to_dict(),
            "campaign_spec_sha256": campaign.campaign_spec_sha256().to_dict(),
            "normalized_campaign_sha256": canonical_sha256(
                campaign.to_campaign_document()
            ).to_dict(),
            "effects_performed": False,
        }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    return campaign, receipt


def normalize_rich_production_science_request_v2(
    value: Mapping[str, Any],
    *,
    resolvers: Mapping[str, ScienceCampaignResolverV1] | None = None,
) -> NormalizedRichScienceRequestV2:
    """Normalize one rich request without selecting or invoking an engine."""

    raw = dict(value)
    discriminants = set(raw) & _RICH_DISCRIMINANTS
    if raw.get("schema_version") != PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION:
        raise ScienceRequestV2Error("science request schema is unsupported")
    if len(discriminants) != 1:
        raise ScienceRequestV2Error(
            "rich science request requires exactly one campaign discriminant"
        )
    unknown = sorted(set(raw) - _COMMON_KEYS - _RICH_DISCRIMINANTS)
    if unknown:
        raise ScienceRequestV2Error(
            f"rich science request contains unknown or flat-only fields: {unknown}"
        )
    request_id = _exact_text(raw.get("request_id"), field="request_id")
    try:
        baseline = BaselineDeclarationV1.from_value(raw.get("baseline"))
        validate_baseline_purpose_v1(baseline, purpose="simulation")
    except BaselineError as error:
        raise ScienceRequestV2Error(str(error)) from error
    campaign, resolution_receipt = _resolve_campaign(raw, resolvers=resolvers)
    core = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "request_id": request_id,
        "baseline": baseline.payload(),
        "science_campaign": campaign.to_campaign_document(),
    }
    request_sha256 = canonical_sha256(core).hex
    if (
        raw.get("request_sha256") is not None
        and raw["request_sha256"] != request_sha256
    ):
        raise ScienceRequestV2Error("science request SHA-256 mismatch")
    return NormalizedRichScienceRequestV2(
        normalized_request={**core, "request_sha256": request_sha256},
        campaign=campaign,
        content_resolution_receipt=resolution_receipt,
    )


__all__ = (
    "SCIENCE_CAMPAIGN_REF_SCHEMA_VERSION",
    "SCIENCE_CAMPAIGN_RESOLUTION_RECEIPT_SCHEMA_VERSION",
    "NormalizedRichScienceRequestV2",
    "ScienceCampaignResolverV1",
    "ScienceRequestV2Error",
    "is_rich_production_science_request_v2",
    "normalize_rich_production_science_request_v2",
)
