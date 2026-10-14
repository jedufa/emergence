"""Canonical digest table for mutation-off linked-block inheritance.

The resident CUDA owner cannot serialize and SHA-256 a Python ``Genome`` on
every birth.  This module builds a complete, immutable digest table before
execution for a two-founder, fixed-linkage allele basis.  A resident genotype
code then identifies one donor allele per linkage block; the table supplies the
exact authoritative hereditary-payload digest ordering for every reachable
mutation-off combination.

The contract is reusable across campaigns.  It is bound to the versioned
inheritance registry, not to a seed, matrix, generation count, checkpoint
schedule, world, or campaign name.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .inheritance_registry_v1 import (
    INHERITANCE_PARENT_ORDER_VERSION,
    INHERITANCE_REGISTRY_VERSION,
    LINKAGE_BLOCKS,
    canonical_hereditary_payload_sha256,
)


SCHEMA_VERSION = "eiel_inheritance_linkage_digest_table_v1"
GENOTYPE_CODE_VERSION = "binary_linkage_block_alleles_lsb_v1"


class InheritanceLinkageDigestTableError(ValueError):
    """Raised when a founder basis cannot provide exact canonical ordering."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _field_value(genome: Any, field: str) -> Any:
    if not hasattr(genome, field):
        return None
    return copy.deepcopy(getattr(genome, field))


def _field_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_sha256(left) == _canonical_sha256(right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


@dataclass(frozen=True)
class LinkageDigestTableV1:
    schema_version: str
    registry_version: str
    parent_order_version: str
    genotype_code_version: str
    block_ids: tuple[str, ...]
    digest_by_code: tuple[str, ...]
    rank_by_code: tuple[int, ...]
    table_sha256: str

    def __post_init__(self) -> None:
        expected = 1 << len(self.block_ids)
        if (
            self.schema_version != SCHEMA_VERSION
            or self.registry_version != INHERITANCE_REGISTRY_VERSION
            or self.parent_order_version != INHERITANCE_PARENT_ORDER_VERSION
            or self.genotype_code_version != GENOTYPE_CODE_VERSION
            or not self.block_ids
            or len(set(self.block_ids)) != len(self.block_ids)
            or len(self.digest_by_code) != expected
            or len(self.rank_by_code) != expected
            or any(type(rank) is not int or rank < 0 for rank in self.rank_by_code)
            or (
                sorted(set(self.rank_by_code))
                != list(range(len(set(self.rank_by_code))))
            )
        ):
            raise InheritanceLinkageDigestTableError(
                "linkage digest table shape or identity is invalid"
            )
        if self.table_sha256 != _canonical_sha256(self.payload(include_hash=False)):
            raise InheritanceLinkageDigestTableError(
                "linkage digest table hash mismatch"
            )

    @property
    def block_count(self) -> int:
        return len(self.block_ids)

    @property
    def genotype_count(self) -> int:
        return len(self.digest_by_code)

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "parent_order_version": self.parent_order_version,
            "genotype_code_version": self.genotype_code_version,
            "block_ids": list(self.block_ids),
            "digest_by_code": list(self.digest_by_code),
            "rank_by_code": list(self.rank_by_code),
        }
        if include_hash:
            payload["table_sha256"] = self.table_sha256
        return payload

    def canonical_pair(
        self, code_a: int, code_b: int
    ) -> tuple[int, int]:
        for code in (code_a, code_b):
            if type(code) is not int or not 0 <= code < self.genotype_count:
                raise InheritanceLinkageDigestTableError(
                    "genotype code is outside the digest table"
                )
        if self.rank_by_code[code_a] <= self.rank_by_code[code_b]:
            return code_a, code_b
        return code_b, code_a

    def child_code(
        self,
        canonical_parent_0_code: int,
        canonical_parent_1_code: int,
        donor_ordinals: Sequence[int],
    ) -> int:
        self.canonical_pair(
            canonical_parent_0_code, canonical_parent_1_code
        )
        if (
            len(donor_ordinals) != self.block_count
            or any(type(value) is not int or value not in {0, 1}
                   for value in donor_ordinals)
        ):
            raise InheritanceLinkageDigestTableError(
                "donor ordinals must contain one binary value per linkage block"
            )
        child = 0
        for block, donor in enumerate(donor_ordinals):
            parent_code = (
                canonical_parent_0_code
                if donor == 0
                else canonical_parent_1_code
            )
            child |= ((parent_code >> block) & 1) << block
        return child


def _build_genome_for_code(
    canonical_parent_0: Any,
    canonical_parent_1: Any,
    *,
    code: int,
    linkage_blocks: Sequence[tuple[str, Sequence[str]]],
) -> Any:
    child = copy.deepcopy(canonical_parent_0)
    for block, (_block_id, fields) in enumerate(linkage_blocks):
        donor = canonical_parent_1 if (code >> block) & 1 else canonical_parent_0
        for field in fields:
            if hasattr(donor, field):
                setattr(child, field, copy.deepcopy(getattr(donor, field)))
            elif hasattr(child, field):
                raise InheritanceLinkageDigestTableError(
                    f"canonical donor is missing linkage field {field!r}"
                )
    refresh = getattr(child, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()
    return child


def build_linkage_digest_table_v1(
    canonical_parent_0: Any,
    canonical_parent_1: Any,
    *,
    linkage_blocks: Sequence[tuple[str, Sequence[str]]] = LINKAGE_BLOCKS,
) -> LinkageDigestTableV1:
    """Build all exact mutation-off hereditary digest combinations.

    The supplied order must already be the authoritative digest order.  This is
    checked directly. Distinct codes may share a digest when a linkage block has
    identical founder alleles; equal ranks deliberately preserve the incoming
    parent argument order, matching the authoritative original-index tie break.
    """

    blocks = tuple(
        (str(block_id), tuple(str(field) for field in fields))
        for block_id, fields in linkage_blocks
    )
    if (
        not blocks
        or len(set(block_id for block_id, _fields in blocks)) != len(blocks)
        or any(not block_id or not fields for block_id, fields in blocks)
    ):
        raise InheritanceLinkageDigestTableError(
            "linkage blocks must be non-empty and uniquely identified"
        )
    digest_0 = canonical_hereditary_payload_sha256(canonical_parent_0)
    digest_1 = canonical_hereditary_payload_sha256(canonical_parent_1)
    if digest_0 >= digest_1:
        raise InheritanceLinkageDigestTableError(
            "founders must be distinct and supplied in authoritative digest order"
        )

    # Every differing top-level Genome field must be owned by one declared
    # linkage block. Otherwise a compact linkage code would omit hereditary
    # state that changes the authoritative digest.
    linked_fields = {field for _block, fields in blocks for field in fields}
    left = vars(canonical_parent_0)
    right = vars(canonical_parent_1)
    differing_unowned = sorted(
        field
        for field in set(left) | set(right)
        if not field.startswith("_")
        and field not in linked_fields
        and not _field_equal(
            _field_value(canonical_parent_0, field),
            _field_value(canonical_parent_1, field),
        )
    )
    if differing_unowned:
        raise InheritanceLinkageDigestTableError(
            "founders differ outside the linkage registry: "
            + ", ".join(differing_unowned)
        )

    count = 1 << len(blocks)
    digests = tuple(
        canonical_hereditary_payload_sha256(
            _build_genome_for_code(
                canonical_parent_0,
                canonical_parent_1,
                code=code,
                linkage_blocks=blocks,
            )
        )
        for code in range(count)
    )
    ordered_digests = {
        digest: rank for rank, digest in enumerate(sorted(set(digests)))
    }
    ranks = [ordered_digests[digest] for digest in digests]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": INHERITANCE_REGISTRY_VERSION,
        "parent_order_version": INHERITANCE_PARENT_ORDER_VERSION,
        "genotype_code_version": GENOTYPE_CODE_VERSION,
        "block_ids": [block_id for block_id, _fields in blocks],
        "digest_by_code": list(digests),
        "rank_by_code": ranks,
    }
    return LinkageDigestTableV1(
        schema_version=SCHEMA_VERSION,
        registry_version=INHERITANCE_REGISTRY_VERSION,
        parent_order_version=INHERITANCE_PARENT_ORDER_VERSION,
        genotype_code_version=GENOTYPE_CODE_VERSION,
        block_ids=tuple(payload["block_ids"]),
        digest_by_code=digests,
        rank_by_code=tuple(ranks),
        table_sha256=_canonical_sha256(payload),
    )


def founder_codes_v1(table: LinkageDigestTableV1) -> tuple[int, int]:
    """Return the all-parent-0 and all-parent-1 founder genotype codes."""

    return 0, table.genotype_count - 1


__all__ = [
    "GENOTYPE_CODE_VERSION",
    "InheritanceLinkageDigestTableError",
    "LinkageDigestTableV1",
    "SCHEMA_VERSION",
    "build_linkage_digest_table_v1",
    "founder_codes_v1",
]
