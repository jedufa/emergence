"""EIEL authoritative-state schema v1 public API."""

from .schema import (
    AuthoritativeStateSchemaError,
    AuthoritativeStateSchemaV1,
    SCHEMA_ROOT,
    SCHEMA_VERSION_V1,
    canonical_json_bytes,
    content_sha256,
    load_schema_v1,
    validate_boundary_v1,
)
from .conversions import (
    checked_finite_float32,
    checked_finite_positive_float32,
    checked_int32,
    checked_lineage_founder_id,
    checked_uint64,
    uint64_from_json_decimal,
    uint64_to_json_decimal,
)
from .identity import (
    AbiVariantIdentityV1,
    BinaryIdentityV1,
    require_identity_equal,
)
from .production_identity import (
    validate_admission_identity_v1,
    validate_identity_against_local_schema_v1,
    validate_production_identity_v1,
)

__all__ = [
    "AuthoritativeStateSchemaError",
    "AuthoritativeStateSchemaV1",
    "SCHEMA_ROOT",
    "SCHEMA_VERSION_V1",
    "canonical_json_bytes",
    "content_sha256",
    "load_schema_v1",
    "validate_boundary_v1",
    "AbiVariantIdentityV1",
    "BinaryIdentityV1",
    "checked_finite_float32",
    "checked_finite_positive_float32",
    "checked_int32",
    "checked_lineage_founder_id",
    "checked_uint64",
    "require_identity_equal",
    "uint64_from_json_decimal",
    "uint64_to_json_decimal",
    "validate_admission_identity_v1",
    "validate_identity_against_local_schema_v1",
    "validate_production_identity_v1",
]
