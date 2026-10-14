"""Frozen loader/build contract for the native V1 compatibility encoder."""

from __future__ import annotations

from typing import Final


NATIVE_SNAPSHOT_ENCODER_ABI_V1: Final[str] = (
    "eiel_native_snapshot_v1_compatibility_encoder_abi_v1"
)
NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1: Final[str] = (
    "eiel_native_snapshot_v1_compatibility_encoder_extension_v1"
)
NATIVE_SNAPSHOT_ENCODER_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_native_snapshot_v1_compatibility_encoding_receipt_v1"
)
NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_native_snapshot_v1_stage_file_receipt_v1"
)
NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1: Final[str] = (
    "EXACT_V1_SINGLE_MEMBER_SEQUENTIAL"
)
NATIVE_SNAPSHOT_ENCODER_BUILD_MANIFEST_V1: Final[str] = (
    "eiel_native_snapshot_v1_compatibility_encoder_build_manifest_v1"
)

REQUIRED_NATIVE_ENCODER_EXPORTS_V1: Final[tuple[str, ...]] = (
    "abi_version",
    "extension_version",
    "zlib_compile_version",
    "zlib_runtime_version",
    "compile_snapshot_encoder_plan_v1",
)


__all__ = [
    "NATIVE_SNAPSHOT_ENCODER_ABI_V1",
    "NATIVE_SNAPSHOT_ENCODER_BUILD_MANIFEST_V1",
    "NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1",
    "NATIVE_SNAPSHOT_ENCODER_RECEIPT_SCHEMA_V1",
    "NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1",
    "NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1",
    "REQUIRED_NATIVE_ENCODER_EXPORTS_V1",
]
