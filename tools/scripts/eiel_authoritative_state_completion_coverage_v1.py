#!/usr/bin/env python3
"""Verify external completion coverage for a sealed authoritative-state schema.

The canonical package remains the authority.  This tool consumes a separate,
hash-bound coverage map and proves that every admitted field, every supported
exact ABI occurrence, every boundary-test profile, every conversion class, and
every declared source anchor is accounted for without changing the schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, NoReturn, Sequence


COVERAGE_MAP_SCHEMA_VERSION_V1 = (
    "eiel_authoritative_state_completion_coverage_map_v1"
)
COVERAGE_RECEIPT_SCHEMA_VERSION_V1 = (
    "eiel_authoritative_state_completion_coverage_receipt_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_MAP_FIELDS = frozenset(
    {
        "schema_version",
        "state_schema_version",
        "state_schema_sha256",
        "boundary_profile_coverage",
        "conversion_policy_coverage",
    }
)
_PROFILE_FIELDS = frozenset({"boundary_test_profile", "evidence_refs"})
_CONVERSION_FIELDS = frozenset(
    {"policy_id", "physical_types", "signedness", "evidence_refs"}
)
_EVIDENCE_FIELDS = frozenset({"path", "sha256", "selectors"})


class CompletionCoverageError(ValueError):
    """Stable fail-closed error for external completion coverage."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise CompletionCoverageError(code, message)


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("EIEL_COVERAGE_NONFINITE", f"{path} is non-finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(
        "EIEL_COVERAGE_VALUE",
        f"{path} has unsupported type {type(value).__name__}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _fail("EIEL_COVERAGE_UNREADABLE", f"{label} cannot be read: {exc}")
    if not isinstance(value, dict):
        _fail("EIEL_COVERAGE_TYPE", f"{label} root must be an object")
    return value


def _exact_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EIEL_COVERAGE_TYPE", f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        _fail(
            "EIEL_COVERAGE_FIELDS",
            f"{path} fields differ: missing={sorted(fields - actual)} "
            f"extra={sorted(actual - fields)}",
        )
    return value


def _sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("EIEL_COVERAGE_TYPE", f"{path} must be an array")
    if not value:
        _fail("EIEL_COVERAGE_MISSING", f"{path} must not be empty")
    return value


def _text(value: Any, *, path: str, token: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("EIEL_COVERAGE_IDENTITY", f"{path} must be a nonempty string")
    if token and _TOKEN.fullmatch(value) is None:
        _fail("EIEL_COVERAGE_IDENTITY", f"{path} has invalid token format")
    return value


def _digest(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("EIEL_COVERAGE_IDENTITY", f"{path} is not a SHA-256 digest")
    return value


def _relative_file(root: Path, raw: Any, *, path: str) -> tuple[str, Path]:
    text = _text(raw, path=path)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        _fail(
            "EIEL_COVERAGE_IDENTITY",
            f"{path} must be a normalized relative POSIX path",
        )
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail("EIEL_COVERAGE_IDENTITY", f"{path} escapes source root")
    if not candidate.is_file():
        _fail("EIEL_COVERAGE_SOURCE_MISSING", f"{path} is not a file: {text}")
    return text, candidate


def _load_sealed_schema(schema_root: Path) -> dict[str, Any]:
    root = schema_root.resolve(strict=True)
    schema_path = root / "schema.json"
    schema = _json(schema_path, label="schema")
    version = _text(schema.get("schema_version"), path="schema.schema_version")
    claimed_schema_hash = _digest(
        schema.get("schema_hash"),
        path="schema.schema_hash",
    )
    unsigned_schema = dict(schema)
    unsigned_schema.pop("schema_hash", None)
    actual_schema_hash = content_sha256(unsigned_schema)
    if actual_schema_hash != claimed_schema_hash:
        _fail(
            "EIEL_COVERAGE_STALE_SCHEMA",
            f"schema hash differs: expected {claimed_schema_hash}, "
            f"got {actual_schema_hash}",
        )
    component_hashes = schema.get("component_hashes")
    if not isinstance(component_hashes, Mapping) or not component_hashes:
        _fail("EIEL_COVERAGE_SCHEMA", "schema component hash map is absent")
    components: dict[str, dict[str, Any]] = {}
    for filename, raw_digest in sorted(component_hashes.items()):
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or filename == "schema.json"
        ):
            _fail("EIEL_COVERAGE_SCHEMA", f"invalid component name {filename!r}")
        expected = _digest(
            raw_digest,
            path=f"schema.component_hashes.{filename}",
        )
        component_path = root / filename
        if not component_path.is_file():
            _fail("EIEL_COVERAGE_SCHEMA", f"component is absent: {filename}")
        actual = file_sha256(component_path)
        if actual != expected:
            _fail(
                "EIEL_COVERAGE_STALE_SCHEMA",
                f"{filename} hash differs: expected {expected}, got {actual}",
            )
        components[filename] = _json(
            component_path,
            label=f"component {filename}",
        )
    registry = components.get("field_registry.json")
    abi = components.get("abi_variants.json")
    if (
        not isinstance(registry, Mapping)
        or not isinstance(registry.get("fields"), list)
        or not isinstance(abi, Mapping)
        or not isinstance(abi.get("variants"), list)
    ):
        _fail(
            "EIEL_COVERAGE_SCHEMA",
            "sealed package lacks field_registry.json or abi_variants.json",
        )
    return {
        "root": root,
        "version": version,
        "hash": claimed_schema_hash,
        "fields": registry["fields"],
        "abi_variants": abi["variants"],
    }


def _validate_evidence_refs(
    raw_refs: Any,
    *,
    source_root: Path,
    path: str,
) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    local_keys: set[tuple[str, str]] = set()
    for index, raw_ref in enumerate(_sequence(raw_refs, path=path)):
        ref_path = f"{path}[{index}]"
        ref = _exact_mapping(raw_ref, fields=_EVIDENCE_FIELDS, path=ref_path)
        relative, source = _relative_file(
            source_root,
            ref["path"],
            path=f"{ref_path}.path",
        )
        expected = _digest(ref["sha256"], path=f"{ref_path}.sha256")
        actual = file_sha256(source)
        if actual != expected:
            _fail(
                "EIEL_COVERAGE_STALE_EVIDENCE",
                f"{relative} hash differs: expected {expected}, got {actual}",
            )
        try:
            source_text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _fail(
                "EIEL_COVERAGE_UNREADABLE",
                f"evidence {relative} is not UTF-8 text: {exc}",
            )
        selectors: list[str] = []
        for selector_index, raw_selector in enumerate(
            _sequence(ref["selectors"], path=f"{ref_path}.selectors")
        ):
            selector = _text(
                raw_selector,
                path=f"{ref_path}.selectors[{selector_index}]",
            )
            if selector in selectors:
                _fail(
                    "EIEL_COVERAGE_DUPLICATE",
                    f"{ref_path} repeats selector {selector!r}",
                )
            if selector not in source_text:
                _fail(
                    "EIEL_COVERAGE_SELECTOR_MISSING",
                    f"{relative} does not contain selector {selector!r}",
                )
            selectors.append(selector)
        for selector in selectors:
            key = (relative, selector)
            if key in local_keys:
                _fail(
                    "EIEL_COVERAGE_DUPLICATE",
                    f"{path} repeats evidence selector {relative}:{selector}",
                )
            local_keys.add(key)
        refs.append(
            {
                "path": relative,
                "sha256": actual,
                "selectors": sorted(selectors),
            }
        )
    refs.sort(key=lambda row: (row["path"], row["selectors"], row["sha256"]))
    return tuple(refs)


def verify_completion_coverage_v1(
    *,
    schema_root: str | Path,
    coverage_map_path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Verify one external coverage map and return a deterministic receipt."""

    sealed = _load_sealed_schema(Path(schema_root))
    source = Path(source_root).resolve(strict=True)
    coverage_path = Path(coverage_map_path)
    coverage = _exact_mapping(
        _json(coverage_path, label="coverage map"),
        fields=_MAP_FIELDS,
        path="coverage_map",
    )
    if coverage["schema_version"] != COVERAGE_MAP_SCHEMA_VERSION_V1:
        _fail(
            "EIEL_COVERAGE_VERSION",
            f"coverage map version differs: {coverage['schema_version']!r}",
        )
    if (
        coverage["state_schema_version"] != sealed["version"]
        or coverage["state_schema_sha256"] != sealed["hash"]
    ):
        _fail(
            "EIEL_COVERAGE_STALE_SCHEMA",
            "coverage map is not bound to the sealed schema identity",
        )

    admitted: list[Mapping[str, Any]] = []
    field_ids: set[str] = set()
    anchor_occurrences: list[dict[str, str]] = []
    for index, raw_field in enumerate(sealed["fields"]):
        if not isinstance(raw_field, Mapping):
            _fail("EIEL_COVERAGE_SCHEMA", f"fields[{index}] is not an object")
        field_id = _text(
            raw_field.get("field_id"),
            path=f"fields[{index}].field_id",
        )
        if field_id in field_ids:
            _fail("EIEL_COVERAGE_DUPLICATE", f"duplicate field ID {field_id}")
        field_ids.add(field_id)
        if raw_field.get("production_admission") != "admitted":
            continue
        if raw_field.get("resolution_status") != "resolved":
            _fail(
                "EIEL_COVERAGE_UNRESOLVED",
                f"admitted field is not resolved: {field_id}",
            )
        for required in (
            "boundary_test_profile",
            "physical_type",
            "signedness",
            "source_anchors",
        ):
            if required not in raw_field:
                _fail(
                    "EIEL_COVERAGE_SCHEMA",
                    f"{field_id} lacks {required}",
                )
        anchors = raw_field["source_anchors"]
        if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
            _fail(
                "EIEL_COVERAGE_SCHEMA",
                f"{field_id} source anchors are not an array",
            )
        if not anchors:
            _fail(
                "EIEL_COVERAGE_MISSING",
                f"{field_id} has no source anchor",
            )
        seen_field_anchors: set[str] = set()
        for anchor_index, raw_anchor in enumerate(anchors):
            anchor_path = f"{field_id}.source_anchors[{anchor_index}]"
            if not isinstance(raw_anchor, Mapping) or set(raw_anchor) != {
                "path",
                "source_sha256",
            }:
                _fail(
                    "EIEL_COVERAGE_SCHEMA",
                    f"{anchor_path} fields differ",
                )
            relative, anchor_file = _relative_file(
                source,
                raw_anchor["path"],
                path=f"{anchor_path}.path",
            )
            if relative in seen_field_anchors:
                _fail(
                    "EIEL_COVERAGE_DUPLICATE",
                    f"{field_id} repeats source anchor {relative}",
                )
            seen_field_anchors.add(relative)
            expected = _digest(
                raw_anchor["source_sha256"],
                path=f"{anchor_path}.source_sha256",
            )
            actual = file_sha256(anchor_file)
            if actual != expected:
                _fail(
                    "EIEL_COVERAGE_STALE_SOURCE",
                    f"{field_id} source anchor {relative} differs: "
                    f"expected {expected}, got {actual}",
                )
            anchor_occurrences.append(
                {"field_id": field_id, "path": relative, "sha256": actual}
            )
        admitted.append(raw_field)
    if not admitted:
        _fail("EIEL_COVERAGE_MISSING", "schema has no admitted fields")

    observed_profiles: dict[str, list[str]] = {}
    for field in admitted:
        profile = _text(
            field["boundary_test_profile"],
            path=f"{field['field_id']}.boundary_test_profile",
            token=True,
        )
        observed_profiles.setdefault(profile, []).append(str(field["field_id"]))

    profile_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    for index, raw_row in enumerate(
        _sequence(
            coverage["boundary_profile_coverage"],
            path="coverage_map.boundary_profile_coverage",
        )
    ):
        row_path = f"coverage_map.boundary_profile_coverage[{index}]"
        row = _exact_mapping(raw_row, fields=_PROFILE_FIELDS, path=row_path)
        profile = _text(
            row["boundary_test_profile"],
            path=f"{row_path}.boundary_test_profile",
            token=True,
        )
        if profile in profile_rows:
            _fail(
                "EIEL_COVERAGE_DUPLICATE",
                f"duplicate boundary profile mapping {profile}",
            )
        if profile not in observed_profiles:
            _fail(
                "EIEL_COVERAGE_UNKNOWN",
                f"coverage map names unknown boundary profile {profile}",
            )
        profile_rows[profile] = _validate_evidence_refs(
            row["evidence_refs"],
            source_root=source,
            path=f"{row_path}.evidence_refs",
        )
    missing_profiles = sorted(set(observed_profiles) - set(profile_rows))
    if missing_profiles:
        _fail(
            "EIEL_COVERAGE_MISSING",
            f"boundary profiles lack coverage: {missing_profiles}",
        )

    conversion_rows: list[dict[str, Any]] = []
    conversion_ids: set[str] = set()
    for index, raw_row in enumerate(
        _sequence(
            coverage["conversion_policy_coverage"],
            path="coverage_map.conversion_policy_coverage",
        )
    ):
        row_path = f"coverage_map.conversion_policy_coverage[{index}]"
        row = _exact_mapping(raw_row, fields=_CONVERSION_FIELDS, path=row_path)
        policy_id = _text(
            row["policy_id"],
            path=f"{row_path}.policy_id",
            token=True,
        )
        if policy_id in conversion_ids:
            _fail(
                "EIEL_COVERAGE_DUPLICATE",
                f"duplicate conversion policy mapping {policy_id}",
            )
        conversion_ids.add(policy_id)
        physical_types = sorted(
            {
                _text(item, path=f"{row_path}.physical_types[{item_index}]")
                for item_index, item in enumerate(
                    _sequence(
                        row["physical_types"],
                        path=f"{row_path}.physical_types",
                    )
                )
            }
        )
        signedness = sorted(
            {
                _text(item, path=f"{row_path}.signedness[{item_index}]")
                for item_index, item in enumerate(
                    _sequence(row["signedness"], path=f"{row_path}.signedness")
                )
            }
        )
        if len(physical_types) != len(row["physical_types"]) or len(
            signedness
        ) != len(row["signedness"]):
            _fail(
                "EIEL_COVERAGE_DUPLICATE",
                f"{policy_id} repeats a conversion selector",
            )
        conversion_rows.append(
            {
                "policy_id": policy_id,
                "physical_types": physical_types,
                "signedness": signedness,
                "evidence_refs": _validate_evidence_refs(
                    row["evidence_refs"],
                    source_root=source,
                    path=f"{row_path}.evidence_refs",
                ),
                "field_ids": [],
            }
        )

    for field in admitted:
        matches = [
            row
            for row in conversion_rows
            if field["physical_type"] in row["physical_types"]
            and field["signedness"] in row["signedness"]
        ]
        if not matches:
            _fail(
                "EIEL_COVERAGE_MISSING",
                f"{field['field_id']} has no conversion-policy mapping for "
                f"{field['physical_type']}/{field['signedness']}",
            )
        if len(matches) != 1:
            _fail(
                "EIEL_COVERAGE_DUPLICATE",
                f"{field['field_id']} matches multiple conversion policies: "
                f"{sorted(row['policy_id'] for row in matches)}",
            )
        matches[0]["field_ids"].append(str(field["field_id"]))
    unused_policies = sorted(
        row["policy_id"] for row in conversion_rows if not row["field_ids"]
    )
    if unused_policies:
        _fail(
            "EIEL_COVERAGE_UNKNOWN",
            f"conversion policies match no admitted field: {unused_policies}",
        )

    abi_occurrence_count = 0
    abi_unique_ids: set[str] = set()
    supported_variant_count = 0
    admitted_ids = {str(field["field_id"]) for field in admitted}
    for variant_index, raw_variant in enumerate(sealed["abi_variants"]):
        if not isinstance(raw_variant, Mapping):
            _fail(
                "EIEL_COVERAGE_SCHEMA",
                f"abi_variants[{variant_index}] is not an object",
            )
        if raw_variant.get("support_status") != "supported_exact_metadata":
            continue
        supported_variant_count += 1
        entries = raw_variant.get("entries")
        if not isinstance(entries, list):
            _fail(
                "EIEL_COVERAGE_SCHEMA",
                f"supported ABI variant {raw_variant.get('variant_id')} "
                "has no entries array",
            )
        if raw_variant.get("count") != len(entries):
            _fail(
                "EIEL_COVERAGE_SCHEMA",
                f"supported ABI variant {raw_variant.get('variant_id')} "
                "count differs from entries",
            )
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                _fail(
                    "EIEL_COVERAGE_SCHEMA",
                    f"ABI entry {variant_index}:{entry_index} is not an object",
                )
            field_id = entry.get("semantic_field_id")
            if field_id not in admitted_ids:
                _fail(
                    "EIEL_COVERAGE_MISSING",
                    f"supported ABI occurrence maps to non-admitted field "
                    f"{field_id!r}",
                )
            abi_occurrence_count += 1
            abi_unique_ids.add(str(field_id))

    evidence_rows: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for refs in profile_rows.values():
        for ref in refs:
            evidence_rows[
                (ref["path"], ref["sha256"], tuple(ref["selectors"]))
            ] = ref
    for row in conversion_rows:
        for ref in row["evidence_refs"]:
            evidence_rows[
                (ref["path"], ref["sha256"], tuple(ref["selectors"]))
            ] = ref

    profile_summary = [
        {
            "boundary_test_profile": profile,
            "field_count": len(observed_profiles[profile]),
            "field_inventory_sha256": content_sha256(
                sorted(observed_profiles[profile])
            ),
            "evidence_ref_count": len(profile_rows[profile]),
        }
        for profile in sorted(observed_profiles)
    ]
    conversion_summary = [
        {
            "policy_id": row["policy_id"],
            "physical_types": row["physical_types"],
            "signedness": row["signedness"],
            "field_count": len(row["field_ids"]),
            "field_inventory_sha256": content_sha256(sorted(row["field_ids"])),
            "evidence_ref_count": len(row["evidence_refs"]),
        }
        for row in sorted(conversion_rows, key=lambda item: item["policy_id"])
    ]
    sorted_anchors = sorted(
        anchor_occurrences,
        key=lambda row: (row["field_id"], row["path"], row["sha256"]),
    )
    unique_anchor_files = sorted(
        {(row["path"], row["sha256"]) for row in sorted_anchors}
    )
    receipt: dict[str, Any] = {
        "schema_version": COVERAGE_RECEIPT_SCHEMA_VERSION_V1,
        "status": "accepted",
        "state_schema_version": sealed["version"],
        "state_schema_sha256": sealed["hash"],
        "coverage_map_file_name": coverage_path.name,
        "coverage_map_sha256": file_sha256(coverage_path),
        "supported_field_count": len(admitted),
        "supported_field_inventory_sha256": content_sha256(sorted(admitted_ids)),
        "supported_abi_variant_count": supported_variant_count,
        "supported_abi_occurrence_count": abi_occurrence_count,
        "supported_abi_unique_field_count": len(abi_unique_ids),
        "boundary_test_profile_count": len(profile_summary),
        "boundary_test_profiles": profile_summary,
        "conversion_policy_count": len(conversion_summary),
        "conversion_policies": conversion_summary,
        "source_anchor_occurrence_count": len(sorted_anchors),
        "unique_source_anchor_file_count": len(unique_anchor_files),
        "source_anchor_inventory_sha256": content_sha256(sorted_anchors),
        "evidence_reference_count": len(evidence_rows),
        "evidence_inventory_sha256": content_sha256(
            [
                evidence_rows[key]
                for key in sorted(evidence_rows)
            ]
        ),
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def publish_receipt_atomic_v1(
    output_path: str | Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Publish a new receipt without overwriting an existing path."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(raw_path)
        payload = json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        _fail(
            "EIEL_COVERAGE_OUTPUT_EXISTS",
            f"refusing to overwrite {destination}",
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-root", type=Path, required=True)
    parser.add_argument("--coverage-map", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify_completion_coverage_v1(
        schema_root=args.schema_root,
        coverage_map_path=args.coverage_map,
        source_root=args.source_root,
    )
    output = publish_receipt_atomic_v1(args.output_receipt, receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "output_receipt": str(output),
                "receipt_sha256": receipt["receipt_sha256"],
                "supported_field_count": receipt["supported_field_count"],
                "supported_abi_occurrence_count": receipt[
                    "supported_abi_occurrence_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
