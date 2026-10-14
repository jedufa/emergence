#!/usr/bin/env python3
"""Issue deterministic preregistration and route-authority documents for P1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (  # noqa: E402
    preregistration_payload_v1 as c1_preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    CAMPAIGNS,
    preregistration_payload_v1 as e1_preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (  # noqa: E402
    preregistration_payload_v1 as environment_preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (  # noqa: E402
    preregistration_payload_v1 as hereditary_preregistration_payload_v1,
    validate_authority_v1 as validate_hereditary_authority_v1,
)
from engine.evolution.genome_upgrade_v2_p10_completion_gate_v1 import (  # noqa: E402
    preregistration_payload_v1 as completion_preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (  # noqa: E402
    checkpoint_authority_binding_v1,
    validate_checkpoint_authority_binding_v1,
)
from tools.scripts.genome_upgrade_v2_p10_completion_collect_v1 import (  # noqa: E402
    CompletionCollectorError,
    SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    validate_execution_route_authority_v1,
    validate_package_authority_v1,
)


SCHEMA_VERSION = "genome_v2_p10_authority_issuer_v1"
PREREGISTRATION_FILES = {
    "c1": "genome_v2_c1_preregistration.json",
    "e1": "genome_v2_e1_preregistration.json",
    "hereditary_capacity": (
        "genome_v2_p10_hereditary_capacity_preregistration.json"
    ),
    "environment_switch": "genome_v2_environment_switch_preregistration.json",
    "p10_completion": "genome_v2_p10_completion_preregistration.json",
}
AUTHORITY_FILES = {
    "patch_checkpoint": "patch_checkpoint_authority_binding.json",
    "wdl_checkpoint": "wdl_checkpoint_authority_binding.json",
    "e1_campaign_routes": "e1_campaign_route_bindings.json",
    "hereditary_capacity": "hereditary_authority.json",
}


class GenomeV2P10AuthorityIssueError(ValueError):
    """Authority inputs are incomplete, stale, or contradictory."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish(output_directory: Path, payloads: Mapping[str, Any]) -> dict[str, Path]:
    output = output_directory.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise GenomeV2P10AuthorityIssueError(
            "authority output directory already exists"
        ) from exc
    paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        path = output / filename
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(payload))
        paths[filename] = path
    return paths


def issue_preregistrations_v1(output_directory: Path) -> dict[str, Path]:
    payloads = {
        PREREGISTRATION_FILES["c1"]: c1_preregistration_payload_v1(),
        PREREGISTRATION_FILES["e1"]: e1_preregistration_payload_v1(),
        PREREGISTRATION_FILES[
            "hereditary_capacity"
        ]: hereditary_preregistration_payload_v1(),
        PREREGISTRATION_FILES[
            "environment_switch"
        ]: environment_preregistration_payload_v1(),
        PREREGISTRATION_FILES[
            "p10_completion"
        ]: completion_preregistration_payload_v1(),
    }
    entries = {
        filename: {
            "content_sha256": _sha256(canonical_json_bytes(payload)),
            "file_sha256": _sha256(canonical_json_bytes(payload)),
        }
        for filename, payload in payloads.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "document_kind": "p10_preregistration_set",
        "entries": entries,
    }
    return _publish(
        output_directory,
        {**payloads, "genome_v2_p10_preregistration_manifest.json": manifest},
    )


def _validated_inputs(
    package_receipt_path: Path,
    patch_route_receipt_path: Path,
    wdl_route_receipt_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Any], dict[str, Any]]:
    try:
        identity, package_id, package_receipt_sha256 = (
            validate_package_authority_v1(
                package_receipt_path
            )
        )
        patch_route, _ = validate_execution_route_authority_v1(
            patch_route_receipt_path,
            label="Patch",
            artifact_sha256=str(identity["cuda_artifact_sha256"]),
        )
        wdl_route, _ = validate_execution_route_authority_v1(
            wdl_route_receipt_path,
            label="WDL",
            artifact_sha256=str(identity["cuda_artifact_sha256"]),
            required_obligations=SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
        )
    except CompletionCollectorError as exc:
        raise GenomeV2P10AuthorityIssueError(str(exc)) from exc
    if (
        patch_route["execution_route_sha256"]
        == wdl_route["execution_route_sha256"]
    ):
        raise GenomeV2P10AuthorityIssueError(
            "Patch and WDL execution routes must be distinct"
        )
    return (
        identity,
        package_id,
        package_receipt_sha256,
        patch_route,
        wdl_route,
    )


def issue_authority_bindings_v1(
    *,
    package_receipt_path: Path,
    patch_route_receipt_path: Path,
    wdl_route_receipt_path: Path,
    output_directory: Path,
) -> dict[str, Path]:
    (
        identity,
        package_id,
        package_receipt_sha256,
        patch_route,
        wdl_route,
    ) = _validated_inputs(
        package_receipt_path,
        patch_route_receipt_path,
        wdl_route_receipt_path,
    )
    common = {
        "package_id_sha256": package_id,
        "package_receipt_sha256": package_receipt_sha256,
    }
    patch_binding = validate_checkpoint_authority_binding_v1(
        checkpoint_authority_binding_v1(
            identity,
            **common,
            execution_route_sha256=str(
                patch_route["execution_route_sha256"]
            ),
        )
    )
    wdl_binding = validate_checkpoint_authority_binding_v1(
        checkpoint_authority_binding_v1(
            identity,
            **common,
            execution_route_sha256=str(wdl_route["execution_route_sha256"]),
        )
    )
    if len(CAMPAIGNS) != 3:
        raise GenomeV2P10AuthorityIssueError(
            "E1 campaign route projection differs from the frozen contract"
        )
    e1_bindings = {
        CAMPAIGNS[0]: patch_binding,
        CAMPAIGNS[1]: patch_binding,
        CAMPAIGNS[2]: wdl_binding,
    }
    hereditary_authority = validate_hereditary_authority_v1(
        {
            name: patch_binding[name]
            for name in (
                "provider_id",
                "runtime_profile_id",
                "source_commit",
                "package_id_sha256",
                "package_receipt_sha256",
                "artifact_sha256",
                "execution_route_sha256",
            )
        }
    )
    payloads = {
        AUTHORITY_FILES["patch_checkpoint"]: patch_binding,
        AUTHORITY_FILES["wdl_checkpoint"]: wdl_binding,
        AUTHORITY_FILES["e1_campaign_routes"]: e1_bindings,
        AUTHORITY_FILES["hereditary_capacity"]: hereditary_authority,
    }
    entries = {
        filename: {
            "file_sha256": _sha256(canonical_json_bytes(payload)),
        }
        for filename, payload in payloads.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "document_kind": "p10_package_route_authority_set",
        "package_id_sha256": package_id,
        "package_receipt_sha256": package_receipt_sha256,
        "artifact_sha256": identity["cuda_artifact_sha256"],
        "patch_execution_route_sha256": patch_route[
            "execution_route_sha256"
        ],
        "wdl_execution_route_sha256": wdl_route[
            "execution_route_sha256"
        ],
        "entries": entries,
    }
    return _publish(
        output_directory,
        {**payloads, "genome_v2_p10_authority_manifest.json": manifest},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--output-dir", type=Path, required=True)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--package-receipt", type=Path, required=True)
    bind.add_argument("--patch-route-receipt", type=Path, required=True)
    bind.add_argument("--wdl-route-receipt", type=Path, required=True)
    bind.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        paths = issue_preregistrations_v1(args.output_dir)
    else:
        paths = issue_authority_bindings_v1(
            package_receipt_path=args.package_receipt,
            patch_route_receipt_path=args.patch_route_receipt,
            wdl_route_receipt_path=args.wdl_route_receipt,
            output_directory=args.output_dir,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "output_files": sorted(str(path) for path in paths.values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
