#!/usr/bin/env python3
"""Reduce a completed Genome V2 E1 campaign matrix into immutable outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    E1AuthorityBindingV1,
    E1CellEvidenceV1,
    PREREGISTRATION_SHA256,
    evaluate_e1_v1,
    preregistration_payload_v1,
)


RUNNER_SCHEMA_VERSION: Final[str] = "genome_v2_p10_e1_runner_v1"
OUTPUT_NAMES: Final[dict[str, str]] = {
    "preregistration": "genome_v2_e1_preregistration.json",
    "paired_json": "genome_v2_e1_paired_deltas.json",
    "paired_csv": "genome_v2_e1_paired_deltas.csv",
    "decision": "genome_v2_e1_decision.json",
    "manifest": "genome_v2_e1_manifest.json",
}


class GenomeV2E1RunnerError(RuntimeError):
    """E1 inputs or immutable output publication are invalid."""


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenomeV2E1RunnerError(
            f"{label} is not readable JSON: {path}"
        ) from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authority_from_payload_v1(value: Any) -> E1AuthorityBindingV1:
    if not isinstance(value, Mapping):
        raise GenomeV2E1RunnerError("authority must be a JSON object")
    expected = {
        "provider_id",
        "runtime_profile_id",
        "source_commit",
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "execution_route_sha256s",
        "checkpoint_authority_bindings",
        "c1_preregistration_sha256",
        "c1_decision_sha256",
        "c1_decision",
        "campaign_plan_sha256s",
        "founder_payload_sha256s",
        "checkpoint_sha256s",
        "parity_receipt_sha256s",
        "reporting_parity_receipt_sha256s",
        "e1_preregistration_sha256",
    }
    if set(value) != expected:
        raise GenomeV2E1RunnerError("authority fields differ from E1 contract")
    try:
        return E1AuthorityBindingV1(**dict(value))
    except TypeError as exc:
        raise GenomeV2E1RunnerError("authority is malformed") from exc


def cells_from_payload_v1(value: Any) -> list[E1CellEvidenceV1]:
    if not isinstance(value, list):
        raise GenomeV2E1RunnerError("cell evidence must be a JSON array")
    fields = {
        "campaign_id",
        "arm",
        "ecology_seed",
        "task_seed",
        "authority_sha256",
        "campaign_plan_sha256",
        "execution_route_sha256",
        "founder_payload_id",
        "founder_payload_sha256",
        "checkpoint_sha256",
        "execution_receipt_sha256",
        "trajectory_sha256",
        "route_parity_receipt_sha256",
        "fresh_repeat_exact",
        "continuation_g16_exact",
        "no_fallback",
        "no_jit",
        "metrics",
    }
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != fields:
            raise GenomeV2E1RunnerError(
                f"cell evidence row {index} fields differ from E1 contract"
            )
        try:
            result.append(E1CellEvidenceV1(**dict(row)))
        except TypeError as exc:
            raise GenomeV2E1RunnerError(
                f"cell evidence row {index} is malformed"
            ) from exc
    return result


def validate_c1_decision_v1(
    value: Any,
    *,
    authority: E1AuthorityBindingV1,
) -> None:
    if not isinstance(value, Mapping):
        raise GenomeV2E1RunnerError("C1 decision must be a JSON object")
    supplied = value.get("decision_sha256")
    unsigned = {key: item for key, item in value.items() if key != "decision_sha256"}
    if (
        value.get("decision") != "C1-A"
        or value.get("preregistration_sha256")
        != authority.c1_preregistration_sha256
        or supplied != content_sha256(unsigned)
        or supplied != authority.c1_decision_sha256
    ):
        raise GenomeV2E1RunnerError(
            "C1 decision does not provide exact content-addressed C1-A admission"
        )


def materialize_e1_outputs_v1(
    *,
    authority: E1AuthorityBindingV1,
    cells: Sequence[E1CellEvidenceV1],
    c1_decision: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate E1 and atomically claim a new immutable output directory."""

    validate_c1_decision_v1(c1_decision, authority=authority)
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise GenomeV2E1RunnerError(
            "E1 output directory already exists; immutable overwrite refused"
        ) from exc
    decision = evaluate_e1_v1(cells, authority=authority)
    preregistration = preregistration_payload_v1()
    paired = {
        "schema_version": "genome_v2_p10_e1_paired_deltas_v1",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority_sha256": authority.authority_sha256(),
        "rows": decision["paired_deltas"],
    }
    _write_json(
        output_directory / OUTPUT_NAMES["preregistration"],
        preregistration,
    )
    _write_json(output_directory / OUTPUT_NAMES["paired_json"], paired)
    rows = list(decision["paired_deltas"])
    csv_path = output_directory / OUTPUT_NAMES["paired_csv"]
    if rows:
        with csv_path.open("x", encoding="ascii", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text(
            "campaign_id,ecology_seed\n",
            encoding="ascii",
        )
    decision_output = {
        **decision,
        "authority_sha256": authority.authority_sha256(),
    }
    unsigned_decision = {
        key: item
        for key, item in decision_output.items()
        if key != "decision_sha256"
    }
    decision_output["decision_sha256"] = content_sha256(unsigned_decision)
    _write_json(output_directory / OUTPUT_NAMES["decision"], decision_output)
    artifacts = {
        name: {
            "path": filename,
            "sha256": _sha256(output_directory / filename),
        }
        for name, filename in OUTPUT_NAMES.items()
        if name != "manifest"
    }
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority_sha256": authority.authority_sha256(),
        "decision": decision_output["decision"],
        "decision_sha256": decision_output["decision_sha256"],
        "cell_count": len(cells),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    _write_json(output_directory / OUTPUT_NAMES["manifest"], manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--c1-decision", type=Path, required=True)
    parser.add_argument("--cell-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    authority = authority_from_payload_v1(
        _read_json(args.authority, label="E1 authority")
    )
    c1_decision = _read_json(args.c1_decision, label="C1 decision")
    cells = cells_from_payload_v1(
        _read_json(args.cell_evidence, label="E1 cell evidence")
    )
    manifest = materialize_e1_outputs_v1(
        authority=authority,
        cells=cells,
        c1_decision=c1_decision,
        output_directory=args.output_directory,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
