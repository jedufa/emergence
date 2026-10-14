#!/usr/bin/env python3
"""Build Phase F v2 relationship evidence from completed battery artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from evolution.phase_f_v2.canonical import canonical_sha256  # noqa: E402
from evolution.phase_f_v2.matrix import load_phase_f_v2_matrix  # noqa: E402
from evolution.phase_f_v2.storage import atomic_write_canonical_json  # noqa: E402
from perf.eiel_phase_f_relationship_analysis_v2 import (  # noqa: E402
    DECISION_A,
    aggregate_phase_f_relationship_terminal_v2,
    bind_phase_f_accelerated_observation_v2,
    bind_phase_f_cpu_expectation_v2,
    build_phase_f_causal_relationships_v2,
    build_phase_f_checkpoint_relationships_v2,
    build_phase_f_repeat_relationships_v2,
)


CANONICAL_MATRIX = (
    ROOT
    / "_spec"
    / "evolutionary_capacity"
    / "phase_f"
    / "v2"
    / "forced_combination_matrix.json"
)


class PhaseFRelationshipCliError(ValueError):
    """An input artifact is missing, corrupt, or contradicts its manifest."""


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_path(raw: Any, *, owner: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PhaseFRelationshipCliError(f"{field} is missing")
    path = Path(raw)
    return path if path.is_absolute() else (owner.parent / path).resolve()


def _verified_record_file(
    row: dict[str, Any],
    *,
    owner: Path,
    path_field: str,
    file_sha_field: str,
) -> tuple[Path, Any]:
    path = _bound_path(row.get(path_field), owner=owner, field=path_field)
    if _file_sha256(path) != row.get(file_sha_field):
        raise PhaseFRelationshipCliError(f"{file_sha_field} mismatch: {path}")
    return path, _json(path)


def _load_cpu_bindings(
    manifest_path: str | Path,
) -> list[dict[str, Any]]:
    owner = Path(manifest_path).resolve()
    manifest = _json(owner)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("records"), list
    ):
        raise PhaseFRelationshipCliError("CPU manifest is malformed")
    unsigned = dict(manifest)
    observed_manifest_sha = unsigned.pop("manifest_sha256", None)
    if observed_manifest_sha != canonical_sha256(unsigned):
        raise PhaseFRelationshipCliError("CPU manifest self digest mismatch")
    bindings = []
    for raw in manifest["records"]:
        if not isinstance(raw, dict):
            raise PhaseFRelationshipCliError("CPU manifest record is malformed")
        _, expectation = _verified_record_file(
            raw,
            owner=owner,
            path_field="path",
            file_sha_field="file_sha256",
        )
        if expectation.get("receipt_sha256") != raw.get("receipt_sha256"):
            raise PhaseFRelationshipCliError(
                "CPU record contradicts expectation receipt"
            )
        binding = bind_phase_f_cpu_expectation_v2(expectation)
        for field in (
            "case_id",
            "expectation_kind",
            "fixture_sha256",
            "source_sha256",
            "expectation_sha256",
        ):
            if binding[field] != raw.get(field):
                raise PhaseFRelationshipCliError(
                    f"CPU manifest record binding mismatch: {field}"
                )
        bindings.append(binding)
    if len(bindings) != manifest.get("expectation_count"):
        raise PhaseFRelationshipCliError("CPU manifest count mismatch")
    return bindings


def _load_accelerated_bindings(
    ledger_path: str | Path,
) -> list[dict[str, Any]]:
    owner = Path(ledger_path).resolve()
    ledger = _json(owner)
    if not isinstance(ledger, dict) or not isinstance(ledger.get("completed"), list):
        raise PhaseFRelationshipCliError("battery ledger is malformed")
    unsigned = dict(ledger)
    observed_ledger_sha = unsigned.pop("ledger_sha256", None)
    if observed_ledger_sha != canonical_sha256(unsigned):
        raise PhaseFRelationshipCliError("battery ledger self digest mismatch")
    bindings = []
    for raw in ledger["completed"]:
        if not isinstance(raw, dict):
            raise PhaseFRelationshipCliError("battery ledger row is malformed")
        _, receipt = _verified_record_file(
            raw,
            owner=owner,
            path_field="receipt_path",
            file_sha_field="receipt_file_sha256",
        )
        _, observation = _verified_record_file(
            raw,
            owner=owner,
            path_field="observation_path",
            file_sha_field="observation_file_sha256",
        )
        if receipt.get("receipt_sha256") != raw.get("receipt_sha256"):
            raise PhaseFRelationshipCliError(
                "ledger row contradicts invocation receipt"
            )
        if canonical_sha256(observation) != raw.get("observed_sha256"):
            raise PhaseFRelationshipCliError(
                "ledger row contradicts persisted observation"
            )
        binding = bind_phase_f_accelerated_observation_v2(
            invocation_receipt=receipt,
            observation=observation,
        )
        for field in ("execution_id", "case_id", "invocation_kind"):
            if binding[field] != raw.get(field):
                raise PhaseFRelationshipCliError(
                    f"battery ledger binding mismatch: {field}"
                )
        bindings.append(binding)
    if len(bindings) != ledger.get("completed_count"):
        raise PhaseFRelationshipCliError("battery ledger completed count mismatch")
    return bindings


def _write_rows(
    root: Path,
    *,
    folder: str,
    rows: Sequence[dict[str, Any]],
    names: Sequence[str],
) -> None:
    if len(rows) != len(names):
        raise PhaseFRelationshipCliError("output row/name count mismatch")
    for row, name in zip(rows, names, strict=True):
        atomic_write_canonical_json(root / folder / f"{name}.json", row)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    matrix = load_phase_f_v2_matrix(args.matrix)
    fixture_registry = _json(args.fixture_registry)
    cpu = _load_cpu_bindings(args.cpu_manifest)
    accelerated = _load_accelerated_bindings(args.battery_ledger)
    repeats = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=accelerated
    )
    causal = build_phase_f_causal_relationships_v2(
        matrix=matrix,
        fixture_registry=fixture_registry,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    checkpoints = build_phase_f_checkpoint_relationships_v2(
        matrix=matrix,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    terminal = aggregate_phase_f_relationship_terminal_v2(
        matrix=matrix,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
        causal_relationships=causal,
        checkpoint_relationships=checkpoints,
        blocked=False,
    )

    output = Path(args.output_directory).resolve()
    _write_rows(
        output,
        folder="cpu_bindings",
        rows=cpu,
        names=[
            f"{row['case_id']}.{row['expectation_kind']}" for row in cpu
        ],
    )
    _write_rows(
        output,
        folder="accelerated_bindings",
        rows=accelerated,
        names=[str(row["execution_id"]) for row in accelerated],
    )
    _write_rows(
        output,
        folder="repeat_relationships",
        rows=repeats,
        names=[
            f"{row['case_id']}.{row['invocation_kind']}" for row in repeats
        ],
    )
    _write_rows(
        output,
        folder="causal_relationships",
        rows=causal,
        names=[str(row["case_id"]) for row in causal],
    )
    _write_rows(
        output,
        folder="checkpoint_relationships",
        rows=checkpoints,
        names=[str(row["case_id"]) for row in checkpoints],
    )
    atomic_write_canonical_json(output / "terminal_ledger.json", terminal)
    summary = {
        "schema_version": "eiel_phase_f_v2_relationship_analysis_summary_v1",
        "matrix_sha256": matrix["matrix_sha256"],
        "cpu_bindings": len(cpu),
        "accelerated_bindings": len(accelerated),
        "repeat_relationships": len(repeats),
        "causal_relationships": len(causal),
        "checkpoint_relationships": len(checkpoints),
        "decision": terminal["decision"],
        "terminal_receipt_sha256": terminal["receipt_sha256"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    atomic_write_canonical_json(output / "analysis_summary.json", summary)
    return {
        "ok": terminal["decision"] == DECISION_A,
        **summary,
        "output_directory": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=CANONICAL_MATRIX)
    parser.add_argument("--fixture-registry", type=Path, required=True)
    parser.add_argument("--cpu-manifest", type=Path, required=True)
    parser.add_argument("--battery-ledger", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
