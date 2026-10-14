#!/usr/bin/env python3
"""Validate GMNW VENV-1 matrix contracts and result envelopes on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scripts import eiel_gmnw_venv1_matrix_contract_v1 as contract  # noqa: E402


SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "run_label",
        "claim_boundary",
        "status",
        "completed_generations",
        "checkpoint_generations",
        "cells",
        "result_sha256",
    }
)
CELL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ordinal",
        "cell_id",
        "profile_id",
        "profile_code",
        "arm_id",
        "policy_arm",
        "context",
        "cue_visible_at_decision",
        "source_seed",
        "source_group",
        "fork_reseeded",
        "status",
        "completed_generations",
        "checkpoints",
    }
)
CHECKPOINT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "generation",
        "alive_count",
        "decision_count",
        "correct_count",
        "incorrect_count",
        "task_death_count",
        "authoritative_state_sha256",
    }
)


class Venv1MatrixHarnessError(RuntimeError):
    """A VENV-1 result is incomplete, unordered, or contradictory."""


def _exact_object(value: object, keys: frozenset[str], *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Venv1MatrixHarnessError(f"{path} must contain exactly {sorted(keys)}")
    return dict(value)


def _exact_int(value: object, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Venv1MatrixHarnessError(
            f"{path} must be an exact integer >= {minimum}"
        )
    return value


def _exact_sha256(value: object, *, path: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise Venv1MatrixHarnessError(f"{path} must be a lowercase SHA-256")
    return value


def result_sha256_v1(value: Mapping[str, Any]) -> str:
    material = dict(value)
    material.pop("result_sha256", None)
    return contract.canonical_sha256_v1(material)


def _validate_checkpoint_v1(
    value: object,
    *,
    path: str,
    expected_generation: int,
) -> dict[str, Any]:
    row = _exact_object(value, CHECKPOINT_KEYS, path=path)
    if row["generation"] != expected_generation:
        raise Venv1MatrixHarnessError(f"{path}.generation is unordered")
    alive = _exact_int(row["alive_count"], path=f"{path}.alive_count")
    if alive > contract.INITIAL_POPULATION:
        raise Venv1MatrixHarnessError(f"{path}.alive_count exceeds capacity")
    decisions = _exact_int(row["decision_count"], path=f"{path}.decision_count")
    correct = _exact_int(row["correct_count"], path=f"{path}.correct_count")
    incorrect = _exact_int(row["incorrect_count"], path=f"{path}.incorrect_count")
    task_deaths = _exact_int(
        row["task_death_count"], path=f"{path}.task_death_count"
    )
    if correct + incorrect != decisions:
        raise Venv1MatrixHarnessError(f"{path} decision counts do not balance")
    if task_deaths > incorrect:
        raise Venv1MatrixHarnessError(f"{path}.task_death_count exceeds incorrect_count")
    _exact_sha256(
        row["authoritative_state_sha256"],
        path=f"{path}.authoritative_state_sha256",
    )
    return row


def _validate_cell_v1(
    value: object,
    *,
    expected: contract.Venv1CellV1,
) -> dict[str, Any]:
    path = f"cells[{expected.ordinal}]"
    row = _exact_object(value, CELL_KEYS, path=path)
    frozen_identity = {
        "ordinal": expected.ordinal,
        "cell_id": expected.cell_id,
        "profile_id": expected.profile_id,
        "profile_code": expected.profile_code,
        "arm_id": expected.arm_id,
        "policy_arm": expected.policy_arm,
        "context": expected.context,
        "cue_visible_at_decision": expected.cue_visible_at_decision,
        "source_seed": expected.source_seed,
        "source_group": expected.source_group,
        "fork_reseeded": False,
        "status": "completed",
        "completed_generations": contract.GENERATIONS,
    }
    observed_identity = {name: row[name] for name in frozen_identity}
    if observed_identity != frozen_identity:
        raise Venv1MatrixHarnessError(f"{path} identity differs from the contract")
    checkpoints = row["checkpoints"]
    if (
        not isinstance(checkpoints, Sequence)
        or isinstance(checkpoints, (str, bytes, bytearray))
        or len(checkpoints) != len(contract.CHECKPOINT_GENERATIONS)
    ):
        raise Venv1MatrixHarnessError(f"{path}.checkpoints must contain 25/50/75/100")
    validated = tuple(
        _validate_checkpoint_v1(
            checkpoint,
            path=f"{path}.checkpoints[{index}]",
            expected_generation=generation,
        )
        for index, (checkpoint, generation) in enumerate(
            zip(checkpoints, contract.CHECKPOINT_GENERATIONS, strict=True)
        )
    )
    for earlier, later in zip(validated, validated[1:]):
        for name in (
            "decision_count",
            "correct_count",
            "incorrect_count",
            "task_death_count",
        ):
            if later[name] < earlier[name]:
                raise Venv1MatrixHarnessError(f"{path}.{name} regressed at a checkpoint")
    return row


def validate_result_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact_object(value, TOP_LEVEL_KEYS, path="result")
    expected_contract = contract.validate_contract_v1()
    expected_header = {
        "schema_version": contract.RESULT_SCHEMA_VERSION,
        "contract_sha256": expected_contract["contract_sha256"],
        "run_label": contract.RUN_LABEL,
        "claim_boundary": contract.CLAIM_BOUNDARY,
        "status": "completed",
        "completed_generations": contract.GENERATIONS,
        "checkpoint_generations": list(contract.CHECKPOINT_GENERATIONS),
    }
    if {name: row[name] for name in expected_header} != expected_header:
        raise Venv1MatrixHarnessError("result header differs from the frozen contract")
    cells = row["cells"]
    if (
        not isinstance(cells, Sequence)
        or isinstance(cells, (str, bytes, bytearray))
        or len(cells) != contract.PHYSICAL_CELLS
    ):
        raise Venv1MatrixHarnessError("result must contain exactly 72 ordered cells")
    validated_cells = [
        _validate_cell_v1(candidate, expected=expected)
        for candidate, expected in zip(cells, contract.cells_v1(), strict=True)
    ]
    actual_hash = _exact_sha256(row["result_sha256"], path="result.result_sha256")
    if actual_hash != result_sha256_v1(row):
        raise Venv1MatrixHarnessError("result_sha256 mismatch")
    return {**row, "cells": validated_cells}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--print-contract",
        action="store_true",
        help="print the frozen device-neutral contract",
    )
    group.add_argument(
        "--validate-result",
        type=Path,
        metavar="PATH",
        help="validate one completed JSON result envelope",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_result is not None:
        try:
            payload = json.loads(args.validate_result.read_text(encoding="ascii"))
            validated = validate_result_v1(payload)
        except (OSError, TypeError, ValueError, Venv1MatrixHarnessError) as exc:
            print(f"VENV-1 validation failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(validated, allow_nan=False, sort_keys=True))
        return 0
    validated_contract = contract.validate_contract_v1()
    if args.print_contract:
        print(json.dumps(validated_contract, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(validated_contract["contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
