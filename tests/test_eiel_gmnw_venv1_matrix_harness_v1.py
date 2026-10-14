from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tools.scripts import eiel_gmnw_venv1_matrix_contract_v1 as contract
from tools.scripts import eiel_gmnw_venv1_matrix_harness_v1 as subject


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _valid_result() -> dict[str, object]:
    cells = []
    for expected in contract.cells_v1():
        checkpoints = []
        for generation in contract.CHECKPOINT_GENERATIONS:
            decisions = generation * 10
            checkpoints.append(
                {
                    "generation": generation,
                    "alive_count": 64,
                    "decision_count": decisions,
                    "correct_count": generation * 8,
                    "incorrect_count": generation * 2,
                    "task_death_count": 0,
                    "authoritative_state_sha256": _digest(
                        f"{expected.cell_id}:{generation}"
                    ),
                }
            )
        cells.append(
            {
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
                "completed_generations": 100,
                "checkpoints": checkpoints,
            }
        )
    result: dict[str, object] = {
        "schema_version": contract.RESULT_SCHEMA_VERSION,
        "contract_sha256": contract.FROZEN_CONTRACT_SHA256,
        "run_label": contract.RUN_LABEL,
        "claim_boundary": contract.CLAIM_BOUNDARY,
        "status": "completed",
        "completed_generations": 100,
        "checkpoint_generations": [25, 50, 75, 100],
        "cells": cells,
    }
    result["result_sha256"] = subject.result_sha256_v1(result)
    return result


def _rehash(result: dict[str, object]) -> dict[str, object]:
    result["result_sha256"] = subject.result_sha256_v1(result)
    return result


def test_strict_cpu_result_validation_accepts_exact_ordered_envelope() -> None:
    result = _valid_result()

    validated = subject.validate_result_v1(result)

    assert len(validated["cells"]) == 72
    assert validated["cells"][0]["cell_id"] == "venv1_p1_gd_s101"
    assert validated["cells"][-1]["cell_id"] == "venv1_p3_fi_s606"
    assert validated["result_sha256"] == subject.result_sha256_v1(validated)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["cells"].reverse(),
            "identity differs from the contract",
        ),
        (
            lambda value: value["cells"][0].__setitem__("profile_code", 2),
            "identity differs from the contract",
        ),
        (
            lambda value: value["cells"][0].__setitem__("fork_reseeded", True),
            "identity differs from the contract",
        ),
        (
            lambda value: value["cells"][0]["checkpoints"].pop(1),
            "checkpoints must contain",
        ),
        (
            lambda value: value["cells"][0]["checkpoints"][0].__setitem__(
                "generation", 50
            ),
            "generation is unordered",
        ),
        (
            lambda value: value["cells"][0]["checkpoints"][0].__setitem__(
                "correct_count", 1
            ),
            "decision counts do not balance",
        ),
    ],
)
def test_strict_result_validation_rejects_semantic_drift(
    mutate,
    message: str,
) -> None:
    changed = copy.deepcopy(_valid_result())
    mutate(changed)
    _rehash(changed)

    with pytest.raises(subject.Venv1MatrixHarnessError, match=message):
        subject.validate_result_v1(changed)


def test_result_validation_rejects_unknown_fields_and_hash_tampering() -> None:
    extra = copy.deepcopy(_valid_result())
    extra["cells"][0]["device"] = "cuda"
    _rehash(extra)
    with pytest.raises(subject.Venv1MatrixHarnessError, match="must contain exactly"):
        subject.validate_result_v1(extra)

    changed = copy.deepcopy(_valid_result())
    changed["cells"][0]["checkpoints"][0]["alive_count"] = 63
    with pytest.raises(subject.Venv1MatrixHarnessError, match="result_sha256 mismatch"):
        subject.validate_result_v1(changed)


def test_cli_validates_contract_and_has_no_execution_mode(capsys) -> None:
    assert subject.main([]) == 0
    assert capsys.readouterr().out.strip() == contract.FROZEN_CONTRACT_SHA256

    with pytest.raises(SystemExit):
        subject.build_parser().parse_args(["--execute"])


def test_cli_validates_one_ascii_json_result(tmp_path, capsys) -> None:
    result = _valid_result()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result, allow_nan=False), encoding="ascii")

    assert subject.main(["--validate-result", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result_sha256"] == result["result_sha256"]
