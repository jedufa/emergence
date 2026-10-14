from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scripts.eiel_campaign_receipt_parser import (
    CampaignReceiptParseError,
    INTERPRETATION_SCHEMA_VERSION,
    build_interpretation_receipt,
    parse_campaign_execution_receipt,
)


def _receipt(campaign_id: str = "eed-b1-science-impact_on-g100") -> dict:
    return {
        "schema_version": "eiel_campaign_execution_receipt_v1",
        "campaign_id": campaign_id,
        "counts": {
            "completed": 18,
            "failed": 0,
            "interrupted": 0,
            "pending": 0,
            "running": 0,
        },
        "execution_route_sha256": "a" * 64,
        "ok": True,
        "plan_sha256": "b" * 64,
        "profile_id": "eiel-production-runtime-v1",
        "provider_id": "eiel-production-provider-v1",
        "reporting_parity_receipt_path": "/tmp/reporting.json",
        "reporting_parity_status": "reporting_parity_proven",
        "status": "completed",
    }


def _json(payload: dict) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _completed_run_root(tmp_path: Path, *, completed: bool = True) -> Path:
    root = tmp_path / "run"
    for idx in range(2):
        batch = root / "batches" / f"batch-{idx:04d}"
        batch.mkdir(parents=True)
        (batch / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed" if completed or idx == 0 else "running",
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
    return root


def test_parse_pure_json_receipt() -> None:
    parsed = parse_campaign_execution_receipt(
        _json(_receipt()), expected_campaign_id="eed-b1-science-impact_on-g100"
    )

    assert parsed.byte_start == 0
    assert parsed.receipt["status"] == "completed"


def test_parse_receipt_with_log_preamble_and_trailing_logs() -> None:
    stdout = (
        b"[gen 0] post-cull injury mean=0.000 max=0.000\n"
        + _json(_receipt())
        + b"\ntrailing diagnostic line\n"
    )

    parsed = parse_campaign_execution_receipt(
        stdout, expected_campaign_id="eed-b1-science-impact_on-g100"
    )

    assert parsed.byte_start > 0
    assert parsed.byte_end < len(stdout)
    assert parsed.receipt["reporting_parity_status"] == "reporting_parity_proven"


def test_parse_rejects_multiple_valid_receipts() -> None:
    stdout = _json(_receipt()) + b"\n" + _json(_receipt())

    with pytest.raises(CampaignReceiptParseError, match="ambiguous"):
        parse_campaign_execution_receipt(
            stdout, expected_campaign_id="eed-b1-science-impact_on-g100"
        )


def test_parse_rejects_malformed_json_without_valid_receipt() -> None:
    with pytest.raises(CampaignReceiptParseError, match="does not contain"):
        parse_campaign_execution_receipt(
            b"log\n{\"schema_version\": ", expected_campaign_id="eed-b1-science-impact_on-g100"
        )


def test_parse_rejects_wrong_campaign_id() -> None:
    with pytest.raises(CampaignReceiptParseError, match="does not contain"):
        parse_campaign_execution_receipt(
            _json(_receipt("wrong-campaign")),
            expected_campaign_id="eed-b1-science-impact_on-g100",
        )


def test_interpretation_receipt_cross_checks_completed_manifests(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_bytes(b"diagnostic\n" + _json(_receipt()))
    run_root = _completed_run_root(tmp_path)

    receipt = build_interpretation_receipt(
        stdout_path=stdout,
        stderr_path=None,
        run_root=run_root,
        expected_campaign_id="eed-b1-science-impact_on-g100",
        command_returncode=0,
    )

    assert receipt["schema_version"] == INTERPRETATION_SCHEMA_VERSION
    assert receipt["manifest_completion"]["manifest_count"] == 2
    assert receipt["manifest_completion"]["all_completed"] is True
    assert receipt["receipt_sha256"]


def test_interpretation_receipt_rejects_incomplete_manifest(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_bytes(_json(_receipt()))
    run_root = _completed_run_root(tmp_path, completed=False)

    with pytest.raises(CampaignReceiptParseError, match="manifest completion"):
        build_interpretation_receipt(
            stdout_path=stdout,
            stderr_path=None,
            run_root=run_root,
            expected_campaign_id="eed-b1-science-impact_on-g100",
            command_returncode=0,
        )


def test_interpretation_receipt_rejects_nonzero_return_code(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_bytes(_json(_receipt()))
    run_root = _completed_run_root(tmp_path)

    with pytest.raises(CampaignReceiptParseError, match="return code"):
        build_interpretation_receipt(
            stdout_path=stdout,
            stderr_path=None,
            run_root=run_root,
            expected_campaign_id="eed-b1-science-impact_on-g100",
            command_returncode=1,
        )
