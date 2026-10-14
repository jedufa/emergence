from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from native_runtime.promotion_certificate import (
    checkpoint_qualification_claim,
    generate_promotion_certificate,
    load_promotion_certificate,
    supported_envelope_claim,
    write_promotion_certificate,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "scripts" / "eiel_flat_native_promotion_certificate.py"
CERTIFICATE_FILENAME = "flat_native_promotion_certificate_v1.json"


def _fields() -> dict[str, Any]:
    hashes = iter(character * 64 for character in "3456789abcdef")
    return {
        "supported_envelope": supported_envelope_claim(),
        "candidate_release_commit_sha": "1" * 40,
        "extension_sha256": "2" * 64,
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "reporting_schema_version": "eiel_flat_native_reporting_v1",
        "checkpoint_qualification": checkpoint_qualification_claim(),
        "parity_evidence_version": "flat-native-cli-test-1",
        "g2_result_sha256": next(hashes),
        "g4_result_sha256": next(hashes),
        "g8_result_sha256": next(hashes),
        "checkpoint_result_sha256": next(hashes),
        "long_horizon_a_result_sha256": next(hashes),
        "long_horizon_b_result_sha256": next(hashes),
        "long_horizon_c_result_sha256": next(hashes),
        "long_horizon_d_result_sha256": next(hashes),
        "full_regression_sha256": next(hashes),
        "output_fairness_result_sha256": next(hashes),
        "output_fairness_ratio": 1.25,
        "benchmark_result_sha256": next(hashes),
        "current_reference_w1_median_wall_s": 15.0,
        "current_native_w1_median_wall_s": 10.0,
        "w1_native_speedup": 1.5,
        "current_reference_w6_median_wall_s": 11.5,
        "current_native_w6_median_wall_s": 10.0,
        "w6_native_speedup": 1.15,
        "measured_arm_coefficients_of_variation": {
            "current_w1_reference": 0.05,
            "current_w1_native": 0.04,
            "current_w6_reference": 0.03,
            "current_w6_native": 0.02,
        },
        "peak_memory_result": {
            "current_reference_peak_rss_bytes": 100,
            "current_native_peak_rss_bytes": 125,
            "native_to_reference_ratio": 1.25,
            "bounded_exception": None,
        },
        "hidden_fallback_false": True,
        "mixed_ownership_false": True,
        "runtime_jit_false": True,
        "decision_timestamp": "2026-07-15T20:00:00Z",
    }


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_generate_writes_fixed_canonical_artifacts_and_validate_accepts_them(
    tmp_path: Path,
) -> None:
    fields_path = tmp_path / "fields.json"
    output_dir = tmp_path / "certificate"
    fields_path.write_text(json.dumps(_fields()), encoding="utf-8")

    generated = _run_cli(
        "generate",
        "--fields-json",
        str(fields_path),
        "--output-dir",
        str(output_dir),
    )

    assert generated.returncode == 0, generated.stdout + generated.stderr
    summary = json.loads(generated.stdout)
    certificate_path = output_dir / CERTIFICATE_FILENAME
    sidecar_path = Path(f"{certificate_path}.sha256")
    assert summary["status"] == "passed"
    assert summary["certificate_path"] == str(certificate_path.resolve())
    assert summary["sha256_path"] == str(sidecar_path.resolve())
    assert certificate_path.read_bytes().endswith(b"\n")
    assert b"\n" not in certificate_path.read_bytes()[:-1]
    assert sidecar_path.read_text(encoding="ascii") == (
        f"{summary['sha256']}  {CERTIFICATE_FILENAME}\n"
    )
    assert load_promotion_certificate(certificate_path)["verdict"]["eligible"] is True
    assert not tuple(output_dir.glob(".*.tmp"))

    validated = _run_cli(
        "validate",
        "--certificate-json",
        str(certificate_path),
    )

    assert validated.returncode == 0, validated.stdout + validated.stderr
    assert json.loads(validated.stdout)["status"] == "passed"


def test_generate_returns_rejected_without_writing_artifacts(tmp_path: Path) -> None:
    fields = _fields()
    fields["current_reference_w1_median_wall_s"] = 14.9
    fields["w1_native_speedup"] = 1.49
    fields_path = tmp_path / "rejected-fields.json"
    output_dir = tmp_path / "certificate"
    fields_path.write_text(json.dumps(fields), encoding="utf-8")

    completed = _run_cli(
        "generate",
        "--fields-json",
        str(fields_path),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert summary["status"] == "rejected"
    assert summary["decision"] == "reject"
    assert summary["reasons"] == ["w1_performance", "benchmark_qualified"]
    assert not (output_dir / CERTIFICATE_FILENAME).exists()
    assert not Path(f"{output_dir / CERTIFICATE_FILENAME}.sha256").exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"field":1,"field":2}', "duplicate JSON field"),
        ('{"field":NaN}', "non-finite JSON number"),
    ],
)
def test_generate_strict_json_parser_rejects_ambiguous_values(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    fields_path = tmp_path / "invalid-fields.json"
    output_dir = tmp_path / "certificate"
    fields_path.write_text(contents, encoding="utf-8")

    completed = _run_cli(
        "generate",
        "--fields-json",
        str(fields_path),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode == 2
    summary = json.loads(completed.stdout)
    assert summary["status"] == "invalid"
    assert message in summary["error"]
    assert not output_dir.exists()


def test_validate_returns_rejected_for_a_canonical_reject_certificate(
    tmp_path: Path,
) -> None:
    fields = _fields()
    fields["current_reference_w1_median_wall_s"] = 14.9
    fields["w1_native_speedup"] = 1.49
    certificate_path = tmp_path / CERTIFICATE_FILENAME
    write_promotion_certificate(
        certificate_path,
        generate_promotion_certificate(fields),
    )

    completed = _run_cli(
        "validate",
        "--certificate-json",
        str(certificate_path),
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert summary["status"] == "rejected"
    assert summary["decision"] == "reject"


@pytest.mark.parametrize("tamper", ["certificate", "sidecar"])
def test_validate_fails_nonzero_on_tampered_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    certificate_path = tmp_path / CERTIFICATE_FILENAME
    write_promotion_certificate(
        certificate_path,
        generate_promotion_certificate(_fields()),
    )
    sidecar_path = Path(f"{certificate_path}.sha256")
    if tamper == "certificate":
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        certificate["decision_timestamp"] = "2026-07-15T20:00:01Z"
        certificate_path.write_text(
            json.dumps(
                certificate,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        sidecar_path.write_text(
            f"{'0' * 64}  {CERTIFICATE_FILENAME}\n",
            encoding="ascii",
        )

    completed = _run_cli(
        "validate",
        "--certificate-json",
        str(certificate_path),
    )

    assert completed.returncode == 2
    summary = json.loads(completed.stdout)
    assert summary["status"] == "invalid"
    assert "SHA-256 mismatch" in summary["error"]
