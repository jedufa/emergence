from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from engine.perf.cuda_full_evolution_v2.workhorse.authority import (
    FullEvolutionWorkhorseAuthorityError,
    _historical_source_path_matches_relative_v2,
    validate_authoritative_micro_denominator_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R11_RECEIPT = (
    PROJECT_ROOT
    / "_artifacts/tw-cuda-full-evolution-reform-20260825"
    / "direct-sparse-autotune-r11-sustained5-r1"
    / "autotune-result.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _r11_value() -> dict[str, object]:
    value = json.loads(R11_RECEIPT.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _write_sealed(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    path.chmod(0o400)


def _validate(path: Path) -> dict[str, object]:
    return validate_authoritative_micro_denominator_v2(
        path,
        expected_device_identity=_r11_value()["device_identity"],
    )


def test_historical_source_path_binding_is_checkout_independent_and_exact() -> None:
    relative = "engine/perf/cuda_full_evolution_v2/direct_sparse_autotune.py"

    assert _historical_source_path_matches_relative_v2(
        "/old/checkout/engine/perf/cuda_full_evolution_v2/direct_sparse_autotune.py",
        relative,
    )
    assert not _historical_source_path_matches_relative_v2(relative, relative)
    assert not _historical_source_path_matches_relative_v2(
        "/old/checkout/engine/perf/cuda_full_evolution_v2/direct_sparse_runtime.py",
        relative,
    )
    assert not _historical_source_path_matches_relative_v2(
        "/old/checkout/../forged/engine/perf/cuda_full_evolution_v2/"
        "direct_sparse_autotune.py",
        relative,
    )


def test_r11_direct_sparse_v3_reopens_as_historical_technical_denominator() -> None:
    binding = _validate(R11_RECEIPT)

    assert binding["schema_version"] == "eiel_workhorse_micro_denominator_binding_v2"
    assert binding["claim_class"] == "TECHNICAL_ONLY"
    assert binding["evidence_role"] == "historical_immutable_micro_denominator_only"
    assert binding["execution_authority_granted"] is False
    assert binding["qualification_authority_granted"] is False
    assert binding["science_authority_granted"] is False
    assert binding["current_execution_compatibility_granted"] is False
    assert binding["sites"] == 1024
    assert binding["initial_population"] == 256
    assert binding["capacity"] == 512
    assert binding["p10_world_ticks_per_second"] == pytest.approx(
        476_945.2092053356
    )
    assert (
        binding["historical_benchmark_identity"]["source_identity"]["identity_role"]
        == "immutable_benchmark_build_identity"
    )
    assert (
        binding["current_source_identity"]["identity_role"]
        == "informational_current_checkout_identity_no_execution_authority"
    )


def test_r11_denominator_rejects_mutable_receipt(tmp_path: Path) -> None:
    mutable = tmp_path / "mutable.json"
    shutil.copyfile(R11_RECEIPT, mutable)
    mutable.chmod(0o600)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="read-only"):
        _validate(mutable)


def test_r11_denominator_rejects_symlink_receipt(tmp_path: Path) -> None:
    link = tmp_path / "receipt.json"
    link.symlink_to(R11_RECEIPT)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="cannot reopen"):
        _validate(link)


def test_r11_denominator_rejects_hardlink_receipt(tmp_path: Path) -> None:
    copied = tmp_path / "copied.json"
    linked = tmp_path / "linked.json"
    shutil.copyfile(R11_RECEIPT, copied)
    copied.chmod(0o400)
    os.link(copied, linked)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="single-link"):
        _validate(copied)


def test_r11_denominator_rejects_unsealed_receipt(tmp_path: Path) -> None:
    value = _r11_value()
    value.pop("result_sha256")
    unsealed = tmp_path / "unsealed.json"
    _write_sealed(unsealed, value)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="receipt or plan seal"):
        _validate(unsealed)


def test_r11_denominator_rejects_tampered_receipt(tmp_path: Path) -> None:
    value = _r11_value()
    value["selected_p10_world_ticks_per_second"] = 1e30
    tampered = tmp_path / "tampered.json"
    _write_sealed(tampered, value)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="receipt or plan seal"):
        _validate(tampered)


def test_r11_denominator_rejects_cross_shape_resealed_receipt(tmp_path: Path) -> None:
    value = _r11_value()
    plan = value["plan"]
    plan["candidates"][0]["initial_capacity_rows_per_world"] = 1024
    plan_body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = _canonical_sha256(plan_body)
    value["measurements"][0]["candidate"]["initial_capacity_rows_per_world"] = 1024
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    value["result_sha256"] = _canonical_sha256(body)
    cross_shape = tmp_path / "cross-shape.json"
    _write_sealed(cross_shape, value)

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="W32/P256/C512"):
        _validate(cross_shape)
