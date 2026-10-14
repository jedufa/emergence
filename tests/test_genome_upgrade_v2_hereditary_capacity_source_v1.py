from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256, file_sha256
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)
from tools.scripts import genome_upgrade_v2_hereditary_capacity_source_v1 as source


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _write_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True).encode("ascii") + b"\n"
    path.write_bytes(gzip.compress(payload, mtime=0))


def _authority() -> dict[str, str]:
    return {
        "provider_id": source.PROVIDER_ID,
        "runtime_profile_id": source.PROFILE_ID,
        "source_commit": "1" * 40,
        "package_id_sha256": "",
        "package_receipt_sha256": "",
        "artifact_sha256": "",
        "execution_route_sha256": "5" * 64,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    run_root = tmp_path / "run"
    artifacts = tmp_path / "package"
    artifact = artifacts / "artifact.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-cuda-artifact")
    authority = _authority()
    authority["artifact_sha256"] = file_sha256(artifact)
    package_path = artifacts / "package.json"
    package = {
        "schema_version": "genome_v2_executable_package_identity_v1",
        "authority_state": "package_candidate_complete",
        "ok": True,
        "identity": {
            "provider_id": source.PROVIDER_ID,
            "runtime_profile_id": source.PROFILE_ID,
            "source_commit": authority["source_commit"],
            "cuda_artifact_sha256": authority["artifact_sha256"],
        },
    }
    package["package_id_sha256"] = package_identity_sha256_v1(package)
    _write(package_path, package)
    authority["package_id_sha256"] = package["package_id_sha256"]
    authority["package_receipt_sha256"] = file_sha256(package_path)
    authority_path = tmp_path / "authority.json"
    _write(authority_path, authority)

    cells = []
    for ordinal, ecology_seed in enumerate(source.EXPECTED_ECOLOGY_SEEDS):
        cells.append(
            {
                "ordinal": ordinal,
                "cell_id": f"cell-{ordinal}",
                "branch_id": f"branch-{ordinal}",
                "rng_lineage_id": f"rng-{ordinal}",
                "dimensions": {
                    "ecology_seed": ecology_seed,
                    "task_seed": source.EXPECTED_TASK_SEED,
                },
                "parameters": dict(source.EXPECTED_PARAMETERS),
                "seeds": {
                    "ecology": ecology_seed,
                    "task": source.EXPECTED_TASK_SEED,
                },
            }
        )
    artifact_hashes = {
        "cuda_extension": authority["artifact_sha256"],
        "genome_upgrade_v2_package_receipt": authority[
            "package_receipt_sha256"
        ],
    }
    plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": source.CAMPAIGN_ID,
        "description": "frozen hereditary source fixture",
        "no_claims": "bounded fixture",
        "resolved_profile": source.PROFILE_ID,
        "qualification_key": {
            "schema_version": "eiel_runtime_qualification_key_v1",
            "profile_id": source.PROFILE_ID,
            "backend": "cuda_resident",
            "admitted_batch_range": [1, 2],
            "fixed_invariants": {
                **source.EXPECTED_INVARIANTS,
                "mechanics_configuration": "patch::fixed_mlp",
            },
            "artifact_hashes": artifact_hashes,
        },
        "qualification_key_sha256": content_sha256(
            {"fixture": "qualification"}
        ),
        "execution": {
            "generations": 3,
            "ticks_per_generation": 80,
            "checkpoints": [0, 3],
            "digest_every_generation": True,
            "checkpoint_storage_mode": "disk",
            "budgets": {
                "vram_bytes": 2_000_000_000,
                "disk_bytes": 10_000_000_000,
                "wall_seconds": 86_400,
            },
            "technical_gate": {"required": False},
        },
        "cells": cells,
        "physical_batches": [
            {
                "ordinal": 0,
                "batch_id": "batch-0000",
                "cell_ids": [cell["cell_id"] for cell in cells],
            }
        ],
        "comparisons": {
            "branches": [
                {
                    "kind": "reporting_parity",
                    "source_cell_id": cells[0]["cell_id"],
                }
            ],
            "reporting_parity": {"required": True},
        },
        "analysis": {},
        "artifacts": {
            "cuda_extension": {
                "path": str(artifact),
                "sha256": authority["artifact_sha256"],
                "kind": "cuda_extension",
                "abi": "test",
            },
            "genome_upgrade_v2_package_receipt": {
                "path": str(package_path),
                "sha256": authority["package_receipt_sha256"],
                "kind": "genome_upgrade_v2_package_receipt",
                "abi": "test",
            },
        },
        "output": {
            "runtime_root": str(run_root),
            "reporting_profile": "compact",
        },
        "resource_estimate": {},
    }
    plan["plan_sha256"] = content_sha256(plan)
    plan_path = run_root / "execution_plan.json"
    _write(plan_path, plan)

    batch_path = run_root / "batches/batch-0000/manifest.json"
    catalog = {}
    for generation in source.EXPECTED_CHECKPOINTS:
        rows = []
        terminal_digests = {}
        for cell in cells:
            bundle = (
                batch_path.parent
                / "checkpoint_bundles_v2"
                / cell["cell_id"]
                / f"bundle_gen{generation:06d}"
            )
            sidecar = bundle / source.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
            _write_gzip_json(
                sidecar,
                {
                    "schema_version": (
                        "native_runtime_checkpoint_projection_v1"
                    ),
                    "status": "present",
                    "generation_index": generation,
                },
            )
            projection_sha256 = content_sha256(
                {
                    "generation": generation,
                    "cell_id": cell["cell_id"],
                    "projection": "exact",
                }
            )
            terminal_digests[cell["cell_id"]] = projection_sha256
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "path": str(bundle),
                    "source_projection_sha256": projection_sha256,
                }
            )
        catalog[str(generation)] = {
            "generation": generation,
            "execution_route_sha256": authority[
                "execution_route_sha256"
            ],
            "checkpoint_binding": {
                "provider_id": source.PROVIDER_ID,
                "profile_id": source.PROFILE_ID,
                "plan_sha256": plan["plan_sha256"],
                "batch_id": "batch-0000",
                "cell_ids": [cell["cell_id"] for cell in cells],
                "execution_route_sha256": authority[
                    "execution_route_sha256"
                ],
                "artifact_sha256s": {
                    "cuda_extension": authority["artifact_sha256"],
                    "genome_upgrade_v2_package_receipt": authority[
                        "package_receipt_sha256"
                    ],
                },
            },
            "cell_bundles": rows,
            "terminal_cell_digests": terminal_digests,
        }
    batch = {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": source.CAMPAIGN_ID,
        "plan_sha256": plan["plan_sha256"],
        "profile_id": source.PROFILE_ID,
        "provider_id": source.PROVIDER_ID,
        "execution_route_sha256": authority["execution_route_sha256"],
        "batch_id": "batch-0000",
        "cell_ids": [cell["cell_id"] for cell in cells],
        "status": "completed",
        "error": None,
        "result": {
            "execution_device": "cuda",
            "fallback_used": False,
            "native_execution_receipts": [
                {
                    "execution_device": "cuda",
                    "explicit_native_no_fallback": True,
                    "hidden_fallback": False,
                    "runtime_jit": False,
                    "start_generation": 0,
                    "end_generation": 3,
                }
            ],
            "checkpoint_catalog": catalog,
        },
    }
    _write(batch_path, batch)
    campaign_path = run_root / "campaign_manifest.json"
    _write(
        campaign_path,
        {
            "schema_version": "eiel_campaign_execution_manifest_v1",
            "campaign_id": source.CAMPAIGN_ID,
            "plan_sha256": plan["plan_sha256"],
            "profile_id": source.PROFILE_ID,
            "provider_id": source.PROVIDER_ID,
            "execution_route_sha256": authority[
                "execution_route_sha256"
            ],
            "status": "completed",
            "batches": [
                {
                    "batch_id": "batch-0000",
                    "status": "completed",
                    "manifest_path": str(batch_path),
                }
            ],
        },
    )
    return {
        "authority": authority_path,
        "plan": plan_path,
        "campaign": campaign_path,
        "batch": batch_path,
        "output": tmp_path / "sources",
    }


def _build(paths: dict[str, Path]) -> dict:
    return source.build_sources_v1(
        authority_path=paths["authority"],
        execution_plan_path=paths["plan"],
        campaign_manifest_path=paths["campaign"],
        output_directory=paths["output"],
    )


def test_builder_emits_two_content_bound_repeat_sources(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    manifest = _build(paths)

    assert [row["ecology_seed"] for row in manifest["sources"]] == [101, 202]
    assert manifest["batch_manifest_sha256"] == file_sha256(paths["batch"])
    for repeat in (0, 1):
        source_path = paths["output"] / f"repeat-{repeat}.json"
        payload = json.loads(source_path.read_text(encoding="ascii"))
        receipt = payload["execution_receipt"]
        assert payload["repeat_index"] == repeat
        assert [row["generation"] for row in payload["observations"]] == [0, 3]
        assert receipt["execution_device"] == "cuda"
        assert receipt["batch_manifest_sha256"] == file_sha256(paths["batch"])
        unsigned = dict(receipt)
        supplied = unsigned.pop("receipt_sha256")
        assert supplied == content_sha256(unsigned)
        assert manifest["sources"][repeat]["sha256"] == file_sha256(source_path)
    assert json.loads(
        (paths["output"] / "source_manifest.json").read_text(encoding="ascii")
    ) == manifest
    with pytest.raises(FileExistsError):
        _build(paths)


def test_builder_rejects_plan_tamper_and_fallback(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="ascii"))
    plan["execution"]["ticks_per_generation"] = 81
    _write(paths["plan"], plan)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="canonical content hash",
    ):
        _build(paths)

    paths = _fixture(tmp_path / "fallback")
    batch = json.loads(paths["batch"].read_text(encoding="ascii"))
    batch["result"]["fallback_used"] = True
    _write(paths["batch"], batch)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="CUDA no-fallback",
    ):
        _build(paths)

    paths = _fixture(tmp_path / "checkpoint-tamper")
    batch = json.loads(paths["batch"].read_text(encoding="ascii"))
    batch["result"]["checkpoint_catalog"]["3"]["checkpoint_binding"][
        "plan_sha256"
    ] = "9" * 64
    _write(paths["batch"], batch)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="checkpoint binding differs",
    ):
        _build(paths)

    paths = _fixture(tmp_path / "projection-tamper")
    batch = json.loads(paths["batch"].read_text(encoding="ascii"))
    batch["result"]["checkpoint_catalog"]["3"]["cell_bundles"][0][
        "source_projection_sha256"
    ] = "8" * 64
    _write(paths["batch"], batch)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="source projection digest differs",
    ):
        _build(paths)


def test_builder_rejects_checkpoint_path_escape_and_symlink(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    batch = json.loads(paths["batch"].read_text(encoding="ascii"))
    escaped = tmp_path / "escaped"
    _write(
        escaped / "native_runtime_checkpoint_projection.json",
        {"status": "present"},
    )
    batch["result"]["checkpoint_catalog"]["3"]["cell_bundles"][0][
        "path"
    ] = str(escaped)
    _write(paths["batch"], batch)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="escapes expected",
    ):
        _build(paths)

    paths = _fixture(tmp_path / "symlink")
    batch = json.loads(paths["batch"].read_text(encoding="ascii"))
    row = batch["result"]["checkpoint_catalog"]["3"]["cell_bundles"][0]
    bundle = Path(row["path"])
    real = tmp_path / "real-bundle"
    _write(
        real / "native_runtime_checkpoint_projection.json",
        {"status": "present"},
    )
    shutil_target = bundle
    for child in bundle.iterdir():
        child.unlink()
    bundle.rmdir()
    shutil_target.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        source.HereditaryCapacitySourceError,
        match="escapes expected|symlink",
    ):
        _build(paths)
