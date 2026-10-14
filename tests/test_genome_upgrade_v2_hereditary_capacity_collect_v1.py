from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256, file_sha256
from engine.evolution.genome_upgrade_v2 import PARAMETER_COUNT, PRODUCT_COUNT
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    DECISION_A,
    evaluate_hereditary_capacity_v1,
)
from tools.scripts.genome_upgrade_v2_hereditary_capacity_collect_v1 import (
    EXECUTION_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    HereditaryCapacityCollectorError,
    collect_repeat_v1,
    materialize_v1,
)


def _authority() -> dict[str, str]:
    return {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": "1" * 40,
        "package_id_sha256": "2" * 64,
        "package_receipt_sha256": "3" * 64,
        "artifact_sha256": "4" * 64,
        "execution_route_sha256": "5" * 64,
    }


def _checkpoint(*, terminal: bool) -> dict[str, object]:
    capacity = 3
    alleles = [
        [
            [
                [
                    1024
                    + (64 if terminal and homolog == 1 and slot == 1 else 0)
                    + (32 if homolog == 1 and slot == 0 else 0)
                    for slot in range(capacity)
                ]
                for _ in range(PARAMETER_COUNT)
            ]
            for _ in range(PRODUCT_COUNT)
        ]
        for homolog in range(2)
    ]
    modifiers = [
        [
            [
                (16 if homolog == 1 and slot == 0 else 0)
                + (8 if terminal and homolog == 0 and slot == 2 else 0)
                for slot in range(capacity)
            ]
            for _ in range(PRODUCT_COUNT)
        ]
        for homolog in range(2)
    ]
    products = [
        [1024 + (64 if terminal and slot == 1 else 0) for slot in range(capacity)]
        for _ in range(PRODUCT_COUNT)
    ]
    phenotypes = [
        [512 + (32 if terminal and slot == 1 else 0) for slot in range(capacity)]
        for _ in range(8)
    ]
    bank = {
        "schema_version": "cuda_genome_upgrade_v2_state_bank_v1",
        "cells": 1,
        "capacity": capacity,
        "tensors": {
            "genome_v2_alleles_q10": [alleles],
            "genome_v2_modifiers_q10": [modifiers],
            "genome_v2_products_q10": [products],
            "genome_v2_phenotypes_q10": [phenotypes],
            "genome_v2_expression_cost_q10": [[4, 5, 6]],
            "genome_v2_saturated_product_count": [[0, 0, 0]],
        },
    }
    return {
        "schema_version": "native_runtime_checkpoint_projection_v1",
        "status": "present",
        "projection": {
            "metadata": {},
            "tensors": {
                "alive_mask": {
                    "kind": "tensor",
                    "data_hex": "010101",
                    "dtype": "|b1",
                    "shape": [capacity],
                    "sha256": hashlib.sha256(bytes([1, 1, 1])).hexdigest(),
                }
            },
            "genome_upgrade_v2_checkpoint_projection": bank,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _source(tmp_path: Path, repeat: int) -> Path:
    baseline = tmp_path / f"r{repeat}-g0.json"
    terminal = tmp_path / f"r{repeat}-g3.json"
    _write_json(baseline, _checkpoint(terminal=False))
    _write_json(terminal, _checkpoint(terminal=True))
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "plan_sha256": "6" * 64,
        "batch_manifest_sha256": "7" * 64,
        "batch_id": "batch-0000",
        "cell_id": f"cell-{repeat}",
        "artifact_sha256": _authority()["artifact_sha256"],
        "execution_route_sha256": _authority()["execution_route_sha256"],
        "execution_device": "cuda",
        "explicit_native_no_fallback": True,
        "hidden_fallback": False,
        "fallback_used": False,
        "runtime_jit": False,
    }
    execution["receipt_sha256"] = content_sha256(execution)
    source = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "repeat_index": repeat,
        "authority": _authority(),
        "execution_receipt": execution,
        "observations": [
            {
                "generation": 0,
                "checkpoint_path": baseline.name,
                "checkpoint_sha256": file_sha256(baseline),
            },
            {
                "generation": 3,
                "checkpoint_path": terminal.name,
                "checkpoint_sha256": file_sha256(terminal),
            },
        ],
    }
    path = tmp_path / f"repeat-{repeat}.json"
    _write_json(path, source)
    return path


def test_collector_materializes_two_reducer_accepted_rows(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    rows = materialize_v1(
        authority=_authority(),
        repeat_sources=[_source(tmp_path, 1), _source(tmp_path, 0)],
        output_path=output,
    )
    assert [row["repeat_index"] for row in rows] == [0, 1]
    assert rows[0]["metrics"]["retained_hidden_allele_count"] > 0
    assert rows[0]["metrics"]["linkage_disequilibrium_abs_q10"] >= 0
    assert rows[0]["mutation_reachable"] is True
    assert (
        evaluate_hereditary_capacity_v1(rows, authority=_authority())["decision"]
        == DECISION_A
    )
    assert json.loads(output.read_text(encoding="utf-8")) == rows
    with pytest.raises(HereditaryCapacityCollectorError, match="already exists"):
        materialize_v1(
            authority=_authority(),
            repeat_sources=[_source(tmp_path, 0), _source(tmp_path, 1)],
            output_path=output,
        )


def test_collector_rejects_tamper_and_terminal_summary(tmp_path: Path) -> None:
    source_path = _source(tmp_path, 0)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / source["observations"][1]["checkpoint_path"]
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HereditaryCapacityCollectorError, match="hash differs"):
        collect_repeat_v1(source_path, authority=_authority())

    source_path = _source(tmp_path, 0)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / source["observations"][1]["checkpoint_path"]
    _write_json(
        checkpoint,
        {
            "schema_version": "genome_upgrade_v2_compact_generation_summary_v2",
            "hereditary_capacity": {"retention_semantics": "terminal_presence_only"},
        },
    )
    source["observations"][1]["checkpoint_sha256"] = file_sha256(checkpoint)
    _write_json(source_path, source)
    with pytest.raises(
        HereditaryCapacityCollectorError,
        match="cannot establish longitudinal retention",
    ):
        collect_repeat_v1(source_path, authority=_authority())


def test_collector_rejects_sparse_homogenization(tmp_path: Path) -> None:
    source_path = _source(tmp_path, 0)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / source["observations"][1]["checkpoint_path"]
    homogeneous = _checkpoint(terminal=True)
    bank = homogeneous["projection"]["genome_upgrade_v2_checkpoint_projection"]
    alleles = bank["tensors"]["genome_v2_alleles_q10"][0]
    modifiers = bank["tensors"]["genome_v2_modifiers_q10"][0]
    for homolog in range(2):
        for product in range(PRODUCT_COUNT):
            for parameter in range(PARAMETER_COUNT):
                alleles[homolog][product][parameter] = [1024, 1024, 1024]
            modifiers[homolog][product] = [0, 0, 0]
    _write_json(checkpoint, homogeneous)
    source["observations"][1]["generation"] = 4
    source["observations"][1]["checkpoint_sha256"] = file_sha256(checkpoint)
    _write_json(source_path, source)
    with pytest.raises(
        HereditaryCapacityCollectorError,
        match="sparse checkpoints cannot establish the exact first",
    ):
        collect_repeat_v1(source_path, authority=_authority())
