#!/usr/bin/env python3
"""Measure the bounded B48/C512 HAPB policy-weight alias rescue.

This is a policy-boundary fixture, not a lifecycle qualification run.  It
reuses the current-source B48 projection prefixes as the canonical bank and
constructs an explicit byte-identical C512 overflow suffix.  That gives the
native baseline and alias routes the same policy workload while making the
retransmission seam observable.  Gate-5 current-source H4 evidence remains
the authority for lifecycle, RNG, birth, checkpoint, and fresh-owner parity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from engine.campaigns.orchestration import atomic_write_json
from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HOT_PREFIX_ROWS_V1,
    POLICY_ROW_BYTES_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    initialize_selected_hapb_owner_v1,
)
from tools.scripts.eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 import (
    _load_exact_module,
    _sha256,
)


SCHEMA_VERSION = "eiel_b48_hapb_policy_traffic_fixture_v1"
OBSERVATION_SIZE = 83
ACTION_COUNT = 7
HIDDEN_SIZE = 64
WEIGHTS = (1.2, 0.9, 1.0, 1.2, 1.4)


def _tensor_digest(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("ascii"))
    h.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    h.update(cpu.numpy().tobytes(order="C"))
    return h.hexdigest()


def _load_source_canonical(
    source_root: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    receipt_path = source_root / "source_factory_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != CELL_COUNT_V1:
        raise RuntimeError("current-source B48 receipt does not contain 48 rows")
    canonical = torch.empty(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1),
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
    projection_hashes: list[str] = []
    for cell, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("current-source row is malformed")
        raw_path = row.get("projection_path")
        expected = row.get("projection_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError("current-source row lacks projection identity")
        path = Path(raw_path).resolve()
        try:
            path.relative_to(source_root.resolve())
        except ValueError as exc:
            raise RuntimeError("current-source projection escapes its root") from exc
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("current-source projection digest differs")
        projection = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(projection, Mapping):
            raise RuntimeError("current-source projection is not a mapping")
        tensors = projection.get("tensors")
        pack = tensors.get("mlp_pack") if isinstance(tensors, Mapping) else None
        if not isinstance(pack, torch.Tensor) or tuple(pack.shape) != (
            SLOT_CAPACITY_V1,
            POLICY_WIDTH_V1,
        ) or pack.dtype != torch.float32:
            raise RuntimeError("current-source projection lacks exact mlp_pack")
        prefix = pack[:HOT_PREFIX_ROWS_V1].contiguous()
        # This is the audit's deterministic duplicate-overflow workload.  The
        # current-source prefix remains the authority; no generated policy
        # bytes are introduced, and the suffix is explicitly labelled fixture.
        canonical[cell, :HOT_PREFIX_ROWS_V1].copy_(prefix)
        canonical[cell, HOT_PREFIX_ROWS_V1:].copy_(prefix)
        projection_hashes.append(expected)
        del projection, tensors, pack, prefix
    return canonical, {
        "source_factory_receipt_sha256": _sha256(receipt_path),
        "projection_sha256": projection_hashes,
        "source_root": str(source_root.resolve()),
    }


def _rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cells = torch.arange(CELL_COUNT_V1, dtype=torch.int64).unsqueeze(1) * SLOT_CAPACITY_V1
    hot = (cells + torch.arange(HOT_PREFIX_ROWS_V1, dtype=torch.int64)).reshape(-1).contiguous()
    overflow = (
        cells + torch.arange(HOT_PREFIX_ROWS_V1, SLOT_CAPACITY_V1, dtype=torch.int64)
    ).reshape(-1).contiguous()
    active_hot = hot.clone()
    active_overflow = overflow.clone()
    content_overflow = (
        (overflow // SLOT_CAPACITY_V1) * HOT_PREFIX_ROWS_V1
        + (overflow % SLOT_CAPACITY_V1 - HOT_PREFIX_ROWS_V1)
    ).contiguous()
    return hot, active_hot, overflow, active_overflow, content_overflow


def _run_case(
    module: Any,
    canonical: torch.Tensor,
    *,
    hot_rows: torch.Tensor,
    hot_active_rows: torch.Tensor,
    overflow_rows: torch.Tensor,
    overflow_active_rows: torch.Tensor,
    content_overflow_rows: torch.Tensor,
    observations: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    cursor_state: torch.Tensor,
    cursor_buffer: torch.Tensor,
    alias_enabled: bool,
    iterations: int,
) -> tuple[list[torch.Tensor], dict[str, int | str], str]:
    owner = initialize_selected_hapb_owner_v1(
        canonical,
        device="cuda",
        populate_hot_prefix=True,
        content_alias_enabled=alias_enabled,
        pool_epoch=1,
    )
    outputs: list[torch.Tensor] = []
    export = module.eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v2
    for _ in range(iterations):
        logits = export(
            owner.canonical_host_bank,
            owner.pinned_staging_buffers[0],
            owner.require_hot_prefix_cache_v1(),
            owner.cuda_tile_arena,
            overflow_rows,
            overflow_active_rows,
            content_overflow_rows,
            owner.content_row_map,
            owner.policy_traffic_counters,
            alias_enabled,
            cursor_state,
            cursor_buffer,
            observations,
            hot_rows,
            hot_active_rows,
            drives,
            attack_bias,
            action_temperature,
            drive_logit_gain,
            HIDDEN_SIZE,
            ACTION_COUNT,
            *WEIGHTS,
        )
        torch.cuda.synchronize()
        outputs.append(logits.detach().cpu().contiguous())
    traffic = owner.policy_traffic_snapshot_v1()
    relation = owner.content_alias_relation_sha256_v1()
    del owner
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, traffic, relation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite an immutable traffic fixture")
    if args.iterations < 10:
        raise ValueError("at least 10 repeated overflow generations are required")
    artifact = args.artifact.resolve()
    manifest = args.manifest.resolve()
    source_root = args.source_root.resolve()
    if not artifact.is_file() or not manifest.is_file():
        raise RuntimeError("artifact and manifest must be files")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("artifact_sha256") != _sha256(artifact):
        raise RuntimeError("artifact does not match its manifest")
    module = _load_exact_module(artifact, eed_overlay=True, hapb_policy_only=True)
    export_name = "eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v2"
    if not hasattr(module, export_name):
        raise RuntimeError("native artifact lacks the explicit alias/traffic export")
    canonical, source_identity = _load_source_canonical(source_root)
    hot_rows, hot_active_rows, overflow_rows, overflow_active_rows, content_rows = _rows()
    total = CELL_COUNT_V1 * SLOT_CAPACITY_V1
    observations = torch.zeros((total, OBSERVATION_SIZE), dtype=torch.float32, device="cuda")
    drives = torch.zeros((total, 4), dtype=torch.float32, device="cuda")
    attack_bias = torch.zeros((total,), dtype=torch.float32, device="cuda")
    action_temperature = torch.ones((total,), dtype=torch.float32, device="cuda")
    drive_logit_gain = torch.ones((total,), dtype=torch.float32, device="cuda")
    cursor_state = torch.zeros((CELL_COUNT_V1, 4), dtype=torch.int64, device="cuda")
    cursor_buffer = torch.zeros((CELL_COUNT_V1, 2), dtype=torch.int64, device="cuda")

    baseline, baseline_traffic, baseline_relation = _run_case(
        module, canonical, hot_rows=hot_rows, hot_active_rows=hot_active_rows,
        overflow_rows=overflow_rows, overflow_active_rows=overflow_active_rows,
        content_overflow_rows=content_rows, observations=observations, drives=drives,
        attack_bias=attack_bias, action_temperature=action_temperature,
        drive_logit_gain=drive_logit_gain, cursor_state=cursor_state,
        cursor_buffer=cursor_buffer, alias_enabled=False, iterations=args.iterations,
    )
    alias, alias_traffic, alias_relation = _run_case(
        module, canonical, hot_rows=hot_rows, hot_active_rows=hot_active_rows,
        overflow_rows=overflow_rows, overflow_active_rows=overflow_active_rows,
        content_overflow_rows=content_rows, observations=observations, drives=drives,
        attack_bias=attack_bias, action_temperature=action_temperature,
        drive_logit_gain=drive_logit_gain, cursor_state=cursor_state,
        cursor_buffer=cursor_buffer, alias_enabled=True, iterations=args.iterations,
    )
    parity = [torch.equal(left, right) for left, right in zip(baseline, alias, strict=True)]
    baseline_bytes = int(baseline_traffic["policy_h2d_bytes"])
    alias_bytes = int(alias_traffic["policy_h2d_bytes"])
    ratio = baseline_bytes / alias_bytes if alias_bytes else None
    expected_hot_rows = CELL_COUNT_V1 * HOT_PREFIX_ROWS_V1
    expected_overflow_rows = expected_hot_rows
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": "B48/C512 duplicate-overflow policy-boundary only",
        "semantics": {
            "canonical_host_policy_bank": "current-source B48 prefix rows; sole authority",
            "overflow_suffix": "byte-identical prefix clone constructed for audit workload",
            "lifecycle_claim": False,
            "qualification_claim": False,
            "production_execution_admission": False,
        },
        "artifact": {
            "path": str(artifact),
            "sha256": _sha256(artifact),
            "manifest_path": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "build_source_commit": manifest_payload.get(
                "authoritative_state_build_provenance_v1", {}
            ).get("build_source_commit"),
            "compiled_source_snapshot_sha256": manifest_payload.get(
                "authoritative_state_build_provenance_v1", {}
            ).get("compiled_source_snapshot_sha256"),
        },
        "source": source_identity,
        "iterations": args.iterations,
        "policy_row_bytes": POLICY_ROW_BYTES_V1,
        "overflow_rows_per_iteration": expected_overflow_rows,
        "baseline": {
            "content_alias_enabled": False,
            "traffic": baseline_traffic,
            "relation_sha256": baseline_relation,
            "output_sha256": [_tensor_digest(value) for value in baseline],
        },
        "alias": {
            "content_alias_enabled": True,
            "traffic": alias_traffic,
            "relation_sha256": alias_relation,
            "output_sha256": [_tensor_digest(value) for value in alias],
        },
        "exact_output_parity_by_iteration": parity,
        "exact_output_parity": all(parity),
        "policy_weight_h2d_bytes_reduction_ratio": ratio,
        "policy_weight_h2d_bytes_reduction_ge_10x": bool(ratio is not None and ratio >= 10.0),
        "expected": {
            "baseline_policy_h2d_rows": expected_hot_rows + args.iterations * expected_overflow_rows,
            "alias_policy_h2d_rows": expected_hot_rows,
            "baseline_policy_h2d_bytes": (expected_hot_rows + args.iterations * expected_overflow_rows) * POLICY_ROW_BYTES_V1,
            "alias_policy_h2d_bytes": expected_hot_rows * POLICY_ROW_BYTES_V1,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "traffic_fixture_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    if not receipt["exact_output_parity"] or not receipt["policy_weight_h2d_bytes_reduction_ge_10x"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
