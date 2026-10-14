from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from engine.native_runtime.compact_summary_host_template import (
    compile_flat_compact_summary_host_programs,
)
from engine.native_runtime.compact_summary_host_token_resolver_plan import (
    CODEBOOKS,
    HOST_TOKEN_RESOLVER_ABI_VERSION,
    PCG64_AUTHORITY_CODES,
    PROJECTION_METADATA_FIELD_CODES,
    PROJECTION_TOP_LEVEL_FIELD_CODES,
    RESOLVER_PLAN_COLUMNS,
    SECONDARY_CODES,
    SEMANTIC_STREAM_CODES,
    SOURCE_KIND_CODES,
    CompactSummaryHostTokenResolverPlanError,
    compile_flat_compact_summary_host_token_resolver_plan,
    validate_compact_summary_host_token_resolver_plan,
)


PLAN_PATH = Path(
    "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
)


@pytest.fixture(scope="module")
def descriptor_plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_g2_bundle(descriptor_plan: dict[str, Any]) -> dict[str, Any]:
    fixture = descriptor_plan["source_contract"]["accepted_flat_projection_fixture"]
    if not Path(fixture["path"]).is_file():
        pytest.skip(
            "the immutable external Flat checkpoint projection fixture is unavailable"
        )
    return compile_flat_compact_summary_host_programs(
        fixture["path"], descriptor_plan, 2, 1
    )


@pytest.fixture(scope="module")
def real_resolver_plan(
    real_g2_bundle: dict[str, Any], descriptor_plan: dict[str, Any]
) -> Any:
    return compile_flat_compact_summary_host_token_resolver_plan(
        real_g2_bundle, descriptor_plan, torch
    )


def _flattened_tokens(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [token for template in bundle["templates"] for token in template["tokens"]]


def _path_row_map(bundle: dict[str, Any], tensor: Any) -> dict[str, tuple[int, ...]]:
    paths: dict[str, tuple[int, ...]] = {}
    for token, raw_row in zip(_flattened_tokens(bundle), tensor.tolist(), strict=True):
        row = tuple(raw_row)
        previous = paths.setdefault(token["source_path"], row)
        assert previous == row
    return paths


def test_real_g2_bundle_is_exactly_ordered_pinned_and_exhaustively_classified(
    real_g2_bundle: dict[str, Any],
    descriptor_plan: dict[str, Any],
    real_resolver_plan: Any,
) -> None:
    tokens = _flattened_tokens(real_g2_bundle)
    tensor = real_resolver_plan.resolver_plan

    assert len(tokens) == 4992
    assert tuple(tensor.shape) == (4992, 6)
    assert tensor.dtype == torch.uint64
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    assert tensor.storage_offset() == 0
    assert tensor.is_pinned()
    assert not tensor.requires_grad
    assert tensor[:, 0].unique().tolist() == [HOST_TOKEN_RESOLVER_ABI_VERSION]
    assert tensor[:, 5].unique().tolist() == [0]
    assert set(tensor[:, 1].unique().tolist()) == set(SOURCE_KIND_CODES.values())

    manifest = real_resolver_plan.manifest
    assert manifest["columns"] == RESOLVER_PLAN_COLUMNS
    assert manifest["shape"] == (4992, 6)
    assert manifest["token_count"] == 4992
    assert manifest["unique_source_path_count"] == 1831
    assert manifest["template_count"] == 12
    assert manifest["generation_count"] == 2
    assert manifest["cell_count"] == 1
    assert manifest["descriptor_count"] == 133
    assert set(manifest["source_kind_counts"]) == set(SOURCE_KIND_CODES)
    assert all(count > 0 for count in manifest["source_kind_counts"].values())
    assert manifest["source_kind_counts"]["python_gauss_cache"] == 4
    assert manifest["pinned_memory"] is True
    assert manifest["test_only_unpinned_fallback"] is False
    assert real_resolver_plan.codebooks == CODEBOOKS
    assert real_resolver_plan.ordered_tensors == real_resolver_plan.tensor_tuple
    assert real_resolver_plan.as_tuple() == (tensor,)

    # Recomputing from the JSON path proves both accepted descriptor-plan inputs.
    validate_compact_summary_host_token_resolver_plan(
        real_resolver_plan, real_g2_bundle, PLAN_PATH, torch
    )
    assert len(_path_row_map(real_g2_bundle, tensor)) == 1831


def test_descriptor_outputs_and_all_non_descriptor_codebooks_are_exact(
    real_g2_bundle: dict[str, Any],
    descriptor_plan: dict[str, Any],
    real_resolver_plan: Any,
) -> None:
    rows = _path_row_map(real_g2_bundle, real_resolver_plan.resolver_plan)
    descriptors = {
        row["tensor_name"]: row for row in descriptor_plan["descriptors"]
    }
    action = descriptors["action_cache_actions"]
    energy = descriptors["policy_energy_observation"]
    identity = descriptors["agent_ids_by_alive_order"]
    policy = descriptors["policy_birth_crossover_mask_tape"]

    def expected(
        kind: str,
        primary: int = 0,
        output: str = "none",
        rank: int = 0,
    ) -> tuple[int, ...]:
        return (
            HOST_TOKEN_RESOLVER_ABI_VERSION,
            SOURCE_KIND_CODES[kind],
            primary,
            SECONDARY_CODES[output],
            rank,
            0,
        )

    assert rows["/checkpoint/trajectory_state_fields/0/sha256"] == expected(
        "checkpoint_trajectory_digest", action["descriptor_index"], "canonical"
    )
    assert rows["/checkpoint/canonical_checkpoint_storage_fields/0/shape"] == expected(
        "checkpoint_canonical_shape",
        action["descriptor_index"],
        "canonical",
        action["canonical_rank"],
    )
    assert rows["/checkpoint/tensor_manifest/action_cache_actions/shape"] == expected(
        "checkpoint_source_manifest_shape",
        action["descriptor_index"],
        "raw",
        action["raw_rank"],
    )
    assert rows[
        "/checkpoint/projection_fields/tensor_manifest/action_cache_actions/bytes"
    ] == expected("checkpoint_stored_manifest_bytes", action["descriptor_index"], "raw")
    assert rows["/checkpoint/projection_fields/event_cursor"] == expected(
        "projection_top_level_integer",
        PROJECTION_TOP_LEVEL_FIELD_CODES["event_cursor"],
    )
    assert rows[
        "/checkpoint/projection_fields/metadata/native_policy_cache_served"
    ] == expected(
        "projection_metadata_boolean",
        PROJECTION_METADATA_FIELD_CODES["native_policy_cache_served"],
    )
    assert rows[
        "/checkpoint/projection_fields/metadata/configured_steps_per_generation"
    ] == expected(
        "projection_metadata_integer",
        PROJECTION_METADATA_FIELD_CODES["configured_steps_per_generation"],
    )
    assert rows[
        "/checkpoint/projection_fields/metadata/native_runtime_numpy_env_state/state/state"
    ] == expected("pcg64_state", PCG64_AUTHORITY_CODES["native_runtime_numpy_env_state"])
    assert rows[
        "/checkpoint/projection_fields/metadata/native_runtime_numpy_env_state/has_uint32"
    ] == expected(
        "pcg64_buffer_flag", PCG64_AUTHORITY_CODES["native_runtime_numpy_env_state"]
    )
    assert rows[
        "/checkpoint/projection_fields/metadata/native_runtime_python_random_state/state/623"
    ] == expected("python_mt_word", 623)
    assert rows[
        "/checkpoint/projection_fields/metadata/native_runtime_python_random_state/index"
    ] == expected("python_mt_index")
    assert rows[
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_python_random_state/gauss_next"
    ] == expected("python_gauss_cache")
    assert rows[
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_rng_manager_stream_indices/metrics/py"
    ] == expected("semantic_stream_counter", SEMANTIC_STREAM_CODES["metrics/py"])
    assert rows["/checkpoint/projection_fields/metadata/import_digest"] == expected(
        "metadata_import_digest"
    )
    assert rows[
        "/checkpoint/projection_fields/metadata/policy_energy_observation_shape"
    ] == expected(
        "policy_energy_shape",
        energy["descriptor_index"],
        "raw",
        energy["raw_rank"],
    )
    assert rows["/identity/tensor_sha256"] == expected(
        "identity_digest", identity["descriptor_index"], "canonical"
    )
    assert rows[
        "/policy_mutation_tape/tensor_sha256/policy_birth_crossover_mask_tape"
    ] == expected(
        "policy_mutation_tape_digest",
        policy["descriptor_index"],
        "canonical",
    )
    assert len(SEMANTIC_STREAM_CODES) == 16
    assert set(SEMANTIC_STREAM_CODES) == {
        f"{semantic}/{backend}"
        for semantic in ("metrics", "mutation", "policy", "stress")
        for backend in ("py", "np", "torch_cpu", "torch_cuda/cuda:0")
    }
    assert len(SOURCE_KIND_CODES) == 28
    assert SOURCE_KIND_CODES["python_gauss_cache"] == 28


def test_manifest_codebooks_and_wrapper_are_immutable(real_resolver_plan: Any) -> None:
    with pytest.raises(TypeError):
        real_resolver_plan.manifest["token_count"] = 0
    with pytest.raises(TypeError):
        real_resolver_plan.manifest["source_kind_counts"][
            "checkpoint_trajectory_digest"
        ] = 0
    with pytest.raises(TypeError):
        real_resolver_plan.codebooks["source_kind_codes"]["unknown"] = 999
    with pytest.raises(FrozenInstanceError):
        real_resolver_plan.manifest = {}


def _unknown_path(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["source_path"] = "/unknown/flat/g2/path"


def _token_type_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["token_type"] = "integer"


def _token_count_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["token_count"] -= 1


def _template_order_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0], bundle["templates"][1] = (
        bundle["templates"][1],
        bundle["templates"][0],
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (_unknown_path, "unknown Flat g2 source path"),
        (_token_type_drift, "validation_rule|token type"),
        (_token_count_drift, "token count"),
        (_template_order_drift, "surface"),
    ],
)
def test_malformed_or_unknown_bundle_fails_closed_before_allocation(
    real_g2_bundle: dict[str, Any],
    descriptor_plan: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    malformed = copy.deepcopy(real_g2_bundle)
    mutate(malformed)
    with pytest.raises(CompactSummaryHostTokenResolverPlanError, match=match):
        compile_flat_compact_summary_host_token_resolver_plan(
            malformed, descriptor_plan, torch
        )


def test_descriptor_plan_drift_fails_closed(
    real_g2_bundle: dict[str, Any], descriptor_plan: dict[str, Any]
) -> None:
    malformed = copy.deepcopy(descriptor_plan)
    malformed["descriptors"][0]["raw_output_index"] = 1
    with pytest.raises(CompactSummaryHostTokenResolverPlanError, match="plan drift"):
        compile_flat_compact_summary_host_token_resolver_plan(
            real_g2_bundle, malformed, torch
        )


class _UnavailableCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _NoPinTorch:
    uint64 = torch.uint64
    uint8 = torch.uint8
    cuda = _UnavailableCuda()

    @staticmethod
    def tensor(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pin_memory"):
            raise RuntimeError("test allocator has no CUDA pin-memory allocator")
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def empty(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pin_memory"):
            raise RuntimeError("test allocator has no CUDA pin-memory allocator")
        return torch.empty(*args, **kwargs)


def test_unpinned_fallback_is_explicit_test_only_and_validator_enforced(
    real_g2_bundle: dict[str, Any], descriptor_plan: dict[str, Any]
) -> None:
    no_pin_torch = _NoPinTorch()
    with pytest.raises(CompactSummaryHostTokenResolverPlanError, match="pinned CPU"):
        compile_flat_compact_summary_host_token_resolver_plan(
            real_g2_bundle, descriptor_plan, no_pin_torch
        )

    compiled = compile_flat_compact_summary_host_token_resolver_plan(
        real_g2_bundle,
        descriptor_plan,
        no_pin_torch,
        allow_unpinned_test_fallback=True,
    )
    assert compiled.manifest["pinned_memory"] is False
    assert compiled.manifest["test_only_unpinned_fallback"] is True
    assert not compiled.resolver_plan.is_pinned()
    validate_compact_summary_host_token_resolver_plan(
        compiled, real_g2_bundle, descriptor_plan, no_pin_torch
    )

    compiled.resolver_plan[0, 5] = 1
    with pytest.raises(CompactSummaryHostTokenResolverPlanError, match="row content"):
        validate_compact_summary_host_token_resolver_plan(
            compiled, real_g2_bundle, descriptor_plan, no_pin_torch
        )
