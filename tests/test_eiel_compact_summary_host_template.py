from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import engine.native_runtime.compact_summary_host_template as host_template_module
from engine.native_runtime.checkpoint_digests import (
    SCHEMA_VERSION as CHECKPOINT_LAYER_SCHEMA_VERSION,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS,
    SOURCE_RNG_SCHEMA_VERSION,
    DigestLayer,
    _canonical_patch_source_projection,
    checkpoint_source_projection_digest_manifest,
)
from engine.native_runtime.compact_summary_checkpoint_descriptor_plan import (
    WORLD_PATCH,
    generate_compact_summary_checkpoint_descriptor_plan,
)
from engine.native_runtime.compact_summary_digests import (
    ACTIVE_AGENT_IDS_SCHEMA_VERSION,
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
    active_agent_ids_digest_from_tensor_sha256,
    checkpoint_layer_digest_from_tensor_manifest,
    checkpoint_rng_digest_from_tensor_manifest,
    checkpoint_source_projection_digest_from_tensor_manifest,
    policy_mutation_tape_digest_from_tensor_sha256,
)
from engine.native_runtime.compact_summary_host_template import (
    HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
    SURFACE_ACTIVE_AGENT_IDS,
    SURFACE_CANONICAL_STORAGE,
    SURFACE_CHECKPOINT_TRAJECTORY,
    SURFACE_NEXT_GENERATION_SOURCE,
    SURFACE_ORDER,
    SURFACE_POLICY_MUTATION_TAPE,
    SURFACE_RNG,
    TOKEN_BOOLEAN,
    TOKEN_INTEGER,
    TOKEN_NULLABLE_FLOAT64,
    TOKEN_PCG_UINT128,
    TOKEN_PYTHON_MT_WORD,
    TOKEN_SHA256_HEX,
    TOKEN_TENSOR_SHAPE,
    CompactSummaryHostTemplateBlocker,
    CompactSummaryHostTemplateError,
    compact_summary_host_template_json_bytes,
    compile_device_neutral_compact_summary_host_programs,
    compile_compact_summary_host_templates,
    compile_flat_compact_summary_host_programs,
    dynamic_token_placeholder,
    render_compact_summary_host_template,
    retained_authority_token_values,
)


PLAN_PATH = Path("docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json")
_TORCH_DTYPE = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _pointer(base: str, component: str | int) -> str:
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def _shape(row: dict[str, Any], output: str) -> list[int]:
    rank = row["canonical_rank"] if output == "canonical" else row["raw_rank"]
    shape = [2] * rank
    if row["tensor_name"] == "agent_ids_by_alive_order":
        shape[0] = 1
    return shape


def _stored_entry(row: dict[str, Any], shape: list[int]) -> dict[str, Any]:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return {
        "shape": list(shape),
        "dtype": row["torch_tensor_manifest_dtype"],
        "device": "cpu",
        "numel": numel,
        "element_size": row["dtype_bytes"],
        "bytes": numel * row["dtype_bytes"],
        "is_cuda": False,
        "is_contiguous": True,
    }


def _pcg(seed: int) -> dict[str, Any]:
    return {
        "bit_generator": "PCG64",
        "state": {"state": seed + 10, "inc": seed * 2 + 1},
        "has_uint32": 0,
        "uinteger": seed,
    }


def _plan_and_skeleton() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    additions = set(
        plan["source_contract"]["accepted_flat_projection_fixture"][
            "current_registry_additions"
        ]
    )
    original_rows = [row for row in rows if row["tensor_name"] not in additions]
    original_manifest = {
        row["tensor_name"]: _stored_entry(
            row,
            _shape(row, row["digest_inclusion"]["source_projection_output"]),
        )
        for row in original_rows
    }
    env = _pcg(11)
    metadata = {
        "world": "flat",
        "generation": 1,
        "tick": 0,
        "configured_steps_per_generation": 4,
        "agent_grid_runner_phase": "generation_boundary",
        "agent_grid_authority_epoch": "generation:1:tick:0",
        "alive_count": 1,
        "event_tape_active_count": 2,
        "event_tape_cursor": 2,
        "event_tape_inactive_tail_count": 0,
        "native_policy_cache_served": False,
        "native_runtime_numpy_env_state": env,
        "native_runtime_numpy_runner_state": _pcg(21),
        "native_runtime_numpy_policy_state": copy.deepcopy(env),
        "native_runtime_env_policy_numpy_alias": True,
        "native_runtime_numpy_population_state": _pcg(31),
        "native_runtime_numpy_birth_spawn_env_state": _pcg(41),
        "native_runtime_python_random_state": {
            "schema_version": "python_random_state_v1",
            "version": 3,
            "state": [101, 202, 303, 404],
            "index": 4,
            "gauss_next": None,
        },
        "native_runtime_rng_manager_stream_indices": {
            "policy": {"py": 1, "np": 2, "torch_cpu": 3, "torch_cuda": {"cuda:0": 4}},
            "mutation": {"py": 5, "np": 6, "torch_cpu": 7, "torch_cuda": {}},
        },
        "native_runtime_rng_state": 99,
        "import_digest": _hash("import"),
        "policy_energy_observation_finite_live_count": 1,
        "policy_energy_observation_live_count": 1,
        "policy_energy_observation_shape": [2],
        "policy_energy_observation_tensor_sha256": _hash("policy-energy"),
        "population_next_id": 9,
        "typed_lifecycle_event_active_count": 2,
        "typed_lifecycle_event_next_sequence": 2,
        "fixed_float": 1.25,
        "fixed_string": "retained-finalizer-fixture",
    }
    projection = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": metadata,
        "tensors": {name: None for name in original_manifest},
        "tensor_manifest": original_manifest,
        "restore_count": 0,
        "generation_calls": 1,
        "event_cursor": 3,
        "runtime_event_cursor": 7,
        "last_applied_actions": 1,
        "last_moved": 1,
        "last_gathered": 0,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 1,
        "pending_generation_free_agent_ids": [],
        "pending_generation_free_slots": [],
        "last_world": "flat",
    }
    assert set(projection) == set(SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS)
    return plan, {"projection": projection}


def _patch_plan_and_skeleton() -> tuple[dict[str, Any], dict[str, Any]]:
    _, skeleton = _plan_and_skeleton()
    plan = generate_compact_summary_checkpoint_descriptor_plan(world=WORLD_PATCH)
    projection = skeleton["projection"]
    metadata = projection["metadata"]
    metadata.update(
        {
            "world": WORLD_PATCH,
            "native_runtime_patch_profile": "P2",
            "native_runtime_patch_profile_scope": "full_profile",
            "native_runtime_patch_contract_schema_version": (
                "eiel_patch_native_import_contract_v1"
            ),
            "patch_world_enable": True,
            "patch_world_schema_version": 1,
            "patch_world_observation_enable": True,
            "patch_world_observation_schema_version": 1,
            "patch_world_diagnostics_enable": True,
            "patch_world_lineage_descriptors_enable": True,
            "patch_world_descriptor_schema_version": 1,
            "patch_movement_cost_apply_enable": True,
            "patch_world_waste_consequence_enable": True,
            "patch_array_import_phase": "post_reference_setup_once",
            "patch_visibility_authority": "derived_from_patch_cover",
            "patch_visibility_from_cover_enable": True,
            "patch_visibility_default": 1.0,
            "patch_visibility_min": 0.0,
            "patch_visibility_max": 1.0,
            "patch_density_authority": "derived_pre_action_from_agent_grid",
            "patch_density_radius": 1,
            "patch_density_include_self": True,
            "patch_density_normalize": True,
            "patch_food_presence_threshold": 0.5,
            "patch_food_binary_projection": (
                "patch_food_amount_strictly_greater_than_threshold"
            ),
            "patch_pending_emission_phase": "queued_until_next_tick",
        }
    )
    projection["last_world"] = WORLD_PATCH
    additions = set(
        plan["source_contract"]["accepted_flat_projection_fixture"][
            "current_registry_additions"
        ]
    )
    rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
        and row["tensor_name"] not in additions
    ]
    projection["tensor_manifest"] = {
        row["tensor_name"]: _stored_entry(
            row,
            _shape(row, row["digest_inclusion"]["source_projection_output"]),
        )
        for row in rows
    }
    projection["tensors"] = {
        name: None for name in projection["tensor_manifest"]
    }
    return plan, skeleton


def _authoritative_patch_projection(
    plan: dict[str, Any],
    skeleton: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection = copy.deepcopy(skeleton["projection"])
    rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    tensors = {
        row["tensor_name"]: torch.zeros(
            _shape(row, row["digest_inclusion"]["source_projection_output"]),
            dtype=_TORCH_DTYPE[row["dtype"]],
        )
        for row in rows
    }
    tensors["alive_mask"][:] = torch.tensor([True, False])
    tensors["slot_tokens"][:] = torch.tensor([11, 22], dtype=torch.int64)
    tensors["action_cache_actions"][:, :] = torch.tensor(
        [[1, 2], [3, 4]],
        dtype=torch.int32,
    )
    tensors["action_cache_token"][:] = torch.tensor([11, -1], dtype=torch.int64)
    tensors["action_cache_cursor"][:] = torch.tensor([0, -1], dtype=torch.int32)
    tensors["action_cache_next"][:] = torch.tensor([1, -1], dtype=torch.int32)
    tensors["agent_grid"][:, :] = torch.tensor(
        [[0, -1], [-1, -1]],
        dtype=torch.int32,
    )
    tensors["patch_food_amount"][:, :] = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2]],
        dtype=torch.float32,
    )
    tensors["food_grid"][:, :] = tensors["patch_food_amount"].gt(0.5).to(
        dtype=torch.int32
    )
    tensors["patch_cover"][:, :] = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=torch.float32,
    )
    for name in (
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    ):
        if name not in {"patch_food_amount", "patch_cover"}:
            tensors[name].fill_(0.25)
    tensors["patch_habitat_lineage_id"][:] = torch.tensor(
        [7, 0],
        dtype=torch.int32,
    )
    # Stale derived values prove the authoritative contract rebuilds both views.
    tensors["patch_visibility"].fill_(-17.0)
    tensors["patch_density"].fill_(23.0)

    projection["tensors"] = tensors
    projection["tensor_manifest"] = {
        row["tensor_name"]: _stored_entry(
            row,
            list(tensors[row["tensor_name"]].shape),
        )
        for row in rows
    }
    canonical, _ = _canonical_patch_source_projection(projection)
    return projection, canonical


def _settled_patch_authority_from_projection(
    plan: dict[str, Any],
    skeleton: dict[str, Any],
    canonical_projection: dict[str, Any],
) -> dict[str, Any]:
    settled = _settled_authority(plan, skeleton)
    settled["checkpoint"]["projection_fields"] = copy.deepcopy(
        {
            name: value
            for name, value in canonical_projection.items()
            if name != "tensors"
        }
    )
    manifest = settled["checkpoint"]["tensor_manifest"]
    for name, tensor in canonical_projection["tensors"].items():
        if name not in manifest:
            continue
        array = tensor.detach().to("cpu").contiguous().numpy()
        manifest[name] = {
            "device": "cpu",
            "dtype": str(array.dtype),
            "shape": [int(dimension) for dimension in array.shape],
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return settled


def _settled_authority(
    plan: dict[str, Any],
    skeleton: dict[str, Any],
    *,
    generation: int = 0,
    cell: int = 0,
) -> dict[str, Any]:
    checkpoint_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    policy_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "policy_mutation_tape"
    ]
    projection = copy.deepcopy(skeleton["projection"])
    projection.pop("tensors")
    projection["metadata"]["generation"] = generation + 1
    projection["metadata"]["tick"] = 0
    projection["metadata"]["agent_grid_authority_epoch"] = (
        f"generation:{generation + 1}:tick:0"
    )
    projection["generation_calls"] = generation + 1
    manifest: dict[str, dict[str, Any]] = {}
    stored: dict[str, dict[str, Any]] = {}
    world = projection["metadata"]["world"]
    for row in checkpoint_rows:
        name = row["tensor_name"]
        output = row["digest_inclusion"]["source_projection_output"]
        shape = _shape(row, output)
        manifest[name] = {
            "device": "cpu",
            "dtype": row["dtype"],
            "shape": shape,
            "sha256": _hash(f"source-{name}-{generation}-{cell}"),
        }
        if not (
            world == WORLD_PATCH and name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS
        ):
            stored[name] = _stored_entry(row, shape)
    projection["tensor_manifest"] = stored

    def layer_fields(layer: str) -> list[dict[str, Any]]:
        return [
            {
                "field": row["tensor_name"],
                "dtype": row["dtype"],
                "shape": _shape(row, "canonical"),
                "sha256": _hash(f"{layer}-{row['tensor_name']}-{generation}-{cell}"),
            }
            for row in checkpoint_rows
            if layer in row["digest_inclusion"]["checkpoint_layers"]
        ]

    active = manifest["agent_ids_by_alive_order"]
    policy_shapes = {
        row["tensor_name"]: [1, 2] if row["canonical_rank"] == 2 else [1]
        for row in policy_rows
    }
    return {
        "generation": generation,
        "cell": cell,
        "world": world,
        "checkpoint": {
            "source_version": CHECKPOINT_LAYER_SCHEMA_VERSION,
            "rng_source_version": SOURCE_RNG_SCHEMA_VERSION,
            "projection_schema_version": projection["schema_version"],
            "projection_fields": projection,
            "tensor_manifest": manifest,
            "trajectory_state_fields": layer_fields("trajectory_state"),
            "canonical_checkpoint_storage_fields": layer_fields(
                "canonical_checkpoint_storage"
            ),
        },
        "identity": {
            "source_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
            "active_agent_count": active["shape"][0],
            "tensor_sha256": active["sha256"],
        },
        "policy_mutation_tape": {
            "source_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
            "cursor": 1,
            "consumed_shapes": policy_shapes,
            "tensor_sha256": {
                name: _hash(f"policy-{name}-{generation}-{cell}")
                for name in policy_shapes
            },
        },
    }


def _set_pointer(root: Any, pointer: str, value: Any) -> None:
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    current = root
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def _placeholder_authority(source: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    placeholder = copy.deepcopy(source)

    def replace(path: str, token_type: str, *, rank: int | None = None) -> None:
        _set_pointer(
            placeholder,
            path,
            dynamic_token_placeholder(token_type, path, rank=rank),
        )

    projection_base = "/checkpoint/projection_fields"
    for name in (
        "restore_count",
        "generation_calls",
        "event_cursor",
        "runtime_event_cursor",
        "last_applied_actions",
        "last_moved",
        "last_gathered",
        "last_policy_refill_calls",
        "last_policy_refill_rows",
    ):
        replace(_pointer(projection_base, name), TOKEN_INTEGER)
    metadata_base = _pointer(projection_base, "metadata")
    metadata = source["checkpoint"]["projection_fields"]["metadata"]
    for name in (
        "configured_steps_per_generation",
        "alive_count",
        "event_tape_active_count",
        "event_tape_cursor",
        "event_tape_inactive_tail_count",
        "native_runtime_rng_state",
        "policy_energy_observation_finite_live_count",
        "policy_energy_observation_live_count",
        "population_next_id",
        "typed_lifecycle_event_active_count",
        "typed_lifecycle_event_next_sequence",
    ):
        replace(_pointer(metadata_base, name), TOKEN_INTEGER)
    replace(_pointer(metadata_base, "native_policy_cache_served"), TOKEN_BOOLEAN)
    for name, value in metadata.items():
        path = _pointer(metadata_base, name)
        if isinstance(value, str) and len(value) == 64 and (
            name.endswith("sha256") or name.endswith("digest")
        ):
            replace(path, TOKEN_SHA256_HEX)
        if name.endswith("_shape") and isinstance(value, list):
            replace(path, TOKEN_TENSOR_SHAPE, rank=len(value))
        if (
            name.startswith("native_runtime_numpy_")
            and name.endswith("_state")
            and isinstance(value, dict)
            and value.get("bit_generator") == "PCG64"
        ):
            replace(_pointer(_pointer(path, "state"), "state"), TOKEN_PCG_UINT128)
            replace(_pointer(_pointer(path, "state"), "inc"), TOKEN_PCG_UINT128)
            replace(_pointer(path, "has_uint32"), TOKEN_INTEGER)
            replace(_pointer(path, "uinteger"), TOKEN_INTEGER)
    python_path = _pointer(metadata_base, "native_runtime_python_random_state")
    for index in range(len(metadata["native_runtime_python_random_state"]["state"])):
        replace(_pointer(_pointer(python_path, "state"), index), TOKEN_PYTHON_MT_WORD)
    replace(_pointer(python_path, "index"), TOKEN_INTEGER)
    replace(_pointer(python_path, "gauss_next"), TOKEN_NULLABLE_FLOAT64)

    def replace_streams(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                replace_streams(child, _pointer(path, key))
        else:
            replace(path, TOKEN_INTEGER)

    streams = metadata["native_runtime_rng_manager_stream_indices"]
    replace_streams(streams, _pointer(metadata_base, "native_runtime_rng_manager_stream_indices"))

    checkpoint_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    for row in checkpoint_rows:
        name = row["tensor_name"]
        output = row["digest_inclusion"]["source_projection_output"]
        rank = row["canonical_rank"] if output == "canonical" else row["raw_rank"]
        manifest_path = _pointer("/checkpoint/tensor_manifest", name)
        replace(_pointer(manifest_path, "shape"), TOKEN_TENSOR_SHAPE, rank=rank)
        replace(_pointer(manifest_path, "sha256"), TOKEN_SHA256_HEX)
        stored_path = _pointer("/checkpoint/projection_fields/tensor_manifest", name)
        if name not in source["checkpoint"]["projection_fields"]["tensor_manifest"]:
            continue
        replace(_pointer(stored_path, "shape"), TOKEN_TENSOR_SHAPE, rank=rank)
        replace(_pointer(stored_path, "numel"), TOKEN_INTEGER)
        replace(_pointer(stored_path, "bytes"), TOKEN_INTEGER)
    for authority_name in (
        "trajectory_state_fields",
        "canonical_checkpoint_storage_fields",
    ):
        entries = source["checkpoint"][authority_name]
        for index, entry in enumerate(entries):
            path = _pointer(_pointer("/checkpoint", authority_name), index)
            replace(_pointer(path, "shape"), TOKEN_TENSOR_SHAPE, rank=len(entry["shape"]))
            replace(_pointer(path, "sha256"), TOKEN_SHA256_HEX)
    replace("/identity/active_agent_count", TOKEN_INTEGER)
    replace("/identity/tensor_sha256", TOKEN_SHA256_HEX)
    replace("/policy_mutation_tape/cursor", TOKEN_INTEGER)
    policy_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "policy_mutation_tape"
    ]
    for row in policy_rows:
        name = row["tensor_name"]
        replace(
            _pointer("/policy_mutation_tape/consumed_shapes", name),
            TOKEN_TENSOR_SHAPE,
            rank=row["canonical_rank"],
        )
        replace(_pointer("/policy_mutation_tape/tensor_sha256", name), TOKEN_SHA256_HEX)
    return placeholder


def _template(bundle: dict[str, Any], surface: str) -> dict[str, Any]:
    return next(template for template in bundle["templates"] if template["surface"] == surface)


def _expected_digest(surface: str, source: dict[str, Any]) -> str:
    checkpoint = source["checkpoint"]
    projection = checkpoint["projection_fields"]
    metadata = projection["metadata"]
    if surface == SURFACE_CHECKPOINT_TRAJECTORY:
        return checkpoint_layer_digest_from_tensor_manifest(
            layer=DigestLayer.TRAJECTORY_STATE,
            projection_schema_version=checkpoint["projection_schema_version"],
            world=source["world"],
            generation=metadata["generation"],
            tick=metadata["tick"],
            fields=checkpoint["trajectory_state_fields"],
        )
    if surface == SURFACE_CANONICAL_STORAGE:
        return checkpoint_layer_digest_from_tensor_manifest(
            layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
            projection_schema_version=checkpoint["projection_schema_version"],
            world=source["world"],
            generation=metadata["generation"],
            tick=metadata["tick"],
            fields=checkpoint["canonical_checkpoint_storage_fields"],
        )
    if surface == SURFACE_NEXT_GENERATION_SOURCE:
        return checkpoint_source_projection_digest_from_tensor_manifest(
            projection_fields=projection,
            tensor_manifest=checkpoint["tensor_manifest"],
        )
    if surface == SURFACE_RNG:
        return checkpoint_rng_digest_from_tensor_manifest(
            metadata=metadata,
            tensor_manifest=checkpoint["tensor_manifest"],
        )
    if surface == SURFACE_ACTIVE_AGENT_IDS:
        return active_agent_ids_digest_from_tensor_sha256(
            source["identity"]["active_agent_count"],
            source["identity"]["tensor_sha256"],
        )
    if surface == SURFACE_POLICY_MUTATION_TAPE:
        policy = source["policy_mutation_tape"]
        return policy_mutation_tape_digest_from_tensor_sha256(
            policy["cursor"], policy["consumed_shapes"], policy["tensor_sha256"]
        )
    raise AssertionError(surface)


def _rendered_digest(template: dict[str, Any], source: dict[str, Any]) -> str:
    values = retained_authority_token_values(template, source)
    return hashlib.sha256(render_compact_summary_host_template(template, values)).hexdigest()


@pytest.fixture(scope="module")
def compiled_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, skeleton = _plan_and_skeleton()
    settled = _settled_authority(plan, skeleton)
    placeholder = _placeholder_authority(settled, plan)
    bundle = compile_compact_summary_host_templates(skeleton, plan, 1, 1, placeholder)
    return plan, skeleton, settled, bundle


def test_compiler_is_deterministic_bounded_and_json_serializable(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    plan, skeleton, settled, actual = compiled_fixture
    placeholder = _placeholder_authority(settled, plan)
    repeated = compile_compact_summary_host_templates(skeleton, plan, 1, 1, placeholder)

    assert actual == repeated
    assert actual["schema_version"] == HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION
    assert actual["surface_order"] == list(SURFACE_ORDER)
    assert actual["template_count"] == 6
    assert json.loads(compact_summary_host_template_json_bytes(actual)) == actual
    for template in actual["templates"]:
        assert template["token_count"] == len(template["tokens"])
        assert template["maximum_rendered_utf8_bytes"] <= actual["bounds"][
            "maximum_rendered_utf8_bytes_per_template"
        ]
        assert all(part["kind"] in {"literal_utf8", "dynamic_token"} for part in template["parts"])


def test_renderer_matches_all_six_existing_outer_digest_helpers(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, _, settled, bundle = compiled_fixture

    assert {
        template["surface"]: _rendered_digest(template, settled)
        for template in bundle["templates"]
    } == {surface: _expected_digest(surface, settled) for surface in SURFACE_ORDER}


def test_generation_cell_matrix_is_generation_major_and_coordinate_isolated() -> None:
    plan, skeleton = _plan_and_skeleton()
    settled = [
        [_settled_authority(plan, skeleton, generation=g, cell=b) for b in range(3)]
        for g in range(2)
    ]
    placeholders = [
        [_placeholder_authority(settled[g][b], plan) for b in range(3)]
        for g in range(2)
    ]
    bundle = compile_compact_summary_host_templates(skeleton, plan, 2, 3, placeholders)

    assert bundle["template_count"] == 2 * 3 * len(SURFACE_ORDER)
    assert [
        (template["generation"], template["cell"], template["surface"])
        for template in bundle["templates"]
    ] == [
        (generation, cell, surface)
        for generation in range(2)
        for cell in range(3)
        for surface in SURFACE_ORDER
    ]
    target = next(
        template
        for template in bundle["templates"]
        if (template["generation"], template["cell"], template["surface"])
        == (1, 2, SURFACE_NEXT_GENERATION_SOURCE)
    )
    assert _rendered_digest(target, settled[1][2]) == _expected_digest(
        SURFACE_NEXT_GENERATION_SOURCE, settled[1][2]
    )


def test_runtime_event_cursor_is_an_exact_source_projection_token(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, _, settled, bundle = compiled_fixture
    template = _template(bundle, SURFACE_NEXT_GENERATION_SOURCE)
    path = "/checkpoint/projection_fields/runtime_event_cursor"
    assert next(token for token in template["tokens"] if token["source_path"] == path)[
        "token_type"
    ] == TOKEN_INTEGER
    changed = copy.deepcopy(settled)
    changed["checkpoint"]["projection_fields"]["runtime_event_cursor"] += 1

    assert _rendered_digest(template, changed) == _expected_digest(
        SURFACE_NEXT_GENERATION_SOURCE, changed
    )
    assert _rendered_digest(template, changed) != _rendered_digest(template, settled)


@pytest.mark.parametrize(
    "token_type,surface,mutate",
    [
        (
            TOKEN_INTEGER,
            SURFACE_NEXT_GENERATION_SOURCE,
            lambda source: source["checkpoint"]["projection_fields"].update(
                runtime_event_cursor=19
            ),
        ),
        (
            TOKEN_BOOLEAN,
            SURFACE_NEXT_GENERATION_SOURCE,
            lambda source: source["checkpoint"]["projection_fields"]["metadata"].update(
                native_policy_cache_served=True
            ),
        ),
        (
            TOKEN_PCG_UINT128,
            SURFACE_RNG,
            lambda source: (
                source["checkpoint"]["projection_fields"]["metadata"]
                ["native_runtime_numpy_env_state"]["state"].update(state=123456789),
                source["checkpoint"]["projection_fields"]["metadata"]
                ["native_runtime_numpy_policy_state"]["state"].update(state=123456789),
            ),
        ),
        (
            TOKEN_PYTHON_MT_WORD,
            SURFACE_RNG,
            lambda source: source["checkpoint"]["projection_fields"]["metadata"]
            ["native_runtime_python_random_state"]["state"].__setitem__(0, 987654321),
        ),
        (
            TOKEN_NULLABLE_FLOAT64,
            SURFACE_RNG,
            lambda source: source["checkpoint"]["projection_fields"]["metadata"]
            ["native_runtime_python_random_state"].update(gauss_next=-0.125),
        ),
        (
            TOKEN_TENSOR_SHAPE,
            SURFACE_POLICY_MUTATION_TAPE,
            lambda source: [
                source["policy_mutation_tape"]["consumed_shapes"][name].__setitem__(1, 3)
                for name in (
                    "policy_birth_crossover_mask_tape",
                    "policy_birth_delta_tape",
                    "policy_birth_mutation_mask_tape",
                )
            ],
        ),
        (
            TOKEN_SHA256_HEX,
            SURFACE_CHECKPOINT_TRAJECTORY,
            lambda source: source["checkpoint"]["trajectory_state_fields"][0].update(
                sha256=_hash("mutated-trajectory")
            ),
        ),
    ],
)
def test_renderer_differential_mutations_cover_every_token_class(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    token_type: str,
    surface: str,
    mutate: Any,
) -> None:
    _, _, settled, bundle = compiled_fixture
    template = _template(bundle, surface)
    assert token_type in {token["token_type"] for token in template["tokens"]}
    changed = copy.deepcopy(settled)
    mutate(changed)

    assert _rendered_digest(template, changed) == _expected_digest(surface, changed)


def test_renderer_rejects_wrong_token_value_types_and_noncanonical_hashes(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, _, settled, bundle = compiled_fixture
    template = _template(bundle, SURFACE_NEXT_GENERATION_SOURCE)
    values = retained_authority_token_values(template, settled)
    count_index = next(
        token["token_index"]
        for token in template["tokens"]
        if token["token_type"] == TOKEN_TENSOR_SHAPE
    )
    bad_shape = list(values)
    bad_shape[count_index] = [True]
    with pytest.raises(TypeError, match="must be an int"):
        render_compact_summary_host_template(template, bad_shape)

    hash_index = next(
        token["token_index"]
        for token in template["tokens"]
        if token["token_type"] == TOKEN_SHA256_HEX
    )
    bad_hash = list(values)
    bad_hash[hash_index] = "A" * 64
    with pytest.raises(ValueError, match="lowercase"):
        render_compact_summary_host_template(template, bad_hash)


def test_nullable_float64_gauss_cache_is_always_dynamic_and_packs_exactly(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, _, settled, bundle = compiled_fixture
    path = (
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_python_random_state/gauss_next"
    )
    representative = -float.fromhex("0x1.fffffffffffffp+1023")
    templates = [
        (_template(bundle, SURFACE_NEXT_GENERATION_SOURCE), 1),
        (_template(bundle, SURFACE_RNG), 2),
    ]

    for template, depth in templates:
        token = next(token for token in template["tokens"] if token["source_path"] == path)
        assert token["token_type"] == TOKEN_NULLABLE_FLOAT64
        assert token["validation_rule"] == {
            "kind": "finite_optional_float64",
            "render_pack_depth": depth,
        }
        assert token["maximum_utf8_bytes"] == len(
            host_template_module._validated_token_bytes(token, representative)
        )
        for value in (None, -0.125, representative):
            rendered = host_template_module._validated_token_bytes(token, value)
            packed: Any = value
            for _ in range(depth):
                packed = host_template_module._pack_concrete_value(packed)
            assert rendered == json.dumps(
                packed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")

    changed = copy.deepcopy(settled)
    changed["checkpoint"]["projection_fields"]["metadata"][
        "native_runtime_python_random_state"
    ]["gauss_next"] = representative
    for surface in (SURFACE_NEXT_GENERATION_SOURCE, SURFACE_RNG):
        template = _template(bundle, surface)
        assert _rendered_digest(template, changed) == _expected_digest(surface, changed)
        assert _rendered_digest(template, changed) != _rendered_digest(template, settled)


@pytest.mark.parametrize(
    "value,exception",
    [
        (False, TypeError),
        (0, TypeError),
        ("0.0", TypeError),
        (float("nan"), CompactSummaryHostTemplateError),
        (float("inf"), CompactSummaryHostTemplateError),
        (-float("inf"), CompactSummaryHostTemplateError),
    ],
)
def test_nullable_float64_renderer_rejects_every_other_value_class(
    compiled_fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    value: Any,
    exception: type[Exception],
) -> None:
    _, _, settled, bundle = compiled_fixture
    template = _template(bundle, SURFACE_NEXT_GENERATION_SOURCE)
    values = retained_authority_token_values(template, settled)
    token_index = next(
        token["token_index"]
        for token in template["tokens"]
        if token["token_type"] == TOKEN_NULLABLE_FLOAT64
    )
    values[token_index] = value

    with pytest.raises(exception, match="None or a finite Python float"):
        render_compact_summary_host_template(template, values)


def test_list_form_python_gauss_cache_element_two_is_always_dynamic() -> None:
    plan, skeleton = _plan_and_skeleton()
    python_state = skeleton["projection"]["metadata"][
        "native_runtime_python_random_state"
    ]
    skeleton["projection"]["metadata"]["native_runtime_python_random_state"] = [
        python_state["version"],
        [*python_state["state"], python_state["index"]],
        None,
    ]
    bundle = compile_flat_compact_summary_host_programs(skeleton, plan, 1, 1)
    path = (
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_python_random_state/2"
    )
    assert [
        token["token_type"]
        for template in bundle["templates"]
        for token in template["tokens"]
        if token["source_path"] == path
    ] == [TOKEN_NULLABLE_FLOAT64, TOKEN_NULLABLE_FLOAT64]

    settled = _settled_authority(plan, skeleton)
    settled["checkpoint"]["projection_fields"]["metadata"][
        "native_runtime_python_random_state"
    ][2] = 0.5
    for surface in (SURFACE_NEXT_GENERATION_SOURCE, SURFACE_RNG):
        template = _template(bundle, surface)
        assert _rendered_digest(template, settled) == _expected_digest(surface, settled)


def test_variable_cardinality_pending_free_authority_reports_exact_blocker() -> None:
    plan, skeleton = _plan_and_skeleton()
    settled = _settled_authority(plan, skeleton)
    placeholder = _placeholder_authority(settled, plan)
    placeholder["checkpoint"]["projection_fields"][
        "pending_generation_free_agent_ids"
    ] = [dynamic_token_placeholder(
        TOKEN_INTEGER,
        "/checkpoint/projection_fields/pending_generation_free_agent_ids/0",
    )]
    placeholder["checkpoint"]["projection_fields"][
        "pending_generation_free_slots"
    ] = [dynamic_token_placeholder(
        TOKEN_INTEGER,
        "/checkpoint/projection_fields/pending_generation_free_slots/0",
    )]

    with pytest.raises(
        CompactSummaryHostTemplateBlocker,
        match="variable-cardinality pending-free authority",
    ):
        compile_compact_summary_host_templates(skeleton, plan, 1, 1, placeholder)


def test_flat_program_helper_builds_generation_major_placeholder_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, skeleton = _plan_and_skeleton()
    captured: dict[str, Any] = {}
    sentinel = {"compiled": True}

    def capture_compile(
        checkpoint_projection_skeleton: Any,
        descriptor_plan: Any,
        generation_count: int,
        cell_count: int,
        placeholder_retained_authorities: Any,
    ) -> dict[str, bool]:
        captured.update(
            skeleton=checkpoint_projection_skeleton,
            plan=descriptor_plan,
            generation_count=generation_count,
            cell_count=cell_count,
            authorities=placeholder_retained_authorities,
        )
        return sentinel

    monkeypatch.setattr(
        host_template_module,
        "compile_compact_summary_host_templates",
        capture_compile,
    )
    actual = compile_flat_compact_summary_host_programs(skeleton, plan, 2, 1)

    assert actual is sentinel
    assert "compile_flat_compact_summary_host_programs" in host_template_module.__all__
    assert captured["skeleton"] is skeleton
    assert captured["plan"] is plan
    assert (captured["generation_count"], captured["cell_count"]) == (2, 1)
    authorities = captured["authorities"]
    assert [len(row) for row in authorities] == [1, 1]
    assert authorities[0][0] is not authorities[1][0]

    for generation, row in enumerate(authorities):
        authority = row[0]
        projection = authority["checkpoint"]["projection_fields"]
        metadata = projection["metadata"]
        assert authority["generation"] == dynamic_token_placeholder(
            TOKEN_INTEGER, "/generation"
        )
        assert authority["cell"] == dynamic_token_placeholder(TOKEN_INTEGER, "/cell")
        assert metadata["generation"] == dynamic_token_placeholder(
            TOKEN_INTEGER,
            "/checkpoint/projection_fields/metadata/generation",
        )
        assert metadata["agent_grid_authority_epoch"] == (
            f"generation:{generation + 1}:tick:0"
        )
        assert projection["generation_calls"] == dynamic_token_placeholder(
            TOKEN_INTEGER,
            "/checkpoint/projection_fields/generation_calls",
        )
        assert metadata["native_policy_cache_served"] == dynamic_token_placeholder(
            TOKEN_BOOLEAN,
            "/checkpoint/projection_fields/metadata/native_policy_cache_served",
        )
        assert metadata["native_runtime_numpy_env_state"]["state"][
            "state"
        ] == dynamic_token_placeholder(
            TOKEN_PCG_UINT128,
            "/checkpoint/projection_fields/metadata/"
            "native_runtime_numpy_env_state/state/state",
        )
        assert metadata["native_runtime_python_random_state"]["state"][
            0
        ] == dynamic_token_placeholder(
            TOKEN_PYTHON_MT_WORD,
            "/checkpoint/projection_fields/metadata/"
            "native_runtime_python_random_state/state/0",
        )
        assert metadata["native_runtime_python_random_state"][
            "gauss_next"
        ] == dynamic_token_placeholder(
            TOKEN_NULLABLE_FLOAT64,
            "/checkpoint/projection_fields/metadata/"
            "native_runtime_python_random_state/gauss_next",
        )
        assert metadata["policy_energy_observation_shape"] == dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            "/checkpoint/projection_fields/metadata/"
            "policy_energy_observation_shape",
            rank=1,
        )
        assert metadata[
            "policy_energy_observation_tensor_sha256"
        ] == dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            "/checkpoint/projection_fields/metadata/"
            "policy_energy_observation_tensor_sha256",
        )
        assert metadata["configured_steps_per_generation"] == dynamic_token_placeholder(
            TOKEN_INTEGER,
            "/checkpoint/projection_fields/metadata/"
            "configured_steps_per_generation",
        )
        assert metadata["fixed_float"] == skeleton["projection"]["metadata"][
            "fixed_float"
        ]
        assert metadata["fixed_string"] == skeleton["projection"]["metadata"][
            "fixed_string"
        ]

        checkpoint_row = plan["descriptors"][0]
        checkpoint_name = checkpoint_row["tensor_name"]
        checkpoint_manifest = authority["checkpoint"]["tensor_manifest"][
            checkpoint_name
        ]
        assert checkpoint_manifest["shape"] == dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            f"/checkpoint/tensor_manifest/{checkpoint_name}/shape",
            rank=checkpoint_row["raw_rank"],
        )
        assert checkpoint_manifest["sha256"] == dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            f"/checkpoint/tensor_manifest/{checkpoint_name}/sha256",
        )
        assert authority["identity"]["active_agent_count"] == dynamic_token_placeholder(
            TOKEN_INTEGER,
            "/identity/active_agent_count",
        )
        assert authority["identity"]["tensor_sha256"] == dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            "/identity/tensor_sha256",
        )
        assert authority["policy_mutation_tape"]["cursor"] == dynamic_token_placeholder(
            TOKEN_INTEGER,
            "/policy_mutation_tape/cursor",
        )
        policy_name = next(
            descriptor["tensor_name"]
            for descriptor in plan["descriptors"]
            if descriptor["source_scope"] == "policy_mutation_tape"
        )
        assert authority["policy_mutation_tape"]["consumed_shapes"][
            policy_name
        ] == dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            f"/policy_mutation_tape/consumed_shapes/{policy_name}",
            rank=2,
        )
        assert authority["policy_mutation_tape"]["tensor_sha256"][
            policy_name
        ] == dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            f"/policy_mutation_tape/tensor_sha256/{policy_name}",
        )


def test_flat_program_helper_g2_b1_is_deterministic_and_renders_all_surfaces() -> None:
    plan, skeleton = _plan_and_skeleton()
    actual = compile_flat_compact_summary_host_programs(skeleton, plan, 2, 1)
    repeated = compile_flat_compact_summary_host_programs(skeleton, plan, 2, 1)

    assert actual == repeated
    assert (actual["generation_count"], actual["cell_count"]) == (2, 1)
    assert actual["template_count"] == 2 * len(SURFACE_ORDER)
    assert [
        (template["generation"], template["cell"], template["surface"])
        for template in actual["templates"]
    ] == [
        (generation, 0, surface)
        for generation in range(2)
        for surface in SURFACE_ORDER
    ]
    assert {
        token["token_type"]
        for template in actual["templates"]
        for token in template["tokens"]
    } == {
        TOKEN_INTEGER,
        TOKEN_BOOLEAN,
        TOKEN_PCG_UINT128,
        TOKEN_PYTHON_MT_WORD,
        TOKEN_TENSOR_SHAPE,
        TOKEN_SHA256_HEX,
        TOKEN_NULLABLE_FLOAT64,
    }
    assert all(
        "/checkpoint/projection_fields/metadata/generation"
        in {token["source_path"] for token in template["tokens"]}
        for template in actual["templates"]
        if template["surface"] == SURFACE_NEXT_GENERATION_SOURCE
    )

    settled = [
        _settled_authority(plan, skeleton, generation=generation, cell=0)
        for generation in range(2)
    ]
    assert {
        (template["generation"], template["surface"]): _rendered_digest(
            template,
            settled[template["generation"]],
        )
        for template in actual["templates"]
    } == {
        (generation, surface): _expected_digest(surface, settled[generation])
        for generation in range(2)
        for surface in SURFACE_ORDER
    }


def test_device_neutral_compiler_preserves_flat_bundle_bytes() -> None:
    plan, skeleton = _plan_and_skeleton()
    frozen = compile_flat_compact_summary_host_programs(skeleton, plan, 2, 1)
    device_neutral = compile_device_neutral_compact_summary_host_programs(
        skeleton,
        plan,
        2,
        1,
    )

    assert compact_summary_host_template_json_bytes(device_neutral) == (
        compact_summary_host_template_json_bytes(frozen)
    )


def test_device_neutral_compiler_accepts_exact_patch_p2_registry_and_surfaces() -> None:
    plan, skeleton = _patch_plan_and_skeleton()
    bundle = compile_device_neutral_compact_summary_host_programs(
        skeleton,
        plan,
        1,
        1,
    )
    direct_projection, canonical_projection = _authoritative_patch_projection(
        plan,
        skeleton,
    )
    settled = _settled_patch_authority_from_projection(
        plan,
        skeleton,
        canonical_projection,
    )
    checkpoint_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    checkpoint_names = {row["tensor_name"] for row in checkpoint_rows}

    full_skeleton = copy.deepcopy(skeleton)
    full_manifest = full_skeleton["projection"]["tensor_manifest"]
    full_tensors = full_skeleton["projection"]["tensors"]
    for row in checkpoint_rows:
        name = row["tensor_name"]
        if name in full_manifest:
            continue
        output = row["digest_inclusion"]["source_projection_output"]
        full_manifest[name] = _stored_entry(row, _shape(row, output))
        full_tensors[name] = None
    full_bundle = compile_device_neutral_compact_summary_host_programs(
        full_skeleton,
        plan,
        1,
        1,
    )

    assert len(checkpoint_names) == 140
    assert set(full_manifest) == checkpoint_names
    assert compact_summary_host_template_json_bytes(full_bundle) == (
        compact_summary_host_template_json_bytes(bundle)
    )
    assert set(SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS) <= checkpoint_names
    assert set(SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS) <= checkpoint_names
    assert set(SOURCE_PATCH_DERIVED_TENSOR_FIELDS) <= checkpoint_names
    rendered_digests = {
        template["surface"]: _rendered_digest(template, settled)
        for template in bundle["templates"]
    }
    expected_digests = {
        surface: _expected_digest(surface, settled)
        for surface in SURFACE_ORDER
        if surface != SURFACE_NEXT_GENERATION_SOURCE
    }
    expected_digests[SURFACE_NEXT_GENERATION_SOURCE] = (
        checkpoint_source_projection_digest_manifest(direct_projection)["digest"]
    )
    assert rendered_digests == expected_digests

    source_template = _template(bundle, SURFACE_NEXT_GENERATION_SOURCE)
    source_manifest_names = {
        token["source_path"].split("/")[3]
        for token in source_template["tokens"]
        if token["source_path"].startswith("/checkpoint/tensor_manifest/")
    }
    stored_manifest_names = {
        token["source_path"].split("/")[4]
        for token in source_template["tokens"]
        if token["source_path"].startswith(
            "/checkpoint/projection_fields/tensor_manifest/"
        )
    }
    assert source_manifest_names == checkpoint_names
    assert stored_manifest_names == checkpoint_names - set(
        SOURCE_PATCH_DERIVED_TENSOR_FIELDS
    )


def test_device_neutral_compiler_fails_closed_on_patch_world_profile_and_registry() -> None:
    patch_plan, patch_skeleton = _patch_plan_and_skeleton()
    flat_plan, _ = _plan_and_skeleton()

    with pytest.raises(CompactSummaryHostTemplateError, match="descriptor_plan"):
        compile_device_neutral_compact_summary_host_programs(
            patch_skeleton,
            flat_plan,
            1,
            1,
        )

    unknown_world = copy.deepcopy(patch_skeleton)
    unknown_world["projection"]["metadata"]["world"] = "jungle"
    unknown_world["projection"]["last_world"] = "jungle"
    with pytest.raises(CompactSummaryHostTemplateError, match="flat.*patch"):
        compile_device_neutral_compact_summary_host_programs(
            unknown_world,
            patch_plan,
            1,
            1,
        )

    mismatched_world = copy.deepcopy(patch_skeleton)
    mismatched_world["projection"]["last_world"] = "flat"
    with pytest.raises(CompactSummaryHostTemplateError, match="exactly match"):
        compile_device_neutral_compact_summary_host_programs(
            mismatched_world,
            patch_plan,
            1,
            1,
        )

    p1 = copy.deepcopy(patch_skeleton)
    p1_metadata = p1["projection"]["metadata"]
    p1_metadata["native_runtime_patch_profile"] = "P1"
    p1_metadata["patch_movement_cost_apply_enable"] = False
    p1_metadata["patch_world_waste_consequence_enable"] = False
    with pytest.raises(CompactSummaryHostTemplateError, match="exact Patch P2"):
        compile_device_neutral_compact_summary_host_programs(
            p1,
            patch_plan,
            1,
            1,
        )

    missing = copy.deepcopy(patch_skeleton)
    missing["projection"]["tensor_manifest"].pop("patch_recovery_activity")
    missing["projection"]["tensors"].pop("patch_recovery_activity")
    with pytest.raises(CompactSummaryHostTemplateError, match="field registry drift"):
        compile_device_neutral_compact_summary_host_programs(
            missing,
            patch_plan,
            1,
            1,
        )

    unknown_field = copy.deepcopy(patch_skeleton)
    unknown_field["projection"]["tensor_manifest"]["patch_unknown"] = copy.deepcopy(
        unknown_field["projection"]["tensor_manifest"]["patch_cover"]
    )
    unknown_field["projection"]["tensors"]["patch_unknown"] = None
    with pytest.raises(CompactSummaryHostTemplateError, match="field registry drift"):
        compile_device_neutral_compact_summary_host_programs(
            unknown_field,
            patch_plan,
            1,
            1,
        )


def test_flat_program_helper_rejects_malformed_skeleton_and_plan() -> None:
    plan, skeleton = _plan_and_skeleton()
    unsupported = copy.deepcopy(skeleton)
    unsupported["projection"]["metadata"]["world"] = "patch"
    with pytest.raises(CompactSummaryHostTemplateError, match="only accepted Flat"):
        compile_flat_compact_summary_host_programs(unsupported, plan, 2, 1)

    pending = copy.deepcopy(skeleton)
    pending["projection"]["pending_generation_free_agent_ids"] = [7]
    with pytest.raises(
        CompactSummaryHostTemplateBlocker,
        match="variable-cardinality pending-free authority",
    ):
        compile_flat_compact_summary_host_programs(pending, plan, 2, 1)

    malformed_plan = copy.deepcopy(plan)
    malformed_plan["schema_version"] = "unsupported"
    with pytest.raises(CompactSummaryHostTemplateError, match="schema_version"):
        compile_flat_compact_summary_host_programs(skeleton, malformed_plan, 2, 1)
