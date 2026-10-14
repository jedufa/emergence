from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pybind11
import pytest

from engine.native_runtime.checkpoint_digests import (
    SCHEMA_VERSION as CHECKPOINT_LAYER_SCHEMA_VERSION,
    SOURCE_RNG_SCHEMA_VERSION,
    DigestLayer,
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
    compile_compact_summary_host_templates,
    dynamic_token_placeholder,
    render_compact_summary_host_template,
    retained_authority_token_values,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_template_renderer_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
BINDING = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_host_template_renderer_v1_binding.cpp"
)
MODULE_NAME = "cuda_architecture_b_compact_summary_host_template_renderer_v1_probe"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _pointer(base: str, component: str | int) -> str:
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def _set_pointer(root: Any, pointer: str, value: Any) -> None:
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    current = root
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    if isinstance(current, list):
        current[int(parts[-1])] = value
    else:
        current[parts[-1]] = value


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


def _exact_sidecar(
    source: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = bytes.fromhex(source["data_hex"])
    values = [value[0] for value in struct.iter_unpack("<f", raw)]
    exact = b"".join(struct.pack("<d", value) for value in values)
    shape = [int(value) for value in source["shape"]]
    tensor = {
        "data_hex": exact.hex(),
        "device": "cpu",
        "dtype": "float64",
        "kind": "tensor",
        "order": "C",
        "sha256": hashlib.sha256(exact).hexdigest(),
        "shape": shape,
    }
    return tensor, _stored_entry(row, shape)


def _real_settled_authority() -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    fixture_contract = plan["source_contract"]["accepted_flat_projection_fixture"]
    fixture_path = Path(fixture_contract["path"])
    raw_bytes = fixture_path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == fixture_contract["sha256"]
    fixture = json.loads(raw_bytes)
    skeleton = fixture["projection"]
    projection = copy.deepcopy(skeleton)
    tensors = projection.pop("tensors")

    checkpoint_rows = [
        row for row in plan["descriptors"] if row["source_scope"] == "checkpoint_projection"
    ]
    rows_by_name = {row["tensor_name"]: row for row in checkpoint_rows}
    for exact_name, source_name in (
        ("early_energy_collected_exact", "early_energy_collected"),
        ("energy_collected_exact", "energy_collected"),
    ):
        tensor, stored = _exact_sidecar(tensors[source_name], rows_by_name[exact_name])
        tensors[exact_name] = tensor
        projection["tensor_manifest"][exact_name] = stored

    tensor_manifest: dict[str, dict[str, Any]] = {}
    for row in checkpoint_rows:
        name = row["tensor_name"]
        entry = tensors[name]
        shape = [int(value) for value in entry["shape"]]
        assert len(shape) == (
            row["canonical_rank"]
            if row["digest_inclusion"]["source_projection_output"] == "canonical"
            else row["raw_rank"]
        )
        assert entry["dtype"] == row["dtype"]
        tensor_manifest[name] = {
            "device": "cpu",
            "dtype": row["dtype"],
            "shape": shape,
            "sha256": entry["sha256"],
        }

    def layer_fields(layer: DigestLayer) -> list[dict[str, Any]]:
        return [
            {
                "field": row["tensor_name"],
                "dtype": row["dtype"],
                "shape": list(tensor_manifest[row["tensor_name"]]["shape"]),
                "sha256": tensor_manifest[row["tensor_name"]]["sha256"],
            }
            for row in checkpoint_rows
            if str(layer) in row["digest_inclusion"]["checkpoint_layers"]
        ]

    policy_rows = [
        row for row in plan["descriptors"] if row["source_scope"] == "policy_mutation_tape"
    ]
    policy_shapes = {
        row["tensor_name"]: (
            [1, 4]
            if row["canonical_rank"] == 2
            else [1]
        )
        for row in policy_rows
    }
    active = tensor_manifest["agent_ids_by_alive_order"]
    settled = {
        "generation": 0,
        "cell": 0,
        "world": "flat",
        "checkpoint": {
            "source_version": CHECKPOINT_LAYER_SCHEMA_VERSION,
            "rng_source_version": SOURCE_RNG_SCHEMA_VERSION,
            "projection_schema_version": projection["schema_version"],
            "projection_fields": projection,
            "tensor_manifest": tensor_manifest,
            "trajectory_state_fields": layer_fields(DigestLayer.TRAJECTORY_STATE),
            "canonical_checkpoint_storage_fields": layer_fields(
                DigestLayer.CANONICAL_CHECKPOINT_STORAGE
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
                name: _hash(f"retained-{name}") for name in policy_shapes
            },
        },
    }
    return plan, fixture_path, settled


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
    python_state = metadata["native_runtime_python_random_state"]
    for index in range(len(python_state["state"])):
        replace(_pointer(_pointer(python_path, "state"), index), TOKEN_PYTHON_MT_WORD)
    replace(_pointer(python_path, "index"), TOKEN_INTEGER)
    replace(_pointer(python_path, "gauss_next"), TOKEN_NULLABLE_FLOAT64)

    def replace_streams(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                replace_streams(child, _pointer(path, key))
        else:
            replace(path, TOKEN_INTEGER)

    replace_streams(
        metadata["native_runtime_rng_manager_stream_indices"],
        _pointer(metadata_base, "native_runtime_rng_manager_stream_indices"),
    )

    checkpoint_rows = [
        row for row in plan["descriptors"] if row["source_scope"] == "checkpoint_projection"
    ]
    for row in checkpoint_rows:
        name = row["tensor_name"]
        output = row["digest_inclusion"]["source_projection_output"]
        rank = row["canonical_rank"] if output == "canonical" else row["raw_rank"]
        manifest_path = _pointer("/checkpoint/tensor_manifest", name)
        replace(_pointer(manifest_path, "shape"), TOKEN_TENSOR_SHAPE, rank=rank)
        replace(_pointer(manifest_path, "sha256"), TOKEN_SHA256_HEX)
        stored_path = _pointer("/checkpoint/projection_fields/tensor_manifest", name)
        replace(_pointer(stored_path, "shape"), TOKEN_TENSOR_SHAPE, rank=rank)
        replace(_pointer(stored_path, "numel"), TOKEN_INTEGER)
        replace(_pointer(stored_path, "bytes"), TOKEN_INTEGER)
    for authority_name in (
        "trajectory_state_fields",
        "canonical_checkpoint_storage_fields",
    ):
        for index, entry in enumerate(source["checkpoint"][authority_name]):
            path = _pointer(_pointer("/checkpoint", authority_name), index)
            replace(_pointer(path, "shape"), TOKEN_TENSOR_SHAPE, rank=len(entry["shape"]))
            replace(_pointer(path, "sha256"), TOKEN_SHA256_HEX)
    replace("/identity/active_agent_count", TOKEN_INTEGER)
    replace("/identity/tensor_sha256", TOKEN_SHA256_HEX)
    replace("/policy_mutation_tape/cursor", TOKEN_INTEGER)
    for row in plan["descriptors"]:
        if row["source_scope"] != "policy_mutation_tape":
            continue
        name = row["tensor_name"]
        replace(
            _pointer("/policy_mutation_tape/consumed_shapes", name),
            TOKEN_TENSOR_SHAPE,
            rank=row["canonical_rank"],
        )
        replace(_pointer("/policy_mutation_tape/tensor_sha256", name), TOKEN_SHA256_HEX)
    return placeholder


def _expected_digest(surface: str, source: dict[str, Any]) -> str:
    checkpoint = source["checkpoint"]
    projection = checkpoint["projection_fields"]
    metadata = projection["metadata"]
    if surface == SURFACE_CHECKPOINT_TRAJECTORY:
        return checkpoint_layer_digest_from_tensor_manifest(
            layer=DigestLayer.TRAJECTORY_STATE,
            projection_schema_version=checkpoint["projection_schema_version"],
            world="flat",
            generation=metadata["generation"],
            tick=metadata["tick"],
            fields=checkpoint["trajectory_state_fields"],
        )
    if surface == SURFACE_CANONICAL_STORAGE:
        return checkpoint_layer_digest_from_tensor_manifest(
            layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
            projection_schema_version=checkpoint["projection_schema_version"],
            world="flat",
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


def _template(bundle: dict[str, Any], surface: str) -> dict[str, Any]:
    return next(item for item in bundle["templates"] if item["surface"] == surface)


def _pack_concrete(value: Any) -> Any:
    if isinstance(value, dict):
        items = [(_pack_concrete(key), _pack_concrete(child)) for key, child in value.items()]
        items.sort(key=lambda row: json.dumps(row[0], sort_keys=True, separators=(",", ":")))
        return {"kind": "mapping", "items": items}
    if isinstance(value, list):
        return {"kind": "list", "items": [_pack_concrete(child) for child in value]}
    return {"kind": type(value).__name__, "value": value}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _nullable_template(depth: int, *, constrained_pair: bool = False) -> dict[str, Any]:
    representative: Any = -float.fromhex("0x1.fffffffffffffp+1023")
    for _ in range(depth):
        representative = _pack_concrete(representative)
    token_maximum = len(_canonical(representative))
    paths = ["/left", "/right"] if constrained_pair else ["/value"]
    tokens = [
        {
            "token_index": index,
            "token_type": TOKEN_NULLABLE_FLOAT64,
            "source_path": path,
            "validation_rule": {
                "kind": "finite_optional_float64",
                "render_pack_depth": depth,
            },
            "maximum_utf8_bytes": token_maximum,
        }
        for index, path in enumerate(paths)
    ]
    parts: list[dict[str, Any]] = []
    for index in range(len(tokens)):
        if index != 0:
            parts.append({"kind": "literal_utf8", "value": ","})
        parts.append({"kind": "dynamic_token", "token_index": index})
    constraints = (
        [
            {
                "kind": "values_equal",
                "left_source_path": "/left",
                "right_source_path": "/right",
            }
        ]
        if constrained_pair
        else []
    )
    return {
        "schema_version": "eiel_compact_summary_host_template_v1",
        "encoding": "utf-8",
        "surface": SURFACE_RNG,
        "generation": 0,
        "cell": 0,
        "token_count": len(tokens),
        "literal_utf8_bytes": max(0, len(tokens) - 1),
        "maximum_rendered_utf8_bytes": (
            len(tokens) * token_maximum + max(0, len(tokens) - 1)
        ),
        "tokens": tokens,
        "parts": parts,
        "constraints": constraints,
    }


@pytest.fixture(scope="module")
def compiled_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, fixture_path, settled = _real_settled_authority()
    placeholder = _placeholder_authority(settled, plan)
    bundle = compile_compact_summary_host_templates(
        fixture_path,
        plan,
        1,
        1,
        placeholder,
    )
    return settled, bundle


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    build = tmp_path_factory.mktemp("compact_summary_host_template_renderer_v1")
    objects: dict[str, Path] = {}
    for compiler in ("g++", "clang++"):
        output = build / f"renderer-{compiler.replace('+', 'p')}.o"
        completed = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-fPIC",
                "-fno-exceptions",
                "-fno-rtti",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                f"-I{ROOT / 'engine/perf'}",
                "-c",
                str(SOURCE),
                "-o",
                str(output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        symbols = subprocess.run(
            ["nm", "-C", "-u", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for forbidden in (
            "operator new",
            "operator delete",
            "__cxa_throw",
            "__cxa_allocate_exception",
            "__dynamic_cast",
            "typeinfo for",
            "__gxx_personality_v0",
        ):
            assert forbidden not in symbols
        objects[compiler] = output

    extension = build / f"{MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'engine/perf'}",
            str(BINDING),
            str(objects["g++"]),
            "-o",
            str(extension),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_pod_abi_and_all_six_real_fixture_surfaces(
    probe: ModuleType,
    compiled_fixture: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    settled, bundle = compiled_fixture
    assert probe.abi_layout() == {
        "abi_version": 1,
        "token_spec_bytes": 80,
        "token_value_bytes": 48,
        "part_bytes": 32,
        "constraint_bytes": 48,
        "program_bytes": 96,
        "request_bytes": 64,
        "result_bytes": 64,
    }
    assert [item["surface"] for item in bundle["templates"]] == list(SURFACE_ORDER)
    assert {
        token["validation_rule"]["render_pack_depth"]
        for item in bundle["templates"]
        for token in item["tokens"]
    } == {0, 1, 2}

    for template in bundle["templates"]:
        values = retained_authority_token_values(template, settled)
        reference = render_compact_summary_host_template(template, values)
        expected = _expected_digest(template["surface"], settled)
        assert hashlib.sha256(reference).hexdigest() == expected

        direct = probe.render(template, values)
        callback = probe.render(template, values, callback=True)
        hash_only = probe.render(template, values, {"mode": 0})
        for result in (direct, callback, hash_only):
            assert result["status_name"] == "success"
            assert result["digest"].hex() == expected
            assert result["rendered_utf8_bytes"] == len(reference)
        assert direct["rendered"] == reference
        assert callback["rendered"] == reference
        assert hash_only["rendered"] == b""


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "value",
    [
        None,
        -0.0,
        0.0,
        1.0,
        math.nextafter(1.0e-4, 0.0),
        1.0e-4,
        math.nextafter(1.0e16, 0.0),
        1.0e16,
        float.fromhex("0x0.0000000000001p-1022"),
        -float.fromhex("0x1.fffffffffffffp+1023"),
    ],
    ids=[
        "null",
        "negative-zero",
        "positive-zero",
        "integral-float",
        "below-fixed-threshold",
        "at-fixed-threshold",
        "below-scientific-threshold",
        "at-scientific-threshold",
        "minimum-subnormal",
        "negative-maximum",
    ],
)
def test_nullable_float64_is_byte_exact_at_all_pack_depths_and_thresholds(
    probe: ModuleType,
    depth: int,
    value: float | None,
) -> None:
    template = _nullable_template(depth)
    expected_value: Any = value
    for _ in range(depth):
        expected_value = _pack_concrete(expected_value)
    result = probe.render(template, [value])
    assert result["status_name"] == "success"
    assert result["rendered"] == _canonical(expected_value)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_nullable_float64_constraints_compare_presence_and_bits(
    probe: ModuleType,
    depth: int,
) -> None:
    template = _nullable_template(depth, constrained_pair=True)
    assert probe.render(template, [None, None])["status_name"] == "success"
    assert probe.render(template, [1.25, 1.25])["status_name"] == "success"
    _assert_failure(
        probe.render(template, [None, 1.25]),
        "constraint_value_contradiction",
    )
    _assert_failure(
        probe.render(template, [-0.0, 0.0]),
        "constraint_value_contradiction",
    )


def test_nullable_float64_rejects_noncanonical_or_nonfinite_values(
    probe: ModuleType,
) -> None:
    template = _nullable_template(0)
    _assert_failure(
        probe.render(template, [None], {"scalar_low": 1}),
        "token_value_bounds",
    )
    _assert_failure(
        probe.render(template, [None], {"scalar_high": 2}),
        "token_value_bounds",
    )
    _assert_failure(probe.render(template, [float("inf")]), "token_value_bounds")
    _assert_failure(probe.render(template, [float("nan")]), "token_value_bounds")


def _mutate_integer(source: dict[str, Any]) -> None:
    source["checkpoint"]["projection_fields"]["runtime_event_cursor"] += 7


def _mutate_boolean(source: dict[str, Any]) -> None:
    source["checkpoint"]["projection_fields"]["metadata"][
        "native_policy_cache_served"
    ] = True


def _mutate_pcg(source: dict[str, Any]) -> None:
    metadata = source["checkpoint"]["projection_fields"]["metadata"]
    for name in ("native_runtime_numpy_env_state", "native_runtime_numpy_policy_state"):
        metadata[name]["state"]["state"] += 1


def _mutate_mt(source: dict[str, Any]) -> None:
    source["checkpoint"]["projection_fields"]["metadata"][
        "native_runtime_python_random_state"
    ]["state"][0] ^= 0x1234


def _mutate_nullable(source: dict[str, Any]) -> None:
    source["checkpoint"]["projection_fields"]["metadata"][
        "native_runtime_python_random_state"
    ]["gauss_next"] = math.nextafter(1.0e-4, 0.0)


def _mutate_shape(source: dict[str, Any]) -> None:
    shapes = source["policy_mutation_tape"]["consumed_shapes"]
    for name in (
        "policy_birth_crossover_mask_tape",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
    ):
        shapes[name][1] += 1


def _mutate_sha(source: dict[str, Any]) -> None:
    source["checkpoint"]["trajectory_state_fields"][0]["sha256"] = _hash(
        "mutated-real-fixture-trajectory"
    )


@pytest.mark.parametrize(
    "token_type,surface,mutate",
    [
        (TOKEN_INTEGER, SURFACE_NEXT_GENERATION_SOURCE, _mutate_integer),
        (TOKEN_BOOLEAN, SURFACE_NEXT_GENERATION_SOURCE, _mutate_boolean),
        (TOKEN_PCG_UINT128, SURFACE_RNG, _mutate_pcg),
        (TOKEN_PYTHON_MT_WORD, SURFACE_RNG, _mutate_mt),
        (TOKEN_NULLABLE_FLOAT64, SURFACE_RNG, _mutate_nullable),
        (TOKEN_TENSOR_SHAPE, SURFACE_POLICY_MUTATION_TAPE, _mutate_shape),
        (TOKEN_SHA256_HEX, SURFACE_CHECKPOINT_TRAJECTORY, _mutate_sha),
    ],
)
def test_differential_mutations_cover_every_compiler_token_class(
    probe: ModuleType,
    compiled_fixture: tuple[dict[str, Any], dict[str, Any]],
    token_type: str,
    surface: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    settled, bundle = compiled_fixture
    changed = copy.deepcopy(settled)
    mutate(changed)
    template = _template(bundle, surface)
    assert token_type in {token["token_type"] for token in template["tokens"]}
    values = retained_authority_token_values(template, changed)
    reference = render_compact_summary_host_template(template, values)
    result = probe.render(template, values)
    assert result["status_name"] == "success"
    assert result["rendered"] == reference
    assert result["digest"].hex() == _expected_digest(surface, changed)


def _assert_failure(result: dict[str, Any], status: str) -> None:
    assert result["status_name"] == status
    assert result["rendered_utf8_bytes"] == 0
    assert result["rendered"] == b""
    assert result["digest"] == bytes(32)


def test_value_bounds_shape_cursor_and_rng_alias_fail_closed(
    probe: ModuleType,
    compiled_fixture: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    settled, bundle = compiled_fixture

    source_template = _template(bundle, SURFACE_NEXT_GENERATION_SOURCE)
    source_values = retained_authority_token_values(source_template, settled)
    integer_index = next(
        token["token_index"]
        for token in source_template["tokens"]
        if token["token_type"] == TOKEN_INTEGER
    )
    wrong_type = list(source_values)
    wrong_type[integer_index] = True
    _assert_failure(
        probe.render(source_template, wrong_type),
        "token_value_type_mismatch",
    )
    boolean_index = next(
        token["token_index"]
        for token in source_template["tokens"]
        if token["token_type"] == TOKEN_BOOLEAN
    )
    bad_boolean = list(source_values)
    bad_boolean[boolean_index] = 2
    _assert_failure(probe.render(source_template, bad_boolean), "token_value_bounds")
    sha_index = next(
        token["token_index"]
        for token in source_template["tokens"]
        if token["token_type"] == TOKEN_SHA256_HEX
    )
    bad_sha = list(source_values)
    bad_sha[sha_index] = "A" * 64
    _assert_failure(
        probe.render(source_template, bad_sha),
        "token_sha256_contradiction",
    )

    rng_template = _template(bundle, SURFACE_RNG)
    rng_values = retained_authority_token_values(rng_template, settled)
    mt_index = next(
        token["token_index"]
        for token in rng_template["tokens"]
        if token["token_type"] == TOKEN_PYTHON_MT_WORD
    )
    bad_word = list(rng_values)
    bad_word[mt_index] = 1 << 32
    _assert_failure(probe.render(rng_template, bad_word), "token_value_bounds")
    alias_index = next(
        token["token_index"]
        for token in rng_template["tokens"]
        if "native_runtime_numpy_env_state/state/state" in token["source_path"]
    )
    alias_drift = list(rng_values)
    alias_drift[alias_index] += 1
    _assert_failure(
        probe.render(rng_template, alias_drift),
        "constraint_value_contradiction",
    )

    policy_template = _template(bundle, SURFACE_POLICY_MUTATION_TAPE)
    policy_values = retained_authority_token_values(policy_template, settled)
    shape_index = next(
        token["token_index"]
        for token in policy_template["tokens"]
        if token["token_type"] == TOKEN_TENSOR_SHAPE
    )
    wrong_rank = list(policy_values)
    wrong_rank[shape_index] = [1]
    _assert_failure(
        probe.render(policy_template, wrong_rank),
        "token_shape_contradiction",
    )
    cursor_index = next(
        token["token_index"]
        for token in policy_template["tokens"]
        if token["source_path"] == "/policy_mutation_tape/cursor"
    )
    cursor_drift = copy.deepcopy(policy_values)
    cursor_drift[cursor_index] += 1
    _assert_failure(
        probe.render(policy_template, cursor_drift),
        "constraint_value_contradiction",
    )
    mask_index = next(
        token["token_index"]
        for token in policy_template["tokens"]
        if "/consumed_shapes/" in token["source_path"]
        and token["source_path"].endswith("/policy_birth_crossover_mask_tape")
    )
    shape_alias_drift = copy.deepcopy(policy_values)
    shape_alias_drift[mask_index][1] += 1
    _assert_failure(
        probe.render(policy_template, shape_alias_drift),
        "constraint_value_contradiction",
    )


@pytest.mark.parametrize(
    "surface,options,status",
    [
        (SURFACE_NEXT_GENERATION_SOURCE, {"program_abi_version": 99}, "unsupported_abi_version"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"encoding": 99}, "unsupported_encoding"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"surface": 99}, "unsupported_surface"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"program_reserved": 1}, "reserved_field_nonzero"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"token_type": 99}, "unsupported_token_type"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"validation_kind": 99}, "unsupported_validation_rule"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"pack_depth": 3}, "token_declaration_contradiction"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"token_maximum_utf8_bytes": 1}, "token_declaration_contradiction"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"duplicate_source_path_with": 1}, "token_declaration_contradiction"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"part_kind": 99}, "unsupported_part_kind"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"part_token_index": 0}, "part_bounds"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"invalid_literal_utf8": True}, "invalid_utf8"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"literal_utf8_bytes": 0}, "rendered_bound_contradiction"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"maximum_rendered_utf8_bytes": 1}, "rendered_bound_contradiction"),
        (SURFACE_NEXT_GENERATION_SOURCE, {"token_value_count": 0}, "token_value_count_mismatch"),
        (SURFACE_POLICY_MUTATION_TAPE, {"constraint_kind": 99}, "unsupported_constraint_kind"),
        (SURFACE_POLICY_MUTATION_TAPE, {"constraint_left_path": "/missing"}, "constraint_path_contradiction"),
    ],
)
def test_malformed_programs_fail_closed(
    probe: ModuleType,
    compiled_fixture: tuple[dict[str, Any], dict[str, Any]],
    surface: str,
    options: dict[str, Any],
    status: str,
) -> None:
    settled, bundle = compiled_fixture
    template = _template(bundle, surface)
    values = retained_authority_token_values(template, settled)
    _assert_failure(probe.render(template, values, options), status)


def test_output_overrun_is_detected_before_writing(
    probe: ModuleType,
    compiled_fixture: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    settled, bundle = compiled_fixture
    template = _template(bundle, SURFACE_ACTIVE_AGENT_IDS)
    values = retained_authority_token_values(template, settled)
    rendered = render_compact_summary_host_template(template, values)
    _assert_failure(
        probe.render(template, values, {"output_capacity": len(rendered) - 1}),
        "output_overrun",
    )
