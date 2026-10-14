"""Strict, route-neutral authoring contract for EIEL_SCIENCE_REQUEST_V1.

This module validates and expands the compact authoring form. It deliberately
does not select a provider, compile a campaign, perform admission, create
execution identity, read repository schemas, or grant science/qualification
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import re
from typing import Final

from ._validation import INT64_MAX, require_instance, require_int
from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .payloads import CanonicalPayloadV1
from .adaptive_science_expedition_v1 import StateClassV1, StateRefV1
from .artifacts import RunArtifactRefV1


EIEL_SCIENCE_REQUEST_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_request_v1"
)
EIEL_SCIENCE_REQUEST_EXPANDED_WORLD_SCHEMA_VERSION: Final[SchemaVersion] = (
    SchemaVersion("eiel_science_request_expanded_world_v1")
)
EIEL_SCIENCE_REQUEST_DIMENSION_VALUES_SCHEMA_VERSION: Final[SchemaVersion] = (
    SchemaVersion("eiel_science_request_dimension_values_v1")
)
EIEL_SCIENCE_REQUEST_RESOLVED_TEMPLATE_SCHEMA_VERSION: Final[SchemaVersion] = (
    SchemaVersion("eiel_science_request_resolved_world_template_v1")
)
EIEL_SCIENCE_REQUEST_BATCH_PLAN_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_request_batch_plan_v1"
)
EIEL_SCIENCE_SELECTED_WORLD_CONTINUATION_SCHEMA_VERSION: Final[SchemaVersion] = (
    SchemaVersion("eiel_science_selected_world_continuation_v1")
)

BATCH_POLICY_AT_MOST: Final[str] = "AT_MOST"
BATCH_POLICY_EXACT: Final[str] = "EXACT"

_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._+-]*$")
_DIMENSION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z][A-Za-z0-9_]*)\}$")
_PLACEHOLDER_OPEN = "$" + "{"

_TOP_LEVEL_REQUIRED = frozenset(
    {
        "schema_version",
        "campaign_id",
        "test",
        "question",
        "world",
        "run",
        "dimensions",
        "world_template",
        "replicate_dimension",
        "observe",
        "reduce",
        "expect_worlds",
    }
)
_TOP_LEVEL_OPTIONAL = frozenset(
    {"exclude", "continuation", "exact_natural_source_state_bank"}
)
_REQUIRED_SEED_OWNERS = frozenset(
    {
        "ecology",
        "founder",
        "initialization",
        "intervention",
        "mutation",
        "policy",
        "task_environment",
    }
)


def _fail(message: str, *, field: str) -> None:
    raise contract_invalid(message, field=field)


def _exact_mapping(
    value: object,
    *,
    field: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if type(value) is not dict:
        _fail("value must be a JSON object", field=field)
    assert isinstance(value, dict)
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        _fail(
            "object fields differ from the strict contract"
            f" (missing={sorted(missing)}, extra={sorted(extra)})",
            field=field,
        )
    if any(type(key) is not str for key in value):
        _fail("JSON object keys must be strings", field=field)
    return value


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail("value must be a JSON object", field=field)
    assert isinstance(value, dict)
    if any(type(key) is not str for key in value):
        _fail("JSON object keys must be strings", field=field)
    return value


def _array(value: object, *, field: str, nonempty: bool = True) -> list[object]:
    if type(value) is not list:
        _fail("value must be a JSON array", field=field)
    assert isinstance(value, list)
    if nonempty and not value:
        _fail("array must not be empty", field=field)
    return value


def _text(value: object, *, field: str, token: bool = False) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail("value must be a non-empty, unpadded text string", field=field)
    assert isinstance(value, str)
    if token and _TOKEN_RE.fullmatch(value) is None:
        _fail("value must be an ASCII contract token", field=field)
    return value


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        _fail("value must be an exact boolean", field=field)
    assert isinstance(value, bool)
    return value


def _integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = INT64_MAX,
) -> int:
    return require_int(value, field=field, minimum=minimum, maximum=maximum)


def _dimension_scalar(value: object, *, field: str) -> bool | int | float | str:
    if type(value) is bool:
        return value
    if type(value) is int:
        _integer(value, field=field, minimum=-(1 << 63), maximum=INT64_MAX)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail("dimension numbers must be finite", field=field)
        return value
    if type(value) is str:
        _text(value, field=field)
        if _PLACEHOLDER_OPEN in value:
            _fail("dimension values cannot contain placeholders", field=field)
        return value
    _fail(
        "dimension values must be exact JSON booleans, integers, numbers, or strings",
        field=field,
    )


def _typed_key(value: bool | int | float | str) -> tuple[str, bytes]:
    return (type(value).__name__, canonical_json_bytes(value))


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and canonical_json_bytes(
        left
    ) == canonical_json_bytes(right)


def _validate_generation_firewall(value: object, *, field: str = "$") -> None:
    if type(value) is dict:
        assert isinstance(value, dict)
        for key, item in value.items():
            if type(key) is not str:
                _fail("JSON object keys must be strings", field=field)
            if "generation" in key.casefold():
                _fail(
                    "generation-shaped fields are forbidden; absolute ticks are authoritative",
                    field=f"{field}.{key}",
                )
            _validate_generation_firewall(item, field=f"{field}.{key}")
    elif type(value) is list:
        assert isinstance(value, list)
        for index, item in enumerate(value):
            _validate_generation_firewall(item, field=f"{field}[{index}]")


def _validate_unique_text_array(value: object, *, field: str) -> tuple[str, ...]:
    rows = _array(value, field=field)
    result = tuple(
        _text(item, field=f"{field}[{index}]") for index, item in enumerate(rows)
    )
    if len(result) != len(set(result)):
        _fail("array values must be unique", field=field)
    return result


def _validate_sha256_id(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 71 or not text.startswith("sha256:") or any(
        character not in "0123456789abcdef" for character in text[7:]
    ):
        _fail("value must be a lowercase sha256 identity", field=field)
    return text


def _validate_source_bank_pointer(value: object, *, field: str) -> None:
    pointer = _exact_mapping(
        value,
        field=field,
        required=frozenset({"path", "file_sha256", "bank_sha256"}),
    )
    _text(pointer["path"], field=f"{field}.path")
    for name in ("file_sha256", "bank_sha256"):
        digest = pointer[name]
        if type(digest) is not str or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            _fail("value must be a lowercase sha256 digest", field=f"{field}.{name}")


def _validate_tick_array(
    value: object,
    *,
    field: str,
    total_ticks: int,
    allow_zero: bool,
    require_zero: bool,
    start_tick: int = 0,
) -> tuple[int, ...]:
    rows = _array(value, field=field)
    minimum = start_tick if allow_zero else start_tick + 1
    ticks = tuple(
        _integer(
            item,
            field=f"{field}[{index}]",
            minimum=minimum,
            maximum=total_ticks,
        )
        for index, item in enumerate(rows)
    )
    if ticks != tuple(sorted(set(ticks))):
        _fail("absolute ticks must be sorted and unique", field=field)
    if require_zero and ticks[0] != start_tick:
        _fail(
            (
                "snapshot ticks must start at absolute tick 0"
                if start_tick == 0
                else "snapshot ticks must start at the selected parent frontier"
            ),
            field=field,
        )
    if ticks[-1] != total_ticks:
        _fail("absolute tick schedule must end at run.ticks", field=field)
    return ticks


def _validate_template_value(
    value: object,
    *,
    field: str,
    dimensions: frozenset[str],
    referenced_dimensions: set[str],
) -> None:
    value_type = type(value)
    if value is None:
        _fail("world templates cannot contain null", field=field)
    if value_type in {bool, int}:
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail("world template numbers must be finite", field=field)
        return
    if value_type is str:
        assert isinstance(value, str)
        _text(value, field=field)
        match = _PLACEHOLDER_RE.fullmatch(value)
        if match is not None:
            name = match.group(1)
            if name not in dimensions:
                _fail("placeholder names an unknown dimension", field=field)
            referenced_dimensions.add(name)
            return
        if _PLACEHOLDER_OPEN in value:
            _fail("placeholders must occupy the whole string", field=field)
        return
    if value_type is list:
        assert isinstance(value, list)
        for index, item in enumerate(value):
            _validate_template_value(
                item,
                field=f"{field}[{index}]",
                dimensions=dimensions,
                referenced_dimensions=referenced_dimensions,
            )
        return
    if value_type is dict:
        assert isinstance(value, dict)
        # A dimension lookup is a generic authoring convenience for controls
        # whose value is a deterministic function of one categorical arm.
        # It resolves before the normal template contract is lowered; it is
        # not a runtime control or a route-specific selector.
        if set(value) == {"dimension_lookup"}:
            lookup = value["dimension_lookup"]
            if not isinstance(lookup, dict) or set(lookup) != {
                "dimension",
                "cases",
            }:
                _fail(
                    "dimension_lookup must contain dimension and cases",
                    field=field,
                )
            dimension = lookup["dimension"]
            if not isinstance(dimension, str) or dimension not in dimensions:
                _fail(
                    "dimension_lookup names an unknown dimension",
                    field=f"{field}.dimension_lookup.dimension",
                )
            referenced_dimensions.add(dimension)
            cases = lookup["cases"]
            if not isinstance(cases, list) or not cases:
                _fail(
                    "dimension_lookup cases must be a nonempty array",
                    field=f"{field}.dimension_lookup.cases",
                )
            seen: set[object] = set()
            for index, case in enumerate(cases):
                if not isinstance(case, dict) or set(case) != {"when", "value"}:
                    _fail(
                        "dimension_lookup cases must contain when and value",
                        field=f"{field}.dimension_lookup.cases[{index}]",
                    )
                marker = case["when"]
                if type(marker) not in {bool, int, float, str}:
                    _fail(
                        "dimension_lookup case markers must be scalar",
                        field=f"{field}.dimension_lookup.cases[{index}].when",
                    )
                if marker in seen:
                    _fail(
                        "dimension_lookup case markers must be unique",
                        field=f"{field}.dimension_lookup.cases",
                    )
                seen.add(marker)
                _validate_template_value(
                    case["value"],
                    field=f"{field}.dimension_lookup.cases[{index}].value",
                    dimensions=dimensions,
                    referenced_dimensions=referenced_dimensions,
                )
            return
        for key, item in value.items():
            _text(key, field=f"{field}.<key>")
            if _PLACEHOLDER_OPEN in key:
                _fail("placeholders are forbidden in object keys", field=field)
            _validate_template_value(
                item,
                field=f"{field}.{key}",
                dimensions=dimensions,
                referenced_dimensions=referenced_dimensions,
            )
        return
    _fail("world templates must contain only JSON values", field=field)


def _substitute_template(
    value: object,
    *,
    dimension_values: dict[str, bool | int | float | str],
) -> object:
    if type(value) is str:
        assert isinstance(value, str)
        match = _PLACEHOLDER_RE.fullmatch(value)
        if match is not None:
            return dimension_values[match.group(1)]
        return value
    if type(value) is list:
        assert isinstance(value, list)
        return [
            _substitute_template(item, dimension_values=dimension_values)
            for item in value
        ]
    if type(value) is dict:
        assert isinstance(value, dict)
        if set(value) == {"dimension_lookup"}:
            lookup = value["dimension_lookup"]
            if not isinstance(lookup, dict) or set(lookup) != {
                "dimension",
                "cases",
            }:
                raise ValueError("dimension_lookup contract is invalid")
            dimension = lookup["dimension"]
            if dimension not in dimension_values:
                raise ValueError("dimension_lookup names an unknown dimension")
            cases = lookup["cases"]
            if not isinstance(cases, list):
                raise ValueError("dimension_lookup cases are invalid")
            matches = [
                case["value"]
                for case in cases
                if isinstance(case, dict)
                and set(case) == {"when", "value"}
                and case["when"] == dimension_values[dimension]
            ]
            if len(matches) != 1:
                raise ValueError("dimension_lookup has no unique matching case")
            return _substitute_template(
                matches[0], dimension_values=dimension_values
            )
        return {
            key: _substitute_template(item, dimension_values=dimension_values)
            for key, item in value.items()
        }
    return value


def _validate_resolved_absolute_ticks(
    value: object,
    *,
    field: str,
    total_ticks: int,
    start_tick: int = 0,
) -> None:
    if type(value) is dict:
        assert isinstance(value, dict)
        for key, item in value.items():
            item_field = f"{field}.{key}"
            if key.endswith("_tick"):
                _integer(
                    item,
                    field=item_field,
                    minimum=start_tick,
                    maximum=total_ticks,
                )
            elif key.endswith("_ticks") and key not in {
                "checkpoint_ticks",
                "snapshot_ticks",
            }:
                if type(item) is int:
                    _integer(
                        item,
                        field=item_field,
                        minimum=start_tick,
                        maximum=total_ticks,
                    )
                elif type(item) is list:
                    for index, tick in enumerate(item):
                        _integer(
                            tick,
                            field=f"{item_field}[{index}]",
                            minimum=start_tick,
                            maximum=total_ticks,
                        )
                else:
                    _fail(
                        "tick fields must be integer or integer arrays",
                        field=item_field,
                    )
            _validate_resolved_absolute_ticks(
                item,
                field=item_field,
                total_ticks=total_ticks,
                start_tick=start_tick,
            )
    elif type(value) is list:
        assert isinstance(value, list)
        for index, item in enumerate(value):
            _validate_resolved_absolute_ticks(
                item,
                field=f"{field}[{index}]",
                total_ticks=total_ticks,
                start_tick=start_tick,
            )


def _selected_world_continuation(
    value: object,
) -> tuple[StateRefV1, RunArtifactRefV1] | None:
    if value is None:
        return None
    row = _exact_mapping(
        value,
        field="continuation",
        required=frozenset(
            {"schema_version", "state_ref", "authority_receipt_ref"}
        ),
    )
    if row["schema_version"] != str(
        EIEL_SCIENCE_SELECTED_WORLD_CONTINUATION_SCHEMA_VERSION
    ):
        _fail("unsupported selected-world continuation schema", field="continuation")
    try:
        state_ref = StateRefV1.from_dict(row["state_ref"])
        authority_ref = RunArtifactRefV1.from_dict(row["authority_receipt_ref"])
    except Exception as exc:
        _fail(
            f"selected-world continuation identity is invalid: {exc}",
            field="continuation",
        )
    if state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT:
        _fail(
            "selected-world continuation requires BRANCHABLE_EXACT state",
            field="continuation.state_ref.state_class",
        )
    return state_ref, authority_ref


def _validate_resolved_initialization(
    value: object,
    *,
    world: dict[str, object],
    field: str,
) -> None:
    initialization = _exact_mapping(
        value,
        field=field,
        required=frozenset(
            {
                "initial_population",
                "initial_live_count",
                "founder_profiles",
                "exact_founder_composition",
                "spatial_placement_profile",
                "standing_variation_profile",
            }
        ),
        optional=frozenset(
            {
                "exact_natural_source_composition",
                "exact_natural_source_states",
                "exact_natural_source_state_bank_binding",
            }
        ),
    )
    population = _integer(
        initialization["initial_population"],
        field=f"{field}.initial_population",
        minimum=1,
    )
    live = _integer(
        initialization["initial_live_count"],
        field=f"{field}.initial_live_count",
        minimum=1,
    )
    if population != world["population"]:
        _fail("initial population must equal world.population", field=field)
    if live > population:
        _fail("initial live count exceeds initial population", field=field)

    founder_profiles = _validate_unique_text_array(
        initialization["founder_profiles"],
        field=f"{field}.founder_profiles",
    )
    composition_rows = _array(
        initialization["exact_founder_composition"],
        field=f"{field}.exact_founder_composition",
    )
    composition_profiles: list[str] = []
    composition_count = 0
    for index, raw_row in enumerate(composition_rows):
        row_field = f"{field}.exact_founder_composition[{index}]"
        row = _exact_mapping(
            raw_row,
            field=row_field,
            required=frozenset({"founder_profile_id", "count"}),
        )
        composition_profiles.append(
            _text(row["founder_profile_id"], field=f"{row_field}.founder_profile_id")
        )
        composition_count += _integer(
            row["count"], field=f"{row_field}.count", minimum=0
        )
    if len(composition_profiles) != len(set(composition_profiles)):
        _fail("founder composition profile IDs must be unique", field=field)
    if tuple(composition_profiles) != founder_profiles:
        _fail(
            "founder_profiles must exactly match composition order",
            field=field,
        )
    if composition_count != live:
        _fail("founder composition count must equal initial_live_count", field=field)
    if not any(
        type(row["count"]) is int and row["count"] > 0
        for row in composition_rows
    ):
        _fail("founder composition must contain at least one live founder", field=field)
    source_rows = initialization.get("exact_natural_source_composition")
    if source_rows is not None:
        source_rows = _array(
            source_rows,
            field=f"{field}.exact_natural_source_composition",
        )
        source_count = 0
        source_positive = False
        for index, raw_row in enumerate(source_rows):
            row_field = f"{field}.exact_natural_source_composition[{index}]"
            row = _exact_mapping(
                raw_row,
                field=row_field,
                required=frozenset({"source_state_ref", "count"}),
                optional=frozenset({"source_state_role"}),
            )
            _text(row["source_state_ref"], field=f"{row_field}.source_state_ref")
            if "source_state_role" in row:
                _text(row["source_state_role"], field=f"{row_field}.source_state_role")
            count = _integer(row["count"], field=f"{row_field}.count", minimum=0)
            source_count += count
            source_positive = source_positive or count > 0
        if source_count != live:
            _fail(
                "exact natural source composition count must equal initial_live_count",
                field=field,
            )
        if not source_positive:
            _fail(
                "exact natural source composition must contain a live state",
                field=field,
            )
    if "exact_natural_source_states" in initialization:
        _array(
            initialization["exact_natural_source_states"],
            field=f"{field}.exact_natural_source_states",
        )
    if "exact_natural_source_state_bank_binding" in initialization:
        _mapping(
            initialization["exact_natural_source_state_bank_binding"],
            field=f"{field}.exact_natural_source_state_bank_binding",
        )
    _text(
        initialization["spatial_placement_profile"],
        field=f"{field}.spatial_placement_profile",
    )
    _text(
        initialization["standing_variation_profile"],
        field=f"{field}.standing_variation_profile",
    )


def _validate_resolved_template(
    value: object,
    *,
    world: dict[str, object],
    total_ticks: int,
    start_tick: int = 0,
    selected_world_continuation: bool = False,
) -> None:
    template = _exact_mapping(
        value,
        field="expanded_world.world_template",
        required=(
            frozenset({"controls"})
            if selected_world_continuation
            else frozenset({"initialization", "controls", "seeds"})
        ),
    )
    if not selected_world_continuation:
        _validate_resolved_initialization(
            template["initialization"],
            world=world,
            field="expanded_world.world_template.initialization",
        )
    controls = _mapping(
        template["controls"], field="expanded_world.world_template.controls"
    )
    if not controls:
        _fail(
            "at least one control is required",
            field="expanded_world.world_template.controls",
        )
    for control_id in controls:
        _text(
            control_id,
            field="expanded_world.world_template.controls.<control_id>",
        )
    if not selected_world_continuation:
        seeds = _mapping(
            template["seeds"], field="expanded_world.world_template.seeds"
        )
        if set(seeds) != _REQUIRED_SEED_OWNERS:
            _fail(
                "seed owners must exactly match the native science owner set",
                field="expanded_world.world_template.seeds",
            )
        for owner, seed in seeds.items():
            _integer(
                seed,
                field=f"expanded_world.world_template.seeds.{owner}",
                minimum=0,
            )
    _validate_resolved_absolute_ticks(
        template,
        field="expanded_world.world_template",
        total_ticks=total_ticks,
        start_tick=start_tick,
    )


def _dimensions(
    raw_dimensions: object,
) -> dict[str, tuple[bool | int | float | str, ...]]:
    dimensions = _mapping(raw_dimensions, field="dimensions")
    if not dimensions:
        _fail("at least one dimension is required", field="dimensions")
    result: dict[str, tuple[bool | int | float | str, ...]] = {}
    for name, raw_values in dimensions.items():
        if _DIMENSION_RE.fullmatch(name) is None:
            _fail(
                "dimension names must be ASCII identifiers",
                field=f"dimensions.{name}",
            )
        values = tuple(
            _dimension_scalar(item, field=f"dimensions.{name}[{index}]")
            for index, item in enumerate(_array(raw_values, field=f"dimensions.{name}"))
        )
        keys = tuple(_typed_key(item) for item in values)
        if len(keys) != len(set(keys)):
            _fail(
                "dimension values must be unique with exact JSON types",
                field=f"dimensions.{name}",
            )
        result[name] = values
    return result


def _exclusions(
    raw_exclusions: object,
    *,
    dimensions: dict[str, tuple[bool | int | float | str, ...]],
) -> tuple[dict[str, bool | int | float | str], ...]:
    if raw_exclusions is None:
        return ()
    rows = _array(raw_exclusions, field="exclude")
    result: list[dict[str, bool | int | float | str]] = []
    identities: set[bytes] = set()
    for index, raw_row in enumerate(rows):
        field = f"exclude[{index}]"
        row = _mapping(raw_row, field=field)
        if not row:
            _fail("exclusion predicates must be non-empty", field=field)
        if not set(row).issubset(dimensions):
            _fail("exclusion names an unknown dimension", field=field)
        parsed: dict[str, bool | int | float | str] = {}
        for name, raw_value in row.items():
            value = _dimension_scalar(raw_value, field=f"{field}.{name}")
            if not any(
                _typed_equal(value, candidate) for candidate in dimensions[name]
            ):
                _fail(
                    "exclusion value is not an exact typed dimension member",
                    field=f"{field}.{name}",
                )
            parsed[name] = value
        identity = canonical_json_bytes(parsed)
        if identity in identities:
            _fail("exclusion predicates must be unique", field=field)
        identities.add(identity)
        result.append(parsed)
    return tuple(result)


def _is_excluded(
    values: dict[str, bool | int | float | str],
    exclusions: tuple[dict[str, bool | int | float | str], ...],
) -> bool:
    return any(
        all(
            _typed_equal(values[name], excluded_value)
            for name, excluded_value in row.items()
        )
        for row in exclusions
    )


def _batch_policy(
    raw_batch: object,
    *,
    logical_world_count: int,
) -> tuple[str, int, tuple[int, ...]]:
    if type(raw_batch) is int:
        worlds = _integer(raw_batch, field="run.batch", minimum=1)
        policy = BATCH_POLICY_AT_MOST
    else:
        batch = _exact_mapping(
            raw_batch,
            field="run.batch",
            required=frozenset({"policy", "worlds"}),
        )
        if batch["policy"] != "exact":
            _fail(
                "batch objects support only the exact policy; integers mean at-most",
                field="run.batch.policy",
            )
        worlds = _integer(batch["worlds"], field="run.batch.worlds", minimum=1)
        policy = BATCH_POLICY_EXACT
    # An integer is an at-most physical packing capacity.  It is valid for a
    # small experiment to request the normal B1024 ceiling and naturally use a
    # single partial batch; only an exact batch may demand more worlds than the
    # semantic experiment contains.
    if policy == BATCH_POLICY_EXACT and worlds > logical_world_count:
        _fail("batch worlds exceed expect_worlds", field="run.batch")
    if policy == BATCH_POLICY_EXACT and logical_world_count % worlds:
        _fail(
            "exact batch worlds must divide expect_worlds without a remainder",
            field="run.batch",
        )
    sizes = tuple(
        min(worlds, logical_world_count - start)
        for start in range(0, logical_world_count, worlds)
    )
    if policy == BATCH_POLICY_EXACT and any(size != worlds for size in sizes):
        _fail("exact batch planning produced a partial batch", field="run.batch")
    return policy, worlds, sizes


def _validate_request_document(value: object) -> dict[str, object]:
    document = _exact_mapping(
        value,
        field="science_request",
        required=_TOP_LEVEL_REQUIRED,
        optional=_TOP_LEVEL_OPTIONAL,
    )
    _validate_generation_firewall(document)
    if document["schema_version"] != str(EIEL_SCIENCE_REQUEST_SCHEMA_VERSION):
        _fail("unsupported science request schema", field="schema_version")
    if "exact_natural_source_state_bank" in document:
        _validate_source_bank_pointer(
            document["exact_natural_source_state_bank"],
            field="exact_natural_source_state_bank",
        )
    _text(document["campaign_id"], field="campaign_id", token=True)
    _text(document["test"], field="test", token=True)
    continuation = _selected_world_continuation(document.get("continuation"))
    start_tick = 0 if continuation is None else continuation[0].next_absolute_tick

    question = _exact_mapping(
        document["question"],
        field="question",
        required=frozenset({"description", "no_claims"}),
    )
    _text(question["description"], field="question.description")
    _boolean(question["no_claims"], field="question.no_claims")

    world = _exact_mapping(
        document["world"],
        field="world",
        required=frozenset({"width", "height", "population", "capacity"}),
    )
    width = _integer(world["width"], field="world.width", minimum=1)
    height = _integer(world["height"], field="world.height", minimum=1)
    population = _integer(world["population"], field="world.population", minimum=1)
    capacity = _integer(world["capacity"], field="world.capacity", minimum=population)
    if population > width * height:
        _fail("world.population exceeds available sites", field="world.population")
    if capacity < population:
        _fail("world.capacity is smaller than population", field="world.capacity")

    run = _exact_mapping(
        document["run"],
        field="run",
        required=frozenset({"ticks", "batch", "checkpoint_ticks"}),
    )
    total_ticks = _integer(run["ticks"], field="run.ticks", minimum=1)
    if continuation is not None and total_ticks <= start_tick:
        _fail(
            "continued run.ticks must be after the selected parent frontier",
            field="run.ticks",
        )
    _validate_tick_array(
        run["checkpoint_ticks"],
        field="run.checkpoint_ticks",
        total_ticks=total_ticks,
        allow_zero=False,
        require_zero=False,
        start_tick=start_tick,
    )

    dimensions = _dimensions(document["dimensions"])
    dimension_names = frozenset(dimensions)

    reduction = _exact_mapping(
        document["reduce"],
        field="reduce",
        required=frozenset({"reducer", "configuration"}),
    )
    _text(reduction["reducer"], field="reduce.reducer", token=True)
    configuration = _exact_mapping(
        reduction["configuration"],
        field="reduce.configuration",
        required=frozenset(
            {
                "analysis_dimensions",
                "axis_a",
                "axis_b",
                "decision_family",
                "reporting_group",
                "tracked_policy_loci",
                "tracked_trait_loci",
            }
        ),
        optional=frozenset(
            {
                "require_direct_genotype_reproductive_attribution",
                "require_reproduction_funnel_denominators",
                "require_full_hereditary_attribution",
                "require_genotype_frequency_trajectory",
                "focal_tracked_genotype_state_ids",
            }
        ),
    )
    analysis_dimensions = _validate_unique_text_array(
        configuration["analysis_dimensions"],
        field="reduce.configuration.analysis_dimensions",
    )
    if frozenset(analysis_dimensions) != dimension_names or len(
        analysis_dimensions
    ) != len(dimensions):
        _fail(
            "analysis_dimensions must be an exact permutation of dimensions",
            field="reduce.configuration.analysis_dimensions",
        )
    axis_a = _text(configuration["axis_a"], field="reduce.configuration.axis_a")
    axis_b = _text(configuration["axis_b"], field="reduce.configuration.axis_b")
    if axis_a not in dimension_names or axis_b not in dimension_names:
        _fail("reducer axes must name dimensions", field="reduce.configuration")
    if axis_a == axis_b:
        _fail("reducer axes must be distinct", field="reduce.configuration")
    _text(
        configuration["decision_family"],
        field="reduce.configuration.decision_family",
        token=True,
    )
    _text(
        configuration["reporting_group"],
        field="reduce.configuration.reporting_group",
        token=True,
    )
    for field_name in (
        "require_direct_genotype_reproductive_attribution",
        "require_reproduction_funnel_denominators",
        "require_full_hereditary_attribution",
        "require_genotype_frequency_trajectory",
    ):
        if field_name in configuration:
            _boolean(
                configuration[field_name],
                field=f"reduce.configuration.{field_name}",
            )
    focal_ids: tuple[str, ...] = ()
    if "focal_tracked_genotype_state_ids" in configuration:
        focal_ids = _validate_unique_text_array(
            configuration["focal_tracked_genotype_state_ids"],
            field="reduce.configuration.focal_tracked_genotype_state_ids",
        )
        for index, focal_id in enumerate(focal_ids):
            _validate_sha256_id(
                focal_id,
                field=f"reduce.configuration.focal_tracked_genotype_state_ids[{index}]",
            )
    if configuration.get("require_genotype_frequency_trajectory", False) and not focal_ids:
        _fail(
            "genotype frequency trajectory requires focal tracked genotype IDs",
            field="reduce.configuration.focal_tracked_genotype_state_ids",
        )
    for field_name in ("tracked_policy_loci", "tracked_trait_loci"):
        loci = tuple(
            _integer(item, field=f"reduce.configuration.{field_name}[{index}]")
            for index, item in enumerate(
                _array(
                    configuration[field_name],
                    field=f"reduce.configuration.{field_name}",
                )
            )
        )
        if loci != tuple(sorted(set(loci))):
            _fail(
                "tracked loci must be sorted and unique",
                field=f"reduce.configuration.{field_name}",
            )

    replicate_dimension = _text(
        document["replicate_dimension"], field="replicate_dimension"
    )
    if replicate_dimension not in dimension_names:
        _fail("replicate_dimension must name a dimension", field="replicate_dimension")
    if replicate_dimension in {axis_a, axis_b}:
        _fail(
            "replicate_dimension must be distinct from reducer axes",
            field="replicate_dimension",
        )

    observe = _exact_mapping(
        document["observe"],
        field="observe",
        required=frozenset(
            {"profile", "events", "state", "runtime_counters", "snapshot_ticks"}
        ),
    )
    _text(observe["profile"], field="observe.profile", token=True)
    for field_name in ("events", "state", "runtime_counters"):
        _validate_unique_text_array(observe[field_name], field=f"observe.{field_name}")
    _validate_tick_array(
        observe["snapshot_ticks"],
        field="observe.snapshot_ticks",
        total_ticks=total_ticks,
        allow_zero=True,
        require_zero=True,
        start_tick=start_tick,
    )

    template = _exact_mapping(
        document["world_template"],
        field="world_template",
        required=(
            frozenset({"controls"})
            if continuation is not None
            else frozenset({"initialization", "controls", "seeds"})
        ),
    )
    seed_template = (
        {}
        if continuation is not None
        else _mapping(template["seeds"], field="world_template.seeds")
    )
    if continuation is None and set(seed_template) != _REQUIRED_SEED_OWNERS:
        _fail(
            "seed owners must exactly match the native science owner set",
            field="world_template.seeds",
        )
    referenced_dimensions: set[str] = set()
    _validate_template_value(
        template,
        field="world_template",
        dimensions=dimension_names,
        referenced_dimensions=referenced_dimensions,
    )
    if continuation is None and referenced_dimensions != dimension_names:
        _fail(
            "every dimension must be referenced by a whole-string placeholder",
            field="world_template",
        )
    replicate_marker = _PLACEHOLDER_OPEN + replicate_dimension + "}"
    if continuation is None and not any(
        value == replicate_marker for value in seed_template.values()
    ):
        _fail(
            "replicate_dimension must bind at least one seed owner",
            field="replicate_dimension",
        )

    exclusions = _exclusions(document.get("exclude"), dimensions=dimensions)
    expanded = _expand_document(
        document,
        dimensions=dimensions,
        analysis_dimensions=analysis_dimensions,
        exclusions=exclusions,
        validate_resolved=True,
    )
    expected = _integer(document["expect_worlds"], field="expect_worlds", minimum=1)
    if continuation is not None and (
        expected != 1
        or any(len(values) != 1 for values in dimensions.values())
        or document.get("exclude") is not None
    ):
        _fail(
            "V1 selected-world continuation requires one unexcluded singleton world",
            field="expect_worlds",
        )
    if len(expanded) != expected:
        _fail(
            f"expect_worlds={expected} differs from expanded world count {len(expanded)}",
            field="expect_worlds",
        )
    _batch_policy(run["batch"], logical_world_count=expected)
    return document


def _expand_document(
    document: dict[str, object],
    *,
    dimensions: dict[str, tuple[bool | int | float | str, ...]] | None = None,
    analysis_dimensions: tuple[str, ...] | None = None,
    exclusions: tuple[dict[str, bool | int | float | str], ...] | None = None,
    validate_resolved: bool,
) -> tuple["ExpandedScienceWorldV1", ...]:
    dimensions = dimensions or _dimensions(document["dimensions"])
    configuration = _mapping(
        _mapping(document["reduce"], field="reduce")["configuration"],
        field="reduce.configuration",
    )
    analysis_dimensions = analysis_dimensions or tuple(
        _text(item, field="reduce.configuration.analysis_dimensions")
        for item in _array(
            configuration["analysis_dimensions"],
            field="reduce.configuration.analysis_dimensions",
        )
    )
    exclusions = (
        exclusions
        if exclusions is not None
        else _exclusions(document.get("exclude"), dimensions=dimensions)
    )
    world = _mapping(document["world"], field="world")
    run = _mapping(document["run"], field="run")
    total_ticks = _integer(run["ticks"], field="run.ticks", minimum=1)
    continuation = _selected_world_continuation(document.get("continuation"))
    start_tick = 0 if continuation is None else continuation[0].next_absolute_tick
    template = _mapping(document["world_template"], field="world_template")

    result: list[ExpandedScienceWorldV1] = []
    ordered_values = tuple(dimensions[name] for name in analysis_dimensions)
    for combination in product(*ordered_values):
        values = dict(zip(analysis_dimensions, combination, strict=True))
        if _is_excluded(values, exclusions):
            continue
        resolved = _substitute_template(template, dimension_values=values)
        if validate_resolved:
            _validate_resolved_template(
                resolved,
                world=world,
                total_ticks=total_ticks,
                start_tick=start_tick,
                selected_world_continuation=continuation is not None,
            )
        result.append(
            ExpandedScienceWorldV1(
                world_ordinal=len(result),
                dimension_values=CanonicalPayloadV1.from_value(
                    EIEL_SCIENCE_REQUEST_DIMENSION_VALUES_SCHEMA_VERSION,
                    values,
                ),
                world_template=CanonicalPayloadV1.from_value(
                    EIEL_SCIENCE_REQUEST_RESOLVED_TEMPLATE_SCHEMA_VERSION,
                    resolved,
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScienceRequestBatchPlanV1:
    policy: str
    worlds: int
    logical_world_count: int
    batch_sizes: tuple[int, ...]
    schema_version: SchemaVersion = EIEL_SCIENCE_REQUEST_BATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.policy not in {BATCH_POLICY_AT_MOST, BATCH_POLICY_EXACT}:
            _fail("unsupported batch policy", field="policy")
        _integer(self.worlds, field="worlds", minimum=1)
        _integer(self.logical_world_count, field="logical_world_count", minimum=1)
        if (
            not isinstance(self.batch_sizes, tuple)
            or not self.batch_sizes
            or any(type(item) is not int or item < 1 for item in self.batch_sizes)
            or sum(self.batch_sizes) != self.logical_world_count
            or any(item > self.worlds for item in self.batch_sizes)
        ):
            _fail("batch_sizes do not cover the logical world set", field="batch_sizes")
        if self.policy == BATCH_POLICY_EXACT and any(
            item != self.worlds for item in self.batch_sizes
        ):
            _fail("exact policy contains a partial batch", field="batch_sizes")
        if self.schema_version != EIEL_SCIENCE_REQUEST_BATCH_PLAN_SCHEMA_VERSION:
            _fail("unsupported batch-plan schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "policy": self.policy,
            "worlds": self.worlds,
            "logical_world_count": self.logical_world_count,
            "physical_batch_count": len(self.batch_sizes),
            "batch_sizes": list(self.batch_sizes),
        }


@dataclass(frozen=True, slots=True)
class ExpandedScienceWorldV1:
    world_ordinal: int
    dimension_values: CanonicalPayloadV1
    world_template: CanonicalPayloadV1
    schema_version: SchemaVersion = EIEL_SCIENCE_REQUEST_EXPANDED_WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _integer(self.world_ordinal, field="world_ordinal", minimum=0)
        require_instance(
            self.dimension_values,
            CanonicalPayloadV1,
            field="dimension_values",
        )
        require_instance(
            self.world_template,
            CanonicalPayloadV1,
            field="world_template",
        )
        if (
            self.dimension_values.schema_version
            != EIEL_SCIENCE_REQUEST_DIMENSION_VALUES_SCHEMA_VERSION
        ):
            _fail("dimension payload schema differs", field="dimension_values")
        if (
            self.world_template.schema_version
            != EIEL_SCIENCE_REQUEST_RESOLVED_TEMPLATE_SCHEMA_VERSION
        ):
            _fail("world-template payload schema differs", field="world_template")
        if self.schema_version != EIEL_SCIENCE_REQUEST_EXPANDED_WORLD_SCHEMA_VERSION:
            _fail("unsupported expanded-world schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        dimensions = self.dimension_values.to_value()
        template = self.world_template.to_value()
        assert isinstance(dimensions, dict)
        assert isinstance(template, dict)
        return {
            "schema_version": str(self.schema_version),
            "world_ordinal": self.world_ordinal,
            "dimension_values": dimensions,
            "world_template": template,
        }


@dataclass(frozen=True, slots=True)
class EielScienceRequestV1:
    """Immutable validated authoring request with deterministic expansion."""

    request: CanonicalPayloadV1
    schema_version: SchemaVersion = EIEL_SCIENCE_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.request, CanonicalPayloadV1, field="request")
        if self.request.schema_version != EIEL_SCIENCE_REQUEST_SCHEMA_VERSION:
            _fail("request payload schema differs", field="request")
        if self.schema_version != EIEL_SCIENCE_REQUEST_SCHEMA_VERSION:
            _fail("unsupported science request schema", field="schema_version")
        _validate_request_document(self.request.to_value())

    @classmethod
    def from_dict(cls, value: object) -> "EielScienceRequestV1":
        validated = _validate_request_document(value)
        return cls(
            request=CanonicalPayloadV1.from_value(
                EIEL_SCIENCE_REQUEST_SCHEMA_VERSION,
                validated,
            )
        )

    def to_dict(self) -> dict[str, object]:
        value = self.request.to_value()
        assert isinstance(value, dict)
        return value

    @property
    def analysis_dimensions(self) -> tuple[str, ...]:
        document = self.to_dict()
        reduction = _mapping(document["reduce"], field="reduce")
        configuration = _mapping(
            reduction["configuration"], field="reduce.configuration"
        )
        return tuple(
            _text(item, field="reduce.configuration.analysis_dimensions")
            for item in _array(
                configuration["analysis_dimensions"],
                field="reduce.configuration.analysis_dimensions",
            )
        )

    @property
    def expect_worlds(self) -> int:
        return _integer(
            self.to_dict()["expect_worlds"], field="expect_worlds", minimum=1
        )

    def request_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def expand_worlds(self) -> tuple[ExpandedScienceWorldV1, ...]:
        return _expand_document(self.to_dict(), validate_resolved=False)

    def batch_plan(self) -> ScienceRequestBatchPlanV1:
        document = self.to_dict()
        run = _mapping(document["run"], field="run")
        policy, worlds, sizes = _batch_policy(
            run["batch"],
            logical_world_count=self.expect_worlds,
        )
        return ScienceRequestBatchPlanV1(
            policy=policy,
            worlds=worlds,
            logical_world_count=self.expect_worlds,
            batch_sizes=sizes,
        )


def validate_eiel_science_request_v1(value: object) -> EielScienceRequestV1:
    """Validate and freeze one compact authoring request."""

    if isinstance(value, EielScienceRequestV1):
        return value
    return EielScienceRequestV1.from_dict(value)


def expand_eiel_science_request_v1(
    value: object,
) -> tuple[ExpandedScienceWorldV1, ...]:
    """Return semantic worlds in declared order, last dimension fastest."""

    return validate_eiel_science_request_v1(value).expand_worlds()


__all__ = (
    "BATCH_POLICY_AT_MOST",
    "BATCH_POLICY_EXACT",
    "EIEL_SCIENCE_REQUEST_BATCH_PLAN_SCHEMA_VERSION",
    "EIEL_SCIENCE_REQUEST_DIMENSION_VALUES_SCHEMA_VERSION",
    "EIEL_SCIENCE_REQUEST_EXPANDED_WORLD_SCHEMA_VERSION",
    "EIEL_SCIENCE_REQUEST_RESOLVED_TEMPLATE_SCHEMA_VERSION",
    "EIEL_SCIENCE_REQUEST_SCHEMA_VERSION",
    "EielScienceRequestV1",
    "ExpandedScienceWorldV1",
    "ScienceRequestBatchPlanV1",
    "expand_eiel_science_request_v1",
    "validate_eiel_science_request_v1",
)
