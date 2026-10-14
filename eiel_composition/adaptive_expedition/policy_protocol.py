"""Pure deterministic policy port for Adaptive Science Expedition V1.

This module deliberately contains no runtime, provider, simulator, storage, or
clock dependency.  Policy implementations consume and return deeply immutable
canonical mappings.  The coordinator remains responsible for validating and
sealing durable decisions; a policy proposal grants no execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest


POLICY_PROTOCOL_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_policy_protocol_v1"
)
POLICY_PROPOSAL_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_policy_proposal_v1"
)
POLICY_INPUT_AUTHORITY_VERSION_V1: Final[str] = (
    "eiel_adaptive_policy_input_authority_v1"
)
SHA256_COUNTER_PRF_VERSION_V1: Final[str] = "eiel_sha256_counter_prf_u64_v1"

_UINT64_MAX: Final[int] = (1 << 64) - 1
_FORBIDDEN_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fitness",
        "fitness_score",
        "reward",
        "reward_scalar",
        "scalar_reward",
        "phenotype",
        "phenotype_id",
        "phenotype_name",
    }
)


class ExpeditionPolicyError(ValueError):
    """Raised when a policy input, profile, or proposal is not deterministic."""


def _plain_canonical_v1(
    value: object,
    *,
    path: str,
    active: set[int],
) -> object:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ExpeditionPolicyError(f"{path} must be a finite canonical number")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ExpeditionPolicyError(f"{path} contains a mapping cycle")
        active.add(identity)
        try:
            if any(type(key) is not str for key in value):
                raise ExpeditionPolicyError(f"{path} mapping keys must be text")
            return {
                key: _plain_canonical_v1(
                    value[key], path=f"{path}.{key}", active=active
                )
                for key in sorted(value)
            }
        finally:
            active.remove(identity)
    if value_type in {list, tuple}:
        identity = id(value)
        if identity in active:
            raise ExpeditionPolicyError(f"{path} contains a sequence cycle")
        active.add(identity)
        try:
            return [
                _plain_canonical_v1(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ExpeditionPolicyError(
        f"{path} contains unsupported canonical type {value_type.__name__}"
    )


def canonical_plain_value_v1(value: object) -> object:
    """Return a detached plain-JSON projection of a canonical immutable value."""

    return _plain_canonical_v1(value, path="$", active=set())


def _freeze_plain_v1(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_plain_v1(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_plain_v1(item) for item in value)
    return value


def freeze_canonical_value_v1(value: object) -> object:
    """Deep-copy and freeze a value after enforcing the canonical JSON domain."""

    return _freeze_plain_v1(canonical_plain_value_v1(value))


def freeze_canonical_mapping_v1(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = freeze_canonical_value_v1(value)
    if not isinstance(frozen, Mapping):
        raise ExpeditionPolicyError("canonical policy value must be a mapping")
    return frozen


def canonical_mapping_sha256_v1(value: Mapping[str, object]) -> ContentDigest:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not dict:
        raise ExpeditionPolicyError("canonical policy value must be a mapping")
    return canonical_sha256(plain)


def sha256_counter_prf_u64_v1(*, seed: int, counter: int, domain: str) -> int:
    """Return one replayable uint64 draw without ambient random state.

    Each call is independently addressed by ``(seed, counter, domain)``.  A
    caller owns cursor advancement and must persist the resulting cursor in its
    proposal/decision record.
    """

    if type(seed) is not int or not 0 <= seed <= _UINT64_MAX:
        raise ExpeditionPolicyError("PRF seed must be a uint64")
    if type(counter) is not int or not 0 <= counter <= _UINT64_MAX:
        raise ExpeditionPolicyError("PRF counter must be a uint64")
    if type(domain) is not str or not domain:
        raise ExpeditionPolicyError("PRF domain must be non-empty text")
    digest = canonical_sha256(
        {
            "protocol_version": SHA256_COUNTER_PRF_VERSION_V1,
            "seed": seed,
            "counter": counter,
            "domain": domain,
        }
    )
    return int(digest.hex[:16], 16)


def sha256_counter_prf_index_v1(
    *, seed: int, counter: int, domain: str, size: int
) -> tuple[int, int]:
    """Select a deterministic bounded index and return the advanced cursor."""

    if type(size) is not int or size <= 0:
        raise ExpeditionPolicyError("PRF choice size must be positive")
    if counter == _UINT64_MAX:
        raise ExpeditionPolicyError("PRF cursor is exhausted")
    return sha256_counter_prf_u64_v1(
        seed=seed, counter=counter, domain=domain
    ) % size, counter + 1


def profile_payload_envelope_v1(
    profile: Mapping[str, object], *, profile_schema_version: str
) -> Mapping[str, object]:
    """Project the exact ``CanonicalPayloadV1.to_dict`` manifest shape."""

    if type(profile_schema_version) is not str or not profile_schema_version:
        raise ExpeditionPolicyError("profile schema version must be non-empty text")
    return freeze_canonical_mapping_v1(
        {
            "schema_version": profile_schema_version,
            "payload": canonical_plain_value_v1(profile),
        }
    )


def profile_sha256_v1(
    profile: Mapping[str, object], *, profile_schema_version: str
) -> ContentDigest:
    return canonical_mapping_sha256_v1(
        profile_payload_envelope_v1(
            profile, profile_schema_version=profile_schema_version
        )
    )


def policy_binding_v1(
    *,
    policy_id: str,
    policy_version: str,
    algorithm_sha256: ContentDigest,
    profile: Mapping[str, object],
    profile_schema_version: str,
) -> Mapping[str, object]:
    if not policy_id or not policy_version:
        raise ExpeditionPolicyError("policy identity must be non-empty text")
    profile_id = profile.get("profile_id")
    profile_version = profile.get("profile_version")
    if type(profile_id) is not str or not profile_id:
        raise ExpeditionPolicyError("profile_id must be non-empty text")
    if type(profile_version) is not str or not profile_version:
        raise ExpeditionPolicyError("profile_version must be non-empty text")
    profile_envelope = profile_payload_envelope_v1(
        profile, profile_schema_version=profile_schema_version
    )
    return freeze_canonical_mapping_v1(
        {
            "policy_id": policy_id,
            "policy_version": policy_version,
            "implementation_sha256": algorithm_sha256.to_dict(),
            "profile_sha256": profile_sha256_v1(
                profile, profile_schema_version=profile_schema_version
            ).to_dict(),
            "profile": canonical_plain_value_v1(profile_envelope),
        }
    )


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExpeditionPolicyError(f"{field} must be a canonical mapping")
    if any(type(key) is not str for key in value):
        raise ExpeditionPolicyError(f"{field} keys must be text")
    return value


def _require_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpeditionPolicyError(f"{field} must be a canonical sequence")
    return value


def _digest_dict(value: ContentDigest) -> dict[str, str]:
    return {"algorithm": value.algorithm, "hex": value.hex}


def validate_bounded_probe_contracts_v1(
    profile: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    raw_contracts = profile.get("probe_contracts", ())
    contracts = _require_sequence(raw_contracts, field="profile.probe_contracts")
    by_id: dict[str, Mapping[str, object]] = {}
    required = {
        "probe_contract_id",
        "phenotype_label",
        "observable_predicates",
        "matched_comparator",
        "authorized_control_ids",
        "action_template",
        "maximum_actions",
        "maximum_replicates",
        "maximum_absolute_ticks",
        "stop_rules",
        "claim_boundary",
    }
    for index, raw in enumerate(contracts):
        contract = _require_mapping(raw, field=f"profile.probe_contracts[{index}]")
        if set(contract) != required:
            raise ExpeditionPolicyError(
                f"profile.probe_contracts[{index}] fields must be exact"
            )
        contract_id = contract.get("probe_contract_id")
        label = contract.get("phenotype_label")
        if type(contract_id) is not str or not contract_id or contract_id in by_id:
            raise ExpeditionPolicyError("probe_contract_id must be unique text")
        if type(label) is not str or not label:
            raise ExpeditionPolicyError("phenotype_label must be non-empty text")
        for field in (
            "maximum_actions",
            "maximum_replicates",
            "maximum_absolute_ticks",
        ):
            if type(contract.get(field)) is not int or int(contract[field]) <= 0:
                raise ExpeditionPolicyError(f"probe contract {field} must be positive")
        for field in ("observable_predicates", "authorized_control_ids", "stop_rules"):
            if not _require_sequence(contract.get(field), field=f"probe.{field}"):
                raise ExpeditionPolicyError(f"probe contract {field} cannot be empty")
        _require_mapping(contract.get("matched_comparator"), field="probe.matched_comparator")
        action_template = _require_mapping(
            contract.get("action_template"), field="probe.action_template"
        )
        if action_template.get("action_kind") != "CAUSAL_PROBE":
            raise ExpeditionPolicyError(
                "bounded probe action_template must emit CAUSAL_PROBE"
            )
        if type(action_template.get("bucket")) is not str or not action_template.get(
            "bucket"
        ):
            raise ExpeditionPolicyError("bounded probe action bucket must be text")
        if type(action_template.get("slot_cost")) is not int or int(
            action_template["slot_cost"]
        ) < 0:
            raise ExpeditionPolicyError(
                "bounded probe action slot_cost must be non-negative"
            )
        claim_boundary = contract.get("claim_boundary")
        if type(claim_boundary) is not str or not claim_boundary:
            raise ExpeditionPolicyError("probe claim_boundary must be non-empty text")
        by_id[contract_id] = contract
    return MappingProxyType(by_id)


_PROJECTION_MISSING = object()
_TRUSTED_IDENTITY_CONTAINER_PATHS: Final[frozenset[str]] = frozenset(
    {"/manifest_sha256", "/policy_binding"}
)


def _allowed_policy_input_paths_v1(
    profile: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    authority = _require_mapping(
        profile.get("policy_input_authority"), field="profile.policy_input_authority"
    )
    if set(authority) != {"schema_version", "allowed_paths"}:
        raise ExpeditionPolicyError("policy input authority fields must be exact")
    if authority.get("schema_version") != POLICY_INPUT_AUTHORITY_VERSION_V1:
        raise ExpeditionPolicyError("policy input authority version is unsupported")
    rows = _require_sequence(
        authority.get("allowed_paths"), field="policy_input_authority.allowed_paths"
    )
    if not rows:
        raise ExpeditionPolicyError("policy input allowed_paths cannot be empty")
    parsed: list[tuple[str, ...]] = []
    probes = validate_bounded_probe_contracts_v1(profile)
    for raw in rows:
        if type(raw) is not str or not raw.startswith("/") or raw.endswith("/"):
            raise ExpeditionPolicyError(
                "policy input paths must be non-empty absolute canonical paths"
            )
        tokens = tuple(raw[1:].split("/"))
        if any(not token or token in {".", ".."} or "~" in token for token in tokens):
            raise ExpeditionPolicyError("policy input path token is invalid")
        probe_indexes = [
            index for index, token in enumerate(tokens) if token == "probes"
        ]
        if probe_indexes:
            if len(probe_indexes) != 1 or probe_indexes[0] + 1 >= len(tokens):
                raise ExpeditionPolicyError("probe input path lacks a probe contract id")
            if tokens[probe_indexes[0] + 1] not in probes:
                raise ExpeditionPolicyError(
                    "probe input path is not authorized by a bound probe contract"
                )
        elif any(token in _FORBIDDEN_AUTHORITY_KEYS for token in tokens):
            raise ExpeditionPolicyError(
                "named phenotype/reward input requires a bounded probe contract"
            )
        parsed.append(tokens)
    if parsed != sorted(set(parsed)):
        raise ExpeditionPolicyError(
            "policy input allowed_paths must be sorted and unique"
        )
    return tuple(parsed)


def validate_policy_input_authority_v1(
    profile: Mapping[str, object],
) -> None:
    """Fail closed unless a profile declares a valid positive input surface."""

    _allowed_policy_input_paths_v1(profile)


def _project_policy_node_v1(
    value: object,
    *,
    path: tuple[str, ...],
    allowed: tuple[tuple[str, ...], ...],
) -> object:
    matches = tuple(row for row in allowed if row[: len(path)] == path)
    if not matches:
        return _PROJECTION_MISSING
    if path in matches:
        plain = canonical_plain_value_v1(value)
        path_text = "/" + "/".join(path)
        digest_leaf = (
            isinstance(plain, dict)
            and set(plain) == {"algorithm", "hex"}
            and all(type(item) is str for item in plain.values())
        )
        if (
            isinstance(plain, dict)
            and not digest_leaf
            and path_text not in _TRUSTED_IDENTITY_CONTAINER_PATHS
        ):
            raise ExpeditionPolicyError(
                f"policy input path {path_text} cannot grant a whole mapping"
            )
        if isinstance(plain, list) and any(
            isinstance(item, list)
            or (
                isinstance(item, dict)
                and not (
                    set(item) == {"algorithm", "hex"}
                    and all(type(part) is str for part in item.values())
                )
            )
            for item in plain
        ):
            raise ExpeditionPolicyError(
                f"policy input path {path_text} cannot grant a composite sequence"
            )
        return plain
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise ExpeditionPolicyError("policy input mapping keys must be text")
            item = _project_policy_node_v1(
                value[key], path=(*path, key), allowed=allowed
            )
            if item is not _PROJECTION_MISSING:
                projected[key] = item
        return projected if projected else _PROJECTION_MISSING
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        projected_rows: list[object] = []
        wildcard_path = (*path, "*")
        wildcard_matches = tuple(
            row for row in allowed if row[: len(wildcard_path)] == wildcard_path
        )
        if not wildcard_matches:
            return _PROJECTION_MISSING
        for item in value:
            projected = _project_policy_node_v1(
                item, path=wildcard_path, allowed=allowed
            )
            if projected is _PROJECTION_MISSING:
                projected = {} if isinstance(item, Mapping) else None
            projected_rows.append(projected)
        return projected_rows
    return _PROJECTION_MISSING


def project_policy_input_v1(
    decision_input: Mapping[str, object],
    *,
    profile: Mapping[str, object],
) -> Mapping[str, object]:
    """Expose only the profile-declared, manifest-bound policy input paths.

    This is the coordinator's positive authority boundary.  Unknown facts,
    descriptors, history fields, and aliases never reach a policy merely
    because they happen to be present in an authenticated observation.
    """

    plain = canonical_plain_value_v1(decision_input)
    if type(plain) is not dict:
        raise ExpeditionPolicyError("policy input must be a canonical mapping")
    raw_frontier = plain.get("decision_frontier")
    if isinstance(raw_frontier, dict) and "absolute_tick" in raw_frontier:
        raise ExpeditionPolicyError(
            "decision frontier cannot claim one absolute tick across trajectories"
        )
    allowed = _allowed_policy_input_paths_v1(profile)
    projected = _project_policy_node_v1(plain, path=(), allowed=allowed)
    if type(projected) is not dict:
        raise ExpeditionPolicyError("policy input authority projected no mapping")
    frontier = projected.setdefault("decision_frontier", {})
    if type(frontier) is not dict:
        raise ExpeditionPolicyError("projected decision frontier must be a mapping")
    frontier.setdefault("observation_frames", [])
    frontier.setdefault("bounded_history", [])
    for field in ("atlas", "retained_parent_bank"):
        value = projected.setdefault(field, {})
        if type(value) is not dict:
            raise ExpeditionPolicyError(f"projected {field} must be a mapping")
        value.setdefault("entries", [])
    projected.setdefault("available_state_refs", [])
    return freeze_canonical_mapping_v1(projected)


def validate_policy_input_v1(
    decision_input: Mapping[str, object],
    *,
    expected_binding: Mapping[str, object],
    profile: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate the small policy-facing view without inventing science truth."""

    value = project_policy_input_v1(decision_input, profile=profile)
    required = {
        "protocol_version",
        "expedition_id",
        "manifest_sha256",
        "policy_binding",
        "decision_frontier",
        "atlas",
        "retained_parent_bank",
        "available_state_refs",
        "logical_capacity",
        "outer_rng",
    }
    missing = required - set(value)
    if missing:
        raise ExpeditionPolicyError(f"policy input is missing {sorted(missing)}")
    if value.get("protocol_version") != POLICY_PROTOCOL_VERSION_V1:
        raise ExpeditionPolicyError("policy protocol version differs")
    expedition_id = value.get("expedition_id")
    if type(expedition_id) is not str or not expedition_id:
        raise ExpeditionPolicyError("expedition_id must be non-empty text")
    _require_mapping(value.get("manifest_sha256"), field="manifest_sha256")
    if canonical_plain_value_v1(value.get("policy_binding")) != canonical_plain_value_v1(
        expected_binding
    ):
        raise ExpeditionPolicyError("manifest policy binding differs from implementation")
    frontier = _require_mapping(value.get("decision_frontier"), field="decision_frontier")
    if type(frontier.get("frontier_id")) is not str or not frontier.get("frontier_id"):
        raise ExpeditionPolicyError("decision frontier_id must be non-empty text")
    if (
        type(frontier.get("frontier_ordinal")) is not int
        or int(frontier["frontier_ordinal"]) < 0
    ):
        raise ExpeditionPolicyError(
            "decision frontier_ordinal must be a non-negative outer ordering"
        )
    if "absolute_tick" in frontier:
        raise ExpeditionPolicyError(
            "decision frontier cannot claim one absolute tick across trajectories"
        )
    for field in ("observation_frames", "bounded_history"):
        rows = _require_sequence(frontier.get(field), field=f"decision_frontier.{field}")
        for index, row in enumerate(rows):
            _require_mapping(row, field=f"decision_frontier.{field}[{index}]")
    _require_mapping(value.get("atlas"), field="atlas")
    _require_mapping(value.get("retained_parent_bank"), field="retained_parent_bank")
    state_refs = _require_sequence(value.get("available_state_refs"), field="available_state_refs")
    for index, row in enumerate(state_refs):
        _require_mapping(row, field=f"available_state_refs[{index}]")
    capacity = value.get("logical_capacity")
    if type(capacity) is not int or capacity < 0:
        raise ExpeditionPolicyError("logical_capacity must be non-negative")
    outer_rng = _require_mapping(value.get("outer_rng"), field="outer_rng")
    if set(outer_rng) != {"seed", "cursor"}:
        raise ExpeditionPolicyError("outer_rng fields must be exact")
    for field in ("seed", "cursor"):
        item = outer_rng.get(field)
        if type(item) is not int or not 0 <= item <= _UINT64_MAX:
            raise ExpeditionPolicyError(f"outer_rng.{field} must be a uint64")
    return value


def _validate_proposal_authority_v1(
    proposal_body: Mapping[str, object],
    *,
    profile: Mapping[str, object],
) -> None:
    probes = validate_bounded_probe_contracts_v1(profile)

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _FORBIDDEN_AUTHORITY_KEYS:
                    raise ExpeditionPolicyError(
                        f"{path}.{key} is hidden scalar/phenotype authority"
                    )
                visit(item, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(proposal_body, "proposal")
    actions = _require_sequence(
        proposal_body.get("semantic_actions"), field="semantic_actions"
    )
    for index, raw in enumerate(actions):
        action = _require_mapping(raw, field=f"semantic_actions[{index}]")
        probe_id = action.get("probe_contract_id")
        if probe_id is not None and probe_id not in probes:
            raise ExpeditionPolicyError(
                f"semantic_actions[{index}] uses an undeclared probe contract"
            )
        if action.get("action_kind") == "CAUSAL_PROBE" and probe_id is None:
            raise ExpeditionPolicyError(
                f"semantic_actions[{index}] causal probe lacks bounded probe authority"
            )


def seal_policy_proposal_v1(
    *,
    policy_binding: Mapping[str, object],
    profile: Mapping[str, object],
    decision_input: Mapping[str, object],
    rng_cursor_after: int,
    atlas_delta: Mapping[str, object],
    retained_parent_bank_delta: Mapping[str, object],
    semantic_actions: Sequence[Mapping[str, object]],
    rejected_candidate_summary: Mapping[str, object],
) -> Mapping[str, object]:
    """Seal a non-authoritative deterministic proposal for coordinator review."""

    checked_input = validate_policy_input_v1(
        decision_input, expected_binding=policy_binding, profile=profile
    )
    outer_rng = _require_mapping(checked_input["outer_rng"], field="outer_rng")
    rng_before = int(outer_rng["cursor"])
    if type(rng_cursor_after) is not int or not rng_before <= rng_cursor_after <= _UINT64_MAX:
        raise ExpeditionPolicyError("proposal RNG cursor must advance monotonically")
    frontier = _require_mapping(
        checked_input["decision_frontier"], field="decision_frontier"
    )
    body = {
        "protocol_version": POLICY_PROPOSAL_VERSION_V1,
        "policy_binding": canonical_plain_value_v1(policy_binding),
        "manifest_sha256": canonical_plain_value_v1(checked_input["manifest_sha256"]),
        "input_sha256": canonical_mapping_sha256_v1(checked_input).to_dict(),
        "frontier_id": frontier["frontier_id"],
        "frontier_ordinal": frontier["frontier_ordinal"],
        "outer_rng": {
            "seed": outer_rng["seed"],
            "cursor_before": rng_before,
            "cursor_after": rng_cursor_after,
        },
        "atlas_delta": canonical_plain_value_v1(atlas_delta),
        "retained_parent_bank_delta": canonical_plain_value_v1(
            retained_parent_bank_delta
        ),
        "semantic_actions": canonical_plain_value_v1(tuple(semantic_actions)),
        "rejected_candidate_summary": canonical_plain_value_v1(
            rejected_candidate_summary
        ),
        "authority_effect": "NONE_PROPOSAL_ONLY",
    }
    _validate_proposal_authority_v1(body, profile=profile)
    sealed = {**body, "proposal_sha256": canonical_sha256(body).to_dict()}
    return freeze_canonical_mapping_v1(sealed)


def validate_policy_proposal_v1(
    proposal: Mapping[str, object],
    *,
    profile: Mapping[str, object],
    decision_input: Mapping[str, object],
    expected_binding: Mapping[str, object],
) -> Mapping[str, object]:
    value = freeze_canonical_mapping_v1(proposal)
    required = {
        "protocol_version",
        "policy_binding",
        "manifest_sha256",
        "input_sha256",
        "frontier_id",
        "frontier_ordinal",
        "outer_rng",
        "atlas_delta",
        "retained_parent_bank_delta",
        "semantic_actions",
        "rejected_candidate_summary",
        "authority_effect",
        "proposal_sha256",
    }
    if set(value) != required:
        raise ExpeditionPolicyError("policy proposal fields must be exact")
    if value.get("protocol_version") != POLICY_PROPOSAL_VERSION_V1:
        raise ExpeditionPolicyError("policy proposal protocol version differs")
    if canonical_plain_value_v1(value.get("policy_binding")) != canonical_plain_value_v1(
        expected_binding
    ):
        raise ExpeditionPolicyError("policy proposal binding differs")
    checked_input = validate_policy_input_v1(
        decision_input, expected_binding=expected_binding, profile=profile
    )
    if value.get("input_sha256") != freeze_canonical_value_v1(
        canonical_mapping_sha256_v1(checked_input).to_dict()
    ):
        raise ExpeditionPolicyError("policy proposal input identity differs")
    core = {
        key: canonical_plain_value_v1(item)
        for key, item in value.items()
        if key != "proposal_sha256"
    }
    if canonical_plain_value_v1(value.get("proposal_sha256")) != _digest_dict(
        canonical_sha256(core)
    ):
        raise ExpeditionPolicyError("policy proposal digest differs")
    _validate_proposal_authority_v1(core, profile=profile)
    return value


@runtime_checkable
class ExpeditionPolicyV1(Protocol):
    """Replaceable pure policy interface; implementations grant no authority."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    @property
    def profile(self) -> Mapping[str, object]: ...

    @property
    def binding(self) -> Mapping[str, object]: ...

    def decide(
        self, decision_input: Mapping[str, object]
    ) -> Mapping[str, object]: ...


__all__ = (
    "ExpeditionPolicyError",
    "ExpeditionPolicyV1",
    "POLICY_PROPOSAL_VERSION_V1",
    "POLICY_PROTOCOL_VERSION_V1",
    "POLICY_INPUT_AUTHORITY_VERSION_V1",
    "SHA256_COUNTER_PRF_VERSION_V1",
    "canonical_mapping_sha256_v1",
    "canonical_plain_value_v1",
    "freeze_canonical_mapping_v1",
    "freeze_canonical_value_v1",
    "policy_binding_v1",
    "project_policy_input_v1",
    "profile_payload_envelope_v1",
    "profile_sha256_v1",
    "seal_policy_proposal_v1",
    "sha256_counter_prf_index_v1",
    "sha256_counter_prf_u64_v1",
    "validate_bounded_probe_contracts_v1",
    "validate_policy_input_authority_v1",
    "validate_policy_input_v1",
    "validate_policy_proposal_v1",
)
