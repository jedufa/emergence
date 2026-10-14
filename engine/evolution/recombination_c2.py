from __future__ import annotations

"""Deterministic homology-aware recombination for Evolvability Phase C2."""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from typing import Any, Mapping

from agents.modular_genome_v1 import (
    b2_capacity_neutral_ordering_payload,
    canonical_payload_from_genome_dict,
    resolve_b2_caps,
)
from agents.structural_identity_v1 import (
    IDENTITY_ORIGIN_LEGACY,
    STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
    STRUCTURAL_IDENTITY_SCHEMA_VERSION,
    module_is_native_homology,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_C2_COMPATIBILITY_VERSION,
    GATED_EQUATION_VERSION,
    GATED_NUMERIC_VERSION,
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_STATE_WIDTH,
    GatedRecordError,
    normalize_gated_module_params,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    C2_COMPATIBILITY_VERSION as SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
    normalize_activity_module_params,
)
from evolution.inheritance_registry_v1 import (
    LEGACY_INHERITANCE_PROFILE,
    validate_production_inheritance_profile,
)
from evolution.diploid_inheritance_v1 import DIPLOID_INHERITANCE_PROFILE


C2_RECOMBINATION_VERSION = "c2_deterministic_homology_recombination_v2"
C2_FROZEN_ALIGNMENT_KEY = "homology_id"
C2_FROZEN_UNMATCHED_POLICY = "keep_all_sorted"
C2_PARENT_ORDER_RULE = "canonical_parent_payload_digest_ascending"
C2_LEGACY_UNRESOLVED_POLICY = "canonical_parent_set_no_cross_alignment"
C2_STRUCTURAL_LINKAGE_RULE = "homology_connected_component_v1"


def _linked_structural_profile(profile: str) -> bool:
    return profile == DIPLOID_INHERITANCE_PROFILE


def c2_strict_enabled() -> bool:
    v1 = str(os.environ.get("EIEL_STRICT_DETERMINISM", "")).strip().lower()
    v2 = str(os.environ.get("DEBUG_DETERMINISM", "")).strip().lower()
    return (v1 in {"1", "true", "yes", "on"}) or (v2 in {"1", "true", "yes", "on"})


def _det_fail(code: str, context: str) -> None:
    raise RuntimeError(f"[DETERMINISM_VIOLATION]\ncode={code}\ncontext={context}\n")


def _draws(rng: Any, n: int) -> list[float]:
    return [float(rng.random()) for _ in range(max(0, int(n)))]


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _normalize_gated_modules_before_draws(
    modules: list[dict[str, Any]],
    *,
    parent_scope: str,
) -> list[dict[str, Any]]:
    """Validate every explicit ESCAP-7 record without consuming donor RNG."""

    out: list[dict[str, Any]] = []
    for module in modules:
        record = deepcopy(dict(module))
        params = record.get("params", {})
        try:
            normalized = normalize_gated_module_params(params)
        except (GatedRecordError, TypeError, ValueError) as exc:
            _det_fail(
                "DET_C2_ESCAP7_GATED_RECORD_INVALID",
                f"parent_scope={parent_scope};module_id={record.get('module_id','')};error={exc}",
            )
        operator = str(
            normalized.get("live_operator", "bounded_scalar_integrator_v1")
            or "bounded_scalar_integrator_v1"
        )
        if operator == GATED_OPERATOR and str(record.get("module_type", "")) != "generic":
            _det_fail(
                "DET_C2_ESCAP7_GATED_OUTER_TYPE_INVALID",
                f"parent_scope={parent_scope};module_id={record.get('module_id','')};"
                f"module_type={record.get('module_type','')}",
            )
        record["params"] = normalized
        out.append(record)
    return out


def _normalize_sustainable_growth_modules_before_draws(
    modules: list[dict[str, Any]],
    *,
    parent_scope: str,
) -> list[dict[str, Any]]:
    """Strictly validate optional ESCAP-8 activity records without donor RNG."""

    out: list[dict[str, Any]] = []
    for module in modules:
        record = deepcopy(dict(module))
        params = record.get("params", {})
        try:
            normalized = normalize_activity_module_params(params)
        except (TypeError, ValueError) as exc:
            _det_fail(
                "DET_C2_ESCAP8_ACTIVITY_RECORD_INVALID",
                f"parent_scope={parent_scope};module_id={record.get('module_id','')};error={exc}",
            )
        record["params"] = normalized
        out.append(record)
    return out


def _activity_contract_signature(params: Mapping[str, Any]) -> tuple[str, str]:
    """Return schema compatibility while excluding the donor dormancy value."""

    normalized = normalize_activity_module_params(params)
    activity = normalized.get(ACTIVITY_RECORD_KEY)
    if activity is None:
        # Absent is the frozen legacy-active representation and is compatible
        # with the v1 architecture contract.  Atomic donor choice still
        # preserves absence versus an explicit active/dormant record.
        return (SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION, ACTIVITY_SCHEMA_VERSION)
    if not isinstance(activity, Mapping):
        _det_fail("DET_C2_ESCAP8_ACTIVITY_RECORD_INVALID", "reason=not_mapping")
    return (
        SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
        str(activity.get("schema_version", "")),
    )


def _validate_escap8_prospective_caps_before_draws(
    *,
    matched_modules: int,
    unmatched_modules_a: int,
    unmatched_modules_b: int,
    matched_connections: int,
    unmatched_connections_a: int,
    unmatched_connections_b: int,
    b2_caps: dict[str, int],
) -> None:
    """Fail closed on the deterministic union envelope before donor draws."""

    module_count = int(matched_modules + unmatched_modules_a + unmatched_modules_b)
    connection_count = int(
        matched_connections + unmatched_connections_a + unmatched_connections_b
    )
    resolved_caps = resolve_b2_caps(b2_caps)
    max_modules = int(resolved_caps["max_modules_per_agent"])
    max_connections = int(resolved_caps["max_connections_per_agent"])
    if module_count > max_modules:
        _det_fail(
            "DET_C2_ESCAP8_PROSPECTIVE_CAP_EXCEEDED",
            f"scope=modules;observed={module_count};max={max_modules}",
        )
    if connection_count > max_connections:
        _det_fail(
            "DET_C2_ESCAP8_PROSPECTIVE_CAP_EXCEEDED",
            f"scope=connections;observed={connection_count};max={max_connections}",
        )


def _gated_contract_signature(params: Mapping[str, Any]) -> tuple[Any, ...] | None:
    operator = str(
        params.get("live_operator", "bounded_scalar_integrator_v1")
        or "bounded_scalar_integrator_v1"
    )
    if operator != GATED_OPERATOR:
        return None
    record = params.get(GATED_RECORD_KEY)
    if not isinstance(record, Mapping):
        _det_fail("DET_C2_ESCAP7_GATED_RECORD_INVALID", "reason=missing_record")
    # Parameter values are deliberately excluded: compatible homologs inherit
    # one complete donor record atomically rather than averaging fields.
    return (
        GATED_C2_COMPATIBILITY_VERSION,
        GATED_RECORD_SCHEMA_VERSION,
        GATED_EQUATION_VERSION,
        GATED_NUMERIC_VERSION,
        GATED_STATE_WIDTH,
        tuple(str(key) for key in record.keys()),
    )


def _validate_policy(alignment_key: str, unmatched_policy: str) -> tuple[str, str]:
    key = str(alignment_key or C2_FROZEN_ALIGNMENT_KEY)
    if key != C2_FROZEN_ALIGNMENT_KEY and c2_strict_enabled():
        _det_fail("DET_C2_ALIGNMENT_KEY_INVALID", f"observed={key};expected={C2_FROZEN_ALIGNMENT_KEY}")
    policy = str(unmatched_policy or C2_FROZEN_UNMATCHED_POLICY).strip().lower()
    if policy != C2_FROZEN_UNMATCHED_POLICY and c2_strict_enabled():
        _det_fail(
            "DET_C2_UNMATCHED_POLICY_INVALID",
            f"observed={policy};expected={C2_FROZEN_UNMATCHED_POLICY}",
        )
    return C2_FROZEN_ALIGNMENT_KEY, C2_FROZEN_UNMATCHED_POLICY


def _trusted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if not isinstance(out.get("genes"), dict):
        _det_fail("DET_C2_TRUSTED_PAYLOAD_MALFORMED", "trusted parent payload missing genes")
    if not isinstance(out.get("modules"), list):
        _det_fail("DET_C2_TRUSTED_PAYLOAD_MALFORMED", "trusted parent payload missing modules")
    if not isinstance(out.get("connections"), list):
        _det_fail("DET_C2_TRUSTED_PAYLOAD_MALFORMED", "trusted parent payload missing connections")
    return out


def _canonical_parent_order(pa: dict[str, Any], pb: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    projected_a = b2_capacity_neutral_ordering_payload(pa)
    projected_b = b2_capacity_neutral_ordering_payload(pb)
    rows = [
        ("caller_a", pa, projected_a, _payload_digest(projected_a)),
        ("caller_b", pb, projected_b, _payload_digest(projected_b)),
    ]
    rows.sort(key=lambda row: (row[3], _canonical_bytes(row[2]), row[0]))
    return [(f"p{index}", row[1], row[3]) for index, row in enumerate(rows)]


def _native_homology_map(modules: list[dict[str, Any]], *, parent_scope: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for module in sorted(
        list(modules or []),
        key=lambda rec: (
            str(rec.get("homology_id", "")),
            str(rec.get("module_instance_id", "")),
            str(rec.get("module_id", "")),
        ),
    ):
        if not module_is_native_homology(module):
            continue
        homology_id = str(module["homology_id"])
        if homology_id in out:
            _det_fail(
                "DET_C2_DUPLICATE_HOMOLOGY_LOCUS",
                f"parent_scope={parent_scope};homology_id={homology_id}",
            )
        out[homology_id] = dict(module)
    return out


def _module_endpoint_keys(
    modules: list[dict[str, Any]],
    *,
    parent_scope: str,
) -> tuple[dict[str, str], dict[str, int]]:
    local_to_key: dict[str, str] = {}
    local_to_ordinal: dict[str, int] = {}
    ordered = sorted(
        list(modules or []),
        key=lambda rec: (
            str(rec.get("homology_id", "")),
            str(rec.get("module_instance_id", "")),
            str(rec.get("module_id", "")),
        ),
    )
    for ordinal, module in enumerate(ordered):
        local_id = str(module.get("module_id", ""))
        if not local_id:
            continue
        local_to_ordinal[local_id] = int(ordinal)
        if module_is_native_homology(module):
            local_to_key[local_id] = f"native:{module['homology_id']}"
        else:
            local_to_key[local_id] = (
                f"legacy_unresolved:{parent_scope}:{ordinal}:"
                f"{module.get('module_instance_id', '')}"
            )
    return local_to_key, local_to_ordinal


def _connection_alignment_map(
    connections: list[dict[str, Any]],
    *,
    endpoint_keys: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for connection in sorted(
        list(connections or []),
        key=lambda rec: (
            str(rec.get("src_module_id", "")),
            str(rec.get("dst_module_id", "")),
            str(rec.get("src_slot", "")),
            str(rec.get("dst_slot", "")),
            str(rec.get("value_type", "")),
            str(rec.get("channel", "")),
            str(rec.get("connection_id", "")),
        ),
    ):
        src_key = endpoint_keys.get(str(connection.get("src_module_id", "")))
        dst_key = endpoint_keys.get(str(connection.get("dst_module_id", "")))
        if not src_key or not dst_key:
            continue
        material = {
            "src": src_key,
            "dst": dst_key,
            "src_slot": str(connection.get("src_slot", "out")),
            "dst_slot": str(connection.get("dst_slot", "in")),
            "value_type": str(connection.get("value_type", "scalar")),
            "channel": str(connection.get("channel", "generic")),
        }
        base_key = hashlib.sha256(_canonical_bytes(material)).hexdigest()
        grouped.setdefault(base_key, []).append(dict(connection))
    out: dict[str, dict[str, Any]] = {}
    for base_key, records in sorted(grouped.items()):
        for ordinal, record in enumerate(records):
            out[f"{base_key}:{ordinal:04d}"] = record
    return out


def _native_endpoint_pair(
    connection: Mapping[str, Any],
    *,
    endpoint_keys: Mapping[str, str],
) -> tuple[str, str] | None:
    """Return the biological endpoint pair only when both ends are native."""

    src_key = endpoint_keys.get(str(connection.get("src_module_id", "")), "")
    dst_key = endpoint_keys.get(str(connection.get("dst_module_id", "")), "")
    native_prefix = "native:"
    if not src_key.startswith(native_prefix) or not dst_key.startswith(native_prefix):
        return None
    return (src_key[len(native_prefix) :], dst_key[len(native_prefix) :])


def _developmental_prerequisites(module: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the normalized hereditary dependency endpoints without mutation."""

    from evolution.structural_development_v1 import (
        DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
        effective_developmental_expression_record,
    )

    params = module.get("params", {})
    if not isinstance(params, Mapping):
        return ()
    block = (
        params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params
        else None
    )
    record = effective_developmental_expression_record(block)
    return tuple(str(value) for value in record.get("prerequisite_homology_ids", []))


def _linked_component_plan(
    *,
    matched_homology: list[str],
    native_a: Mapping[str, Mapping[str, Any]],
    native_b: Mapping[str, Mapping[str, Any]],
    matched_conn_keys: list[str],
    conn_a: Mapping[str, Mapping[str, Any]],
    conn_b: Mapping[str, Mapping[str, Any]],
    endpoint_a: Mapping[str, str],
    endpoint_b: Mapping[str, str],
) -> tuple[list[tuple[str, ...]], dict[str, int], dict[str, int]]:
    """Build the frozen connected-component donor plan before any draw.

    Every matched native homology is a node.  A matched ordinary connection,
    or a developmental dependency present in either parent, joins its endpoint
    nodes.  A matched connection that cannot be assigned to native endpoints is
    a static incompatibility and therefore fails before donor RNG is consumed.
    """

    nodes = tuple(sorted(str(value) for value in matched_homology))
    node_set = set(nodes)
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Root choice is lexical so component identity cannot depend on edge
        # traversal, caller parent order, or local module packing.
        low, high = sorted((left_root, right_root))
        parent[high] = low

    connection_endpoints: dict[str, tuple[str, str]] = {}
    for conn_key in sorted(matched_conn_keys):
        pair_a = _native_endpoint_pair(conn_a[conn_key], endpoint_keys=endpoint_a)
        pair_b = _native_endpoint_pair(conn_b[conn_key], endpoint_keys=endpoint_b)
        if pair_a is None or pair_b is None or pair_a != pair_b:
            _det_fail(
                "DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION",
                f"connection_key={conn_key};parent_a_endpoints={pair_a!r};"
                f"parent_b_endpoints={pair_b!r}",
            )
        src_homology, dst_homology = pair_a
        if src_homology not in node_set or dst_homology not in node_set:
            _det_fail(
                "DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION",
                f"connection_key={conn_key};endpoints={pair_a!r};"
                "reason=endpoint_not_matched_native_homology",
            )
        connection_endpoints[conn_key] = pair_a
        union(src_homology, dst_homology)

    for homology_id in nodes:
        for module in (native_a[homology_id], native_b[homology_id]):
            for prerequisite in _developmental_prerequisites(module):
                if prerequisite in node_set:
                    union(homology_id, prerequisite)

    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)
    components = sorted(tuple(sorted(values)) for values in grouped.values())
    homology_to_component = {
        homology_id: component_index
        for component_index, component in enumerate(components)
        for homology_id in component
    }
    connection_to_component: dict[str, int] = {}
    for conn_key, (src_homology, dst_homology) in sorted(connection_endpoints.items()):
        src_component = homology_to_component[src_homology]
        dst_component = homology_to_component[dst_homology]
        if src_component != dst_component:
            _det_fail(
                "DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION",
                f"connection_key={conn_key};reason=endpoints_not_in_one_component",
            )
        connection_to_component[conn_key] = src_component
    return components, homology_to_component, connection_to_component


@dataclass(frozen=True)
class C2RecombinationSummary:
    matched_modules: int
    unmatched_modules_parent_a: int
    unmatched_modules_parent_b: int
    matched_connections: int
    unmatched_connections_parent_a: int
    unmatched_connections_parent_b: int
    unmatched_policy: str
    alignment_key: str
    budget_draws: int
    legacy_unresolved_modules: int = 0
    parent_order_rule: str = C2_PARENT_ORDER_RULE


def recombine_lineage_payloads(
    *,
    parent_a_payload: dict[str, Any],
    parent_b_payload: dict[str, Any],
    rng: Any,
    rng_manager: Any = None,
    alignment_key: str = C2_FROZEN_ALIGNMENT_KEY,
    unmatched_policy: str = C2_FROZEN_UNMATCHED_POLICY,
    trusted_canonical_parent_payloads: bool = False,
    sustainable_growth_enabled: bool = False,
    inheritance_profile: str = LEGACY_INHERITANCE_PROFILE,
) -> tuple[dict[str, Any], C2RecombinationSummary]:
    key, policy = _validate_policy(alignment_key, unmatched_policy)
    profile = validate_production_inheritance_profile(inheritance_profile)
    if bool(trusted_canonical_parent_payloads):
        raw_a = _trusted_payload(parent_a_payload)
        raw_b = _trusted_payload(parent_b_payload)
    else:
        raw_a = canonical_payload_from_genome_dict(dict(parent_a_payload))
        raw_b = canonical_payload_from_genome_dict(dict(parent_b_payload))

    parents = _canonical_parent_order(raw_a, raw_b)
    scope_a, pa, digest_a = parents[0]
    scope_b, pb, digest_b = parents[1]
    caps_a = resolve_b2_caps(pa.get("b2_caps", None))
    caps_b = resolve_b2_caps(pb.get("b2_caps", None))
    if caps_a != caps_b:
        _det_fail(
            "DET_C2_CAPACITY_EPOCH_MISMATCH",
            f"caps_a={caps_a!r};caps_b={caps_b!r}",
        )
    modules_a = [dict(v) for v in list(pa.get("modules", []))]
    modules_b = [dict(v) for v in list(pb.get("modules", []))]
    from evolution.structural_development_v1 import (
        DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
        DEVELOPMENT_OPERATOR_PROFILE,
        agent_has_developmental_records,
        effective_developmental_expression_record,
        normalize_agent_developmental_modules,
        validate_agent_developmental_graph,
    )

    # ESCAP-6 schema, stable identities, endpoints, caps, and parent DAGs are
    # all validated before the donor budget is computed or any draw occurs.
    modules_a = normalize_agent_developmental_modules(modules_a)
    modules_b = normalize_agent_developmental_modules(modules_b)
    if agent_has_developmental_records(modules_a):
        validate_agent_developmental_graph(modules_a)
    if agent_has_developmental_records(modules_b):
        validate_agent_developmental_graph(modules_b)
    modules_a = _normalize_gated_modules_before_draws(modules_a, parent_scope=scope_a)
    modules_b = _normalize_gated_modules_before_draws(modules_b, parent_scope=scope_b)
    if bool(sustainable_growth_enabled):
        modules_a = _normalize_sustainable_growth_modules_before_draws(
            modules_a,
            parent_scope=scope_a,
        )
        modules_b = _normalize_sustainable_growth_modules_before_draws(
            modules_b,
            parent_scope=scope_b,
        )
    native_a = _native_homology_map(modules_a, parent_scope=scope_a)
    native_b = _native_homology_map(modules_b, parent_scope=scope_b)

    matched_homology = sorted(set(native_a).intersection(native_b))
    from evolution.structural_observation_routing_v1 import (
        ROUTING_BLOCK_KEY,
        normalize_routing_block,
    )
    from evolution.structural_output_routing_v1 import (
        OUTPUT_ROUTING_BLOCK_KEY,
        normalize_output_routing_block,
    )

    for homology_id in matched_homology:
        ma = native_a[homology_id]
        mb = native_b[homology_id]
        pa_live = ma.get("params", {}) if isinstance(ma.get("params", {}), dict) else {}
        pb_live = mb.get("params", {}) if isinstance(mb.get("params", {}), dict) else {}
        op_a = str(pa_live.get("live_operator", "bounded_scalar_integrator_v1") or "bounded_scalar_integrator_v1")
        op_b = str(pb_live.get("live_operator", "bounded_scalar_integrator_v1") or "bounded_scalar_integrator_v1")
        if str(ma.get("module_type", "")) != str(mb.get("module_type", "")) or op_a != op_b:
            _det_fail(
                "DET_C2_INCOMPATIBLE_LIVE_MODULE",
                f"homology_id={homology_id};module_types={ma.get('module_type','')},{mb.get('module_type','')};operators={op_a},{op_b}",
            )
        gated_signature_a = _gated_contract_signature(pa_live)
        gated_signature_b = _gated_contract_signature(pb_live)
        if gated_signature_a != gated_signature_b:
            _det_fail(
                "DET_C2_INCOMPATIBLE_ESCAP7_GATED_CONTRACT",
                f"homology_id={homology_id};signature_a={gated_signature_a!r};"
                f"signature_b={gated_signature_b!r}",
            )
        if bool(sustainable_growth_enabled):
            activity_signature_a = _activity_contract_signature(pa_live)
            activity_signature_b = _activity_contract_signature(pb_live)
            if activity_signature_a != activity_signature_b:
                _det_fail(
                    "DET_C2_INCOMPATIBLE_ESCAP8_ACTIVITY_CONTRACT",
                    f"homology_id={homology_id};signature_a={activity_signature_a!r};"
                    f"signature_b={activity_signature_b!r}",
                )
        # Validate optional module-local routing records before the donor draw.
        # A missing legacy block is the canonical disabled/empty v1 record;
        # binding contents may differ because the complete donor module is
        # inherited atomically by the existing C2 rule.
        routing_a = normalize_routing_block(pa_live.get(ROUTING_BLOCK_KEY))
        routing_b = normalize_routing_block(pb_live.get(ROUTING_BLOCK_KEY))
        if (
            routing_a["schema_version"],
            routing_a["registry_version"],
        ) != (
            routing_b["schema_version"],
            routing_b["registry_version"],
        ):
            _det_fail(
                "DET_C2_INCOMPATIBLE_OBSERVATION_ROUTING",
                f"homology_id={homology_id};routing_a={routing_a['schema_version']},{routing_a['registry_version']};"
                f"routing_b={routing_b['schema_version']},{routing_b['registry_version']}",
            )
        output_a = normalize_output_routing_block(pa_live.get(OUTPUT_ROUTING_BLOCK_KEY))
        output_b = normalize_output_routing_block(pb_live.get(OUTPUT_ROUTING_BLOCK_KEY))
        if (
            output_a["schema_version"],
            output_a["registry_version"],
        ) != (
            output_b["schema_version"],
            output_b["registry_version"],
        ):
            _det_fail(
                "DET_C2_INCOMPATIBLE_OUTPUT_ROUTING",
                f"homology_id={homology_id};output_a={output_a['schema_version']},{output_a['registry_version']};"
                f"output_b={output_b['schema_version']},{output_b['registry_version']}",
            )
        development_a = effective_developmental_expression_record(
            pa_live.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
            if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in pa_live
            else None
        )
        development_b = effective_developmental_expression_record(
            pb_live.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
            if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in pb_live
            else None
        )
        compatibility_a = (
            development_a["schema_version"],
            development_a["stage_profile_version"],
            DEVELOPMENT_OPERATOR_PROFILE,
        )
        compatibility_b = (
            development_b["schema_version"],
            development_b["stage_profile_version"],
            DEVELOPMENT_OPERATOR_PROFILE,
        )
        if compatibility_a != compatibility_b:
            _det_fail(
                "DET_C2_INCOMPATIBLE_DEVELOPMENTAL_EXPRESSION",
                f"homology_id={homology_id};development_a={compatibility_a!r};"
                f"development_b={compatibility_b!r}",
            )
    matched_a_instances = {str(native_a[h]["module_instance_id"]) for h in matched_homology}
    matched_b_instances = {str(native_b[h]["module_instance_id"]) for h in matched_homology}
    unmatched_a = [
        module
        for module in modules_a
        if str(module.get("module_instance_id", "")) not in matched_a_instances
    ]
    unmatched_b = [
        module
        for module in modules_b
        if str(module.get("module_instance_id", "")) not in matched_b_instances
    ]

    # Legacy-derived identifiers prove neither common ancestry nor distinct
    # loci.  Unioning both unresolved sets would therefore manufacture
    # structural growth on every cross and eventually violate frozen caps.
    # When both parents carry unresolved legacy material, retain the complete
    # set from the canonical first parent and treat the other set as
    # conservatively unresolvable.  This is whole-set inheritance, not
    # cross-parent alignment: legacy loci remain unmatched, consume no donor
    # draws, and never acquire native homology status.
    legacy_a = [
        module
        for module in unmatched_a
        if str(module.get("identity_origin", "")) == IDENTITY_ORIGIN_LEGACY
    ]
    legacy_b = [
        module
        for module in unmatched_b
        if str(module.get("identity_origin", "")) == IDENTITY_ORIGIN_LEGACY
    ]
    legacy_parent_set_fallback = bool(legacy_a and legacy_b)
    if legacy_parent_set_fallback:
        unmatched_b = [
            module
            for module in unmatched_b
            if str(module.get("identity_origin", "")) != IDENTITY_ORIGIN_LEGACY
        ]

    endpoint_a, _ordinal_a = _module_endpoint_keys(modules_a, parent_scope=f"{scope_a}:{digest_a}")
    endpoint_b, _ordinal_b = _module_endpoint_keys(modules_b, parent_scope=f"{scope_b}:{digest_b}")
    conn_a = _connection_alignment_map(list(pa.get("connections", [])), endpoint_keys=endpoint_a)
    conn_b = _connection_alignment_map(list(pb.get("connections", [])), endpoint_keys=endpoint_b)
    matched_conn_keys = sorted(set(conn_a).intersection(conn_b))
    unmatched_conn_a = sorted(set(conn_a) - set(conn_b))
    unmatched_conn_b = sorted(set(conn_b) - set(conn_a))

    linked_components: list[tuple[str, ...]] = []
    homology_to_component: dict[str, int] = {}
    connection_to_component: dict[str, int] = {}
    if _linked_structural_profile(profile):
        (
            linked_components,
            homology_to_component,
            connection_to_component,
        ) = _linked_component_plan(
            matched_homology=matched_homology,
            native_a=native_a,
            native_b=native_b,
            matched_conn_keys=matched_conn_keys,
            conn_a=conn_a,
            conn_b=conn_b,
            endpoint_a=endpoint_a,
            endpoint_b=endpoint_b,
        )

    if bool(sustainable_growth_enabled):
        _validate_escap8_prospective_caps_before_draws(
            matched_modules=len(matched_homology),
            unmatched_modules_a=len(unmatched_a),
            unmatched_modules_b=len(unmatched_b),
            matched_connections=len(matched_conn_keys),
            unmatched_connections_a=len(unmatched_conn_a),
            unmatched_connections_b=len(unmatched_conn_b),
            b2_caps=caps_a,
        )

    budget_draws = int(
        len(linked_components)
        if _linked_structural_profile(profile)
        else len(matched_homology) + len(matched_conn_keys)
    )
    expected = None
    if rng_manager is not None:
        expected = rng_manager.expected_delta_for_boundary(
            "mutation.c2.recombination_lineage",
            ctx={"recomb_draws": int(budget_draws)},
        )
    scope = (
        rng_manager.budget_scope("mutation.c2.recombination_lineage", expected_delta=expected)
        if rng_manager is not None
        else None
    )
    if scope is None:
        draws = _draws(rng, budget_draws)
    else:
        with scope:
            draws = _draws(rng, budget_draws)
    draw_index = 0

    component_donor_scope: dict[int, str] = {}
    if _linked_structural_profile(profile):
        for component_index, _component in enumerate(linked_components):
            component_donor_scope[component_index] = (
                scope_a if draws[draw_index] < 0.5 else scope_b
            )
            draw_index += 1

    selected: list[dict[str, Any]] = []
    for homology_id in matched_homology:
        if _linked_structural_profile(profile):
            donor_scope = component_donor_scope[homology_to_component[homology_id]]
        else:
            donor_scope = scope_a if draws[draw_index] < 0.5 else scope_b
            draw_index += 1
        donor = native_a[homology_id] if donor_scope == scope_a else native_b[homology_id]
        selected.append(
            {
                "record": dict(donor),
                "source_scope": donor_scope,
                "source_local_id": str(donor.get("module_id", "")),
                "matched_homology_id": homology_id,
            }
        )

    for module in unmatched_a:
        selected.append(
            {
                "record": dict(module),
                "source_scope": scope_a,
                "source_local_id": str(module.get("module_id", "")),
                "matched_homology_id": "",
            }
        )
    for module in unmatched_b:
        selected.append(
            {
                "record": dict(module),
                "source_scope": scope_b,
                "source_local_id": str(module.get("module_id", "")),
                "matched_homology_id": "",
            }
        )

    selected.sort(
        key=lambda item: (
            str(item["record"].get("homology_id", "")),
            str(item["record"].get("module_instance_id", "")),
            str(item["source_scope"]),
            str(item["source_local_id"]),
        )
    )
    out_modules: list[dict[str, Any]] = []
    source_local_to_child: dict[tuple[str, str], str] = {}
    homology_to_child: dict[str, str] = {}
    for local_index, item in enumerate(selected):
        child_local_id = f"m{local_index:03d}"
        record = dict(item["record"])
        record["module_id"] = child_local_id
        out_modules.append(record)
        source_local_to_child[(str(item["source_scope"]), str(item["source_local_id"]))] = child_local_id
        if module_is_native_homology(record):
            homology_to_child[str(record["homology_id"])] = child_local_id

    # Shared native homologs map both parental local slots to the selected child
    # slot; unmatched records map only through their owning parent.
    for parent_scope, modules in ((scope_a, modules_a), (scope_b, modules_b)):
        for module in modules:
            if module_is_native_homology(module):
                child_local = homology_to_child.get(str(module.get("homology_id", "")))
                if child_local:
                    source_local_to_child[(parent_scope, str(module.get("module_id", "")))] = child_local

    connection_candidates: list[tuple[str, dict[str, Any]]] = []
    for conn_key in matched_conn_keys:
        if _linked_structural_profile(profile):
            donor_scope = component_donor_scope[connection_to_component[conn_key]]
        else:
            donor_scope = scope_a if draws[draw_index] < 0.5 else scope_b
            draw_index += 1
        donor = conn_a[conn_key] if donor_scope == scope_a else conn_b[conn_key]
        connection_candidates.append((donor_scope, dict(donor)))
    connection_candidates.extend((scope_a, dict(conn_a[k])) for k in unmatched_conn_a)
    connection_candidates.extend((scope_b, dict(conn_b[k])) for k in unmatched_conn_b)

    remapped_connections: list[dict[str, Any]] = []
    for source_scope, connection in connection_candidates:
        src_child = source_local_to_child.get((source_scope, str(connection.get("src_module_id", ""))))
        dst_child = source_local_to_child.get((source_scope, str(connection.get("dst_module_id", ""))))
        if not src_child or not dst_child:
            continue
        out = dict(connection)
        out["src_module_id"] = src_child
        out["dst_module_id"] = dst_child
        out.pop("locality_distance", None)
        out.pop("locality_cost", None)
        out["_source_scope"] = source_scope
        remapped_connections.append(out)
    remapped_connections.sort(
        key=lambda rec: (
            str(rec.get("src_module_id", "")),
            str(rec.get("dst_module_id", "")),
            str(rec.get("src_slot", "")),
            str(rec.get("dst_slot", "")),
            str(rec.get("value_type", "")),
            str(rec.get("channel", "")),
            float(rec.get("weight", 0.0) or 0.0),
            str(rec.get("_source_scope", "")),
            str(rec.get("connection_id", "")),
        )
    )
    out_connections: list[dict[str, Any]] = []
    for local_index, connection in enumerate(remapped_connections):
        out = dict(connection)
        out.pop("_source_scope", None)
        out["connection_id"] = f"c{local_index:03d}"
        out_connections.append(out)

    out_payload = dict(pa)
    out_payload["modules"] = out_modules
    out_payload["connections"] = out_connections
    out_payload["b2_invalid_policy"] = "reject"
    genes = dict(out_payload.get("genes", {}))
    genes.update(
        {
            "c2_recombination_version": C2_RECOMBINATION_VERSION,
            "c2_alignment_key": C2_FROZEN_ALIGNMENT_KEY,
            "c2_unmatched_policy": C2_FROZEN_UNMATCHED_POLICY,
            "c2_parent_order_rule": C2_PARENT_ORDER_RULE,
            "c2_legacy_unresolved_policy": C2_LEGACY_UNRESOLVED_POLICY,
            "c2_legacy_parent_set_fallback": int(legacy_parent_set_fallback),
            "c2_matched_modules": int(len(matched_homology)),
            "c2_unmatched_modules_parent_a": int(len(unmatched_a)),
            "c2_unmatched_modules_parent_b": int(len(unmatched_b)),
            "c2_matched_connections": int(len(matched_conn_keys)),
            "c2_unmatched_connections_parent_a": int(len(unmatched_conn_a)),
            "c2_unmatched_connections_parent_b": int(len(unmatched_conn_b)),
            "c2_budget_draws": int(budget_draws),
            "c2_legacy_unresolved_modules": int(
                sum(
                    1
                    for module in out_modules
                    if str(module.get("identity_origin", "")) == IDENTITY_ORIGIN_LEGACY
                )
            ),
            "structural_identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
            "structural_ancestry_schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
        }
    )
    if _linked_structural_profile(profile):
        genes.update(
            {
                "c2_inheritance_profile": profile,
                "c2_structural_linkage_rule": C2_STRUCTURAL_LINKAGE_RULE,
                "c2_linked_component_count": int(len(linked_components)),
            }
        )
    out_payload["genes"] = genes
    out_payload = canonical_payload_from_genome_dict(out_payload)
    _normalize_gated_modules_before_draws(
        list(out_payload.get("modules", []) or []),
        parent_scope="child",
    )
    if bool(sustainable_growth_enabled):
        _normalize_sustainable_growth_modules_before_draws(
            list(out_payload.get("modules", []) or []),
            parent_scope="child",
        )
    if agent_has_developmental_records(list(out_payload.get("modules", []) or [])):
        validate_agent_developmental_graph(list(out_payload.get("modules", []) or []))

    summary = C2RecombinationSummary(
        matched_modules=int(len(matched_homology)),
        unmatched_modules_parent_a=int(len(unmatched_a)),
        unmatched_modules_parent_b=int(len(unmatched_b)),
        matched_connections=int(len(matched_conn_keys)),
        unmatched_connections_parent_a=int(len(unmatched_conn_a)),
        unmatched_connections_parent_b=int(len(unmatched_conn_b)),
        unmatched_policy=policy,
        alignment_key=key,
        budget_draws=int(budget_draws),
        legacy_unresolved_modules=int(genes["c2_legacy_unresolved_modules"]),
    )
    return out_payload, summary


def recombine_self_lineage_payload(
    *,
    parent_payload: dict[str, Any],
    rng: Any,
    rng_manager: Any = None,
    alignment_key: str = C2_FROZEN_ALIGNMENT_KEY,
    unmatched_policy: str = C2_FROZEN_UNMATCHED_POLICY,
    parent_payload_is_canonical: bool = False,
    skip_output_canonicalize: bool = False,
    sustainable_growth_enabled: bool = False,
    inheritance_profile: str = LEGACY_INHERITANCE_PROFILE,
) -> tuple[dict[str, Any], C2RecombinationSummary]:
    """Self-reproduction preserves payload identity while consuming its budget."""

    key, policy = _validate_policy(alignment_key, unmatched_policy)
    profile = validate_production_inheritance_profile(inheritance_profile)
    parent = (
        _trusted_payload(parent_payload)
        if bool(parent_payload_is_canonical)
        else canonical_payload_from_genome_dict(dict(parent_payload))
    )
    parent_caps = resolve_b2_caps(parent.get("b2_caps", None))
    modules = [dict(v) for v in list(parent.get("modules", []))]
    connections = [dict(v) for v in list(parent.get("connections", []))]
    from evolution.structural_development_v1 import (
        agent_has_developmental_records,
        normalize_agent_developmental_modules,
        validate_agent_developmental_graph,
    )

    modules = normalize_agent_developmental_modules(modules)
    if agent_has_developmental_records(modules):
        validate_agent_developmental_graph(modules)
    modules = _normalize_gated_modules_before_draws(modules, parent_scope="self_parent")
    if bool(sustainable_growth_enabled):
        modules = _normalize_sustainable_growth_modules_before_draws(
            modules,
            parent_scope="self_parent",
        )
        _validate_escap8_prospective_caps_before_draws(
            matched_modules=len(modules),
            unmatched_modules_a=0,
            unmatched_modules_b=0,
            matched_connections=len(connections),
            unmatched_connections_a=0,
            unmatched_connections_b=0,
            b2_caps=parent_caps,
        )
    linked_component_count = 0
    if _linked_structural_profile(profile):
        native = _native_homology_map(modules, parent_scope="self_parent")
        endpoint_keys, _ordinals = _module_endpoint_keys(
            modules,
            parent_scope="self_parent",
        )
        conn_map = _connection_alignment_map(connections, endpoint_keys=endpoint_keys)
        linked_conn_keys = [
            conn_key
            for conn_key, connection in sorted(conn_map.items())
            if _native_endpoint_pair(connection, endpoint_keys=endpoint_keys) is not None
        ]
        linked_components, _homology_components, _connection_components = (
            _linked_component_plan(
                matched_homology=sorted(native),
                native_a=native,
                native_b=native,
                matched_conn_keys=linked_conn_keys,
                conn_a=conn_map,
                conn_b=conn_map,
                endpoint_a=endpoint_keys,
                endpoint_b=endpoint_keys,
            )
        )
        linked_component_count = len(linked_components)
    budget_draws = int(
        linked_component_count
        if _linked_structural_profile(profile)
        else len(modules) + len(connections)
    )
    expected = None
    if rng_manager is not None:
        expected = rng_manager.expected_delta_for_boundary(
            "mutation.c2.recombination_lineage",
            ctx={"recomb_draws": int(budget_draws)},
        )
    scope = (
        rng_manager.budget_scope("mutation.c2.recombination_lineage", expected_delta=expected)
        if rng_manager is not None
        else None
    )
    if scope is None:
        _draws(rng, budget_draws)
    else:
        with scope:
            _draws(rng, budget_draws)

    out_payload = dict(parent)
    out_payload["modules"] = modules
    out_payload["connections"] = connections
    genes = dict(out_payload.get("genes", {}))
    genes.update(
        {
            "c2_recombination_version": C2_RECOMBINATION_VERSION,
            "c2_alignment_key": C2_FROZEN_ALIGNMENT_KEY,
            "c2_unmatched_policy": C2_FROZEN_UNMATCHED_POLICY,
            "c2_parent_order_rule": "self_identity",
            "c2_legacy_unresolved_policy": C2_LEGACY_UNRESOLVED_POLICY,
            "c2_legacy_parent_set_fallback": 0,
            "c2_matched_modules": int(len(modules)),
            "c2_unmatched_modules_parent_a": 0,
            "c2_unmatched_modules_parent_b": 0,
            "c2_matched_connections": int(len(connections)),
            "c2_unmatched_connections_parent_a": 0,
            "c2_unmatched_connections_parent_b": 0,
            "c2_budget_draws": int(budget_draws),
            "c2_legacy_unresolved_modules": int(
                sum(
                    1
                    for module in modules
                    if str(module.get("identity_origin", "")) == IDENTITY_ORIGIN_LEGACY
                )
            ),
        }
    )
    if _linked_structural_profile(profile):
        genes.update(
            {
                "c2_inheritance_profile": profile,
                "c2_structural_linkage_rule": C2_STRUCTURAL_LINKAGE_RULE,
                "c2_linked_component_count": int(linked_component_count),
            }
        )
    out_payload["genes"] = genes
    if not bool(skip_output_canonicalize):
        out_payload = canonical_payload_from_genome_dict(out_payload)
    if agent_has_developmental_records(list(out_payload.get("modules", []) or [])):
        validate_agent_developmental_graph(list(out_payload.get("modules", []) or []))
    _normalize_gated_modules_before_draws(
        list(out_payload.get("modules", []) or []),
        parent_scope="self_child",
    )
    if bool(sustainable_growth_enabled):
        _normalize_sustainable_growth_modules_before_draws(
            list(out_payload.get("modules", []) or []),
            parent_scope="self_child",
        )
    summary = C2RecombinationSummary(
        matched_modules=int(len(modules)),
        unmatched_modules_parent_a=0,
        unmatched_modules_parent_b=0,
        matched_connections=int(len(connections)),
        unmatched_connections_parent_a=0,
        unmatched_connections_parent_b=0,
        unmatched_policy=policy,
        alignment_key=key,
        budget_draws=int(budget_draws),
        legacy_unresolved_modules=int(genes["c2_legacy_unresolved_modules"]),
        parent_order_rule="self_identity",
    )
    return out_payload, summary
