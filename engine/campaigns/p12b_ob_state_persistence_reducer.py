"""P12B ordinary-birth Genome V2 state-persistence reducer.

This reducer is post-hoc over authoritative checkpoint projections. It extends
the P12A state identity semantics with a stricter origin test: a non-founder
carrier is counted as ordinary-birth persistence only when the target checkpoint
contains a validated causal event/terminal/join receipt for an ordinary-birth
ancestor or direct ordinary-born child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_causal_transaction_projection_v1 import (
    join_causal_transaction_records_v1,
)

from . import p12a_gv2_persistence_reducer as p12a
from .canonical import canonical_sha256, pretty_json


SCHEMA_VERSION = "p12b_ob_state_persistence_reducer_schema_v1"
RESULT_SCHEMA_VERSION = "p12b_ob_state_persistence_reducer_result_v1"
REDUCER_ID = "p12b_ob_state_persistence_b12_g30_confirmation_v1"
REDUCER_VERSION = "p12b_ob_state_persistence_reducer_v1"
SUPPORTED_IDENTITY_TYPES = p12a.SUPPORTED_IDENTITY_TYPES
ORIGIN_CLASSES = (
    "founder",
    "ordinary_birth",
    "turnover_refill",
    "mixed_descendant_path",
    "unresolved",
)

EXPECTED_AUTHORITY = {
    "provider_id": "eiel-production-provider-v1",
    "runtime_profile": "eiel-production-runtime-v1",
    "cuda_artifact_sha256": "b0413642f9781452774377bf811927619bfa4ff1c754b474e7028c324e09280d",
    "cuda_manifest_sha256": "799c951fd5c0d60deb09a48a8801f1d20fa1b477ffd48245328f863cb014ead2",
    "execution_route_sha256": "4e4137e376a3c74bab67237cd3f9ef4cfceb4e7c95e2292482b772eadf5028cf",
    "execution_route_receipt_sha256": "690a9d5fcb7201e5a553d445117da2af5f29e99a839a1f71bd6f0a8ca11eda4e",
    "package_id_sha256": "5d9b11e08e818849558e45589ec5d678170f68386351dbd4df01415353bb3500",
    "artifact_source_commit": "ab5b5c65968676d2676289abdc335d39a07401de",
}


class P12BOrdinaryBirthStatePersistenceReducerError(RuntimeError):
    """Reducer input is not authoritative enough for P12B confirmation."""


def _fail(message: str) -> None:
    raise P12BOrdinaryBirthStatePersistenceReducerError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12BOrdinaryBirthStatePersistenceReducerError(
            f"cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"JSON document must be an object: {path}")
    return value


def _validate_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(authority, Mapping):
        _fail("authority binding must be an object")
    result: dict[str, Any] = {}
    for key, expected in EXPECTED_AUTHORITY.items():
        actual = authority.get(key)
        if actual != expected:
            _fail(
                f"authority binding mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
        result[key] = actual
    return result


def reducer_schema(*, source_commit: str, authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_id": REDUCER_ID,
        "reducer_version": REDUCER_VERSION,
        "source_commit": source_commit,
        "authority": _validate_authority(authority),
        "identity_types": list(SUPPORTED_IDENTITY_TYPES),
        "origin_classes": list(ORIGIN_CLASSES),
        "unit_of_observation": "cell x preregistered hereditary state x milestone",
        "positive_ordinary_birth_persistence_rule": {
            "nonfounder_carrier": True,
            "requires_event_kind": "ordinary_birth",
            "requires_exact_event_terminal_join": True,
            "counted_origin_classes": ["ordinary_birth", "mixed_descendant_path"],
            "turnover_only_carriers_counted": False,
        },
    }


def _receipt_sha(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    if type(claimed) is not str:
        _fail("causal receipt lacks receipt_sha256")
    observed = content_sha256(payload)
    if observed != claimed:
        _fail("causal receipt SHA-256 does not match its content")
    return claimed


def _causal_projection(projection: Mapping[str, Any]) -> Mapping[str, Any]:
    gv2 = projection.get("genome_upgrade_v2_checkpoint_projection")
    if not isinstance(gv2, Mapping):
        _fail("checkpoint projection missing Genome V2 projection")
    causal = gv2.get("causal_transaction_projection")
    if not isinstance(causal, Mapping):
        _fail("checkpoint projection missing causal transaction projection")
    if causal.get("capture_status") != "capture_enabled_joined":
        _fail("causal transaction projection is not capture_enabled_joined")
    if causal.get("evaluable") is not True:
        _fail("causal transaction projection is not evaluable")
    dropped = causal.get("dropped_counts")
    if not isinstance(dropped, Mapping) or int(dropped.get("total", -1)) != 0:
        _fail("causal transaction projection has dropped records")
    return causal


def _validate_causal_projection(
    projection: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    causal = _causal_projection(projection)
    event_receipts = causal.get("event_receipts")
    terminal_receipts = causal.get("terminal_receipts")
    join_receipts = causal.get("join_receipts")
    if not isinstance(event_receipts, list) or not isinstance(terminal_receipts, list):
        _fail("causal projection receipts must be arrays")
    if not isinstance(join_receipts, list):
        _fail("causal projection join receipts must be an array")
    event_records: list[Mapping[str, Any]] = []
    terminal_records: list[Mapping[str, Any]] = []
    event_sha_by_tx: dict[str, str] = {}
    terminal_sha_by_tx: dict[str, str] = {}
    for receipt in event_receipts:
        if not isinstance(receipt, Mapping):
            _fail("event receipt must be an object")
        record = receipt.get("transaction_record")
        if not isinstance(record, Mapping):
            _fail("event receipt lacks transaction_record")
        identity = record.get("identity")
        if not isinstance(identity, Mapping) or type(identity.get("transaction_id")) is not str:
            _fail("event receipt lacks transaction_id")
        receipt_sha = _receipt_sha(receipt)
        event_records.append(record)
        event_sha_by_tx[str(identity["transaction_id"])] = receipt_sha
    for receipt in terminal_receipts:
        if not isinstance(receipt, Mapping):
            _fail("terminal receipt must be an object")
        record = receipt.get("transaction_record")
        if not isinstance(record, Mapping):
            _fail("terminal receipt lacks transaction_record")
        identity = record.get("identity")
        if not isinstance(identity, Mapping) or type(identity.get("transaction_id")) is not str:
            _fail("terminal receipt lacks transaction_id")
        receipt_sha = _receipt_sha(receipt)
        terminal_records.append(record)
        terminal_sha_by_tx[str(identity["transaction_id"])] = receipt_sha
    joins = join_causal_transaction_records_v1(event_records, terminal_records)
    join_by_tx = {join.transaction_id: join for join in joins}
    if len(join_by_tx) != len(join_receipts):
        _fail("causal join receipt count differs from validated joins")

    origin_by_agent: dict[int, dict[str, Any]] = {}
    stats = {
        "validated_join_count": len(joins),
        "ordinary_birth_join_count": 0,
        "turnover_join_count": 0,
        "direct_child_survival_count": 0,
        "event_ledger_descendant_retention_count": 0,
        "terminal_loss_count": 0,
    }
    for receipt in join_receipts:
        if not isinstance(receipt, Mapping):
            _fail("join receipt must be an object")
        tx = receipt.get("transaction_id")
        if type(tx) is not str or tx not in join_by_tx:
            _fail("join receipt references an unknown transaction_id")
        if receipt.get("event_receipt_sha256") != event_sha_by_tx[tx]:
            _fail("join receipt event SHA-256 does not match event receipt")
        if receipt.get("terminal_receipt_sha256") != terminal_sha_by_tx[tx]:
            _fail("join receipt terminal SHA-256 does not match terminal receipt")
        _receipt_sha(receipt)
        joined = join_by_tx[tx]
        identity = joined.event_time.identity
        terminal_state = dict(joined.terminal.resulting_state)
        relation = str(terminal_state.get("retention_relation", ""))
        event_kind = identity.event_kind
        if event_kind == "ordinary_birth":
            stats["ordinary_birth_join_count"] += 1
        else:
            stats["turnover_join_count"] += 1
        if relation == "direct_child_survival":
            stats["direct_child_survival_count"] += 1
        elif relation == "event_ledger_descendant_retention":
            stats["event_ledger_descendant_retention_count"] += 1
        elif relation == "terminal_loss":
            stats["terminal_loss_count"] += 1
        terminal_agent_id = terminal_state.get("agent_id")
        survived = terminal_state.get("survived_to_terminal")
        if survived is not True or type(terminal_agent_id) is not int:
            continue
        origin_class = (
            "ordinary_birth"
            if event_kind == "ordinary_birth" and relation == "direct_child_survival"
            else (
                "mixed_descendant_path"
                if event_kind == "ordinary_birth"
                and relation == "event_ledger_descendant_retention"
                else "turnover_refill"
            )
        )
        candidate = {
            "origin_class": origin_class,
            "transaction_id": tx,
            "event_kind": event_kind,
            "retention_relation": relation,
            "event_child_id": identity.child_id,
            "terminal_agent_id": terminal_agent_id,
            "parent_id": identity.parent_id,
            "parent_slot": identity.parent_slot,
            "mate_parent_id": identity.mate_parent_id,
            "mate_parent_slot": identity.mate_parent_slot,
            "lineage_founder_id": identity.lineage_founder_id,
            "lineage_depth": identity.lineage_depth,
            "generation": identity.generation,
            "tick": identity.tick,
            "event_sequence": identity.event_sequence,
            "event_receipt_sha256": event_sha_by_tx[tx],
            "terminal_receipt_sha256": terminal_sha_by_tx[tx],
            "join_receipt_sha256": str(receipt["receipt_sha256"]),
        }
        previous = origin_by_agent.get(terminal_agent_id)
        if previous is None or (
            previous["origin_class"] == "turnover_refill"
            and origin_class in {"ordinary_birth", "mixed_descendant_path"}
        ):
            origin_by_agent[terminal_agent_id] = candidate
    return origin_by_agent, stats


def _carrier_origin(
    carrier: Mapping[str, int],
    *,
    founder_id: int,
    origin_by_agent: Mapping[int, Mapping[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if carrier["agent_id"] == founder_id and carrier["lineage_depth"] == 0:
        return "founder", None
    origin = origin_by_agent.get(int(carrier["agent_id"]))
    if origin is None:
        return "unresolved", None
    origin_class = str(origin.get("origin_class", "unresolved"))
    if origin_class not in ORIGIN_CLASSES:
        return "unresolved", dict(origin)
    return origin_class, dict(origin)


def reduce_ob_state_persistence(
    *,
    g0_path: Path,
    target_path: Path,
    target_milestone: int,
    authority: Mapping[str, Any],
    source_commit: str,
    campaign_id: str,
    cell_id: str,
    seed: int,
    evidence_root: Path,
) -> dict[str, Any]:
    if type(target_milestone) is not int or target_milestone <= 0:
        _fail("target milestone must be a positive integer")
    schema = reducer_schema(source_commit=source_commit, authority=authority)
    g0_doc = _read_json(g0_path)
    target_doc = _read_json(target_path)
    try:
        g0 = p12a._projection(g0_doc, milestone=0)
        target = p12a._projection(target_doc, milestone=target_milestone)
        g0_live = p12a._live_slots(g0)
        target_live = p12a._live_slots(target)
    except p12a.P12APersistenceReducerError as exc:
        raise P12BOrdinaryBirthStatePersistenceReducerError(str(exc)) from exc
    capacity = len(p12a._tensor(g0, "alive_mask")[0])
    if len(p12a._tensor(target, "alive_mask")[0]) != capacity:
        _fail(f"G0/G{target_milestone} tensor capacity mismatch")
    origin_by_agent, origin_stats = _validate_causal_projection(target)
    founders = [
        row
        for row in g0_live
        if row["lineage_depth"] == 0 and row["lineage_founder_id"] == row["agent_id"]
    ]
    if not founders:
        _fail("G0 contains no authoritative founders")

    founder_state_table: list[dict[str, Any]] = []
    carrier_table: list[dict[str, Any]] = []
    reducer_rows: list[dict[str, Any]] = []
    for founder in founders:
        for identity_type in SUPPORTED_IDENTITY_TYPES:
            state = p12a._state_definition(
                g0,
                founder,
                identity_type=identity_type,
                capacity=capacity,
            )
            state_id = f"state-{state['state_definition_sha256'][:16]}-{identity_type}"
            founder_state_table.append(
                {
                    "metric_id": REDUCER_ID,
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "milestone": 0,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "founder_slot": founder["slot"],
                    "founder_slot_token": founder["slot_token"],
                    "canonical_source_representation": state,
                }
            )
            matches: list[dict[str, Any]] = []
            for carrier in target_live:
                try:
                    matched, selected, exact = p12a._matches(
                        state, target, carrier["slot"], capacity=capacity
                    )
                except p12a.P12APersistenceReducerError as exc:
                    raise P12BOrdinaryBirthStatePersistenceReducerError(str(exc)) from exc
                if not matched:
                    continue
                origin_class, origin = _carrier_origin(
                    carrier,
                    founder_id=int(founder["agent_id"]),
                    origin_by_agent=origin_by_agent,
                )
                positive = (
                    origin_class in {"ordinary_birth", "mixed_descendant_path"}
                    and int(carrier["lineage_depth"]) > 0
                    and int(carrier["lineage_founder_id"]) == int(founder["agent_id"])
                )
                row = {
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "milestone": target_milestone,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "carrier_stable_id": carrier["agent_id"],
                    "carrier_slot": carrier["slot"],
                    "carrier_slot_token": carrier["slot_token"],
                    "lineage_depth": carrier["lineage_depth"],
                    "lineage_founder_id": carrier["lineage_founder_id"],
                    "carrier_origin_class": origin_class,
                    "positive_ordinary_birth_persistence": positive,
                    "selected_values": selected,
                    "exact_values": exact,
                    "origin_evidence": origin,
                }
                matches.append(row)
                carrier_table.append(row)
            founder_count = sum(1 for item in matches if item["carrier_origin_class"] == "founder")
            positive_rows = [
                item for item in matches if item["positive_ordinary_birth_persistence"]
            ]
            ordinary_direct = sum(
                1 for item in positive_rows if item["carrier_origin_class"] == "ordinary_birth"
            )
            mixed_path = sum(
                1 for item in positive_rows if item["carrier_origin_class"] == "mixed_descendant_path"
            )
            turnover_count = sum(1 for item in matches if item["carrier_origin_class"] == "turnover_refill")
            unresolved_count = sum(1 for item in matches if item["carrier_origin_class"] == "unresolved")
            reducer_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "metric_id": REDUCER_ID,
                    "reducer_version": REDUCER_VERSION,
                    "source_commit": source_commit,
                    "provider_runtime": {
                        "provider_id": schema["authority"]["provider_id"],
                        "runtime_profile": schema["authority"]["runtime_profile"],
                    },
                    "artifact_package_route_identities": {
                        key: schema["authority"][key]
                        for key in (
                            "cuda_artifact_sha256",
                            "cuda_manifest_sha256",
                            "execution_route_sha256",
                            "execution_route_receipt_sha256",
                            "package_id_sha256",
                        )
                    },
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "milestone": target_milestone,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "match_result": "match" if matches else "no_match",
                    "carrier_count": len(matches),
                    "founder_carrier_count": founder_count,
                    "positive_ordinary_birth_carrier_count": len(positive_rows),
                    "ordinary_birth_direct_child_carrier_count": ordinary_direct,
                    "mixed_descendant_path_carrier_count": mixed_path,
                    "turnover_only_carrier_count": turnover_count,
                    "unresolved_carrier_count": unresolved_count,
                    "alive_denominator": len(target_live),
                    "frequency": (len(matches) / len(target_live)) if target_live else 0.0,
                    "positive_ordinary_birth_frequency": (
                        len(positive_rows) / len(target_live) if target_live else 0.0
                    ),
                    "classification": {
                        "state_present": len(matches) > 0,
                        "founder_present": founder_count > 0,
                        "positive_ordinary_birth_persistence": len(positive_rows) > 0,
                        "ordinary_birth_direct_child_persistence": ordinary_direct > 0,
                        "ordinary_birth_descendant_persistence": mixed_path > 0,
                        "turnover_only_not_counted": turnover_count > 0,
                        "unresolved_present": unresolved_count > 0,
                        "state_lost": len(matches) == 0,
                    },
                }
            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "metric_id": REDUCER_ID,
        "reducer_version": REDUCER_VERSION,
        "schema": schema,
        "source_commit": source_commit,
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "seed": seed,
        "target_milestone": target_milestone,
        "checkpoint_identity": {
            "g0": {
                "path": str(g0_path),
                "sha256": hashlib.sha256(g0_path.read_bytes()).hexdigest(),
            },
            f"g{target_milestone}": {
                "path": str(target_path),
                "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            },
        },
        "evidence_root": str(evidence_root),
        "causal_origin_summary": origin_stats,
        "founder_state_table": founder_state_table,
        "target_carrier_table": carrier_table,
        "reducer_rows": reducer_rows,
        "summary": {
            "g0_founder_count": len(founders),
            "g0_alive_denominator": len(g0_live),
            "target_milestone": target_milestone,
            "target_alive_denominator": len(target_live),
            "reducer_row_count": len(reducer_rows),
            "carrier_row_count": len(carrier_table),
            "positive_ordinary_birth_carrier_rows": sum(
                1 for row in carrier_table if row["positive_ordinary_birth_persistence"]
            ),
            "supported_identity_types": list(SUPPORTED_IDENTITY_TYPES),
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def write_reducer_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output_dir / "p12b_ob_state_persistence_reducer_result_v1.json",
        "schema": output_dir / "p12b_ob_state_persistence_reducer_schema_v1.json",
        "founder_state_table": output_dir / "p12b_ob_founder_state_table_v1.json",
        "target_carrier_table": output_dir / "p12b_ob_state_carrier_table_v1.json",
        "reducer_rows": output_dir / "p12b_ob_state_persistence_reducer_rows_v1.json",
        "summary": output_dir / "p12b_ob_state_persistence_reducer_summary_v1.json",
    }
    payloads = {
        "result": dict(result),
        "schema": result["schema"],
        "founder_state_table": {"rows": result["founder_state_table"]},
        "target_carrier_table": {"rows": result["target_carrier_table"]},
        "reducer_rows": {"rows": result["reducer_rows"]},
        "summary": result["summary"],
    }
    written: dict[str, str] = {}
    for name, path in paths.items():
        path.write_text(pretty_json(payloads[name]), encoding="utf-8")
        written[name] = str(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-milestone", type=int, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_ob_state_persistence(
        g0_path=args.g0,
        target_path=args.target,
        target_milestone=args.target_milestone,
        authority=_read_json(args.authority),
        source_commit=args.source_commit,
        campaign_id=args.campaign_id,
        cell_id=args.cell_id,
        seed=args.seed,
        evidence_root=args.evidence_root,
    )
    written = write_reducer_outputs(result, args.output_dir)
    print(pretty_json({"ok": True, "result_sha256": result["result_sha256"], "paths": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
