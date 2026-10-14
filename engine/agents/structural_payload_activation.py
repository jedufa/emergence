from __future__ import annotations

"""Default-off structural payload activation helpers for controlled runs."""

from typing import Any

from agents.genome import Genome
from agents.modular_genome_v1 import MODULAR_GENOME_SCHEMA_VERSION


STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION = "structural_payload_seed_v1"
STRUCTURAL_PAYLOAD_SEED_PROFILE_NONE = "none"
STRUCTURAL_PAYLOAD_SEED_PROFILE_MINIMAL = "minimal_canonical_v1"
STRUCTURAL_PAYLOAD_SEED_PROFILES = (
    STRUCTURAL_PAYLOAD_SEED_PROFILE_NONE,
    STRUCTURAL_PAYLOAD_SEED_PROFILE_MINIMAL,
)
STRUCTURAL_PAYLOAD_SEED_ABLATION_NONE = "none"
STRUCTURAL_PAYLOAD_SEED_ABLATION_ZERO_CONNECTION_GATES = "zero_connection_gates"
STRUCTURAL_PAYLOAD_SEED_ABLATION_REMOVE_CONNECTIONS = "remove_connections"
STRUCTURAL_PAYLOAD_SEED_ABLATIONS = (
    STRUCTURAL_PAYLOAD_SEED_ABLATION_NONE,
    STRUCTURAL_PAYLOAD_SEED_ABLATION_ZERO_CONNECTION_GATES,
    STRUCTURAL_PAYLOAD_SEED_ABLATION_REMOVE_CONNECTIONS,
)


def structural_payload_activation_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_payload_activation_enable", False)) if sim_cfg is not None else False


def validate_structural_payload_activation_config(sim_cfg: Any) -> None:
    if not structural_payload_activation_enabled(sim_cfg):
        return
    profile = str(getattr(sim_cfg, "structural_payload_seed_profile", STRUCTURAL_PAYLOAD_SEED_PROFILE_NONE) or "")
    if profile not in STRUCTURAL_PAYLOAD_SEED_PROFILES:
        raise ValueError(f"unsupported structural_payload_seed_profile={profile!r}")
    if profile == STRUCTURAL_PAYLOAD_SEED_PROFILE_NONE:
        raise ValueError("structural_payload_activation_enable requires a non-none structural_payload_seed_profile")
    schema = str(getattr(sim_cfg, "structural_payload_seed_schema_version", STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION) or "")
    if schema != STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION:
        raise ValueError(
            "structural_payload_seed_schema_version must be "
            f"{STRUCTURAL_PAYLOAD_SEED_SCHEMA_VERSION!r}, got {schema!r}"
        )
    ablation = str(getattr(sim_cfg, "structural_payload_seed_ablation", STRUCTURAL_PAYLOAD_SEED_ABLATION_NONE) or "")
    if ablation not in STRUCTURAL_PAYLOAD_SEED_ABLATIONS:
        raise ValueError(f"unsupported structural_payload_seed_ablation={ablation!r}")


def minimal_canonical_seed_payload(genome: Genome) -> dict[str, Any]:
    """Return a deterministic, existing-type structural seed for one genome.

    The module params mirror existing scalar genes so developmental expression
    reaches the structural boundary without changing the effective scalar values.
    """

    return {
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "hidden_size": int(getattr(genome, "hidden_size", 64)),
        "exploration_bias": float(getattr(genome, "exploration_bias", 0.0)),
        "aggression_bias": float(getattr(genome, "aggression_bias", 0.0)),
        "reproduction_threshold": float(getattr(genome, "reproduction_threshold", 30.0)),
        "mutation_rate": float(getattr(genome, "mutation_rate", 0.05)),
        "action_temperature": float(getattr(genome, "action_temperature", 1.0)),
        "use_gru": bool(getattr(genome, "use_gru", False)),
        "memory_dim": int(getattr(genome, "memory_dim", 0)),
        "drive_inertia": float(getattr(genome, "drive_inertia", 0.85)),
        "drive_noise": float(getattr(genome, "drive_noise", 0.02)),
        "drive_logit_gain": float(getattr(genome, "drive_logit_gain", 1.0)),
        "pred_lr": float(getattr(genome, "pred_lr", 0.08)),
        "modules": [
            {
                "module_id": "m000_policy_core",
                "module_type": "policy_core",
                "params": {
                    "hidden_size": int(getattr(genome, "hidden_size", 64)),
                    "use_gru": bool(getattr(genome, "use_gru", False)),
                    "memory_dim": int(getattr(genome, "memory_dim", 0)),
                },
            },
            {
                "module_id": "m010_behavior_bias",
                "module_type": "behavior_bias",
                "params": {
                    "exploration_bias": float(getattr(genome, "exploration_bias", 0.0)),
                    "aggression_bias": float(getattr(genome, "aggression_bias", 0.0)),
                    "action_temperature": float(getattr(genome, "action_temperature", 1.0)),
                },
            },
            {
                "module_id": "m030_identity_regulation",
                "module_type": "identity_regulation",
                "params": {
                    "drive_inertia": float(getattr(genome, "drive_inertia", 0.85)),
                    "drive_noise": float(getattr(genome, "drive_noise", 0.02)),
                    "drive_logit_gain": float(getattr(genome, "drive_logit_gain", 1.0)),
                    "pred_lr": float(getattr(genome, "pred_lr", 0.08)),
                },
            },
        ],
        "connections": [
            {
                "connection_id": "c000_behavior_to_policy",
                "src_module_id": "m010_behavior_bias",
                "dst_module_id": "m000_policy_core",
                "src_slot": "bias_out",
                "dst_slot": "bias_in",
                "value_type": "scalar",
                "channel": "logit_bias",
                "weight": 1.0,
            },
            {
                "connection_id": "c010_identity_to_policy",
                "src_module_id": "m030_identity_regulation",
                "dst_module_id": "m000_policy_core",
                "src_slot": "mod_out",
                "dst_slot": "mod_in",
                "value_type": "gate",
                "channel": "modulation",
                "weight": float(getattr(genome, "drive_inertia", 0.85)),
            },
        ],
        "developmental_program": getattr(genome, "developmental_program", None),
    }


def ablate_seed_payload(payload: dict[str, Any], ablation: str) -> dict[str, Any]:
    """Return a matched diagnostic payload control without changing mechanics."""

    out = dict(payload)
    out["modules"] = [dict(module) for module in list(payload.get("modules", []) or [])]
    out["connections"] = [dict(conn) for conn in list(payload.get("connections", []) or [])]
    if ablation == STRUCTURAL_PAYLOAD_SEED_ABLATION_NONE:
        return out
    if ablation == STRUCTURAL_PAYLOAD_SEED_ABLATION_ZERO_CONNECTION_GATES:
        for conn in out["connections"]:
            if conn.get("value_type") == "gate":
                conn["weight"] = 0.0
        return out
    if ablation == STRUCTURAL_PAYLOAD_SEED_ABLATION_REMOVE_CONNECTIONS:
        out["connections"] = []
        return out
    raise ValueError(f"unsupported structural_payload_seed_ablation={ablation!r}")


def apply_structural_payload_activation_to_founder(genome: Genome, sim_cfg: Any) -> Genome:
    validate_structural_payload_activation_config(sim_cfg)
    if not structural_payload_activation_enabled(sim_cfg):
        return genome
    ablation = str(getattr(sim_cfg, "structural_payload_seed_ablation", STRUCTURAL_PAYLOAD_SEED_ABLATION_NONE) or "")
    profile = str(getattr(sim_cfg, "structural_payload_seed_profile", STRUCTURAL_PAYLOAD_SEED_PROFILE_NONE) or "")
    if profile == STRUCTURAL_PAYLOAD_SEED_PROFILE_MINIMAL:
        if list(getattr(genome, "modules", []) or []) or list(getattr(genome, "connections", []) or []):
            return genome
        payload = minimal_canonical_seed_payload(genome)
        return Genome.from_dict(ablate_seed_payload(payload, ablation))
    raise ValueError(f"unsupported structural_payload_seed_profile={profile!r}")
