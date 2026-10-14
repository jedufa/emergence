from __future__ import annotations

"""Frozen run configuration and storage preflight for Scientific Chronicle V1."""

from dataclasses import asdict, dataclass
from typing import Any

from simulation.chronicle.schema import canonical_payload, stable_hash


PROFILES = ("off", "summary", "notable", "scientific_full", "debug_trace")
HEAVY_PROFILES = frozenset({"scientific_full", "debug_trace"})


@dataclass(frozen=True)
class ChronicleConfig:
    enabled: bool = False
    profile: str = "notable"
    terminal_enabled: bool = False
    episode_enabled: bool = True
    context_window_enabled: bool = True
    context_events_before: int = 8
    context_events_after: int = 4
    max_context_windows_per_generation: int = 16
    max_live_messages_per_generation: int = 8
    max_stored_events_per_generation: int = 32
    max_events_per_category_per_generation: int = 12
    subject_cooldown_generations: int = 1
    region_cooldown_generations: int = 1
    min_magnitude: float = 0.0
    minimum_persistence_generations: int = 1
    aggregation_window_generations: int = 1
    episode_gap_generations: int = 1
    dominance_threshold: float = 0.50
    rapid_lineage_share_delta: float = 0.15
    near_extinction_count: int = 1
    food_depleted_fraction_threshold: float = 0.25
    food_recovery_fraction_threshold: float = 0.10
    crowding_hotspot_fraction_threshold: float = 0.20
    waste_hotspot_threshold: float = 0.25
    bottleneck_survival_fraction: float = 0.50
    estimated_event_bytes: int = 2048
    allow_large_output: bool = False

    @classmethod
    def from_sim_cfg(cls, sim_cfg: Any) -> "ChronicleConfig":
        def read(name: str, default: Any) -> Any:
            return getattr(sim_cfg, name, default)

        profile = str(read("chronicle_profile", "notable") or "notable").strip()
        if profile not in PROFILES:
            raise ValueError(f"unknown chronicle_profile {profile!r}; expected one of {list(PROFILES)}")
        enabled = bool(read("chronicle_enable", False)) and profile != "off"
        config = cls(
            enabled=enabled,
            profile=profile,
            terminal_enabled=bool(read("chronicle_terminal_enable", False)),
            episode_enabled=bool(read("chronicle_episode_enable", True)),
            context_window_enabled=bool(read("chronicle_context_window_enable", True)),
            context_events_before=max(0, int(read("chronicle_context_events_before", 8) or 0)),
            context_events_after=max(0, int(read("chronicle_context_events_after", 4) or 0)),
            max_context_windows_per_generation=max(
                0, int(read("chronicle_max_context_windows_per_generation", 16) or 0)
            ),
            max_live_messages_per_generation=max(
                0, int(read("chronicle_max_live_messages_per_generation", 8) or 0)
            ),
            max_stored_events_per_generation=max(
                1, int(read("chronicle_max_stored_events_per_generation", 32) or 1)
            ),
            max_events_per_category_per_generation=max(
                1, int(read("chronicle_max_events_per_category_per_generation", 12) or 1)
            ),
            subject_cooldown_generations=max(
                0, int(read("chronicle_subject_cooldown_generations", 1) or 0)
            ),
            region_cooldown_generations=max(
                0, int(read("chronicle_region_cooldown_generations", 1) or 0)
            ),
            min_magnitude=max(0.0, float(read("chronicle_min_magnitude", 0.0) or 0.0)),
            minimum_persistence_generations=max(
                1, int(read("chronicle_minimum_persistence_generations", 1) or 1)
            ),
            aggregation_window_generations=max(
                0, int(read("chronicle_aggregation_window_generations", 1) or 0)
            ),
            episode_gap_generations=max(0, int(read("chronicle_episode_gap_generations", 1) or 0)),
            dominance_threshold=float(read("chronicle_dominance_threshold", 0.50) or 0.50),
            rapid_lineage_share_delta=float(read("chronicle_rapid_lineage_share_delta", 0.15) or 0.15),
            near_extinction_count=max(0, int(read("chronicle_near_extinction_count", 1) or 0)),
            food_depleted_fraction_threshold=float(
                read("chronicle_food_depleted_fraction_threshold", 0.25) or 0.25
            ),
            food_recovery_fraction_threshold=float(
                read("chronicle_food_recovery_fraction_threshold", 0.10) or 0.10
            ),
            crowding_hotspot_fraction_threshold=float(
                read("chronicle_crowding_hotspot_fraction_threshold", 0.20) or 0.20
            ),
            waste_hotspot_threshold=float(read("chronicle_waste_hotspot_threshold", 0.25) or 0.25),
            bottleneck_survival_fraction=float(
                read("chronicle_bottleneck_survival_fraction", 0.50) or 0.50
            ),
            estimated_event_bytes=max(256, int(read("chronicle_estimated_event_bytes", 2048) or 2048)),
            allow_large_output=bool(
                read("allow_large_output_profile", False) or read("chronicle_allow_large_output", False)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"unsupported Chronicle profile: {self.profile}")
        for name in (
            "dominance_threshold",
            "rapid_lineage_share_delta",
            "food_depleted_fraction_threshold",
            "food_recovery_fraction_threshold",
            "crowding_hotspot_fraction_threshold",
            "bottleneck_survival_fraction",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.enabled and self.profile in HEAVY_PROFILES and not self.allow_large_output:
            raise ValueError(
                f"Chronicle profile {self.profile!r} requires allow_large_output_profile=true"
            )

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))

    @property
    def config_hash(self) -> str:
        return "sha256:" + stable_hash("eiel-chronicle-config-v1", self.to_dict(), length=64)


def storage_preflight(
    config: ChronicleConfig,
    *,
    generations: int,
    planned_cells: int = 1,
) -> dict[str, Any]:
    generation_count = max(0, int(generations))
    cells = max(1, int(planned_cells))
    events_per_generation = 0 if not config.enabled else int(config.max_stored_events_per_generation)
    estimated_events = generation_count * cells * events_per_generation
    estimated_bytes = estimated_events * int(config.estimated_event_bytes)
    risk = "off"
    acknowledgements: list[str] = []
    if config.enabled:
        risk = "compact" if config.profile in {"summary", "notable"} else "large"
    if config.profile in HEAVY_PROFILES:
        acknowledgements.append("allow_large_output_profile=true")
    return {
        "schema_version": "eiel_chronicle_storage_preflight_v1",
        "enabled": bool(config.enabled),
        "profile": config.profile,
        "generations": generation_count,
        "planned_cells": cells,
        "maximum_events_per_generation": events_per_generation,
        "estimated_events": estimated_events,
        "estimated_bytes": estimated_bytes,
        "estimated_mebibytes": round(float(estimated_bytes) / float(1024 * 1024), 6),
        "risk_classification": risk,
        "required_acknowledgements": acknowledgements,
        "accepted": not acknowledgements or bool(config.allow_large_output),
        "config_hash": config.config_hash,
    }
