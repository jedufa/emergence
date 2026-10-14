from __future__ import annotations

"""
env/gridworld_pkg/state.py

Gridworld environment state.

This module maintains:
- The static terrain grid (food/empty)
- A dense occupancy grid mapping (y,x) -> resident world slot handle
- Capacity-bounded slot rows for per-agent position/alive state
- Fixed-size low-ID compatibility shadows for legacy direct readers
- reset(), in_bounds(), etc.

Last modified: 2026-02-24 03:45:00 CET (Europe/Vienna)
"""

from typing import Optional, Tuple
import logging
import os
import numpy as np

from env.env_config import EnvConfig
from env.gridworld_pkg.constants import EMPTY



logger = logging.getLogger(__name__)
_ONCE_DEBUG_NOTES: set[str] = set()


def _log_debug_once(*, key: str, msg: str) -> None:
    """Log a debug message at most once per process for a given key.

    This exists to prevent console spam in shadow-clone runs (EPI) where
    temporary SimulationRunner instances may initialize env state repeatedly.

    Safety rule: logging must never be able to crash or perturb the sim.
    """
    try:
        if key in _ONCE_DEBUG_NOTES:
            return
        _ONCE_DEBUG_NOTES.add(key)
        logger.debug(msg)
    except Exception:
        return


BIOME_SAFE = np.int8(0)
BIOME_WILD = np.int8(1)


class StateMixin:
    """State container mixed into the environment class."""

    def __init__(self, cfg: EnvConfig, seed: int = 0, run_dir: str | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.base_seed = int(seed)
        # Phase 1: prefer RNG injection from RNGManager named streams.
        # Phase 1 determinism rule: no RNG islands outside RNGManager.
        # In debug mode, require explicit rng injection.
        if rng is None:
            raise RuntimeError("StateMixin requires rng injection from RNGManager (no RNG islands allowed)")
        self.rng = rng

        # -----------------------------
        # Tier 4: Juvenile spatial vulnerability (config clamp + derived flags)
        # -----------------------------
        # Tier 4 is strictly age-based and must remain orthogonal to Tier 3 CC/spawn.
        # We clamp here once at env init to keep all callers deterministic and to avoid
        # accidental extinction from extreme settings.
        self.tier4_enabled = bool(getattr(cfg, "tier4_juvenile_vulnerability_enable", False))
        # Pull raw values (defaults preserve behavior)
        self.tier4_juvenile_steps = int(getattr(cfg, "tier4_juvenile_steps", 0) or 0)
        self.tier4_juvenile_grace_steps = int(getattr(cfg, "tier4_juvenile_grace_steps", 0) or 0)
        self.tier4_juvenile_steps = max(0, self.tier4_juvenile_steps)
        self.tier4_juvenile_grace_steps = max(0, self.tier4_juvenile_grace_steps)

        # Clamp multipliers to explicit bounds
        def _clamp(v: float, lo: float, hi: float) -> float:
            try:
                fv = float(v)
            except Exception:
                fv = lo
            if fv < lo:
                return lo
            if fv > hi:
                return hi
            return fv

        raw_h = getattr(cfg, "tier4_juvenile_hazard_multiplier", 1.0)
        raw_m = getattr(cfg, "tier4_juvenile_movement_penalty", 1.0)
        raw_y = getattr(cfg, "tier4_juvenile_harvest_multiplier", 1.0)

        h = _clamp(raw_h, 1.0, 10.0)
        m = _clamp(raw_m, 1.0, 10.0)
        y = _clamp(raw_y, 0.0, 1.0)

        # Store clamped values on cfg (so runner and logs see the true values)
        try:
            setattr(cfg, "tier4_juvenile_hazard_multiplier", h)
            setattr(cfg, "tier4_juvenile_movement_penalty", m)
            setattr(cfg, "tier4_juvenile_harvest_multiplier", y)
            setattr(cfg, "tier4_juvenile_steps", self.tier4_juvenile_steps)
            setattr(cfg, "tier4_juvenile_grace_steps", self.tier4_juvenile_grace_steps)
        except Exception:
            pass

        self.tier4_juvenile_hazard_multiplier = float(h)
        self.tier4_juvenile_movement_penalty = float(m)
        self.tier4_juvenile_harvest_multiplier = float(y)

        # Best-effort warning when clamping occurs (avoid crashing, but leave breadcrumbs)
        if self.tier4_enabled:
            if float(raw_h) != float(h) or float(raw_m) != float(m) or float(raw_y) != float(y):
                print(
                    "[tier4] clamp: hazard_mult=%s->%s move_penalty=%s->%s harvest_mult=%s->%s"
                    % (raw_h, h, raw_m, m, raw_y, y)
                )


        # Runtime multipliers
        self.food_spawn_mult = 1.0
        self.action_cost_mult = 1.0
        self.rest_cost_mult = 1.0

        h, w = cfg.height, cfg.width

        # Fixed resident capacity. Biological IDs are monotonic, but world state is
        # keyed by reusable population slots and therefore never grows with ID history.
        pop = int(getattr(cfg, "population_size", 2048))
        cap_mult = int(getattr(cfg, "max_agent_capacity_mult", 4))
        cap_mult = max(1, min(64, cap_mult))
        max_agents = int(pop * cap_mult)

        # World grids. agent_grid stores resident world slots, while public
        # reporting resolves biological IDs through slot_to_agent_id.
        self.grid = np.zeros((h, w), dtype=np.int8)
        self.agent_grid = np.full((h, w), -1, dtype=np.int32)

        # ---------
        # Capacity-bounded authoritative world membership and state.
        # ---------
        self.slot_to_agent_id = np.full((max_agents,), -1, dtype=np.int64)
        self.slot_tokens = np.zeros((max_agents,), dtype=np.int64)
        self._agent_id_to_world_slot: dict[int, tuple[int, int]] = {}
        self._world_slot_capacity_configured = False
        self._world_authority_migration_required = False
        self.agent_pos_y_by_slot = np.full((max_agents,), -1, dtype=np.int32)
        self.agent_pos_x_by_slot = np.full((max_agents,), -1, dtype=np.int32)
        self.agent_world_alive_by_slot = np.zeros((max_agents,), dtype=np.bool_)

        # Fixed-size low-ID compatibility shadow. High biological IDs never index
        # or resize these arrays.
        self.agent_pos_y = np.full((max_agents,), -1, dtype=np.int32)
        self.agent_pos_x = np.full((max_agents,), -1, dtype=np.int32)
        self.agent_alive = np.zeros((max_agents,), dtype=np.bool_)

        # ---------
        # Biomes
        # ---------
        self.biome_grid = np.full((h, w), BIOME_SAFE, dtype=np.int8)
        self._init_biomes()

        # ---------
        # Patch World / Jungle V0 Step 1-9
        # ---------
        # Feature-gated substrate only; disabled mode keeps legacy behavior.
        self._init_patch_world()

        # ---------
        # Tier 1: Spatial ecology (static heterogeneity)
        # ---------
        # These maps are world-level only and must be reproducible.
        # They are generated once here at env init time.
        self._init_spatial_ecology(run_dir=run_dir)
        self._init_spatial_droughts(run_dir=run_dir, base_seed=seed)

        # === Tier 3 change: region partition init ===
        self._init_tier3_regions(run_dir=run_dir)
        # === Tier 3 change: end region partition init ===

        self._views_dirty = True
        self._occ_p = None
        self._food_p = None

        r = int(cfg.view_radius)
        offs = np.arange(-r, r + 1, dtype=np.int32)
        self._dy = offs.reshape(1, -1, 1)
        self._dx = offs.reshape(1, 1, -1)

    # -----------------------
    # Patch World schema shell
    # -----------------------
    def _init_patch_world(self) -> None:
        """Initialize the Patch World feature gate, schema version, and arrays.

        Validation and array allocation are active only when the feature is
        enabled so old/default configs remain behaviorally unchanged.
        """
        from env.gridworld_pkg.jungle import validate_jungle_config

        validate_jungle_config(self.cfg)
        enable = bool(getattr(self.cfg, "patch_world_enable", False))
        if bool(getattr(self.cfg, "prediction_demand_enable", False)) and not enable:
            raise ValueError("prediction_demand_enable requires patch_world_enable")
        version = int(getattr(self.cfg, "patch_world_schema_version", 1) or 1)

        self.patch_world_enable = bool(enable)
        if enable:
            from env.gridworld_pkg.patch_world import (
                initialize_patch_world_arrays,
                initialize_patch_world_runtime_fields,
                validate_patch_world_schema_version,
            )

            version = validate_patch_world_schema_version(version)
            arrays = initialize_patch_world_arrays(self.cfg, self.grid)
            for name, arr in arrays.items():
                setattr(self, name, arr)
            initialize_patch_world_runtime_fields(self)
            self.patch_food_regrowth_total_last = 0.0
            self.patch_food_harvest_total_last = 0.0
            self.patch_movement_energy_extra_last = 0.0
            self.patch_movement_energy_event_count_last = 0
            self.patch_recovery_positive_total_last = 0.0
            self.patch_recovery_negative_total_last = 0.0
            self.patch_recovery_decay_total_last = 0.0
        self.patch_world_schema_version = int(version)

    def _reset_patch_world_arrays(self) -> None:
        """Reset passive Patch World arrays from current grid state."""
        if not bool(getattr(self, "patch_world_enable", False)):
            return
        from env.gridworld_pkg.patch_world import (
            initialize_patch_world_arrays,
            initialize_patch_world_runtime_fields,
        )

        arrays = initialize_patch_world_arrays(self.cfg, self.grid)
        for name, arr in arrays.items():
            setattr(self, name, arr)
        initialize_patch_world_runtime_fields(self)
        self.patch_food_regrowth_total_last = 0.0
        self.patch_food_harvest_total_last = 0.0
        self.patch_movement_energy_extra_last = 0.0
        self.patch_movement_energy_event_count_last = 0
        self.patch_recovery_positive_total_last = 0.0
        self.patch_recovery_negative_total_last = 0.0
        self.patch_recovery_decay_total_last = 0.0

    def patch_world_step(self, abs_step: int) -> None:
        """Run the Patch World per-tick update hook.

        Step 5 decays/emits activity traces before Step 8 optionally computes a
        bounded regrowth multiplier for Step 3 food amount regrowth and binary
        compatibility projection.
        """
        if not bool(getattr(self, "patch_world_enable", False)):
            return
        from env.gridworld_pkg.patch_world import (
            apply_delayed_crowding_waste_step,
            apply_delayed_resource_depletion_step,
            apply_mem_max_delayed_choice_step,
            apply_prediction_demand_step,
            apply_wdl_delayed_hazard_step,
            apply_patch_world_trace_step,
            apply_patch_world_food_step,
            validate_patch_world_schema_version,
        )

        validate_patch_world_schema_version(int(getattr(self, "patch_world_schema_version", 1) or 1))
        apply_wdl_delayed_hazard_step(self, int(abs_step))
        apply_prediction_demand_step(self, int(abs_step))
        apply_delayed_resource_depletion_step(self, int(abs_step))
        apply_delayed_crowding_waste_step(self, int(abs_step))
        apply_mem_max_delayed_choice_step(self, int(abs_step))
        apply_patch_world_trace_step(self, int(abs_step))
        apply_patch_world_food_step(self, int(abs_step))

    # -----------------------
    # Tier 1: Spatial ecology
    # -----------------------
    def _init_spatial_ecology(self, run_dir: str | None) -> None:
        """Initialize Tier 1 spatial ecology maps and logging.

        This is strictly a run-start operation (no mid-run adaptation).
        """
        from env.gridworld_pkg.spatial_ecology import (
            generate_map,
            log_spatial_ecology_artifacts,
            run_environment_only_validation,
        )

        enable = bool(getattr(self.cfg, "spatial_ecology_enable", False))
        self.spatial_ecology_enable = bool(enable)

        h, w = int(self.cfg.height), int(self.cfg.width)

        # Defaults: implicit uniform maps (preserve behavior)
        self.resource_regrowth_map = np.ones((h, w), dtype=np.float32)
        self.resource_spawn_density_map = np.ones((h, w), dtype=np.float32)
        self.hazard_rate_map = np.ones((h, w), dtype=np.float32)

        self._spatial_ecology_resolved_specs = {}
        self._spatial_ecology_region_id_maps = {}

        if not enable:
            # Console note for clarity (do not spam per tick)
            _log_debug_once(
                key="spatial_ecology_disabled",
                msg="[spatial-ecology] disabled: using implicit uniform multipliers",
            )
            return

        spatial_seed = int(getattr(self.cfg, "spatial_ecology_seed", 0) or 0)

        # Generate maps
        reg_map, reg_rid, reg_spec = generate_map(
            getattr(self.cfg, "resource_regrowth_map_spec"),
            h,
            w,
            spatial_ecology_seed=spatial_seed,
            rng=self.rng,
            allow_region_id_map=bool(getattr(self.cfg, "spatial_ecology_log_region_id_map", False)),
        )
        spn_map, spn_rid, spn_spec = generate_map(
            getattr(self.cfg, "resource_spawn_density_map_spec"),
            h,
            w,
            spatial_ecology_seed=spatial_seed,
            rng=self.rng,
            allow_region_id_map=bool(getattr(self.cfg, "spatial_ecology_log_region_id_map", False)),
        )
        haz_map, haz_rid, haz_spec = generate_map(
            getattr(self.cfg, "hazard_rate_map_spec"),
            h,
            w,
            spatial_ecology_seed=spatial_seed,
            rng=self.rng,
            allow_region_id_map=bool(getattr(self.cfg, "spatial_ecology_log_region_id_map", False)),
        )

        self.resource_regrowth_map = reg_map
        self.resource_spawn_density_map = spn_map
        self.hazard_rate_map = haz_map

        self._spatial_ecology_resolved_specs = {
            "resource_regrowth_map": reg_spec,
            "resource_spawn_density_map": spn_spec,
            "hazard_rate_map": haz_spec,
            "spatial_ecology_seed": spatial_seed,
            "spatial_ecology_enable": bool(getattr(self.cfg, "spatial_ecology_enable", False)),
        }

        if reg_rid is not None:
            self._spatial_ecology_region_id_maps["resource_regrowth_map"] = reg_rid
        if spn_rid is not None:
            self._spatial_ecology_region_id_maps["resource_spawn_density_map"] = spn_rid
        if haz_rid is not None:
            self._spatial_ecology_region_id_maps["hazard_rate_map"] = haz_rid

        # Basic sanity asserts
        assert self.resource_regrowth_map.shape == (h, w)
        assert self.resource_spawn_density_map.shape == (h, w)
        assert self.hazard_rate_map.shape == (h, w)

        # Log artifacts if run_dir is available
        if run_dir:
            formats = list(getattr(self.cfg, "spatial_ecology_log_formats", ["npy", "png"]) or ["npy", "png"])
            # Log effective hazard probability field (after multiplying + clamping).
            base_haz_prob = float(getattr(self.cfg, "biome_wild_hazard_prob", 0.0))
            eff_haz_prob_map = np.clip(base_haz_prob * self.hazard_rate_map, 0.0, 1.0).astype(np.float32, copy=False)
            log_spatial_ecology_artifacts(
                run_dir,
                maps={
                    "resource_regrowth_map": self.resource_regrowth_map,
                    "resource_spawn_density_map": self.resource_spawn_density_map,
                    "hazard_rate_map": self.hazard_rate_map,
                    "hazard_prob_effective": eff_haz_prob_map,
                },
                resolved_specs=self._spatial_ecology_resolved_specs,
                region_id_maps=self._spatial_ecology_region_id_maps or None,
                formats=formats,
                write_stats=bool(getattr(self.cfg, "spatial_ecology_log_stats", True)),
            )

            # Deterministic environment-only validation
            # Spawn weights use density map (distribution) and regrowth map as a global rate proxy in this codebase.
            out_dir = os.path.join(run_dir, "maps")
            base_spawn = float(getattr(self.cfg, "food_spawn_per_step", 0.0))
            base_haz = float(getattr(self.cfg, "biome_wild_hazard_prob", 0.0))
            # For Tier 1 validation, we treat the regrowth map as a global multiplier on spawn intensity.
            reg_mean = float(np.mean(self.resource_regrowth_map))
            eff_spawn = max(0.0, base_spawn * reg_mean)
            # Spawn weights for sampling locations
            spawn_w = np.maximum(self.resource_spawn_density_map, 0.0)
            run_environment_only_validation(
                out_dir=out_dir,
                spawn_weights=spawn_w,
                hazard_map=self.hazard_rate_map,
                steps=500,
                seed=int(spatial_seed) ^ 0xA5A5A5A5,
                base_spawn_per_step=eff_spawn,
                base_hazard_prob=base_haz,
                rng=self.rng,
            )

    def get_regrowth_multiplier(self, x: int, y: int) -> float:
        """Return the effective regrowth multiplier at (x,y)."""
        return float(self.resource_regrowth_map[int(y), int(x)])

    def get_spawn_multiplier(self, x: int, y: int) -> float:
        """Return the effective spawn multiplier at (x,y).

        Tier 1 semantics: multiplier on spawn *distribution weights*.
        """
        return float(self.resource_spawn_density_map[int(y), int(x)])

    def get_hazard_multiplier(self, x: int, y: int) -> float:
        """Return the effective hazard multiplier at (x,y)."""
        return float(self.hazard_rate_map[int(y), int(x)])

    # -----------------------
    # Capacity management
    # -----------------------
    def _ensure_agent_capacity(self, agent_id: int) -> None:
        """Validate a biological ID without growing compatibility state."""
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id > np.iinfo(np.int32).max:
            raise ValueError("agent_id must be in signed int32 range")

    def _configure_world_slot_capacity(self, capacity: int) -> None:
        """Bind slot-keyed world capacity once to the population SlotPool."""
        capacity = int(capacity)
        if capacity <= 0 or capacity > np.iinfo(np.int32).max:
            raise ValueError(f"invalid world slot capacity: capacity={capacity}")
        current = int(self.slot_to_agent_id.size)
        if bool(self._world_slot_capacity_configured):
            if capacity != current:
                raise RuntimeError(
                    "world slot capacity is already configured: "
                    f"current={current};requested={capacity}"
                )
            return

        canonical = (
            not self._agent_id_to_world_slot
            and not bool(np.any(self.agent_grid != -1))
            and bool(np.all(self.slot_to_agent_id == -1))
            and bool(np.all(self.slot_tokens == 0))
            and bool(np.all(self.agent_pos_y_by_slot == -1))
            and bool(np.all(self.agent_pos_x_by_slot == -1))
            and not bool(np.any(self.agent_world_alive_by_slot))
            and not bool(np.any(self.agent_alive))
            and bool(np.all(self.agent_pos_y == -1))
            and bool(np.all(self.agent_pos_x == -1))
        )
        if not canonical:
            raise RuntimeError(
                "world slot capacity can only be configured on pristine world state"
            )

        if capacity != current:
            self.slot_to_agent_id = np.full((capacity,), -1, dtype=np.int64)
            self.slot_tokens = np.zeros((capacity,), dtype=np.int64)
            self.agent_pos_y_by_slot = np.full((capacity,), -1, dtype=np.int32)
            self.agent_pos_x_by_slot = np.full((capacity,), -1, dtype=np.int32)
            self.agent_world_alive_by_slot = np.zeros((capacity,), dtype=np.bool_)
            if bool(
                getattr(
                    self.cfg,
                    "respec0b_resource_specialization_enable",
                    False,
                )
            ):
                self.respec0b_resource_active_y = np.full(
                    (capacity,), -1, dtype=np.int32
                )
                self.respec0b_resource_active_x = np.full(
                    (capacity,), -1, dtype=np.int32
                )
                self.respec0b_resource_active_class = np.zeros(
                    (capacity,), dtype=np.int32
                )
                self.respec0b_resource_handling_progress = np.zeros(
                    (capacity,), dtype=np.float32
                )
        self._world_slot_capacity_configured = True
        self._world_authority_migration_required = False

    def _validate_world_slot_row(self, slot: int) -> tuple[int, int]:
        slot = int(slot)
        if slot < 0 or slot >= int(self.slot_to_agent_id.size):
            raise ValueError(f"world slot out of range: slot={slot}")
        agent_id = int(self.slot_to_agent_id[slot])
        token = int(self.slot_tokens[slot])
        if agent_id == -1:
            return agent_id, token
        if self._agent_id_to_world_slot.get(agent_id) != (slot, token):
            raise RuntimeError(
                f"world reverse resolver mismatch: agent_id={agent_id};"
                f"slot={slot};token={token}"
            )
        return agent_id, token

    def _world_slot_snapshot(self, slot: int) -> tuple[object, ...]:
        agent_id, token = self._validate_world_slot_row(slot)
        slot = int(slot)
        snapshot: tuple[object, ...] = (
            agent_id,
            token,
            int(self.agent_pos_y_by_slot[slot]),
            int(self.agent_pos_x_by_slot[slot]),
            bool(self.agent_world_alive_by_slot[slot]),
        )
        if hasattr(self, "respec0b_resource_handling_progress"):
            snapshot += (
                int(self.respec0b_resource_active_y[slot]),
                int(self.respec0b_resource_active_x[slot]),
                int(self.respec0b_resource_active_class[slot]),
                float(self.respec0b_resource_handling_progress[slot]),
            )
        return snapshot

    def _restore_world_slot_snapshot(
        self,
        slot: int,
        snapshot: tuple[object, ...],
    ) -> None:
        slot = int(slot)
        agent_id, token, y, x, alive = snapshot[:5]
        existing = self._agent_id_to_world_slot.get(int(agent_id))
        current_id, _ = self._validate_world_slot_row(slot)
        if (
            int(agent_id) != -1
            and int(agent_id) != current_id
            and existing is not None
        ):
            raise RuntimeError(
                f"cannot restore duplicate world membership: agent_id={int(agent_id)};"
                f"existing={existing}"
            )
        if current_id != -1:
            del self._agent_id_to_world_slot[current_id]
        self.slot_to_agent_id[slot] = int(agent_id)
        self.slot_tokens[slot] = int(token)
        self.agent_pos_y_by_slot[slot] = int(y)
        self.agent_pos_x_by_slot[slot] = int(x)
        self.agent_world_alive_by_slot[slot] = bool(alive)
        if int(agent_id) != -1:
            self._agent_id_to_world_slot[int(agent_id)] = (slot, int(token))
        if len(snapshot) == 9:
            (
                self.respec0b_resource_active_y[slot],
                self.respec0b_resource_active_x[slot],
                self.respec0b_resource_active_class[slot],
                self.respec0b_resource_handling_progress[slot],
            ) = snapshot[5:]

    def _bind_world_slot(self, agent_id: int, slot: int, slot_token: int) -> None:
        """Reserve one capacity-bounded world row for a biological identity."""
        self._ensure_agent_capacity(agent_id)
        agent_id = int(agent_id)
        slot = int(slot)
        slot_token = int(slot_token)
        if slot < 0 or slot >= int(self.slot_to_agent_id.size):
            raise ValueError(f"world slot out of range: slot={slot}")
        if slot_token <= 0:
            raise ValueError("slot_token must be positive")

        resident_id, resident_token = self._validate_world_slot_row(slot)
        if resident_id == agent_id and resident_token == slot_token:
            return
        if resident_id != -1:
            raise RuntimeError(
                f"world slot already occupied: slot={slot};agent_id={resident_id}"
            )
        if slot_token <= resident_token:
            raise RuntimeError(
                f"stale world slot token: slot={slot};token={slot_token};"
                f"last_token={resident_token}"
            )
        if agent_id in self._agent_id_to_world_slot:
            raise RuntimeError(f"agent_id already has world membership: agent_id={agent_id}")

        self.slot_to_agent_id[slot] = agent_id
        self.slot_tokens[slot] = slot_token
        self._agent_id_to_world_slot[agent_id] = (slot, slot_token)
        self.agent_pos_y_by_slot[slot] = -1
        self.agent_pos_x_by_slot[slot] = -1
        self.agent_world_alive_by_slot[slot] = False
        if hasattr(self, "respec0b_resource_handling_progress"):
            self.respec0b_resource_active_y[slot] = -1
            self.respec0b_resource_active_x[slot] = -1
            self.respec0b_resource_active_class[slot] = 0
            self.respec0b_resource_handling_progress[slot] = np.float32(0.0)

    def _resolve_world_slot(
        self,
        agent_id: int,
        *,
        slot: int | None = None,
        slot_token: int | None = None,
        require_alive: bool = False,
    ) -> tuple[int, int]:
        """Resolve and verify ID -> (slot, token) using only O(capacity) state."""
        self._ensure_agent_capacity(agent_id)
        agent_id = int(agent_id)
        if bool(self._world_authority_migration_required):
            raise RuntimeError(
                "world_state_v2_to_v3 migration is required before world resolution"
            )
        resolved = self._agent_id_to_world_slot.get(agent_id)
        if resolved is None:
            raise RuntimeError(
                f"world membership resolution failed: agent_id={agent_id};matches=0"
            )
        resolved_slot, resolved_token = resolved
        row_agent_id, row_token = self._validate_world_slot_row(resolved_slot)
        if row_agent_id != agent_id or row_token != resolved_token:
            raise RuntimeError(
                f"world membership table mismatch: agent_id={agent_id};"
                f"slot={resolved_slot};token={resolved_token}"
            )
        if slot is not None and int(slot) != resolved_slot:
            raise RuntimeError(
                f"world slot mismatch: agent_id={agent_id};"
                f"expected={int(slot)};actual={resolved_slot}"
            )
        if slot_token is not None and int(slot_token) != resolved_token:
            raise RuntimeError(
                f"world slot token mismatch: agent_id={agent_id};"
                f"expected={int(slot_token)};actual={resolved_token}"
            )
        alive = bool(self.agent_world_alive_by_slot[resolved_slot])
        if require_alive and not alive:
            raise RuntimeError(f"world member is not alive: agent_id={agent_id}")

        if agent_id < int(self.agent_alive.size):
            shadow = (
                bool(self.agent_alive[agent_id]),
                int(self.agent_pos_y[agent_id]),
                int(self.agent_pos_x[agent_id]),
            )
            authoritative = (
                alive,
                int(self.agent_pos_y_by_slot[resolved_slot]),
                int(self.agent_pos_x_by_slot[resolved_slot]),
            )
            if shadow != authoritative:
                raise RuntimeError(
                    f"legacy world-state shadow mismatch: agent_id={agent_id};"
                    f"shadow={shadow};authoritative={authoritative}"
                )
        return resolved_slot, resolved_token


    # -----------------------
    # Tier 2: Spatial droughts
    # -----------------------
    def _init_spatial_droughts(self, run_dir: str | None, base_seed: int) -> None:
        """Initialize Tier 2 spatial droughts controller (time-varying regional stress).

        Rules:
        - Defaults preserve behavior: if disabled, all multipliers are 1.0 and no logging occurs.
        - Depends only on time, space, and seed. No agent inspection.

        Args:
            run_dir: Run directory for logging artifacts (may be None).
            base_seed: Run seed used for stochastic scheduling.
        """
        from env.gridworld_pkg.spatial_droughts import SpatialDroughtSystem

        enable = bool(getattr(self.cfg, "spatial_droughts_enable", False))
        self.spatial_droughts_enable = bool(enable)

        h, w = int(self.cfg.height), int(self.cfg.width)

        # Defaults: identity multipliers
        self.drought_regrowth_mult_map = np.ones((h, w), dtype=np.float32)
        self.drought_spawn_mult_map = np.ones((h, w), dtype=np.float32)

        self._spatial_droughts = None
        if not enable:
            # Note: this path may be hit frequently in shadow-clone runners; keep silent by default.
            _log_debug_once(
                key="spatial_droughts_disabled",
                msg="[spatial-droughts] disabled: using implicit uniform multipliers",
            )
            return

        spec = getattr(self.cfg, "spatial_droughts_spec", None) or {}

        self._spatial_droughts = SpatialDroughtSystem(
            width=w,
            height=h,
            enabled=True,
            spec=spec,
            base_seed=int(base_seed),
            run_dir=run_dir,
            periodic_stats_every=int(getattr(self.cfg, "spatial_droughts_log_periodic_every", 100) or 100),
            timeseries_every=int(getattr(self.cfg, "spatial_droughts_log_timeseries_every", 1) or 1),
        )

    # === Tier 3 change: region partition init ===
    def _init_tier3_regions(self, run_dir: str | None = None) -> None:
        """Initialize Tier 3 primary region partition.

        Tier 3 regions are a PRIMARY partition:
        - Every cell has exactly one region_id in [0..R-1]
        - Tier 2 droughts remain overlays (masks) and may span multiple regions

        Supported v1 modes:
        - mode="grid": deterministic rectangular tiling (region_rows × region_cols)
        - mode="file": load explicit region_id_grid from .npy

        Logs (when run_dir is provided and Tier 3 is enabled):
        - maps/tier3_region_id.npy
        - maps/tier3_regions_stats.json
        """
        cfg = self.cfg
        enabled = bool(getattr(cfg, "migration_pressure_enable", False))
        # NOTE: We want Tier 3 region artifacts available for A/B tests where
        # the baseline has Tier 3 pressures disabled but still uses the same
        # region partition as the variant.
        # Therefore, we initialize/log regions whenever a regions_spec is provided,
        # even if migration pressure itself is disabled.
        h, w = int(cfg.height), int(cfg.width)

        spec_raw = getattr(cfg, "tier3_regions_spec", None)
        spec = spec_raw or {}
        mode = str(spec.get("mode", "grid")).strip().lower()

        region_id_grid: np.ndarray

        if mode == "file":
            path = (spec.get("file", {}) or {}).get("region_grid_path", None)
            if path is None:
                raise ValueError("Tier 3 regions_spec mode='file' requires file.region_grid_path")
            region_id_grid = np.load(str(path))
        else:
            # Default: deterministic grid tiling
            grid = spec.get("grid", {}) or {}
            rr = int(grid.get("region_rows", 1))
            rc = int(grid.get("region_cols", 1))
            rr = max(1, rr)
            rc = max(1, rc)

            region_id_grid = np.empty((h, w), dtype=np.int32)
            # Compute row/col boundaries (last absorbs remainder)
            y_edges = [int(round(i * h / rr)) for i in range(rr + 1)]
            x_edges = [int(round(i * w / rc)) for i in range(rc + 1)]
            for ry in range(rr):
                y0, y1 = y_edges[ry], y_edges[ry + 1]
                for rx in range(rc):
                    x0, x1 = x_edges[rx], x_edges[rx + 1]
                    rid = int(ry * rc + rx)
                    region_id_grid[y0:y1, x0:x1] = rid

        # Validate partition
        if region_id_grid.shape != (h, w):
            raise ValueError(f"Tier 3 region_id_grid has shape {region_id_grid.shape}, expected {(h, w)}")
        if not np.issubdtype(region_id_grid.dtype, np.integer):
            # Fail fast: force explicit integer ids
            raise ValueError(f"Tier 3 region_id_grid dtype must be integer, got {region_id_grid.dtype}")
        region_id_grid = region_id_grid.astype(np.int32, copy=False)

        r_min = int(region_id_grid.min(initial=0))
        r_max = int(region_id_grid.max(initial=0))
        if r_min < 0:
            raise ValueError(f"Tier 3 region_id_grid contains negative ids (min={r_min})")
        region_count = int(r_max + 1)
        if region_count <= 0:
            region_count = 1

        # Enforce that ids are in [0..R-1] by construction (file mode can be sparse)
        # Sparse ids are allowed but we normalize by compressing to contiguous ids.
        # This keeps downstream arrays small and predictable.
        uniq = np.unique(region_id_grid)
        if uniq.size != region_count or uniq[0] != 0 or uniq[-1] != (region_count - 1):
            # Compress ids
            mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
            flat = region_id_grid.ravel()
            for i in range(flat.size):
                flat[i] = mapping[int(flat[i])]
            region_id_grid = flat.reshape((h, w)).astype(np.int32, copy=False)
            region_count = int(np.max(region_id_grid) + 1)

        # Histogram for sanity
        cells_per_region = np.bincount(region_id_grid.ravel(), minlength=region_count).astype(int)

        self.tier3_region_id_grid = region_id_grid
        self.tier3_region_count = int(region_count)

        # Log when (a) Tier 3 is enabled OR (b) regions_spec was explicitly provided.
        # This keeps defaults no-op (no spec, disabled => no extra artifacts), while
        # supporting harness baseline/variant comparisons that expect region maps.
        should_log = bool(run_dir) and (enabled or (spec_raw is not None))
        if should_log:
            maps_dir = os.path.join(str(run_dir), "maps")
            os.makedirs(maps_dir, exist_ok=True)
            np.save(os.path.join(maps_dir, "tier3_region_id.npy"), region_id_grid)
            stats = {
                "tier": 3,
                "mode": mode,
                "height": h,
                "width": w,
                "region_count": int(region_count),
                "cells_per_region": cells_per_region.tolist(),
                "cells_min": int(cells_per_region.min(initial=0)),
                "cells_max": int(cells_per_region.max(initial=0)),
                "spec": spec,
            }
            import json as _json
            with open(os.path.join(maps_dir, "tier3_regions_stats.json"), "w", encoding="utf-8") as f:
                _json.dump(stats, f, indent=2, sort_keys=True)
    # === Tier 3 change: end region partition init ===

    def spatial_droughts_step(self, abs_step: int) -> None:
        """Update drought multipliers for the given absolute step.

        This should be called exactly once per environment tick by the runner.
        """
        if not bool(getattr(self, "spatial_droughts_enable", False)):
            return
        sys = getattr(self, "_spatial_droughts", None)
        if sys is None:
            return
        reg_map, spawn_map = sys.step(int(abs_step))
        # Store references (arrays owned by the drought system)
        self.drought_regrowth_mult_map = reg_map
        self.drought_spawn_mult_map = spawn_map

    def spatial_droughts_close(self) -> None:
        """Close drought log writers (best-effort)."""
        sys = getattr(self, "_spatial_droughts", None)
        if sys is None:
            return
        try:
            sys.close()
        except Exception:
            pass

    # -----------------------
    # Biomes
    # -----------------------
    def _init_biomes(self):
        if not bool(getattr(self.cfg, "enable_biomes", False)):
            self.biome_grid.fill(BIOME_SAFE)
            return

        safe_frac = float(getattr(self.cfg, "biome_safe_frac", 0.55))
        safe_frac = float(np.clip(safe_frac, 0.0, 1.0))

        noise = self.rng.random(self.biome_grid.shape, dtype=np.float32)
        mask_safe = (noise < safe_frac)

        passes = int(getattr(self.cfg, "biome_smooth_passes", 4))
        for _ in range(max(0, min(20, passes))):
            s = mask_safe.astype(np.int16)
            neigh = (
                s
                + np.roll(s, 1, 0) + np.roll(s, -1, 0)
                + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            )
            mask_safe = (neigh >= 3)

        self.biome_grid[:] = np.where(mask_safe, BIOME_SAFE, BIOME_WILD).astype(np.int8, copy=False)

    # -----------------------
    # Core API
    # -----------------------
    def reset(self):
        """Reset world grids and clear all agent state."""
        self.grid.fill(EMPTY)
        self.agent_grid.fill(-1)
        self.agent_alive[:] = False
        self.agent_pos_y[:] = -1
        self.agent_pos_x[:] = -1
        self.slot_to_agent_id[:] = -1
        self.slot_tokens[:] = 0
        self._agent_id_to_world_slot.clear()
        self._world_authority_migration_required = False
        self.agent_pos_y_by_slot[:] = -1
        self.agent_pos_x_by_slot[:] = -1
        self.agent_world_alive_by_slot[:] = False
        self._reset_patch_world_arrays()
        self._views_dirty = True
        self._occ_p = None
        self._food_p = None

    def in_bounds(self, y: int, x: int) -> bool:
        """Return True if (y,x) is inside the grid."""
        return 0 <= y < self.cfg.height and 0 <= x < self.cfg.width

    def add_agent(
        self,
        agent_id: int,
        pos: Optional[Tuple[int, int]] = None,
        *,
        slot: int | None = None,
        slot_token: int | None = None,
    ):
        """Place agent `agent_id` into the world at `pos` or a random empty cell."""
        # Phase 3 determinism: membership write localization (optional).
        try:
            from simulation.membership_trace import note_write  # noqa: WPS433
            note_write(kind="env.add_agent", agent_id=int(agent_id), detail=f"pos={'None' if pos is None else pos}")
        except Exception:
            pass

        agent_id = int(agent_id)
        self._ensure_agent_capacity(agent_id)
        world_slot_before = None
        if agent_id in self._agent_id_to_world_slot:
            resolved_slot, _ = self._resolve_world_slot(
                agent_id,
                slot=slot,
                slot_token=slot_token,
            )
        else:
            if slot is None and slot_token is None and agent_id < int(self.slot_to_agent_id.size):
                slot = agent_id
                slot_token = int(self.slot_tokens[slot]) + 1
            if slot is None or slot_token is None:
                raise RuntimeError(
                    "high-ID world insertion requires a reserved slot and slot_token"
                )
            world_slot_before = self._world_slot_snapshot(int(slot))
            self._bind_world_slot(agent_id, int(slot), int(slot_token))
            resolved_slot, _ = self._resolve_world_slot(
                agent_id,
                slot=slot,
                slot_token=slot_token,
            )
        if bool(self.agent_world_alive_by_slot[resolved_slot]):
            current_pos = (
                int(self.agent_pos_y_by_slot[resolved_slot]),
                int(self.agent_pos_x_by_slot[resolved_slot]),
            )
            if pos is None or tuple(pos) == current_pos:
                return
            raise RuntimeError(
                f"agent already occupies world position: agent_id={agent_id};"
                f"position={current_pos}"
            )

        try:
            if pos is None:
                for _ in range(200):
                    y = int(self.rng.integers(0, self.cfg.height))
                    x = int(self.rng.integers(0, self.cfg.width))
                    if self.agent_grid[y, x] == -1:
                        pos = (y, x)
                        break
                if pos is None:
                    raise RuntimeError("No empty cells available")

            y, x = pos
            if self.agent_grid[y, x] != -1:
                raise ValueError("Cell occupied")
        except BaseException:
            if world_slot_before is not None:
                self._restore_world_slot_snapshot(int(resolved_slot), world_slot_before)
            raise

        self.agent_pos_y_by_slot[resolved_slot] = y
        self.agent_pos_x_by_slot[resolved_slot] = x
        self.agent_world_alive_by_slot[resolved_slot] = True
        if agent_id < int(self.agent_alive.size):
            self.agent_pos_y[agent_id] = y
            self.agent_pos_x[agent_id] = x
            self.agent_alive[agent_id] = True
        self.agent_grid[y, x] = resolved_slot
        self._views_dirty = True

    def remove_agent(
        self,
        agent_id: int,
        *,
        slot: int | None = None,
        slot_token: int | None = None,
    ):
        """Remove agent `agent_id` from the world (if currently alive)."""
        agent_id = int(agent_id)
        # Phase 3 determinism: membership write localization (optional).
        try:
            from simulation.membership_trace import note_write  # noqa: WPS433
            note_write(kind="env.remove_agent", agent_id=int(agent_id), detail="")
        except Exception:
            pass

        if agent_id < 0 or agent_id > np.iinfo(np.int32).max:
            return
        if agent_id not in self._agent_id_to_world_slot:
            if slot is None and slot_token is None:
                return
            if slot is None or slot_token is None:
                raise RuntimeError(
                    f"incomplete world removal identity: agent_id={agent_id}"
                )
            resident_id, resident_token = self._validate_world_slot_row(int(slot))
            if resident_id == -1 and resident_token == int(slot_token):
                return
            raise RuntimeError(
                f"stale world removal identity: agent_id={agent_id};slot={int(slot)};"
                f"token={int(slot_token)};resident_id={resident_id};"
                f"resident_token={resident_token}"
            )
        resolved_slot, _ = self._resolve_world_slot(
            agent_id,
            slot=slot,
            slot_token=slot_token,
        )
        if self.agent_world_alive_by_slot[resolved_slot]:
            y = int(self.agent_pos_y_by_slot[resolved_slot])
            x = int(self.agent_pos_x_by_slot[resolved_slot])
            if self.in_bounds(y, x) and self.agent_grid[y, x] == resolved_slot:
                self.agent_grid[y, x] = -1

        self.agent_world_alive_by_slot[resolved_slot] = False
        self.agent_pos_y_by_slot[resolved_slot] = -1
        self.agent_pos_x_by_slot[resolved_slot] = -1
        if hasattr(self, "respec0b_resource_handling_progress"):
            self.respec0b_resource_active_y[resolved_slot] = -1
            self.respec0b_resource_active_x[resolved_slot] = -1
            self.respec0b_resource_active_class[resolved_slot] = 0
            self.respec0b_resource_handling_progress[resolved_slot] = np.float32(
                0.0
            )
        if agent_id < int(self.agent_alive.size):
            self.agent_alive[agent_id] = False
            self.agent_pos_y[agent_id] = -1
            self.agent_pos_x[agent_id] = -1
        del self._agent_id_to_world_slot[agent_id]
        self.slot_to_agent_id[resolved_slot] = -1
        self._views_dirty = True

    def move_agent(
        self,
        agent_id: int,
        pos: Tuple[int, int],
        *,
        slot: int | None = None,
        slot_token: int | None = None,
    ) -> None:
        """Move a live world member without changing its slot identity."""
        agent_id = int(agent_id)
        resolved_slot, _ = self._resolve_world_slot(
            agent_id,
            slot=slot,
            slot_token=slot_token,
            require_alive=True,
        )
        y, x = (int(pos[0]), int(pos[1]))
        if not self.in_bounds(y, x):
            raise ValueError(f"world position out of bounds: pos={(y, x)}")
        occupant = int(self.agent_grid[y, x])
        if occupant not in (-1, resolved_slot):
            raise ValueError("Cell occupied")
        old_y = int(self.agent_pos_y_by_slot[resolved_slot])
        old_x = int(self.agent_pos_x_by_slot[resolved_slot])
        if self.in_bounds(old_y, old_x) and int(self.agent_grid[old_y, old_x]) == resolved_slot:
            self.agent_grid[old_y, old_x] = -1
        self.agent_grid[y, x] = resolved_slot
        self.agent_pos_y_by_slot[resolved_slot] = y
        self.agent_pos_x_by_slot[resolved_slot] = x
        if agent_id < int(self.agent_alive.size):
            self.agent_pos_y[agent_id] = y
            self.agent_pos_x[agent_id] = x
        self._views_dirty = True
