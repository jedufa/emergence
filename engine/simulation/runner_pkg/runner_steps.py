from __future__ import annotations

"""
simulation/runner_pkg/runner_steps.py

Compatibility shim for the runner's per-generation time loop.

Status:
- The core implementation has been split into smaller, future-proof modules.
- "step" has been renamed conceptually to "tick" (a clearer term for "one unit of sim time").

Why keep this file?
- Existing imports across the codebase may still reference runner_pkg.steps.StepLoopMixin
- Keeping this shim avoids a wide refactor across unrelated modules right now

Public API provided:
- TickLoopMixin: preferred API (tick terminology)
- StepLoopMixin: backwards-compatible wrapper that forwards to TickLoopMixin

Implementation modules:
- runner_ticks_loop.py (orchestration)
- runner_ticks_env_apply.py (env commit)
- runner_ticks_drives.py (drives + prediction error)
- runner_ticks_recovery.py (rest + recovery)
- runner_ticks_combat.py (combat bookkeeping)
- runner_ticks_reproduction.py (births)
- runner_ticks_coherence.py (switch/oscillation)
- runner_ticks_init.py (per-generation resets)
"""

from typing import Dict, Tuple

from .runner_ticks_init import TickInitMixin
from .runner_ticks_coherence import TickCoherenceMixin
from .runner_ticks_env_apply import TickEnvApplyMixin
from .runner_ticks_drives import TickDrivesMixin
from .runner_ticks_chemistry import TickChemistryMixin
from .runner_ticks_maintenance import TickMaintenanceMixin
from .runner_ticks_identity_drift import TickIdentityDriftMixin
from .runner_ticks_recovery import TickRecoveryMixin
from .runner_ticks_combat import TickCombatMixin
from .runner_ticks_reproduction import TickReproductionMixin
from .runner_ticks_loop import TickLoopMixin as _TickLoopImpl


class TickLoopMixin(
    TickInitMixin,
    TickCoherenceMixin,
    TickEnvApplyMixin,
    TickDrivesMixin,
    TickChemistryMixin,
    TickMaintenanceMixin,
    TickIdentityDriftMixin,
    TickRecoveryMixin,
    TickCombatMixin,
    TickReproductionMixin,
    _TickLoopImpl,
):
    """
    Preferred mixin: tick-based simulation time loop.

    Exposes:
    - _run_ticks(...)
    - _apply_env_tick(...)
    - and the rest of the tick subsystem methods
    """
    pass


class StepLoopMixin(TickLoopMixin):
    """
    Backwards-compatible wrapper.

    Keeps the old method names so other code doesn't have to be updated immediately.
    """

    def _run_steps(
        self,
        gen: int,
        buckets: Dict[str, float],
        injury_recover_mult: float,
        deaths_this_gen: int,
        successful_births: int,
        failed_births: int,
        gen_peak_injury: float,
        rest_streaks: Dict[int, int],
    ) -> Tuple[int, int, int, float, int]:
        return self._run_ticks(
            gen=gen,
            buckets=buckets,
            injury_recover_mult=injury_recover_mult,
            deaths_this_gen=deaths_this_gen,
            successful_births=successful_births,
            failed_births=failed_births,
            gen_peak_injury=gen_peak_injury,
            rest_streaks=rest_streaks,
        )

    def _apply_env_step(self, alive, agent_ids_np, actions_np, buckets, deaths_this_gen: int, gen_peak_injury: float):
        return self._apply_env_tick(alive, agent_ids_np, actions_np, buckets, deaths_this_gen, gen_peak_injury)
