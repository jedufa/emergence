# Emergence simulation slot monitor

The read-only monitor is integrated into the main Causyl WebUI:

`http://127.0.0.1:8787/#live-simulations`

The main server exposes its data at `/api/emergence/live_simulations`; no
separate monitor service or port is required.

Managed runners publish operational heartbeat files through
`engine.monitoring.slot_telemetry`. Set `EIEL_SLOT_MONITOR=off` to disable that
output for a process. The heartbeat files are not simulation state and are not
included in checkpoints or scientific digests.

Current heartbeat schema `eiel_simulation_slot_heartbeat_v2` exposes the
half-open committed tick cursor as `next_absolute_tick`, together with
`target_absolute_tick` and `rate_ticks_per_second`. Retained v1 heartbeats are
projected as their exact committed boundary tick and marked boundary-only; the
monitor never manufactures an intra-boundary live tick from them.
