# EIEL WRC Reducer v1

The WRC reducer is one deterministic core used in two modes:

```text
live simulator observations ─┐
                             ├─> WorldReducer ─> parameter/Atlas summaries
retained metrics/Chronicle ──┘
```

The source adapter changes. The reduction semantics do not.

## Reduction stages

For every numeric metric, the world reducer preserves and summarizes four
classes of evidence:

1. **Anchors**: first and terminal observations.
2. **Local behavior**: abrupt changes and local extrema.
3. **Accumulated behavior**: cumulative drift and drift from the initial
   baseline, so repeated sub-threshold steps are not lost.
4. **Trajectory structure**: extrema, online outliers, trend, integral,
   variation, and compressed stable spans.

Every retained point carries a source locator. The raw source remains
scientific authority; reducer output is a compact index and summary layer.

## Live use

```python
from engine.wrc.reducer import LiveReducerSink, WorldIdentity

sink = LiveReducerSink(
    WorldIdentity(
        world_id="p023-level-1p0-seed-101",
        parameter_id="env_cfg.move_cost",
        parameter_value=1.0,
        seed=101,
        ticks_per_observation=10,
    )
)

# Call at the simulation's normal reporting boundary.
sink.ingest(sequence=0, tick=10, metrics={"alive": 256, "mean_fitness": 42.0})
state = sink.snapshot()          # checkpointable reducer state
world_summary = sink.finalize()
```

The v1.18 live adapter is enabled at the existing reporting boundary by
default. It remains observational: disabling the adapter or comparing live
output with metrics replay must not change the simulation trajectory.

Report bounded operator status from a checkpointable live snapshot:

```bash
python -m tools.scripts.eiel_wrc_reduce status \
  --snapshot RUN/reducer_state.json \
  --retention-manifest RUN/retained/retention_manifest.json
```

The status output is limited to reducer enablement/version, retention mode,
finalization, retention outcome, and available byte/file counters. Without a
retention manifest, the outcome is `not_materialized` and counters are `null`.

## Post-run use

Reduce one metrics file:

```bash
python -m tools.scripts.eiel_wrc_reduce world RUN/metrics.csv \
  --parameter-id env_cfg.move_cost \
  --parameter-value 1.0 \
  --seed 101 \
  --ticks-per-observation 10 \
  --output-dir reduction/world-101
```

Reduce a completed serialized WRC row or Atlas directory/ZIP:

```bash
python -m tools.scripts.eiel_wrc_reduce reduce atlas.zip \
  --output-dir reduction/atlas

python -m tools.scripts.eiel_wrc_reduce reduce atlas.zip \
  --row P011 \
  --output-dir reduction/p011
```

Replay compact Chronicle generation summaries:

```bash
python -m tools.scripts.eiel_wrc_reduce chronicle \
  RUN/chronicle/formal_events.jsonl \
  --parameter-id env_cfg.move_cost \
  --parameter-value 1.0 \
  --seed 101 \
  --output-dir reduction/chronicle
```

## Standard outputs

At Atlas level:

- `wrc_master.csv`: one standardized machine-readable row per parameter.
- `wrc_master.md`: a compact human-readable screening table.
- `wrc_family_summaries.csv|json|md`: grouped family-level summaries for
  comparing related controls without opening every parameter card.
- `atlas_reduction_status.json`: completed, skipped, and failed rows.
- `atlas_reduction_manifest.json`: hashes and source identity.

For each parameter:

- `parameter_card.json`: compact machine-readable result divided into control
  transport, activation, direct, ecology, organism, population, and dynamics
  layers.
- `parameter_card.md`: the same standardized result as a short human-readable
  card.
- `parameter_summary.json`: fuller machine-readable parameter reduction.
- `metric_effects.csv`: paired-seed effects for all available metrics and
  supported trajectory features. It preserves the raw normalized `effect_score`
  and also exposes `scientific_effect_score`, which adds conservative
  materiality, evidence-coverage, seed-consistency, and mechanism-relevance
  weights for human-facing ranking.
- `response_curves.csv`: response-shape classification across tested levels.
- `world_summaries.jsonl`: bounded per-world trajectory summaries for selected
  detail metrics.
- `selected_points.csv`: retained anchors, changes, drift points, extrema, and
  outliers.
- `raw_row_index.csv`: source locators for exact drill-back.

Extract only the indexed metrics from referenced raw rows:

```bash
python -m tools.scripts.eiel_wrc_reduce extract \
  --index reduction/p011/raw_row_index.csv \
  --output reduction/p011/raw_rows_compact.jsonl
```

Use `--metric METRIC` to restrict extraction further. `--full-row` is explicit
because a full EIEL metrics row may contain more than one thousand columns and
large JSON payloads. Humans have suffered enough accidental 200 MB "small"
files.

## Standard interpretation

Every parameter card uses the same vocabulary:

- control transport: `CONFIRMED`, `PARTIAL`, `MISMATCH`, or `NOT_AVAILABLE`;
- effect layers: `STRONG`, `MODERATE`, `WEAK`, `NO_EFFECT_DETECTED`, or
  `NOT_AVAILABLE`;
- response curves: Boolean contrast, flat, monotonic, threshold, saturation,
  U/inverted-U, non-monotonic, catastrophic threshold, or unresolved;
- dynamics: early/mid/late timing and persistent/transient/delayed/threshold
  trajectory patterns.

These are screening descriptions, not qualification or scientific decision
authority. The scientific ranking score is deliberately separate from the raw
normalized effect so tiny-denominator metrics remain auditable without
automatically becoming headline results.

## Porting to v1.18

The implementation is intentionally isolated to:

```text
engine/wrc/
tools/scripts/eiel_wrc_reduce.py
tests/wrc/
```

No existing simulator, campaign runner, checkpoint, or metrics producer is
modified. Porting is therefore a directory-level comparison or cherry-pick,
followed by the existing reporting-boundary integration and a bounded status
check.

## Retention qualification

Materialize a verified compact copy without modifying the source:

```bash
python -m tools.scripts.eiel_wrc_reduce retain RUN \
  --mode retain_compact \
  --output-dir retained \
  --expected-world-count 1
```

The emitted `retention_manifest.json` records source and retained bytes,
artifact counts, hashes, and whether an explicitly requested raw prune
completed. One-run storage and timing measurements are descriptive; missing
CPU/GPU counters remain unavailable rather than being estimated.

### Interpretation semantics (v1.1)

Human-facing summaries keep three concepts separate:

- `control_path_activation`: transport/readback evidence only; it does not claim a scientific effect.
- `mechanism_response`: the declared activation layer when it moves materially, otherwise the direct-response layer as an explicit fallback.
- `downstream_response`: broader ecology/organism/population consequence consistency.

A supported persistent/threshold effect with no retained first-significant tick is reported as
`EFFECT_PRESENT_ONSET_UNRESOLVED`, never as if the effect itself were absent. Layer headline metrics prefer declared response hints and canonical metrics over helper quantiles/dispersion fields.
