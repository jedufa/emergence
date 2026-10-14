# EIEL Living Lineage Canopy V0.1

Standalone read-only browser prototype for visualizing lineage dynamics as
fluid particle canopies.

## Run

From the Emergence project directory:

```bash
python -m http.server 8769 --directory tools/visualization/living_lineage_canopy
```

Open:

```text
http://127.0.0.1:8769/
```

Runtime prerequisites for the prototype are only Python 3 and a modern browser.
No Node, build step, GPU, database, engine service, campaign runner, Playwright,
or production runtime is required.

Playwright is optional and was used only for local screenshot/performance
evidence capture.

Near-live relay review also uses Python 3. A browser must load the app over
local HTTP, not `file://`, because relay mode fetches JSON manifests and uses
Web Crypto hash verification. Playwright and ffmpeg are evidence tools only;
they are not runtime prerequisites for using the app.

## Included Datasets

- `balanced_coexistence.json`: coexistence with modest branch formation.
- `dominant_sweep.json`: one lineage gradually takes over and others decline.
- `weak_lineage_extinction.json`: one weak lineage declines to extinction.
- `late_bloomer.json`: one lineage blooms late after a quiet start.
- `branch_split.json`: one lineage forms two branches.
- `sibling_branch_competition.json`: sibling descendant branches diverge.
- `boom_collapse.json`: one lineage booms and collapses while neighbors recover.
- `dominance_reversal.json`: an early dominant lineage is overtaken.
- `cyclic_lineage_fortunes.json`: lineages expand and contract cyclically.
- `founder_bottleneck.json`: a founder lineage bottlenecks, then branches regrow.
- `scale_stress_5000.json`: synthetic stress fixture with at least 5,000 living
  particles at peak.
- `p12b_partial_birth_events.json`: partial real-data adapter output from P12B
  birth-event reductions.

Synthetic fixtures are labeled as synthetic demographic dynamics, not
scientific fitness.

The browser exposes these datasets through an in-app scenario gallery and a
Showcase mode. Reviewers do not need to manually pick JSON files.

## Generate And Validate

```bash
python -m tools.visualization.living_lineage_canopy.python.cli generate stress \
  --out /tmp/scale_stress_5000.json

python -m tools.visualization.living_lineage_canopy.python.cli validate \
  tools/visualization/living_lineage_canopy/fixtures/scale_stress_5000.json
```

## Offline Production Export

The exporter is read-only and writes generated playback/provenance files only
to an explicit output path, normally under `/tmp/emergence-example/runtime` or
`/tmp`.

```bash
python -m tools.visualization.living_lineage_canopy.python.export_cli audit \
  --source /path/to/run-or-evidence-root \
  --out /tmp/source_audit.json

python -m tools.visualization.living_lineage_canopy.python.export_cli export \
  --source /path/to/run-or-evidence-root \
  --out /tmp/playback.json \
  --provenance-out /tmp/provenance.json \
  --summary-out /tmp/summary.json \
  --cross-check-out /tmp/cross_check.json \
  --generated-at 2026-07-30T14:20:00Z

python -m tools.visualization.living_lineage_canopy.python.export_cli validate \
  --playback /tmp/playback.json \
  --provenance /tmp/provenance.json
```

The browser can load exported playback with `Load playback file` and an
optional paired `Load provenance` file. Files are read locally by the browser;
there is no upload, network write, live engine connection, or simulation
feedback path.

Source classifications:

- `EXP-A`: full actual lifecycle playback.
- `EXP-B`: partial actual lifecycle playback with explicit missing fields.
- `EXP-C`: aggregate or representative playback.
- `EXP-D`: insufficient or integrity-blocked source.

Current strongest tested export source is the GV2 R1 checkpoint-bundle source:

```text
/tmp/emergence-example/runtime/evidence_raw/gv2_b48_r1_neutral_package_identity_20260730/runs/gv2-b48-r1-neutral-package-identity-b12-partition-0000
```

It exports as `EXP-B / partial_agents`: agent identities, lineage founder ids,
parent edges, and birth event sequences are actual production records; death is
checkpoint-boundary disappearance projection because the selected projection
does not expose successful exact death events.

## Near-Live Playback Relay

`LLC-NEAR-LIVE-PLAYBACK-RELAY-V0` is a read-only file relay for watching
completed EIEL source artifacts appear with bounded delay. It runs outside the
production engine and does not import engine modules, use callbacks, open a
WebSocket, write into a run directory, or influence simulation state.

Typical one-shot update:

```bash
python -m tools.visualization.living_lineage_canopy.python.relay_cli update-once \
  --source-root /path/to/eiel-run-root \
  --output-root /tmp/emergence-example/runtime/live/<run-id>/lineage-playback-relay \
  --source-kind gv2_checkpoint_bundle_v2
```

Typical polling mode:

```bash
python -m tools.visualization.living_lineage_canopy.python.relay_cli watch \
  --source-root /path/to/eiel-run-root \
  --output-root /tmp/emergence-example/runtime/live/<run-id>/lineage-playback-relay \
  --poll-seconds 2
```

The relay writes an isolated output directory containing:

- `relay_manifest.json`;
- `playback_snapshot.json`;
- `provenance.json`;
- `relay_status.json`;
- `cross_check.json`;
- `relay_performance.json`;
- `receipts/update_*.json`.

Writes are atomic: files are written to temporary paths, flushed, validated,
hashed, and atomically replaced. `relay_status.json` is updated last. The
browser follows `relay_manifest.json`, verifies playback/provenance hashes,
verifies playback/provenance binding, and keeps the last valid playback if a
new update fails verification.

Relay source files are processed only after stable observation across scans.
Accepted source hashes are stored in receipts; later source mutation, missing
files, skipped boundaries, regressed boundaries, duplicate boundaries, schema
changes, identity discontinuity, or cross-check failure set
`integrity_error` and retain the last valid snapshot.

Browser modes:

- `Scenarios`: built-in synthetic and partial-adapter gallery plus Showcase.
- `Saved playback`: local playback/provenance file loading.
- `Near-live relay`: HTTP polling of a relay manifest with a `Follow live`
  control. Manual scrubbing turns follow-live off, while paused playback,
  historical tick, selected lineage, and isolation are preserved across valid
  updates when still available.

The staged G88 review fixture is explicit test data reconstructed from the
accepted compact export because the original raw run tree was intentionally
removed after review acceptance. It is valid relay evidence for manifest,
atomicity, browser polling, interruption semantics, and idempotency. It is not
a replacement for raw source hash re-verification.

## Test

```bash
pytest -q tools/visualization/living_lineage_canopy/tests
```

## Optional Browser Evidence

Start the local server first, then:

```bash
python -m tools.visualization.living_lineage_canopy.python.browser_smoke \
  --out-dir /tmp/emergence-example/runtime/evidence_raw/eiel_living_lineage_canopy_v0_20260730/v01_screenshots

python -m tools.visualization.living_lineage_canopy.python.browser_checks \
  --out /tmp/emergence-example/runtime/evidence_raw/eiel_living_lineage_canopy_v0_20260730/v01_browser_checks.json
```

This requires local Playwright installation and is not required to run the
prototype.

## Scientific Boundary

This prototype visualizes lineage and agent lifecycle data.

It does not show or prove intelligence, emergence, cognition, memory selection,
prediction selection, ecological specialization, durable strategy diversity,
consciousness, culture, or open-ended evolution.

Visual clustering, lobe formation, dominance, extinction, or persistence are
display projections over the supplied lineage data. They are not new scientific
findings.

Future CUDA world-search and passive lifecycle journals are outside this relay.
The relay V0 decision gate is whether existing completed artifacts are
sufficient for useful near-live replay, or whether a later bounded passive
lifecycle journal is needed for lower latency or fuller lifecycle completeness.
