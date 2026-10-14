import { collectElements, bindControls } from "./controls.js";
import { Playback } from "./playback.js";
import { ParticleSystem } from "./physics.js";
import { Renderer } from "./renderer.js";

const FIXED_DT = 1000 / 60;

const DATASETS = [
  {
    key: "balanced",
    url: "fixtures/balanced_coexistence.json",
    label: "Balanced coexistence",
    description: "Multiple founder canopies persist together without a single takeover.",
    observe: "Watch similarly sized colonies repack softly while branch lobes stay distinct.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 20,
    focusEnd: 190,
  },
  {
    key: "dominant",
    url: "fixtures/dominant_sweep.json",
    label: "Dominant sweep",
    description: "One lineage grows gradually while several competitors decline.",
    observe: "The expanding canopy should push neighboring bodies aside instead of growing in a fixed slot.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 40,
    focusEnd: 250,
  },
  {
    key: "weak_extinction",
    url: "fixtures/weak_lineage_extinction.json",
    label: "Weak lineage extinction",
    description: "A weak lineage contracts to zero recorded living agents.",
    observe: "Its released space should be reclaimed by adjacent surviving canopies.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 20,
    focusEnd: 220,
  },
  {
    key: "late_bloomer",
    url: "fixtures/late_bloomer.json",
    label: "Late bloomer",
    description: "A small lineage stays quiet before blooming late in the playback.",
    observe: "The late lobe should emerge from lifecycle births and displace surrounding bodies.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 80,
    focusEnd: 270,
  },
  {
    key: "boom_collapse",
    url: "fixtures/boom_collapse.json",
    label: "Boom and collapse",
    description: "A lineage expands rapidly, then collapses while neighbors recover space.",
    observe: "Look for a push-out phase followed by visible free-space reclamation.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 35,
    focusEnd: 250,
  },
  {
    key: "branch_split",
    url: "fixtures/branch_split.json",
    label: "Branch split",
    description: "A major lineage splits into multiple visible branch lobes.",
    observe: "Look for a shared lineage wash with distinct but related sub-canopies.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 40,
    focusEnd: 225,
  },
  {
    key: "sibling_competition",
    url: "fixtures/sibling_branch_competition.json",
    label: "Sibling branch competition",
    description: "Two descendant branches from one lineage diverge in success.",
    observe: "Sibling lobes should remain related while one becomes visibly stronger.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 40,
    focusEnd: 225,
  },
  {
    key: "dominance_reversal",
    url: "fixtures/dominance_reversal.json",
    label: "Dominance reversal",
    description: "The early dominant lineage is overtaken by a later expanding lineage.",
    observe: "The composition should reorganize as the leader changes.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 35,
    focusEnd: 275,
  },
  {
    key: "cyclic_fortunes",
    url: "fixtures/cyclic_lineage_fortunes.json",
    label: "Cyclic lineage fortunes",
    description: "Lineages take turns expanding and contracting in bounded demographic waves.",
    observe: "No lineage should be visually pinned; the field should breathe and repack.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 20,
    focusEnd: 300,
  },
  {
    key: "founder_bottleneck",
    url: "fixtures/founder_bottleneck.json",
    label: "Founder bottleneck",
    description: "A founder lineage narrows sharply, then surviving branches recover.",
    observe: "The bottleneck should briefly thin the canopy before descendant lobes regrow.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 25,
    focusEnd: 270,
  },
  {
    key: "p12b",
    url: "fixtures/p12b_partial_birth_events.json",
    label: "P12B partial real-data playback",
    description: "Partial adapter playback from recorded P12B birth receipts.",
    observe: "Use this only as a data-shape preview; current living population is not established.",
    status: "Partial real-data adapter",
    semantics: "partial_agents",
    focusStart: 837,
    focusEnd: 1814,
  },
  {
    key: "stress",
    url: "fixtures/scale_stress_5000.json",
    label: "5,000 particle stress",
    description: "Dense synthetic stress case for Canvas 2D responsiveness.",
    observe: "The field should remain stable and controls should stay responsive.",
    status: "Synthetic demographic dynamics",
    semantics: "actual_agents",
    focusStart: 60,
    focusEnd: 190,
  },
];

const elements = collectElements();
const canvas = document.getElementById("canopyCanvas");
const system = new ParticleSystem();
const renderer = new Renderer(canvas, system);

let playback = null;
let currentScenario = DATASETS[0];
let currentState = null;
let currentTick = 0;
let playing = false;
let speed = 1;
let selectedAgentId = null;
let selectedLineageId = null;
let isolated = false;
let externalPlayback = null;
let externalProvenance = null;
let appMode = "scenarios";
let lastFrame = performance.now();
let lastUiRefresh = 0;
let physicsAccumulator = 0;
let fpsSmoother = 60;
let physicsMs = 0;
let renderMs = 0;
let initialParamsConsumed = false;
const initialParams = new URLSearchParams(window.location.search);
const datasetCache = new Map();
const scenarioStats = new Map();
const showcase = {
  active: false,
  index: 0,
  playUntil: null,
  transitioning: false,
};
const relay = {
  active: false,
  manifestUrl: "",
  timer: null,
  inFlight: false,
  followLive: true,
  pollMs: 2000,
  sequence: null,
  sourceRunId: null,
  sourceState: "idle",
  sourceBoundary: null,
  playbackBoundary: null,
  sourceLag: null,
  warningCount: 0,
  lastPollAt: null,
  lastSuccessAt: null,
  hashVerified: false,
  error: null,
  selectionNotice: null,
};
const playbackLibrary = {
  rows: [],
  query: "",
};

function optionValue() {
  return {
    particleScale: Number(elements.particleSizeRange.value),
    motion: Number(elements.motionRange.value),
    branchPull: Number(elements.branchPullRange.value),
    lineagePull: Number(elements.lineagePullRange.value),
    showBirthPulse: elements.birthPulseToggle.checked,
    showDying: elements.dyingToggle.checked,
  };
}

function scenarioByKey(key) {
  return DATASETS.find((item) => item.key === key) || DATASETS[0];
}

function computeScenarioStats(data) {
  const events = [];
  for (const agent of data.agents) {
    events.push([agent.birth_tick, 1]);
    if (agent.death_tick !== null && agent.death_tick !== undefined) events.push([agent.death_tick, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  let current = 0;
  let peak = 0;
  for (const [, delta] of events) {
    current += delta;
    peak = Math.max(peak, current);
  }
  const duration = data.metadata.last_tick - data.metadata.first_tick;
  return {
    peak,
    lineages: data.lineages.length,
    duration,
  };
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
  }
  return value;
}

function canonicalJson(data) {
  return `${JSON.stringify(canonicalize(data))}\n`;
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function hashPlaybackData(data) {
  return sha256Hex(canonicalJson(data));
}

async function fetchTextAndSha256(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const text = await response.text();
  return { text, sha256: await sha256Hex(text) };
}

function playbackSearchText(row) {
  return [row.label, row.location, row.storage, row.source_state, row.semantics_mode]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase();
}

function renderPlaybackLibrary() {
  const query = playbackLibrary.query.trim().toLocaleLowerCase();
  const rows = playbackLibrary.rows.filter((row) => !query || playbackSearchText(row).includes(query));
  elements.playbackLibrarySelect.innerHTML = "";
  if (!rows.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = playbackLibrary.rows.length ? "No matching saved playbacks" : "No compatible saved playbacks found";
    elements.playbackLibrarySelect.appendChild(option);
    elements.playbackLibrarySelect.disabled = true;
    elements.playbackLibraryPlay.disabled = true;
    elements.playbackLibraryStatus.textContent = playbackLibrary.rows.length
      ? "Try a shorter search."
      : "Scanned runtime and cold storage for verified relay playbacks.";
    return;
  }
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a saved playback…";
  elements.playbackLibrarySelect.appendChild(placeholder);
  for (const row of rows) {
    const option = document.createElement("option");
    option.value = row.key;
    option.textContent = `${row.label} · ${row.storage} · ${row.source_state}`;
    elements.playbackLibrarySelect.appendChild(option);
  }
  elements.playbackLibrarySelect.disabled = false;
  elements.playbackLibraryPlay.disabled = true;
  elements.playbackLibraryStatus.textContent = `${rows.length} of ${playbackLibrary.rows.length} verified playback${playbackLibrary.rows.length === 1 ? "" : "s"}.`;
}

async function refreshPlaybackLibrary() {
  elements.playbackLibrarySelect.disabled = true;
  elements.playbackLibraryPlay.disabled = true;
  elements.playbackLibraryStatus.textContent = "Discovering saved playbacks…";
  try {
    const response = await fetch("/api/canopy/playbacks", { cache: "no-store" });
    if (!response.ok) throw new Error(`Playback library request failed: ${response.status}`);
    const payload = await response.json();
    playbackLibrary.rows = Array.isArray(payload.playbacks) ? payload.playbacks : [];
    renderPlaybackLibrary();
  } catch (error) {
    playbackLibrary.rows = [];
    renderPlaybackLibrary();
    elements.playbackLibraryStatus.textContent = `Playback library unavailable: ${error.message}`;
  }
}

async function loadLibraryPlayback(key) {
  const row = playbackLibrary.rows.find((candidate) => candidate.key === key);
  if (!row) return;
  stopShowcase();
  elements.relayManifestInput.value = new URL(row.manifest_url, window.location.origin).toString();
  await connectRelay({ mode: "saved", poll: false });
  elements.playbackLibraryStatus.textContent = `Loaded ${row.label} from ${row.storage}.`;
}

function selectLibraryPlayback(key) {
  const row = playbackLibrary.rows.find((candidate) => candidate.key === key);
  elements.playbackLibraryPlay.disabled = !row;
  if (row) elements.playbackLibraryStatus.textContent = `Ready to play ${row.label} from ${row.storage}.`;
}

async function playSelectedLibraryPlayback() {
  const key = elements.playbackLibrarySelect.value;
  if (!playbackLibrary.rows.some((row) => row.key === key)) return;
  elements.playbackLibraryPlay.disabled = true;
  try {
    await loadLibraryPlayback(key);
    playing = true;
    updatePanel();
    const row = playbackLibrary.rows.find((candidate) => candidate.key === key);
    elements.playbackLibraryStatus.textContent = `Playing ${row.label} from ${row.storage}.`;
  } finally {
    elements.playbackLibraryPlay.disabled = !playbackLibrary.rows.some((row) => row.key === elements.playbackLibrarySelect.value);
  }
}

function verifyProvenanceForCurrentPlayback(provenance) {
  if (!externalPlayback) throw new Error("Load a playback file before loading provenance.");
  if (provenance.schema_version !== "eiel_lineage_playback_export_provenance_v1") {
    throw new Error("Unsupported provenance schema.");
  }
  if (provenance.output_sha256 !== externalPlayback.outputSha256) {
    throw new Error("Provenance hash does not match the loaded playback.");
  }
  const metadata = externalPlayback.data.metadata || {};
  if (provenance.playback_schema_version !== metadata.schema_version) {
    throw new Error("Provenance playback schema does not match the loaded playback.");
  }
  for (const key of ["source_classification", "semantics_mode"]) {
    if (provenance[key] !== metadata[key]) throw new Error(`Provenance ${key} does not match the loaded playback.`);
  }
  if (provenance.dataset_id !== metadata.dataset_id) throw new Error("Provenance dataset_id does not match the loaded playback.");
  if ((provenance.source_cell_id ?? null) !== (metadata.source_cell_id ?? null)) {
    throw new Error("Provenance source_cell_id does not match the loaded playback.");
  }
  if (!provenance.exporter_git_commit) throw new Error("Provenance exporter commit is missing.");
  return {
    outputSha256: provenance.output_sha256,
    exporterGitCommit: provenance.exporter_git_commit,
  };
}

async function fetchDataset(entry) {
  if (datasetCache.has(entry.key)) return datasetCache.get(entry.key);
  const response = await fetch(entry.url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${entry.url}: ${response.status}`);
  const data = await response.json();
  datasetCache.set(entry.key, data);
  scenarioStats.set(entry.key, computeScenarioStats(data));
  renderScenarioGallery();
  return data;
}

function setDatasetOptions() {
  elements.datasetSelect.innerHTML = "";
  for (const entry of DATASETS) {
    const option = document.createElement("option");
    option.value = entry.key;
    option.textContent = entry.label;
    elements.datasetSelect.appendChild(option);
  }
}

function setExternalDatasetOption(label) {
  let option = Array.from(elements.datasetSelect.options).find((item) => item.value === "external");
  if (!option) {
    option = document.createElement("option");
    option.value = "external";
    elements.datasetSelect.appendChild(option);
  }
  option.textContent = label;
  elements.datasetSelect.value = "external";
}

function setMode(mode) {
  appMode = mode;
  elements.scenariosModeButton.classList.toggle("is-active", mode === "scenarios");
  elements.savedModeButton.classList.toggle("is-active", mode === "saved");
  elements.relayModeButton.classList.toggle("is-active", mode === "relay");
  elements.datasetSelect.hidden = mode !== "scenarios";
  elements.scenarioGallery.closest(".panel").hidden = mode !== "scenarios";
  elements.relayControls.hidden = mode !== "relay";
  elements.playbackFileInput.closest(".file-load-row").hidden = mode === "relay";
  updateRelayPanel();
}

function renderScenarioGallery() {
  elements.scenarioGallery.innerHTML = "";
  for (const entry of DATASETS) {
    const stats = scenarioStats.get(entry.key);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `scenario-card${entry.key === currentScenario.key ? " is-active" : ""}`;
    card.addEventListener("click", () => {
      stopShowcase();
      loadDataset(entry.key, { tick: entry.focusStart }).catch(reportLoadError);
    });

    const top = document.createElement("span");
    top.className = "scenario-card-top";
    const swatch = document.createElement("span");
    swatch.className = `scenario-status ${entry.semantics}`;
    const name = document.createElement("strong");
    name.textContent = entry.label;
    top.append(swatch, name);

    const description = document.createElement("span");
    description.className = "scenario-description";
    description.textContent = entry.description;

    const observe = document.createElement("span");
    observe.className = "scenario-observe";
    observe.textContent = entry.observe;

    const meta = document.createElement("span");
    meta.className = "scenario-meta";
    const peak = stats ? stats.peak.toLocaleString() : "...";
    const lineages = stats ? stats.lineages.toString() : "...";
    const duration = stats ? `~${stats.duration} ticks` : "~...";
    meta.textContent = `${entry.status} · ${entry.semantics} · peak ${peak} · ${lineages} lineages · ${duration}`;

    card.append(top, description, observe, meta);
    elements.scenarioGallery.appendChild(card);
  }
}

async function preloadScenarioStats() {
  for (const entry of DATASETS) {
    fetchDataset(entry).catch(() => {});
  }
}

function applySemanticState() {
  const mode = playback?.metadata.semantics_mode || currentScenario.semantics;
  const display = playback?.metadata.display_semantics || {};
  const datasetId = playback?.metadata.dataset_id || "";
  const defaultPartialSummary = datasetId.startsWith("p12b")
    ? "Birth receipts only. Death timing and the complete initial population are unavailable. Particle count is not current living population."
    : "Partial source semantics: particle count is not established as complete current living production population.";
  const notes = playback?.metadata.notes || currentScenario.description;
  const warnings = playback?.metadata.warnings || [];
  const classification = playback?.metadata.source_classification;
  const timeUnit = playback?.metadata.time_unit;
  const branchSemantics = playback?.metadata.branch_semantics;
  const birthSemantics = playback?.metadata.birth_record_semantics;
  const deathSemantics = playback?.metadata.death_record_semantics;
  const initialSemantics = playback?.metadata.initial_population_semantics;
  const countSemantics = playback?.metadata.population_count_semantics;
  const shareSemantics = playback?.metadata.share_semantics;
  const provenanceHash = externalProvenance?.data?.output_sha256;
  const provenanceVerified = externalProvenance?.verified;
  const compactWarnings = warnings.map((warning) => warning.includes("They do not show or prove:")
    ? "Scientific boundary: visualization only; no intelligence, emergence, cognition, memory-selection, prediction-selection, or consciousness claims."
    : warning);
  elements.semanticPanel.hidden = mode === "actual_agents";
  elements.livingMetricLabel.textContent = display.metric_label || (mode === "partial_agents" ? "Recorded particles" : mode === "representative_particles" ? "Representative particles" : "Living");
  elements.shareMetricLabel.textContent = display.share_label || (mode === "partial_agents" ? "Recorded share" : mode === "representative_particles" ? "Representative share" : "Largest share");
  elements.dyingToggle.disabled = mode !== "actual_agents";
  elements.dyingToggle.parentElement.classList.toggle("is-disabled", mode !== "actual_agents");
  if (mode !== "actual_agents") elements.dyingToggle.checked = false;
  if (mode === "partial_agents") {
    elements.semanticTitle.textContent = display.notice_title || "Partial Data Warning";
    elements.semanticNotice.textContent = [
      display.notice_summary || defaultPartialSummary,
      classification ? `Classification: ${classification}.` : "",
      timeUnit ? `Time unit: ${timeUnit}.` : "",
      branchSemantics ? `Branch semantics: ${branchSemantics}.` : "",
      birthSemantics ? `Birth semantics: ${birthSemantics}.` : "",
      deathSemantics ? `Death semantics: ${deathSemantics}.` : "",
      initialSemantics ? `Initial population: ${initialSemantics}.` : "",
      countSemantics ? `Count semantics: ${countSemantics}.` : "",
      shareSemantics ? `Share semantics: ${shareSemantics}.` : "",
      provenanceHash ? `Provenance verified: ${provenanceVerified ? "yes" : "no"}; output hash ${provenanceHash.slice(0, 12)}...` : externalPlayback ? "Provenance not loaded or not verified." : "",
      notes,
      ...compactWarnings,
    ].filter(Boolean).join("\n");
  } else if (mode === "representative_particles") {
    elements.semanticTitle.textContent = "Representative Data";
    elements.semanticNotice.textContent = [
      "Particles are representatives, not complete individual lifecycle records.",
      classification ? `Classification: ${classification}.` : "",
      timeUnit ? `Time unit: ${timeUnit}.` : "",
      branchSemantics ? `Branch semantics: ${branchSemantics}.` : "",
      notes,
      ...compactWarnings,
    ].filter(Boolean).join("\n");
  }
}

function colorCss(color) {
  return `hsl(${color.h.toFixed(1)} ${color.s}% ${color.l}%)`;
}

function renderLineageLegend() {
  elements.lineageLegend.innerHTML = "";
  if (!playback || !currentState) return;
  const rows = playback.lineages.map((lineage) => {
    const count = currentState.lineageCounts.get(lineage.lineage_id) || 0;
    const share = currentState.living.length ? count / currentState.living.length : 0;
    return { lineage, count, share };
  }).sort((a, b) => b.count - a.count || a.lineage.lineage_id.localeCompare(b.lineage.lineage_id));
  for (const row of rows.slice(0, 8)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `lineage-row${selectedLineageId === row.lineage.lineage_id ? " is-selected" : ""}${row.count === 0 ? " is-extinct" : ""}`;
    button.addEventListener("click", () => {
      selectedLineageId = row.lineage.lineage_id;
      selectedAgentId = null;
      elements.lineageSelect.value = selectedLineageId;
      updatePanel();
    });
    const swatch = document.createElement("span");
    swatch.className = "lineage-swatch";
    swatch.style.background = colorCss(system.lineageColors.get(row.lineage.lineage_id));
    const label = document.createElement("span");
    label.className = "lineage-row-label";
    label.textContent = row.lineage.display_label || row.lineage.lineage_id;
    const value = document.createElement("span");
    value.className = "lineage-row-value";
    value.textContent = row.count ? `${row.count.toLocaleString()} · ${Math.round(row.share * 100)}%` : "extinct";
    button.append(swatch, label, value);
    elements.lineageLegend.appendChild(button);
  }
}

async function loadDataset(key, options = {}) {
  const entry = scenarioByKey(key);
  currentScenario = entry;
  externalPlayback = null;
  externalProvenance = null;
  elements.datasetSelect.value = entry.key;
  elements.datasetSelect.disabled = false;
  elements.returnGalleryButton.hidden = true;
  const data = await fetchDataset(entry);
  playback = new Playback(data);
  let startTick = options.tick ?? playback.firstTick;
  if (!initialParamsConsumed && options.applyInitialParams !== false) {
    const requestedTick = initialParams.get("tick");
    if (requestedTick !== null) startTick = Number(requestedTick);
  }
  currentTick = startTick;
  selectedAgentId = null;
  selectedLineageId = null;
  isolated = false;
  physicsAccumulator = 0;
  system.configure(playback);
  system.setOptions(optionValue());
  renderer.resize();
  currentState = playback.stateAt(currentTick);
  system.rebuild(currentState);
  elements.timelineRange.min = playback.firstTick;
  elements.timelineRange.max = playback.lastTick;
  elements.timelineRange.value = currentState.tick;
  elements.datasetSubtitle.textContent = `${entry.label} · ${playback.metadata.semantics_mode}`;
  elements.qualityBadge.textContent = playback.metadata.source_kind === "synthetic" ? "synthetic demographic dynamics" : "partial real adapter";
  refreshLineageSelect();
  if (!initialParamsConsumed && options.applyInitialParams !== false) {
    const requestedLineage = initialParams.get("lineage");
    if (requestedLineage) {
      selectedLineageId = requestedLineage;
      elements.lineageSelect.value = requestedLineage;
    }
    if (initialParams.get("isolated") === "1") isolated = Boolean(selectedLineageId);
    initialParamsConsumed = true;
  }
  applySemanticState();
  renderScenarioGallery();
  updateShowcaseOverlay();
  updatePanel();
}

function installPlayback(data, entry, options = {}) {
  const nextPlayback = new Playback(data);
  currentScenario = entry;
  if (entry.key === "external") setExternalDatasetOption(entry.label);
  playback = nextPlayback;
  currentTick = options.tick ?? playback.firstTick;
  selectedAgentId = null;
  selectedLineageId = null;
  isolated = false;
  playing = false;
  physicsAccumulator = 0;
  system.configure(playback);
  system.setOptions(optionValue());
  renderer.resize();
  currentState = playback.stateAt(currentTick);
  system.rebuild(currentState);
  elements.timelineRange.min = playback.firstTick;
  elements.timelineRange.max = playback.lastTick;
  elements.timelineRange.value = currentState.tick;
  elements.datasetSubtitle.textContent = `${entry.label} · ${playback.metadata.semantics_mode}`;
  elements.qualityBadge.textContent = entry.status;
  refreshLineageSelect();
  applySemanticState();
  renderScenarioGallery();
  updateShowcaseOverlay();
  updatePanel();
}

async function loadExternalPlayback(file) {
  if (!file) return;
  const prior = { playback, currentScenario, currentTick, selectedAgentId, selectedLineageId, isolated, externalPlayback, externalProvenance };
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    const outputSha256 = await hashPlaybackData(data);
    const entry = {
      key: "external",
      label: data.metadata?.dataset_id || file.name,
      description: data.metadata?.notes || "Externally loaded read-only playback.",
      observe: "Review the imported lifecycle projection with its warnings and provenance.",
      status: data.metadata?.source_kind === "real_export" ? "external real export" : "external playback",
      semantics: data.metadata?.semantics_mode || "partial_agents",
      focusStart: data.metadata?.first_tick ?? 0,
      focusEnd: data.metadata?.last_tick ?? 0,
    };
    const nextPlayback = new Playback(data);
    externalPlayback = { fileName: file.name, data, outputSha256 };
    externalProvenance = null;
    currentScenario = entry;
    playback = nextPlayback;
    elements.datasetSelect.disabled = true;
    elements.returnGalleryButton.hidden = false;
    installPlayback(data, entry);
  } catch (error) {
    playback = prior.playback;
    currentScenario = prior.currentScenario;
    currentTick = prior.currentTick;
    selectedAgentId = prior.selectedAgentId;
    selectedLineageId = prior.selectedLineageId;
    isolated = prior.isolated;
    externalPlayback = prior.externalPlayback;
    externalProvenance = prior.externalProvenance;
    reportLoadError(error);
  } finally {
    elements.playbackFileInput.value = "";
  }
}

async function loadExternalProvenance(file) {
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const binding = verifyProvenanceForCurrentPlayback(data);
    externalProvenance = { fileName: file.name, data, binding, verified: true };
    applySemanticState();
    updatePanel();
  } catch (error) {
    reportLoadError(error);
  } finally {
    elements.provenanceFileInput.value = "";
  }
}

function validateRelayManifest(manifest) {
  if (!manifest || typeof manifest !== "object") throw new Error("Relay manifest must be an object.");
  if (manifest.schema_version !== "llc_near_live_playback_relay_v0_manifest") {
    throw new Error(`Unsupported relay manifest schema: ${manifest.schema_version}`);
  }
  if (!Number.isInteger(manifest.sequence_number) || manifest.sequence_number < 1) {
    throw new Error("Relay sequence must be a positive integer.");
  }
  for (const key of ["source_run_id", "playback_file_name", "playback_sha256", "provenance_file_name", "provenance_sha256"]) {
    if (typeof manifest[key] !== "string" || !manifest[key]) throw new Error(`Relay manifest missing ${key}.`);
  }
  return manifest;
}

function resolveRelayUrl(fileName) {
  return new URL(fileName, relay.manifestUrl).toString();
}

async function fetchRelayStatus(manifest) {
  if (!manifest.status_file_name) return null;
  try {
    const { text } = await fetchTextAndSha256(resolveRelayUrl(manifest.status_file_name));
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function relayEntryFor(playbackData, manifest) {
  return {
    key: "external",
    label: playbackData.metadata?.dataset_id || manifest.source_run_id,
    description: playbackData.metadata?.notes || "Near-live read-only relay playback.",
    observe: "Follow the verified normalized playback projection as source boundaries arrive.",
    status: "near-live relay",
    semantics: playbackData.metadata?.semantics_mode || manifest.semantics_mode || "partial_agents",
    focusStart: playbackData.metadata?.first_tick ?? 0,
    focusEnd: playbackData.metadata?.last_tick ?? 0,
  };
}

function installRelayPlayback(data, provenance, manifest, status) {
  const priorPlaying = playing;
  const priorTick = currentTick;
  const priorSelectedLineageId = selectedLineageId;
  const priorSelectedAgentId = selectedAgentId;
  const priorIsolated = isolated;
  const nextPlayback = new Playback(data);
  const nextTick = relay.followLive ? nextPlayback.lastTick : Math.max(nextPlayback.firstTick, Math.min(nextPlayback.lastTick, priorTick));
  externalPlayback = { fileName: manifest.playback_file_name, data, outputSha256: manifest.playback_sha256 };
  const binding = verifyProvenanceForCurrentPlayback(provenance);
  externalProvenance = { fileName: manifest.provenance_file_name, data: provenance, binding, verified: true };
  currentScenario = relayEntryFor(data, manifest);
  playback = nextPlayback;
  currentTick = nextTick;
  playing = priorPlaying;
  selectedLineageId = priorSelectedLineageId && playback.lineageById.has(priorSelectedLineageId) ? priorSelectedLineageId : null;
  selectedAgentId = priorSelectedAgentId && playback.agentById.has(priorSelectedAgentId) ? priorSelectedAgentId : null;
  isolated = priorIsolated && Boolean(selectedLineageId);
  relay.selectionNotice = priorSelectedLineageId && !selectedLineageId ? "Prior selection is unavailable in the latest relay playback." : null;
  physicsAccumulator = 0;
  setExternalDatasetOption(currentScenario.label);
  elements.datasetSelect.disabled = true;
  elements.returnGalleryButton.hidden = false;
  system.configure(playback);
  system.setOptions(optionValue());
  renderer.resize();
  currentState = playback.stateAt(currentTick);
  system.rebuild(currentState);
  elements.timelineRange.min = playback.firstTick;
  elements.timelineRange.max = playback.lastTick;
  elements.timelineRange.value = currentState.tick;
  elements.datasetSubtitle.textContent = `${currentScenario.label} · relay sequence ${manifest.sequence_number}`;
  elements.qualityBadge.textContent = `${status?.state || manifest.source_state || "near-live relay"}`;
  refreshLineageSelect();
  if (selectedLineageId) elements.lineageSelect.value = selectedLineageId;
  applySemanticState();
  updateShowcaseOverlay();
  relay.active = true;
  relay.sequence = manifest.sequence_number;
  relay.sourceRunId = manifest.source_run_id;
  relay.sourceState = status?.state || manifest.source_state || "watching";
  relay.sourceBoundary = status?.source_boundary ?? manifest.latest_completed_source_boundary;
  relay.playbackBoundary = status?.playback_boundary ?? manifest.latest_exported_playback_boundary;
  relay.sourceLag = status?.source_lag ?? manifest.source_lag ?? 0;
  relay.warningCount = (status?.incomplete_artifact_warnings || manifest.warnings || []).length;
  relay.lastSuccessAt = status?.last_successful_update || manifest.last_successful_update || new Date().toISOString();
  relay.hashVerified = true;
  relay.error = null;
  updatePanel();
}

async function pollRelayOnce() {
  if (!relay.manifestUrl) throw new Error("Relay manifest URL is empty.");
  if (relay.inFlight) return { skipped: true };
  relay.inFlight = true;
  relay.lastPollAt = new Date().toISOString();
  try {
    const manifestResult = await fetchTextAndSha256(relay.manifestUrl);
    const manifest = validateRelayManifest(JSON.parse(manifestResult.text));
    relay.pollMs = Math.max(1000, Math.round(Number(manifest.poll_seconds || 2) * 1000));
    const status = await fetchRelayStatus(manifest);
    relay.sourceState = status?.state || manifest.source_state || "watching";
    relay.sourceBoundary = status?.source_boundary ?? manifest.latest_completed_source_boundary;
    relay.playbackBoundary = status?.playback_boundary ?? manifest.latest_exported_playback_boundary;
    relay.sourceLag = status?.source_lag ?? manifest.source_lag ?? 0;
    relay.warningCount = (status?.incomplete_artifact_warnings || manifest.warnings || []).length;
    if (relay.sequence !== null && manifest.sequence_number <= relay.sequence) {
      relay.error = null;
      updateRelayPanel();
      return { updated: false, manifest };
    }
    const playbackResult = await fetchTextAndSha256(resolveRelayUrl(manifest.playback_file_name));
    if (playbackResult.sha256 !== manifest.playback_sha256) throw new Error("Relay playback hash mismatch.");
    const provenanceResult = await fetchTextAndSha256(resolveRelayUrl(manifest.provenance_file_name));
    if (provenanceResult.sha256 !== manifest.provenance_sha256) throw new Error("Relay provenance hash mismatch.");
    const data = JSON.parse(playbackResult.text);
    const provenance = JSON.parse(provenanceResult.text);
    installRelayPlayback(data, provenance, manifest, status);
    return { updated: true, manifest };
  } catch (error) {
    relay.hashVerified = false;
    relay.error = error.message;
    relay.sourceState = "integrity_error";
    updateRelayPanel();
    return { updated: false, error: error.message };
  } finally {
    relay.inFlight = false;
  }
}

function scheduleRelayPoll() {
  if (!relay.active) return;
  clearTimeout(relay.timer);
  relay.timer = setTimeout(() => {
    pollRelayOnce().finally(scheduleRelayPoll);
  }, Math.max(1000, relay.pollMs));
}

async function connectRelay({ mode = "relay", poll = true } = {}) {
  const url = elements.relayManifestInput.value.trim();
  if (!url) throw new Error("Enter a relay manifest URL.");
  stopShowcase();
  setMode(mode);
  clearTimeout(relay.timer);
  relay.active = true;
  relay.manifestUrl = new URL(url, window.location.href).toString();
  relay.followLive = elements.followLiveToggle.checked;
  relay.error = null;
  await pollRelayOnce();
  if (poll) scheduleRelayPoll();
  else relay.active = false;
}

function disconnectRelay() {
  clearTimeout(relay.timer);
  relay.timer = null;
  relay.active = false;
  relay.inFlight = false;
  relay.sourceState = "stopped";
  updateRelayPanel();
}

function updateRelayPanel() {
  if (!elements.relayStateValue) return;
  elements.relayStateValue.textContent = relay.error ? `error: ${relay.error}` : relay.sourceState;
  elements.relayRunValue.textContent = relay.sourceRunId || "-";
  elements.relayBoundaryValue.textContent = relay.playbackBoundary === null || relay.playbackBoundary === undefined ? "-" : `${relay.playbackBoundary}`;
  elements.relayLagValue.textContent = relay.sourceLag === null || relay.sourceLag === undefined ? "-" : `${relay.sourceLag}`;
  elements.relaySequenceValue.textContent = relay.sequence === null || relay.sequence === undefined ? "-" : `${relay.sequence}`;
  elements.relayVerifiedValue.textContent = relay.hashVerified ? "yes" : "no";
  elements.relayWarningValue.textContent = `${relay.warningCount}`;
}

function refreshLineageSelect() {
  elements.lineageSelect.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "No lineage selected";
  elements.lineageSelect.appendChild(all);
  for (const lineage of playback.lineages) {
    const option = document.createElement("option");
    option.value = lineage.lineage_id;
    option.textContent = lineage.display_label || lineage.lineage_id;
    elements.lineageSelect.appendChild(option);
  }
}

function updateState(nextTick, rebuild = false, refreshPanel = true) {
  currentTick = Math.max(playback.firstTick, Math.min(playback.lastTick, Math.round(nextTick)));
  currentState = playback.stateAt(currentTick);
  if (rebuild) {
    physicsAccumulator = 0;
    system.rebuild(currentState);
  } else {
    system.syncToState(currentState);
  }
  if (refreshPanel) {
    elements.timelineRange.value = currentTick;
    updatePanel();
  }
}

function publishMetrics() {
  window.__LLC_METRICS = {
  dataset_id: playback?.metadata.dataset_id,
  scenario_key: currentScenario.key,
    app_mode: appMode,
    source_classification: playback?.metadata.source_classification || null,
    time_unit: playback?.metadata.time_unit || null,
    branch_semantics: playback?.metadata.branch_semantics || null,
    external_loaded: Boolean(externalPlayback),
    provenance_verified: Boolean(externalProvenance?.verified),
    provenance_output_sha256: externalProvenance?.data?.output_sha256 || null,
    semantics_mode: playback?.metadata.semantics_mode,
    tick: currentTick,
    living: currentState ? currentState.living.length : 0,
    lineages: currentState ? currentState.lineageCounts.size : 0,
    branches: currentState ? currentState.branchCounts.size : 0,
    largest_share: currentState ? currentState.largestLineageShare : 0,
    fps: fpsSmoother,
    physics_ms: physicsMs,
    render_ms: renderMs,
    selected_lineage_id: selectedLineageId,
    isolated,
    showcase_active: showcase.active,
    relay_active: relay.active,
    relay_manifest_url: relay.manifestUrl,
    relay_sequence: relay.sequence,
    relay_source_state: relay.sourceState,
    relay_source_boundary: relay.sourceBoundary,
    relay_playback_boundary: relay.playbackBoundary,
    relay_source_lag: relay.sourceLag,
    relay_follow_live: relay.followLive,
    relay_last_poll_at: relay.lastPollAt,
    relay_last_success_at: relay.lastSuccessAt,
    relay_hash_verified: relay.hashVerified,
    relay_error: relay.error,
  };
}

function updatePanel() {
  elements.tickValue.textContent = `tick ${currentTick}`;
  elements.generationValue.textContent = playback ? playback.generationForTick(currentTick) : "";
  elements.livingMetric.textContent = currentState ? currentState.living.length.toLocaleString() : "0";
  elements.lineageMetric.textContent = currentState ? currentState.lineageCounts.size.toString() : "0";
  elements.branchMetric.textContent = currentState ? currentState.branchCounts.size.toString() : "0";
  elements.shareMetric.textContent = currentState ? `${Math.round(currentState.largestLineageShare * 100)}%` : "0%";
  elements.fpsMetric.textContent = Math.round(fpsSmoother).toString();
  elements.physicsMetric.textContent = `${physicsMs.toFixed(1)} ms`;
  elements.renderMetric.textContent = `${renderMs.toFixed(1)} ms`;
  elements.playButton.textContent = playing ? "Pause" : "Play";
  elements.endButton.textContent = "Jump to end";
  elements.speedValue.textContent = `${speed.toFixed(2)}x`;
  elements.showcaseButton.textContent = showcase.active ? (playing ? "Pause Showcase" : "Resume Showcase") : "Showcase";
  updateRelayPanel();
  renderLineageLegend();
  publishMetrics();
  if (selectedAgentId && playback) {
    const agent = playback.agentById.get(selectedAgentId);
    elements.selectionDetails.innerHTML = agent ? [
      `<strong>${agent.agent_id}</strong>`,
      `lineage: ${agent.lineage_id}`,
      `branch: ${agent.branch_id}`,
      `born: ${agent.birth_tick}`,
      `death: ${agent.death_tick ?? "not recorded"}`,
      agent.death_interval ? `death interval: ${agent.death_interval.join(" to ")}` : "",
      agent.death_tick_source ? `death source: ${agent.death_tick_source}` : "",
      `parents: ${(agent.parent_agent_ids || []).join(", ") || "none"}`,
    ].filter(Boolean).join("<br>") : "No particle selected.";
  } else if (selectedLineageId) {
    const count = currentState?.lineageCounts.get(selectedLineageId) || 0;
    const share = currentState?.living.length ? Math.round((count / currentState.living.length) * 100) : 0;
    elements.selectionDetails.textContent = `Selected lineage: ${selectedLineageId} · ${count.toLocaleString()} particles · ${share}%`;
  } else if (relay.selectionNotice) {
    elements.selectionDetails.textContent = relay.selectionNotice;
  } else {
    elements.selectionDetails.textContent = "No particle selected.";
  }
}

function updateShowcaseOverlay() {
  if (!showcase.active) {
    elements.showcaseOverlay.hidden = true;
    elements.showcaseOverlay.textContent = "";
    return;
  }
  elements.showcaseOverlay.hidden = false;
  elements.showcaseOverlay.innerHTML = "";
  const title = document.createElement("strong");
  title.textContent = currentScenario.label;
  const body = document.createElement("span");
  body.textContent = currentScenario.observe;
  elements.showcaseOverlay.append(title, body);
}

async function startShowcase(index = 0) {
  showcase.active = true;
  showcase.index = index;
  const entry = DATASETS[showcase.index];
  showcase.playUntil = entry.focusEnd;
  speed = 2.25;
  elements.speedRange.value = speed;
  await loadDataset(entry.key, { tick: entry.focusStart, applyInitialParams: false });
  playing = true;
  updatePanel();
}

function stopShowcase() {
  showcase.active = false;
  showcase.transitioning = false;
  showcase.playUntil = null;
  updateShowcaseOverlay();
  updatePanel();
}

async function advanceShowcase() {
  if (showcase.transitioning) return;
  showcase.transitioning = true;
  playing = false;
  let next = showcase.index + 1;
  if (next >= DATASETS.length) {
    if (!elements.loopShowcaseToggle.checked) {
      stopShowcase();
      return;
    }
    next = 0;
  }
  await new Promise((resolve) => setTimeout(resolve, 300));
  showcase.transitioning = false;
  await startShowcase(next);
}

function animationLoop(now) {
  const dt = Math.min(80, now - lastFrame);
  lastFrame = now;
  if (playing && playback) {
    const advance = (dt / 1000) * speed * 10;
    const limit = showcase.active ? Math.min(showcase.playUntil ?? playback.lastTick, playback.lastTick) : playback.lastTick;
    if (currentTick + advance >= limit) {
      updateState(limit, false, false);
      if (showcase.active) advanceShowcase().catch(reportLoadError);
      else {
        playing = false;
        updateState(playback.lastTick);
      }
    } else {
      updateState(currentTick + advance, false, false);
    }
  }
  if (currentState) {
    physicsAccumulator += dt;
    physicsMs = 0;
    let steps = 0;
    while (physicsAccumulator >= FIXED_DT && steps < 6) {
      physicsMs += system.step(FIXED_DT, currentState);
      physicsAccumulator -= FIXED_DT;
      steps += 1;
    }
    if (steps === 6) physicsAccumulator = 0;
    renderMs = renderer.render({ selectedLineageId, selectedAgentId, isolated, highDensity: currentState.living.length > 2600 });
    fpsSmoother = fpsSmoother * 0.92 + (1000 / Math.max(1, dt)) * 0.08;
    publishMetrics();
    if (now - lastUiRefresh > 150) {
      lastUiRefresh = now;
      elements.timelineRange.value = currentTick;
      updatePanel();
    }
  }
  requestAnimationFrame(animationLoop);
}

function handleCanvasClick(event) {
  const rect = canvas.getBoundingClientRect();
  const hit = renderer.hitTest(event.clientX - rect.left, event.clientY - rect.top);
  if (hit) {
    selectedAgentId = hit.id;
    selectedLineageId = hit.lineageId;
    elements.lineageSelect.value = selectedLineageId;
  } else {
    selectedAgentId = null;
  }
  updatePanel();
}

function reportLoadError(error) {
  elements.datasetSubtitle.textContent = error.message;
  console.error(error);
}

async function initializeFromUrl() {
  const relayUrl = initialParams.get("relay_manifest") || initialParams.get("relay");
  await loadDataset(initialParams.get("dataset") || "balanced");
  if (!relayUrl) return;
  stopShowcase();
  setMode("relay");
  elements.relayManifestInput.value = relayUrl;
  const followLive = initialParams.get("follow_live");
  if (followLive !== null) {
    relay.followLive = followLive !== "0";
    elements.followLiveToggle.checked = relay.followLive;
  }
  await connectRelay();
}

window.__LLC_TEST_API = {
  scenarios: () => DATASETS.map((entry) => ({ ...entry, stats: scenarioStats.get(entry.key) || null })),
  load: async (key, tick = null) => {
    const entry = scenarioByKey(key);
    await loadDataset(entry.key, { tick: tick ?? entry.focusStart, applyInitialParams: false });
    publishMetrics();
    return window.__LLC_METRICS;
  },
  loadData: async (data, label = "test external playback") => {
    externalProvenance = null;
    externalPlayback = { fileName: label, data, outputSha256: await hashPlaybackData(data) };
    const entry = {
      key: "external",
      label,
      description: data.metadata?.notes || "Test-loaded playback",
      observe: "Test-loaded playback",
      status: data.metadata?.source_kind || "external playback",
      semantics: data.metadata?.semantics_mode || "partial_agents",
      focusStart: data.metadata?.first_tick ?? 0,
      focusEnd: data.metadata?.last_tick ?? 0,
    };
    installPlayback(data, entry);
    elements.datasetSelect.disabled = true;
    elements.returnGalleryButton.hidden = false;
    publishMetrics();
    return window.__LLC_METRICS;
  },
  loadProvenanceData: (data) => {
    const binding = verifyProvenanceForCurrentPlayback(data);
    externalProvenance = { fileName: "test provenance", data, binding, verified: true };
    applySemanticState();
    updatePanel();
    return { output_sha256: externalProvenance?.data?.output_sha256 || null, verified: externalProvenance?.verified || false, warningText: elements.semanticNotice.textContent };
  },
  connectRelay: async (manifestUrl) => {
    setMode("relay");
    elements.relayManifestInput.value = manifestUrl;
    await connectRelay();
    publishMetrics();
    return window.__LLC_METRICS;
  },
  pollRelayOnce: async () => {
    const result = await pollRelayOnce();
    publishMetrics();
    return { result, metrics: window.__LLC_METRICS };
  },
  relayStatus: () => ({ ...relay }),
  setFollowLive: (value) => {
    relay.followLive = Boolean(value);
    elements.followLiveToggle.checked = relay.followLive;
    publishMetrics();
    return window.__LLC_METRICS;
  },
  loadInvalidData: (data) => {
    const before = window.__LLC_METRICS;
    try {
      installPlayback(data, { key: "invalid", label: "invalid", description: "", observe: "", status: "", semantics: "partial_agents", focusStart: 0, focusEnd: 0 });
    } catch (error) {
      publishMetrics();
      return { rejected: true, message: error.message, before, after: window.__LLC_METRICS };
    }
    return { rejected: false };
  },
  setTick: (tick, rebuild = true) => {
    updateState(tick, rebuild, true);
    publishMetrics();
    return window.__LLC_METRICS;
  },
  stepFixed: (steps = 1) => {
    if (!currentState) return null;
    let total = 0;
    for (let i = 0; i < steps; i += 1) total += system.step(FIXED_DT, currentState);
    physicsMs = total;
    renderMs = renderer.render({ selectedLineageId, selectedAgentId, isolated, highDensity: currentState.living.length > 2600 });
    publishMetrics();
    return system.snapshot();
  },
  advanceTicks: (ticks = 1, stepsPerTick = 2) => {
    for (let i = 0; i < ticks; i += 1) {
      updateState(currentTick + 1, false, false);
      for (let s = 0; s < stepsPerTick; s += 1) system.step(FIXED_DT, currentState);
    }
    renderMs = renderer.render({ selectedLineageId, selectedAgentId, isolated, highDensity: currentState.living.length > 2600 });
    publishMetrics();
    return system.snapshot();
  },
  snapshot: () => system.snapshot(),
  metrics: () => window.__LLC_METRICS,
};

setDatasetOptions();
renderScenarioGallery();
setMode("scenarios");
bindControls(elements, {
  onPlayToggle: () => { playing = !playing; updatePanel(); },
  onReset: () => {
    playing = false;
    selectedAgentId = null;
    const tick = showcase.active ? currentScenario.focusStart : playback.firstTick;
    updateState(tick, true);
  },
  onEnd: () => { playing = false; updateState(playback.lastTick, true); },
  onShowcaseToggle: () => {
    if (showcase.active) {
      playing = !playing;
      updatePanel();
    } else {
      startShowcase(0).catch(reportLoadError);
    }
  },
  onShowcaseSkip: () => {
    if (showcase.active) advanceShowcase().catch(reportLoadError);
    else startShowcase(0).catch(reportLoadError);
  },
  onScenariosMode: () => {
    disconnectRelay();
    setMode("scenarios");
    if (currentScenario.key === "external") loadDataset("balanced").catch(reportLoadError);
  },
  onSavedMode: () => {
    disconnectRelay();
    setMode("saved");
  },
  onRelayMode: () => {
    stopShowcase();
    setMode("relay");
  },
  onRelayConnect: () => { connectRelay().catch(reportLoadError); },
  onRelayDisconnect: disconnectRelay,
  onFollowLive: (value) => {
    relay.followLive = value;
    if (value && relay.active && playback) updateState(playback.lastTick, true);
  },
  onDataset: (key) => {
    disconnectRelay();
    setMode("scenarios");
    stopShowcase();
    loadDataset(key).catch(reportLoadError);
  },
  onPlaybackFile: (file) => {
    disconnectRelay();
    setMode("saved");
    loadExternalPlayback(file).catch(reportLoadError);
  },
  onProvenanceFile: (file) => { loadExternalProvenance(file).catch(reportLoadError); },
  onPlaybackSearch: (query) => {
    playbackLibrary.query = query;
    renderPlaybackLibrary();
  },
  onPlaybackLibrarySelect: selectLibraryPlayback,
  onPlaybackLibraryPlay: () => { playSelectedLibraryPlayback().catch(reportLoadError); },
  onPlaybackLibraryRefresh: () => { refreshPlaybackLibrary(); },
  onReturnGallery: () => {
    disconnectRelay();
    externalPlayback = null;
    externalProvenance = null;
    setDatasetOptions();
    elements.datasetSelect.disabled = false;
    elements.returnGalleryButton.hidden = true;
    setMode("scenarios");
    loadDataset(currentScenario.key === "external" ? "balanced" : currentScenario.key).catch(reportLoadError);
  },
  onScrub: (tick) => {
    playing = false;
    selectedAgentId = null;
    if (relay.active) {
      relay.followLive = false;
      elements.followLiveToggle.checked = false;
    }
    updateState(tick, true);
  },
  onSpeed: (value) => { speed = value; updatePanel(); },
  onViewChange: () => { system.setOptions(optionValue()); },
  onLineage: (lineageId) => { selectedLineageId = lineageId || null; selectedAgentId = null; updatePanel(); },
  onIsolate: () => { if (selectedLineageId) isolated = true; },
  onShowAll: () => { isolated = false; selectedLineageId = null; selectedAgentId = null; elements.lineageSelect.value = ""; updatePanel(); },
});

canvas.addEventListener("click", handleCanvasClick);
window.addEventListener("resize", () => {
  renderer.resize();
});

initializeFromUrl().catch(reportLoadError);
preloadScenarioStats();
refreshPlaybackLibrary();
requestAnimationFrame(animationLoop);
