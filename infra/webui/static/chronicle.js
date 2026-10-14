"use strict";

const state = {
  runs: [],
  filteredRuns: [],
  runKey: "",
  audience: "dry_humour",
  requestNumber: 0,
};

const $ = (id) => document.getElementById(id);

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function textElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = String(text ?? "");
  return node;
}

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== "" && value !== null && value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  });
  const response = await fetch(url, { credentials: "same-origin" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function runTimestamp(run) {
  const value = Number(run.modified_at_ns || 0);
  if (Number.isFinite(value) && value > 0) return value;
  const match = String(run.archive_run_id || run.run_id || "").match(/20\d{6}(?:T?\d{6})?/);
  return match ? Number(match[0].replace("T", "").padEnd(14, "0")) : 0;
}

function runLabel(run) {
  const campaign = run.campaign_id ? `${run.campaign_id} · ` : "";
  const world = run.world_id ? ` · ${run.world_id}` : "";
  const status = run.storage_kind === "verified_zip" ? " · archived" : (run.finalized ? " · finalized" : " · live");
  const compact = run.storage_summary_available === false ? " · compact" : "";
  return `${campaign}${run.archive_run_id || run.run_id || run.run_key}${world}${status}${compact}`;
}

function runSearchText(run) {
  return [
    run.campaign_id,
    run.archive_run_id,
    run.run_id,
    run.world_id,
    run.branch_id,
    run.source_commit,
    run.profile,
    run.storage_kind,
    run.run_key,
    run.run_path,
  ].filter(Boolean).join(" ").toLocaleLowerCase();
}

function matchingRuns() {
  const terms = $("runSearch").value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return state.runs.filter((run) => {
    const searchable = runSearchText(run);
    return terms.every((term) => searchable.includes(term));
  });
}

function renderRunOptions() {
  const select = $("simulationSelect");
  state.filteredRuns = matchingRuns();
  clear(select);
  state.filteredRuns.forEach((run) => {
    const option = textElement("option", "", runLabel(run));
    option.value = run.run_key;
    select.appendChild(option);
  });
  select.disabled = state.filteredRuns.length === 0;
  select.value = state.runKey;
  if (select.value !== state.runKey) select.selectedIndex = -1;
  $("runSearch").title = `Showing ${state.filteredRuns.length} of ${state.runs.length} dialogue-ready Chronicle runs.`;
}

function systemBubble(message) {
  return textElement("div", "system-bubble", message);
}

function renderMessage(message) {
  const speaker = message.speaker === "creator" ? "creator" : "scientist";
  const bubble = textElement("article", `bubble ${speaker}`, "");
  bubble.appendChild(textElement("div", "speaker", message.speaker_label || speaker));
  bubble.appendChild(textElement("div", "message", message.text || ""));
  bubble.appendChild(textElement("div", "meta", `Generation ${message.generation}`));
  return bubble;
}

async function loadDialogue() {
  const root = $("dialogue");
  clear(root);
  if (!state.runKey) {
    root.appendChild(systemBubble("No Chronicle simulations are available."));
    return;
  }
  root.appendChild(systemBubble("Loading the conversation…"));
  const requestNumber = ++state.requestNumber;
  try {
    const data = await api("/api/chronicle/dialogue", {
      run_key: state.runKey,
      audience: state.audience,
      limit: 1000,
    });
    if (requestNumber !== state.requestNumber) return;
    clear(root);
    (data.messages || []).forEach((message) => root.appendChild(renderMessage(message)));
    if (!(data.messages || []).length) {
      root.appendChild(systemBubble("This Chronicle contains no dialogue-ready events."));
    } else if (data.truncated) {
      root.appendChild(systemBubble(`Showing ${data.selected_event_count} of ${data.event_count} Chronicle events.`));
    }
  } catch (error) {
    if (requestNumber !== state.requestNumber) return;
    clear(root);
    root.appendChild(systemBubble(error instanceof Error ? error.message : String(error)));
  }
}

function selectInitialRun(ready) {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("run_key");
  return ready.find((run) => run.run_key === requested) || ready[0] || null;
}

async function loadRuns() {
  const root = $("dialogue");
  root.appendChild(systemBubble("Finding Chronicle simulations…"));
  try {
    const data = await api("/api/chronicle/runs");
    state.runs = (data.runs || [])
      .filter((run) => run.status === "ready")
      .sort((left, right) => (
        runTimestamp(right) - runTimestamp(left)
        || runLabel(left).localeCompare(runLabel(right))
      ));
    const initial = selectInitialRun(state.runs);
    state.runKey = initial ? initial.run_key : "";
    renderRunOptions();
    await loadDialogue();
  } catch (error) {
    clear(root);
    root.appendChild(systemBubble(error instanceof Error ? error.message : String(error)));
  }
}

$("simulationSelect").addEventListener("change", async (event) => {
  state.runKey = event.target.value;
  const url = new URL(window.location.href);
  url.searchParams.set("run_key", state.runKey);
  history.replaceState({}, "", url);
  await loadDialogue();
});

$("runSearch").addEventListener("input", renderRunOptions);

$("audienceSelect").addEventListener("change", async (event) => {
  state.audience = event.target.value;
  await loadDialogue();
});

loadRuns();
