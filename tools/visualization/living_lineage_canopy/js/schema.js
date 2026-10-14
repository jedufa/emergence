export const SCHEMA_VERSION = "eiel_lineage_playback_v1";
const PROJECTED_DEATH_SOURCES = new Set(["checkpoint_boundary_projection", "terminal_boundary_projection"]);
const DEATH_TICK_SOURCES = new Set(["checkpoint_boundary_projection", "terminal_boundary_projection", "typed_lifecycle_death_event", "implicit_lifecycle_record", "not_recorded"]);

export function validatePlayback(data) {
  if (!data || typeof data !== "object") throw new Error("Playback must be an object");
  const metadata = data.metadata || {};
  if (metadata.schema_version !== SCHEMA_VERSION) {
    throw new Error(`Unsupported schema version: ${metadata.schema_version}`);
  }
  if (!["actual_agents", "partial_agents", "representative_particles"].includes(metadata.semantics_mode)) {
    throw new Error(`Unsupported semantics mode: ${metadata.semantics_mode}`);
  }
  if (metadata.source_classification && !["EXP-A", "EXP-B", "EXP-C", "EXP-D"].includes(metadata.source_classification)) {
    throw new Error(`Unsupported source classification: ${metadata.source_classification}`);
  }
  if (!Array.isArray(data.lineages) || !Array.isArray(data.branches) || !Array.isArray(data.agents)) {
    throw new Error("Playback requires lineages, branches, and agents arrays");
  }
  const lineageIds = new Set(data.lineages.map((item) => item.lineage_id));
  const branchIds = new Set(data.branches.map((item) => item.branch_id));
  const agentIds = new Set();
  for (const agent of data.agents) {
    if (agentIds.has(agent.agent_id)) throw new Error(`Duplicate agent id: ${agent.agent_id}`);
    agentIds.add(agent.agent_id);
    if (!lineageIds.has(agent.lineage_id)) throw new Error(`Missing lineage: ${agent.lineage_id}`);
    if (!branchIds.has(agent.branch_id)) throw new Error(`Missing branch: ${agent.branch_id}`);
    if (agent.death_tick !== null && agent.death_tick !== undefined && agent.death_tick < agent.birth_tick) {
      throw new Error(`Invalid lifecycle ordering: ${agent.agent_id}`);
    }
    const rawDeathSource = agent.death_tick_source;
    const deathSource = rawDeathSource ?? (agent.death_tick !== null && agent.death_tick !== undefined ? "implicit_lifecycle_record" : "not_recorded");
    if (!DEATH_TICK_SOURCES.has(deathSource)) throw new Error(`Unsupported death source: ${agent.agent_id}`);
    if (agent.death_interval !== null && agent.death_interval !== undefined) {
      if (!Array.isArray(agent.death_interval) || agent.death_interval.length !== 2 || !agent.death_interval.every(Number.isInteger)) {
        throw new Error(`Invalid death interval: ${agent.agent_id}`);
      }
      const [lower, upper] = agent.death_interval;
      if (upper < lower || upper < agent.birth_tick) throw new Error(`Invalid death interval bounds: ${agent.agent_id}`);
      if (agent.death_tick !== null && agent.death_tick !== undefined && agent.death_tick !== upper) {
        throw new Error(`Death tick must match interval upper bound: ${agent.agent_id}`);
      }
    }
    if (PROJECTED_DEATH_SOURCES.has(deathSource)) {
      if (!agent.death_interval || agent.death_tick === null || agent.death_tick === undefined) {
        throw new Error(`Projected death requires interval and death tick: ${agent.agent_id}`);
      }
      if (metadata.semantics_mode === "actual_agents" || metadata.source_classification === "EXP-A") {
        throw new Error("Projected deaths require partial non-EXP-A semantics");
      }
    }
    if (deathSource === "typed_lifecycle_death_event" && (agent.death_tick === null || agent.death_tick === undefined)) {
      throw new Error(`Typed death source requires death tick: ${agent.agent_id}`);
    }
    if (rawDeathSource === "not_recorded" && agent.death_tick !== null && agent.death_tick !== undefined) {
      throw new Error(`Unrecorded death source cannot have death tick: ${agent.agent_id}`);
    }
  }
  return data;
}
