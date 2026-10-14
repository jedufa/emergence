import { validatePlayback } from "./schema.js";

export class Playback {
  constructor(data) {
    this.data = validatePlayback(data);
    this.metadata = data.metadata;
    this.firstTick = data.metadata.first_tick;
    this.lastTick = data.metadata.last_tick;
    this.lineages = data.lineages;
    this.branches = data.branches;
    this.agents = data.agents.slice().sort((a, b) => a.birth_tick - b.birth_tick);
    this.agentById = new Map(this.agents.map((agent) => [agent.agent_id, agent]));
    this.lineageById = new Map(this.lineages.map((lineage) => [lineage.lineage_id, lineage]));
    this.branchById = new Map(this.branches.map((branch) => [branch.branch_id, branch]));
  }

  stateAt(rawTick) {
    const tick = Math.max(this.firstTick, Math.min(this.lastTick, Math.round(rawTick)));
    const living = [];
    const lineageCounts = new Map();
    const branchCounts = new Map();
    for (const agent of this.agents) {
      if (agent.birth_tick > tick) break;
      if (agent.death_tick === null || agent.death_tick === undefined || agent.death_tick > tick) {
        living.push(agent);
        lineageCounts.set(agent.lineage_id, (lineageCounts.get(agent.lineage_id) || 0) + 1);
        branchCounts.set(agent.branch_id, (branchCounts.get(agent.branch_id) || 0) + 1);
      }
    }
    let largest = 0;
    for (const count of lineageCounts.values()) largest = Math.max(largest, count);
    return {
      tick,
      living,
      lineageCounts,
      branchCounts,
      largestLineageShare: living.length ? largest / living.length : 0,
    };
  }

  generationForTick(tick) {
    const mapping = this.metadata.generation_mapping;
    if (!mapping || !mapping.ticks_per_generation) return "";
    return `gen ${Math.floor(tick / mapping.ticks_per_generation)}`;
  }
}
