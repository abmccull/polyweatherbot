import { philosopherThreatDefs, type FactionId, type PhilosopherThreatDef } from "@the-oracle/content";

import type { FactionState, GameState, PhilosophersState, PhilosopherThreatLevel } from "./gameState";

function hash(seed: number): number {
  let value = seed + 0x6d2b79f5;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function agendaDomain(agenda: FactionState["currentAgenda"]): PhilosopherThreatDef["domain"] {
  if (agenda === "trade") {
    return "economic";
  }
  if (agenda === "faith") {
    return "spiritual";
  }
  return "military";
}

function worldviewForFaction(state: Pick<GameState, "worldGeneration">, factionId: FactionId): { worldview: string; pressure: number; unrest: number } {
  const controlledRegion = [...state.worldGeneration.regions]
    .filter((region) => region.controllingFactionId === factionId)
    .sort((left, right) => right.pressure - left.pressure || right.unrest - left.unrest || left.id.localeCompare(right.id))[0];
  const fallbackRegion = [...state.worldGeneration.regions]
    .sort((left, right) => right.pressure - left.pressure || right.unrest - left.unrest || left.id.localeCompare(right.id))[0];
  const region = controlledRegion ?? fallbackRegion;

  return {
    worldview: region?.philosophy ?? "Civic piety",
    pressure: region?.pressure ?? 38,
    unrest: region?.unrest ?? 26
  };
}

function philosopherAffinity(definition: PhilosopherThreatDef, faction: FactionState, worldview: string): number {
  let score = 0;

  if (definition.preferredProfiles.includes(faction.profile)) {
    score += 10;
  }
  if (definition.favoredAgenda === faction.currentAgenda) {
    score += 8;
  }
  if (definition.domain === agendaDomain(faction.currentAgenda)) {
    score += 4;
  }
  if (definition.preferredWorldviews.includes(worldview)) {
    score += 8;
  }

  return score;
}

export function philosopherThreatStageForPressure(definition: PhilosopherThreatDef, pressure: number): PhilosopherThreatLevel {
  if (pressure >= definition.stageThresholds.crisis) {
    return "crisis";
  }
  if (pressure >= definition.stageThresholds.sect) {
    return "sect";
  }
  if (pressure >= definition.stageThresholds.circle) {
    return "circle";
  }
  if (pressure >= Math.max(30, definition.stageThresholds.circle - 12)) {
    return "rumor";
  }
  return "dormant";
}

export function philosopherThreatStageRank(stage: PhilosopherThreatLevel): number {
  switch (stage) {
    case "rumor":
      return 1;
    case "circle":
      return 2;
    case "sect":
      return 3;
    case "crisis":
      return 4;
    default:
      return 0;
  }
}

export function createInitialPhilosophersState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">
): PhilosophersState {
  const byFaction = Object.fromEntries(
    Object.values(state.factions)
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((faction, index) => {
        const regionalContext = worldviewForFaction(state, faction.id);
        const selectedDefinition = [...philosopherThreatDefs].sort((left, right) => {
          const affinity = philosopherAffinity(right, faction, regionalContext.worldview) - philosopherAffinity(left, faction, regionalContext.worldview);
          if (affinity !== 0) {
            return affinity;
          }
          const leftRoll = hash(state.worldSeed + index * 31 + left.name.length * 13 + regionalContext.pressure);
          const rightRoll = hash(state.worldSeed + index * 31 + right.name.length * 13 + regionalContext.pressure);
          if (leftRoll !== rightRoll) {
            return rightRoll - leftRoll;
          }
          return left.id.localeCompare(right.id);
        })[0]!;
        const affinity = philosopherAffinity(selectedDefinition, faction, regionalContext.worldview);
        const influence = clamp(
          Math.round(
            12
            + affinity * 1.1
            + Math.max(0, regionalContext.pressure - 30) * 0.18
            + Math.max(0, regionalContext.unrest - 20) * 0.12
          ),
          8,
          68
        );
        const suspicion = clamp(
          Math.round(
            6
            + Math.max(0, 54 - faction.credibility) * 0.18
            + Math.max(0, faction.dependence - 24) * 0.14
            + (faction.currentAgenda === "faith" ? 3 : 0)
            + (selectedDefinition.domain === "spiritual" ? 2 : 0)
          ),
          4,
          52
        );
        const pressure = clamp(
          Math.round(
            influence * 0.58
            + suspicion * 0.42
            + Math.max(0, regionalContext.pressure - 38) * 0.3
          ),
          0,
          100
        );
        const stage = philosopherThreatStageForPressure(selectedDefinition, pressure);

        return [
          faction.id,
          {
            philosopherId: selectedDefinition.id,
            worldview: regionalContext.worldview,
            influence,
            suspicion,
            pressure,
            stage,
            active: stage !== "dormant"
          }
        ];
      })
  ) as PhilosophersState["byFaction"];

  return {
    byFaction,
    spotlightFactionIds: Object.entries(byFaction)
      .sort((left, right) =>
        right[1].pressure - left[1].pressure
        || philosopherThreatStageRank(right[1].stage) - philosopherThreatStageRank(left[1].stage)
        || left[0].localeCompare(right[0])
      )
      .slice(0, 3)
      .map(([factionId]) => factionId as FactionId)
  };
}

export function normalizePhilosophersState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions" | "philosophers">
): PhilosophersState {
  if (!state.philosophers) {
    return createInitialPhilosophersState(state);
  }

  const fallback = createInitialPhilosophersState(state);

  return {
    byFaction: Object.fromEntries(
      (Object.keys(state.factions) as FactionId[]).map((factionId) => [
        factionId,
        state.philosophers?.byFaction[factionId] ?? fallback.byFaction[factionId]
      ])
    ) as PhilosophersState["byFaction"],
    spotlightFactionIds: state.philosophers.spotlightFactionIds?.length
      ? state.philosophers.spotlightFactionIds.filter((factionId) => Boolean(state.factions[factionId]))
      : fallback.spotlightFactionIds
  };
}
