import {
  rivalOracleDefById,
  rivalOracleDefs,
  rivalOracleOperationDefById,
  type FactionId,
  type RivalOracleDef,
  type RivalOracleOperationDef
} from "@the-oracle/content";

import type {
  EventFeedItem,
  FactionState,
  GameState,
  RivalOracleDiscovery,
  RivalOracleIncident,
  RivalOraclesState,
  RivalOracleState
} from "./gameState";

const MAX_INCIDENTS = 18;

function hash(seed: number): number {
  let value = seed + 0x6d2b79f5;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampPercent(value: number): number {
  return clamp(Math.round(value), 0, 100);
}

function clampDebt(value: number): number {
  return clamp(Math.round(value), 0, 40);
}

function prophecyDepthSignal(state: Pick<GameState, "pythia">): number {
  const tranceDepth = typeof state.pythia?.tranceDepth === "number" ? state.pythia.tranceDepth : 50;
  const clarity = typeof state.pythia?.mentalClarity === "number" ? state.pythia.mentalClarity : 55;
  const purification = typeof state.pythia?.needs?.purification === "number" ? state.pythia.needs.purification : 40;

  return clamp(Math.round(tranceDepth * 0.55 + clarity * 0.3 + (100 - purification) * 0.15), 20, 95);
}

function patronAffinity(definition: RivalOracleDef, faction: FactionState, controlsHomeRegion: boolean): number {
  let score = 0;

  if (definition.patronPool.includes(faction.id)) {
    score += 18;
  }
  if (definition.preferredPatronProfiles.includes(faction.profile)) {
    score += 12;
  }
  if (definition.preferredPatronAgendas.includes(faction.currentAgenda)) {
    score += 10;
  }
  if (controlsHomeRegion) {
    score += 8;
  }

  score += faction.favour * 0.18;
  score += faction.credibility * 0.12;
  score -= faction.debt * 0.45;

  return score;
}

function choosePatronFaction(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">,
  definition: RivalOracleDef,
  index: number
): FactionId {
  const homeRegion = state.worldGeneration.regions.find((region) => region.id === definition.homeRegionId);
  const candidates = definition.patronPool.filter((factionId) => Boolean(state.factions[factionId]));

  return [...candidates]
    .sort((leftId, rightId) => {
      const left = state.factions[leftId];
      const right = state.factions[rightId];
      const leftScore = patronAffinity(definition, left, homeRegion?.controllingFactionId === leftId)
        + hash(state.worldSeed + index * 29 + leftId.length * 11) * 6;
      const rightScore = patronAffinity(definition, right, homeRegion?.controllingFactionId === rightId)
        + hash(state.worldSeed + index * 29 + rightId.length * 11) * 6;
      return rightScore - leftScore || leftId.localeCompare(rightId);
    })[0] ?? candidates[0] ?? "athens";
}

function initialIntrigue(definition: RivalOracleDef, patron: FactionState, worldStress: number, depthSignal: number): number {
  let intrigue = definition.secrecy * 0.68 + worldStress * 0.15 - depthSignal * 0.1;

  if (patron.profile === "scheming") {
    intrigue += 8;
  }
  if (patron.currentAgenda === "succession") {
    intrigue += 6;
  }

  return clampPercent(intrigue);
}

function discoveryBand(visibility: number, intel: number): RivalOracleDiscovery {
  const combined = visibility * 0.55 + intel * 0.45;
  if (combined >= 64 || (visibility >= 60 && intel >= 56)) {
    return "confirmed";
  }
  if (combined >= 34) {
    return "suspected";
  }
  return "shadow";
}

function discoveryRank(discovery: RivalOracleDiscovery): number {
  switch (discovery) {
    case "confirmed":
      return 3;
    case "suspected":
      return 2;
    default:
      return 1;
  }
}

function totalPressure(roster: RivalOracleState[]): number {
  if (roster.length === 0) {
    return 0;
  }

  return Math.round(roster.reduce((sum, rival) => sum + rival.pressure, 0) / roster.length);
}

function chooseTargetRegionId(
  state: Pick<GameState, "worldGeneration">,
  rival: RivalOracleState,
  patron: FactionState
): string {
  const patronRegion = state.worldGeneration.regions
    .filter((region) => region.controllingFactionId === patron.id)
    .sort((left, right) => right.pressure - left.pressure || right.unrest - left.unrest || left.id.localeCompare(right.id))[0];

  return patronRegion?.id ?? rival.homeRegionId;
}

function operationScore(
  state: Pick<GameState, "clock" | "worldSeed" | "pythia">,
  rival: RivalOracleState,
  patron: FactionState,
  operation: RivalOracleOperationDef
): number {
  const depthSignal = prophecyDepthSignal(state);
  const lastExecutedDay = rival.operations.find((entry) => entry.id === operation.id)?.lastExecutedDay ?? -99;
  const daysSinceUse = state.clock.day - lastExecutedDay;
  const reusePenalty = daysSinceUse <= 3 ? 12 : daysSinceUse <= 6 ? 5 : 0;
  let score = operation.basePressure + operation.patronageShift * 0.7 - reusePenalty;

  if (operation.kind === "espionage") {
    score += rival.visibility >= 52 ? 18 : 4;
    score += rival.intel >= 44 ? 8 : 0;
    score += rival.intrigue * 0.16;
  }
  if (operation.kind === "patronage") {
    score += patron.debt >= 14 ? 14 : 4;
    score += patron.currentAgenda === "trade" || patron.currentAgenda === "succession" ? 8 : 0;
  }
  if (operation.kind === "pressure") {
    score += rival.pressure >= 48 ? 12 : 5;
    score += patron.activeConflicts.length * 4;
  }
  if (operation.domain === rival.favoredDomain) {
    score += 7;
  }

  return score + hash(state.worldSeed + state.clock.day * 17 + rival.id.length * 13 + operation.id.length * 19) * 9 - depthSignal * 0.05;
}

function buildIncidentSummary(
  state: Pick<GameState, "clock" | "worldGeneration" | "factions">,
  rival: RivalOracleState,
  operation: RivalOracleOperationDef,
  targetRegionId: string,
  discovery: RivalOracleDiscovery
): string {
  const regionLabel = state.worldGeneration.regions.find((region) => region.id === targetRegionId)?.label ?? targetRegionId;
  const patronName = state.factions[rival.patronFactionId]?.name ?? "an unseen patron";
  const rivalLabel = discovery === "shadow" ? "A rival oracle" : rival.name;
  const patronLabel = discovery === "confirmed"
    ? `for ${patronName}`
    : discovery === "suspected"
      ? `under suspected ${patronName} patronage`
      : "for an unseen patron";

  switch (operation.id) {
    case "poach-supplicants":
      return `Day ${state.clock.day}: ${rivalLabel} is drawing pilgrims and gifts away from Delphi around ${regionLabel} ${patronLabel}.`;
    case "plant-whispers":
      return `Day ${state.clock.day}: shadow traffic thickens near ${regionLabel} as ${rivalLabel.toLowerCase()} seeds whispers ${patronLabel}.`;
    case "court-patron":
      return `Day ${state.clock.day}: ${rivalLabel} tightens its purse strings in ${regionLabel}, courting fresh backing ${patronLabel}.`;
    case "counter-rite":
      return `Day ${state.clock.day}: ${rivalLabel} stages a counter-rite near ${regionLabel} ${patronLabel}, pressuring Delphi's authority in the open.`;
    default:
      return `Day ${state.clock.day}: ${rivalLabel} moves near ${regionLabel} ${patronLabel}.`;
  }
}

export function createInitialRivalOraclesState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions" | "pythia">
): RivalOraclesState {
  const depthSignal = prophecyDepthSignal(state);
  const roster = rivalOracleDefs
    .map((definition, index) => {
      const homeRegion = state.worldGeneration.regions.find((region) => region.id === definition.homeRegionId);
      const patronFactionId = choosePatronFaction(state, definition, index);
      const patron = state.factions[patronFactionId];
      const homePressure = homeRegion?.pressure ?? 40;
      const homeUnrest = homeRegion?.unrest ?? 28;

      return {
        id: definition.id,
        name: definition.name,
        title: definition.title,
        homeRegionId: definition.homeRegionId,
        favoredDomain: definition.favoredDomain,
        active: true,
        patronFactionId,
        patronage: clampPercent(36 + patron.favour * 0.18 - patron.debt * 0.3),
        pressure: clampPercent(definition.baselinePressure + homePressure * 0.14 + patron.favour * 0.06 - depthSignal * 0.08),
        visibility: clampPercent(18 + homePressure * 0.16 - definition.secrecy * 0.12 + depthSignal * 0.08),
        intel: clampPercent(14 + homeUnrest * 0.18 - definition.secrecy * 0.1 + depthSignal * 0.12),
        intrigue: initialIntrigue(definition, patron, homePressure + homeUnrest, depthSignal),
        pressureCap: definition.pressureCap,
        operations: definition.operationIds.map((operationId) => ({
          id: operationId,
          successCount: 0
        }))
      } satisfies RivalOracleState;
    })
    .sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id));

  return {
    roster,
    spotlightRivalIds: roster.slice(0, 3).map((rival) => rival.id),
    incidents: [],
    totalPressure: totalPressure(roster)
  };
}

export function normalizeRivalOraclesState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions" | "pythia" | "rivalOracles">
): RivalOraclesState {
  const fallback = createInitialRivalOraclesState(state);
  if (!state.rivalOracles) {
    return fallback;
  }

  const byId = new Map(state.rivalOracles.roster.map((rival) => [rival.id, rival]));
  const roster = fallback.roster.map((generatedRival) => {
    const saved = byId.get(generatedRival.id);
    if (!saved) {
      return generatedRival;
    }

    return {
      ...generatedRival,
      ...saved,
      patronFactionId: state.factions[saved.patronFactionId] ? saved.patronFactionId : generatedRival.patronFactionId,
      operations: generatedRival.operations.map((generatedOperation) => ({
        ...generatedOperation,
        ...(saved.operations.find((operation) => operation.id === generatedOperation.id) ?? {})
      }))
    };
  });

  return {
    roster,
    spotlightRivalIds: state.rivalOracles.spotlightRivalIds?.length
      ? state.rivalOracles.spotlightRivalIds.filter((rivalId) => roster.some((entry) => entry.id === rivalId))
      : fallback.spotlightRivalIds,
    incidents: [...(state.rivalOracles.incidents ?? [])].slice(0, MAX_INCIDENTS),
    totalPressure: typeof state.rivalOracles.totalPressure === "number"
      ? clampPercent(state.rivalOracles.totalPressure)
      : totalPressure(roster),
    lastPressureDay: state.rivalOracles.lastPressureDay
  };
}

export function advanceRivalOracles(
  state: Pick<GameState, "clock" | "worldSeed" | "worldGeneration" | "factions" | "pythia" | "rivalOracles">
): {
  rivalOracles: RivalOraclesState;
  factions: Record<FactionId, FactionState>;
  feedItems: EventFeedItem[];
  incidents: RivalOracleIncident[];
} {
  const previousState = normalizeRivalOraclesState(state);
  const nextFactions = { ...state.factions };
  const incidents: RivalOracleIncident[] = [];
  const nextRoster = previousState.roster.map((rival) => {
    const definition = rivalOracleDefById[rival.id];
    const fallbackPatronId = definition.patronPool[0] ?? "athens";
    const patron = nextFactions[rival.patronFactionId] ?? nextFactions[fallbackPatronId];
    const targetRegionId = chooseTargetRegionId(state, rival, patron);
    const targetRegion = state.worldGeneration.regions.find((region) => region.id === targetRegionId);
    const operation = [...definition.operationIds]
      .map((operationId) => rivalOracleOperationDefById[operationId])
      .sort((left, right) => operationScore(state, rival, patron, right) - operationScore(state, rival, patron, left) || left.id.localeCompare(right.id))[0]!;
    const depthSignal = prophecyDepthSignal(state);
    const patronStress = patron.debt * 0.5 + patron.activeConflicts.length * 7 + Math.max(0, 56 - patron.credibility) * 0.28;
    const regionalStress = (targetRegion?.pressure ?? 40) * 0.18 + (targetRegion?.unrest ?? 24) * 0.12;
    const concealment = rival.intrigue * 0.14 + definition.secrecy * 0.08;
    const pressureDelta = clampPercent(operation.basePressure + patronStress * 0.16 + regionalStress * 0.12 - depthSignal * 0.1);
    const nextPressure = clampPercent(clamp(rival.pressure + pressureDelta - 4, 0, rival.pressureCap));
    const nextVisibility = clampPercent(rival.visibility + operation.baseVisibility - concealment * 0.22 + Math.max(0, nextPressure - 52) * 0.12);
    const nextIntel = clampPercent(
      rival.intel
      + operation.baseIntel
      + depthSignal * 0.14
      + Math.max(0, nextVisibility - 36) * 0.18
      - (operation.kind === "espionage" ? 9 : 0)
      - definition.secrecy * 0.05
    );
    const nextPatronage = clampPercent(rival.patronage + operation.patronageShift + patronStress * 0.08 - Math.max(0, nextVisibility - 68) * 0.14);
    const nextIntrigue = clampPercent(rival.intrigue + (operation.kind === "espionage" ? 4 : 1) - depthSignal * 0.06);
    const discovery = discoveryBand(nextVisibility, nextIntel);
    const summary = buildIncidentSummary(state, rival, operation, targetRegionId, discovery);

    nextFactions[rival.patronFactionId] = {
      ...patron,
      favour: clampPercent(patron.favour + (operation.kind === "patronage" ? 2 : 1)),
      credibility: clampPercent(patron.credibility + (operation.id === "counter-rite" ? 1 : 0)),
      dependence: clampPercent(patron.dependence + (operation.kind === "espionage" ? 2 : 1)),
      debt: clampDebt(patron.debt + (operation.kind === "patronage" ? 0 : 1))
    };

    incidents.push({
      id: `rival-incident-${state.clock.day}-${rival.id}`,
      day: state.clock.day,
      rivalId: rival.id,
      operationId: operation.id,
      targetRegionId,
      targetFactionId: rival.patronFactionId,
      discovery,
      pressureDelta,
      visibility: nextVisibility,
      intel: nextIntel,
      summary
    });

    return {
      ...rival,
      patronage: nextPatronage,
      pressure: nextPressure,
      visibility: nextVisibility,
      intel: nextIntel,
      intrigue: nextIntrigue,
      lastOperationDay: state.clock.day,
      lastKnownOperationId: discovery === "shadow" ? rival.lastKnownOperationId : operation.id,
      lastTargetRegionId: targetRegionId,
      operations: rival.operations.map((entry) =>
        entry.id === operation.id
          ? {
              ...entry,
              successCount: entry.successCount + 1,
              lastExecutedDay: state.clock.day
            }
          : entry
      )
    };
  });

  const sortedRoster = [...nextRoster].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id));

  return {
    rivalOracles: {
      roster: sortedRoster,
      spotlightRivalIds: [...sortedRoster]
        .sort((left, right) =>
          right.pressure - left.pressure
          || discoveryRank(discoveryBand(right.visibility, right.intel)) - discoveryRank(discoveryBand(left.visibility, left.intel))
          || left.id.localeCompare(right.id)
        )
        .slice(0, 3)
        .map((rival) => rival.id),
      incidents: [...incidents, ...previousState.incidents].slice(0, MAX_INCIDENTS),
      totalPressure: totalPressure(sortedRoster),
      lastPressureDay: state.clock.day
    },
    factions: nextFactions,
    incidents,
    feedItems: incidents
      .filter((incident) => incident.discovery !== "shadow" || incident.pressureDelta >= 10)
      .sort((left, right) => right.pressureDelta - left.pressureDelta || left.rivalId.localeCompare(right.rivalId))
      .slice(0, 2)
      .map((incident) => ({
        id: `event-${incident.id}`,
        day: state.clock.day,
        text: incident.summary
      }))
  };
}
