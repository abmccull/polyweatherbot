import { philosopherThreatDefById, type FactionId, type PhilosopherThreatDef } from "@the-oracle/content";

import type { EventFeedItem, FactionState, GameState, PhilosophersState, PhilosopherThreatLevel, PhilosopherThreatState } from "../state/gameState";
import { normalizePhilosophersState, philosopherThreatStageForPressure, philosopherThreatStageRank } from "../state/philosophers";

const EMPTY_EFFECT = {
  credibility: 0,
  favour: 0,
  debt: 0,
  dependence: 0
} as const;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampStat(value: number): number {
  return clamp(Math.round(value), 0, 100);
}

function clampDebt(value: number): number {
  return clamp(Math.round(value), 0, 40);
}

function stageFocus(definition: PhilosopherThreatDef, stage: PhilosopherThreatLevel): string {
  switch (definition.domain) {
    case "military":
      return stage === "crisis"
        ? "officers and magistrates are now testing obedience against prophetic order"
        : "captains and magistrates are debating whether strife makes a stronger city";
    case "economic":
      return stage === "crisis"
        ? "merchants and magistrates now price legitimacy above inherited rite"
        : "merchants and speakers are learning to weigh legitimacy like a market bargain";
    default:
      return stage === "crisis"
        ? "rites, legitimacy, and civic order are being contested in the open"
        : "priests and students are recasting legitimacy through doctrine and rite";
  }
}

function buildThreatNote(month: number, faction: FactionState, threat: PhilosopherThreatState): string {
  const definition = philosopherThreatDefById[threat.philosopherId];
  const focus = stageFocus(definition, threat.stage);

  switch (threat.stage) {
    case "rumor":
      return `Month ${month}: ${faction.name} hears ${definition.name}'s doctrine in private salons, and debate over legitimacy and civic order begins to spread.`;
    case "circle":
      return `Month ${month}: A ${definition.name} circle finds patrons in ${faction.name}; ${focus}.`;
    case "sect":
      return `Month ${month}: ${faction.name} hardens around a ${definition.name} sect, turning philosophy into leverage over legitimacy, rites, and public order.`;
    case "crisis":
      return `Month ${month}: ${faction.name} faces a ${definition.name} schism as ${focus}.`;
    default:
      return "";
  }
}

function withThreatEffects(faction: FactionState, threat: PhilosopherThreatState): FactionState {
  if (threat.stage === "dormant") {
    return faction;
  }

  const definition = philosopherThreatDefById[threat.philosopherId];
  const effect = definition.stageEffects[threat.stage] ?? EMPTY_EFFECT;
  let nextFaction: FactionState = {
    ...faction,
    credibility: clampStat(faction.credibility + effect.credibility),
    favour: clampStat(faction.favour + effect.favour),
    debt: clampDebt(faction.debt + effect.debt),
    dependence: clampStat(faction.dependence + effect.dependence)
  };

  if ((threat.stage === "sect" || threat.stage === "crisis") && nextFaction.currentAgenda !== definition.favoredAgenda && threat.pressure >= definition.stageThresholds.sect + 4) {
    nextFaction = {
      ...nextFaction,
      currentAgenda: definition.favoredAgenda
    };
  }

  if (threat.stage === "crisis" && (definition.domain === "economic" || definition.domain === "military")) {
    nextFaction = {
      ...nextFaction,
      tradeAccess: false
    };
  }

  return nextFaction;
}

function evolveThreat(faction: FactionState, previous: PhilosopherThreatState): PhilosopherThreatState {
  const definition = philosopherThreatDefById[previous.philosopherId];
  const debtPressure = faction.debt * definition.pressureBias.debt;
  const dependencePressure = Math.max(0, faction.dependence - 24) * definition.pressureBias.dependence;
  const lowCredibilityPressure = Math.max(0, 58 - faction.credibility) * definition.pressureBias.lowCredibility;
  const conflictPressure = faction.activeConflicts.length * definition.pressureBias.conflicts;
  const embargoPressure = faction.embargoes.length * definition.pressureBias.embargoes;
  const agendaPressure = faction.currentAgenda === definition.favoredAgenda ? definition.pressureBias.agendaAlignment : 0;
  const worldviewPressure = definition.preferredWorldviews.includes(previous.worldview) ? definition.pressureBias.worldviewAlignment : 0;
  const stabilityRelief =
    (faction.credibility >= 64 ? 4 : 0)
    + (faction.favour >= 58 ? 3 : 0)
    + (faction.debt <= 8 ? 3 : 0)
    + (faction.activeConflicts.length === 0 ? 2 : 0);
  const influence = clamp(
    Math.round(
      previous.influence
      + 1
      + worldviewPressure * 0.18
      + agendaPressure * 0.15
      + debtPressure * 0.12
      + dependencePressure * 0.12
      + conflictPressure * 0.22
      + embargoPressure * 0.18
      - stabilityRelief * 0.55
    ),
    0,
    100
  );
  const suspicion = clamp(
    Math.round(
      previous.suspicion
      + 1
      + lowCredibilityPressure * 0.16
      + conflictPressure * 0.14
      + embargoPressure * 0.1
      + (previous.stage === "dormant" ? 0 : 2)
      - stabilityRelief * 0.4
    ),
    0,
    100
  );
  const pressure = clamp(
    Math.round(
      influence * 0.45
      + suspicion * 0.35
      + debtPressure
      + dependencePressure
      + lowCredibilityPressure
      + conflictPressure
      + embargoPressure
      + agendaPressure
      + worldviewPressure
      - stabilityRelief
    ),
    0,
    100
  );
  const stage = philosopherThreatStageForPressure(definition, pressure);

  return {
    ...previous,
    influence,
    suspicion,
    pressure,
    stage,
    active: stage !== "dormant"
  };
}

export function advancePhilosopherThreats(
  state: Pick<GameState, "clock" | "worldSeed" | "worldGeneration" | "factions" | "philosophers">
): {
  philosophers: PhilosophersState;
  factions: Record<FactionId, FactionState>;
  feedItems: EventFeedItem[];
} {
  const previousState = normalizePhilosophersState(state);
  const stories: Array<{ factionId: FactionId; salience: number; note: string }> = [];
  const nextFactions = { ...state.factions };
  const nextByFaction = {} as PhilosophersState["byFaction"];

  for (const faction of Object.values(state.factions).sort((left, right) => left.id.localeCompare(right.id))) {
    const previousThreat = previousState.byFaction[faction.id];
    const nextThreat = evolveThreat(faction, previousThreat);
    const nextFaction = withThreatEffects(nextFactions[faction.id], nextThreat);
    const stageRaised = philosopherThreatStageRank(nextThreat.stage) > philosopherThreatStageRank(previousThreat.stage);
    const severe = philosopherThreatStageRank(nextThreat.stage) >= philosopherThreatStageRank("sect");

    nextByFaction[faction.id] = {
      ...nextThreat,
      lastShiftMonth: stageRaised ? state.clock.month : previousThreat.lastShiftMonth,
      lastEventDay: stageRaised || severe ? state.clock.day : previousThreat.lastEventDay
    };
    nextFactions[faction.id] = nextFaction;

    if (nextThreat.stage !== "dormant" && (stageRaised || severe)) {
      const note = buildThreatNote(state.clock.month, nextFaction, nextThreat);
      nextFactions[faction.id] = {
        ...nextFactions[faction.id],
        lastOutcome: `${nextFactions[faction.id].lastOutcome ?? ""} ${note}`.trim(),
        history: [note, ...nextFactions[faction.id].history].slice(0, 4)
      };
      stories.push({
        factionId: faction.id,
        salience: nextThreat.pressure + philosopherThreatStageRank(nextThreat.stage) * 10 + (stageRaised ? 8 : 0),
        note
      });
    }
  }

  const philosophers: PhilosophersState = {
    byFaction: nextByFaction,
    spotlightFactionIds: Object.entries(nextByFaction)
      .sort((left, right) =>
        right[1].pressure - left[1].pressure
        || philosopherThreatStageRank(right[1].stage) - philosopherThreatStageRank(left[1].stage)
        || left[0].localeCompare(right[0])
      )
      .slice(0, 3)
      .map(([factionId]) => factionId as FactionId)
  };

  return {
    philosophers,
    factions: nextFactions,
    feedItems: stories
      .sort((left, right) => right.salience - left.salience || left.factionId.localeCompare(right.factionId))
      .slice(0, 2)
      .map((story) => ({
        id: `event-philosopher-threat-${state.clock.year}-${state.clock.month}-${story.factionId}`,
        day: state.clock.day,
        text: story.note
      }))
  };
}
