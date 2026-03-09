import type { FactionId } from "./gameState";

/** Structured world history event — replaces string-based faction history */
export type WorldHistoryEvent = {
  id: string;
  day: number;
  year: number;
  month: number;
  kind: WorldHistoryEventKind;
  factionIds: FactionId[];
  regionId?: string;
  title: string;
  description: string;
  impact: WorldHistoryImpact;
  oracleInvolved: boolean;
  prophecyId?: string;
};

export type WorldHistoryEventKind =
  | "diplomatic_shift"
  | "alliance_formed"
  | "alliance_broken"
  | "trade_agreement"
  | "embargo"
  | "conflict_started"
  | "conflict_resolved"
  | "hegemon_emerged"
  | "hegemon_declined"
  | "revolution"
  | "regime_change"
  | "philosophy_spread"
  | "pilgrimage_surge"
  | "oracle_triumph"
  | "oracle_disgrace"
  | "crisis_escalation"
  | "crisis_resolution";

export type WorldHistoryImpact = {
  credibilityDelta?: number;
  factionRelationDeltas?: Array<{ factionA: FactionId; factionB: FactionId; delta: number }>;
  pressureDelta?: number;
  tradeFlowDelta?: number;
  pilgrimDelta?: number;
};

/** Alliance between two factions */
export type FactionAlliance = {
  id: string;
  factionA: FactionId;
  factionB: FactionId;
  formedDay: number;
  kind: "trade" | "military" | "cultural";
  strength: number;
  oracleMediated: boolean;
  brokenDay?: number;
  breakReason?: string;
};

/** Hegemon state — tracks dominant faction */
export type HegemonState = {
  currentHegemonId?: FactionId;
  hegemonScore: Record<string, number>;
  lastChangeDay?: number;
  hegemonHistory: Array<{
    factionId: FactionId;
    startDay: number;
    endDay?: number;
    peakScore: number;
  }>;
};

/** Root world history state */
export type WorldHistoryState = {
  events: WorldHistoryEvent[];
  alliances: FactionAlliance[];
  hegemon: HegemonState;
  revolutionCount: number;
  lastEventDay: number;
};

/** Creates initial empty world history state */
export function createInitialWorldHistoryState(): WorldHistoryState {
  return {
    events: [],
    alliances: [],
    hegemon: {
      hegemonScore: {},
      hegemonHistory: []
    },
    revolutionCount: 0,
    lastEventDay: 0
  };
}

/** Creates a world history event */
export function createWorldHistoryEvent(
  id: string,
  day: number,
  year: number,
  month: number,
  kind: WorldHistoryEventKind,
  factionIds: FactionId[],
  title: string,
  description: string,
  impact: WorldHistoryImpact = {},
  oracleInvolved = false,
  prophecyId?: string,
  regionId?: string
): WorldHistoryEvent {
  return {
    id,
    day,
    year,
    month,
    kind,
    factionIds,
    regionId,
    title,
    description,
    impact,
    oracleInvolved,
    prophecyId
  };
}

/** Checks if any faction qualifies as hegemon based on scores */
export function evaluateHegemon(
  hegemon: HegemonState,
  factionScores: Record<string, number>,
  day: number,
  threshold = 75
): HegemonState {
  const topFaction = Object.entries(factionScores).sort((a, b) => b[1] - a[1])[0];
  if (!topFaction) return { ...hegemon, hegemonScore: factionScores };

  const [factionId, score] = topFaction;
  const newState = { ...hegemon, hegemonScore: factionScores };

  if (score >= threshold && factionId !== hegemon.currentHegemonId) {
    // New hegemon emerged
    const history = [...hegemon.hegemonHistory];
    if (hegemon.currentHegemonId) {
      // Close previous hegemon's era
      const lastEntry = history[history.length - 1];
      if (lastEntry && !lastEntry.endDay) {
        history[history.length - 1] = { ...lastEntry, endDay: day };
      }
    }
    history.push({
      factionId: factionId as FactionId,
      startDay: day,
      peakScore: score
    });
    return {
      ...newState,
      currentHegemonId: factionId as FactionId,
      lastChangeDay: day,
      hegemonHistory: history
    };
  }

  if (hegemon.currentHegemonId && score < threshold * 0.6) {
    // Hegemon declined
    const history = [...hegemon.hegemonHistory];
    const lastEntry = history[history.length - 1];
    if (lastEntry && !lastEntry.endDay) {
      history[history.length - 1] = { ...lastEntry, endDay: day };
    }
    return {
      ...newState,
      currentHegemonId: undefined,
      lastChangeDay: day,
      hegemonHistory: history
    };
  }

  // Update peak score
  if (hegemon.currentHegemonId) {
    const history = [...hegemon.hegemonHistory];
    const lastEntry = history[history.length - 1];
    if (lastEntry && score > lastEntry.peakScore) {
      history[history.length - 1] = { ...lastEntry, peakScore: score };
      return { ...newState, hegemonHistory: history };
    }
  }

  return newState;
}
