import type { FactionId, GameState } from "./gameState";
import { getAbsoluteDay } from "../simulation/clock";

// --- Types ---

export type LegacyPhase = "thriving" | "declining" | "terminal";

export type LegacyProphecySummary = {
  prophecyId: string;
  factionName: string;
  domain: string;
  outcome: string;
  credibilityImpact: number;
};

export type LegacyPatronSummary = {
  factionName: string;
  peakCredibility: number;
  finalCredibility: number;
  totalConsultations: number;
};

export type LegacyTurningPoint = {
  day: number;
  year: number;
  description: string;
  impact: "positive" | "negative" | "neutral";
};

export type LegacyFigureSummary = {
  name: string;
  role: string;
  relationship: string;
};

export type LegacyArtifact = {
  generatedDay: number;
  totalDays: number;
  totalYears: number;
  finalAge?: string;
  majorProphecies: LegacyProphecySummary[];
  notablePatrons: LegacyPatronSummary[];
  turningPoints: LegacyTurningPoint[];
  namedFigures: LegacyFigureSummary[];
  finalReputation: string;
  finalScore: number;
  epitaph: string;
};

export type LegacyState = {
  phase: LegacyPhase;
  declineStartDay?: number;
  declineSeverity: number;
  comebackAttempts: number;
  legacyScore: number;
  legacyArtifact?: LegacyArtifact;
};

// --- Factory ---

export function createInitialLegacyState(): LegacyState {
  return {
    phase: "thriving",
    declineSeverity: 0,
    comebackAttempts: 0,
    legacyScore: 0
  };
}

// --- Score Calculation ---

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function computeLegacyScore(state: GameState): number {
  const absoluteDay = getAbsoluteDay(state.clock);

  // Survival time component (up to 200 points)
  const survivalMonths = Math.floor(absoluteDay / 30);
  const survivalPoints = Math.min(200, survivalMonths * 4);

  // Reputation component (up to 200 points)
  const reputationPoints = Math.round(state.campaign.reputation.score * 2);

  // Prophecy outcomes component (up to 250 points)
  const resolvedProphecies = state.consultation.history.filter((p) => p.resolved);
  const totalCredibilityGain = resolvedProphecies.reduce((sum, p) => sum + Math.max(0, p.credibilityDelta ?? 0), 0);
  const prophecyPoints = Math.min(250, Math.round(totalCredibilityGain * 1.5) + resolvedProphecies.length * 5);

  // World influence component (up to 150 points)
  const factionValues = Object.values(state.factions);
  const averageCredibility = factionValues.length > 0
    ? factionValues.reduce((sum, f) => sum + f.credibility, 0) / factionValues.length
    : 0;
  const treatyCount = factionValues.reduce((sum, f) => sum + f.treaties.length, 0);
  const tradeAccess = factionValues.filter((f) => f.tradeAccess).length;
  const worldPoints = Math.min(150, Math.round(averageCredibility * 0.8 + treatyCount * 8 + tradeAccess * 6));

  // Treasury component (up to 100 points)
  const treasuryPoints = Math.min(100, state.campaign.treasury.completed * 25 + Math.round(state.resources.gold.amount * 0.05));

  // Crisis resolution bonus (up to 100 points)
  const resolvedCrises = state.campaign.worldMap.crisisChains.filter((c) => c.stage === "resolution").length;
  const crisisPoints = Math.min(100, resolvedCrises * 30);

  return clamp(survivalPoints + reputationPoints + prophecyPoints + worldPoints + treasuryPoints + crisisPoints, 0, 1000);
}

// --- Artifact Generation ---

function ageLabel(totalDays: number): string {
  const years = Math.floor(totalDays / 360);
  if (years < 2) return "Fledgling Oracle";
  if (years < 5) return "Established Oracle";
  if (years < 10) return "Venerable Oracle";
  return "Legendary Oracle";
}

function reputationLabel(tier: string): string {
  switch (tier) {
    case "panhellenic": return "Panhellenic — known across all Greece";
    case "revered": return "Revered — sought by kings and generals";
    case "recognized": return "Recognized — a rising voice among the sanctuaries";
    case "obscure":
    default:
      return "Obscure — a forgotten shrine on a dusty hillside";
  }
}

function generateEpitaph(state: GameState, score: number): string {
  const totalDays = getAbsoluteDay(state.clock);
  const years = Math.floor(totalDays / 360);
  const phase = state.legacy?.phase ?? "thriving";
  const resolvedProphecies = state.consultation.history.filter((p) => p.resolved).length;

  if (score >= 800) {
    return `For ${years} years, Delphi stood as the unrivaled voice of the gods. ${resolvedProphecies} prophecies shaped the fate of nations. Its name will echo through the ages.`;
  }
  if (score >= 500) {
    return `Delphi endured ${years} years as a beacon of divine wisdom. Though not without missteps, its ${resolvedProphecies} prophecies left marks upon the Greek world that time cannot erase.`;
  }
  if (score >= 250) {
    return `The oracle at Delphi served for ${years} years, delivering ${resolvedProphecies} prophecies of mixed fortune. A modest legacy, but one that pilgrims will remember.`;
  }
  if (phase === "terminal") {
    return `After ${years} years, Delphi's fires dimmed. The Pythia fell silent, the priests scattered, and the sanctuary returned to the mountain. ${resolvedProphecies} prophecies were all it left behind.`;
  }
  return `Delphi's brief tenure of ${years} years produced ${resolvedProphecies} prophecies before fading into the margins of Greek memory.`;
}

export function generateLegacyArtifact(state: GameState): LegacyArtifact {
  const absoluteDay = getAbsoluteDay(state.clock);
  const totalYears = Math.floor(absoluteDay / 360);
  const score = computeLegacyScore(state);

  // Major prophecies
  const majorProphecies: LegacyProphecySummary[] = state.consultation.history
    .filter((p) => p.resolved && Math.abs(p.credibilityDelta ?? 0) >= 3)
    .sort((a, b) => Math.abs(b.credibilityDelta ?? 0) - Math.abs(a.credibilityDelta ?? 0))
    .slice(0, 8)
    .map((p) => ({
      prophecyId: p.id,
      factionName: state.factions[p.factionId]?.name ?? p.factionId,
      domain: p.semantics[0]?.domain ?? "unknown",
      outcome: p.resolutionReport ?? (p.credibilityDelta && p.credibilityDelta > 0 ? "Fulfilled" : "Failed"),
      credibilityImpact: p.credibilityDelta ?? 0
    }));

  // Notable patrons
  const notablePatrons: LegacyPatronSummary[] = (Object.values(state.factions) as Array<{ id: FactionId; name: string; credibility: number }>)
    .map((f) => {
      const consultations = state.consultation.history.filter((p) => p.factionId === f.id);
      const peakCredibility = consultations.reduce(
        (peak, p) => Math.max(peak, f.credibility + (p.credibilityDelta ?? 0)),
        f.credibility
      );
      return {
        factionName: f.name,
        peakCredibility: Math.min(100, peakCredibility),
        finalCredibility: f.credibility,
        totalConsultations: consultations.length
      };
    })
    .filter((p) => p.totalConsultations > 0)
    .sort((a, b) => b.totalConsultations - a.totalConsultations);

  // Turning points from world history
  const turningPoints: LegacyTurningPoint[] = [];
  if (state.worldHistory) {
    const significantEvents = state.worldHistory.events
      .filter((e) => e.oracleInvolved || e.kind === "hegemon_emerged" || e.kind === "revolution" || e.kind === "oracle_triumph" || e.kind === "oracle_disgrace")
      .sort((a, b) => a.day - b.day)
      .slice(0, 10);

    for (const event of significantEvents) {
      const impact: LegacyTurningPoint["impact"] =
        event.kind === "oracle_triumph" || event.kind === "alliance_formed" || event.kind === "crisis_resolution"
          ? "positive"
          : event.kind === "oracle_disgrace" || event.kind === "revolution" || event.kind === "conflict_started"
            ? "negative"
            : "neutral";
      turningPoints.push({
        day: event.day,
        year: event.year,
        description: event.title,
        impact
      });
    }
  }

  // Named figures
  const namedFigures: LegacyFigureSummary[] = [];
  if (state.characters) {
    const notable = state.characters.roster
      .filter((c) => c.prominence >= 40 || c.status === "legendary")
      .sort((a, b) => b.prominence - a.prominence)
      .slice(0, 6);

    for (const character of notable) {
      const rel = character.relationship;
      let relationship = "acquaintance";
      if (rel.trust >= 60) relationship = "trusted ally";
      else if (rel.trust >= 30) relationship = "friendly";
      else if (rel.hostility >= 60) relationship = "bitter enemy";
      else if (rel.hostility >= 30) relationship = "rival";
      else if (rel.fear >= 50) relationship = "fearful subject";

      namedFigures.push({
        name: character.displayName,
        role: character.role,
        relationship
      });
    }
  }

  return {
    generatedDay: absoluteDay,
    totalDays: absoluteDay,
    totalYears,
    finalAge: ageLabel(absoluteDay),
    majorProphecies,
    notablePatrons,
    turningPoints,
    namedFigures,
    finalReputation: reputationLabel(state.campaign.reputation.currentTier),
    finalScore: score,
    epitaph: generateEpitaph(state, score)
  };
}
