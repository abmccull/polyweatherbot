import type { OriginId } from "@the-oracle/content";
import type { LegacyArtifact } from "./legacy";

// --- Burden Types ---

export type BurdenId =
  | "no_trade"
  | "hostile_factions"
  | "fragile_pythia"
  | "scarce_resources"
  | "aggressive_rivals"
  | "short_seasons";

export type BurdenDef = {
  id: BurdenId;
  name: string;
  description: string;
  scoreMultiplier: number;
};

export const burdenDefs: BurdenDef[] = [
  {
    id: "no_trade",
    name: "Closed Roads",
    description: "All factions begin with trade access revoked. Caravans must be earned.",
    scoreMultiplier: 1.25
  },
  {
    id: "hostile_factions",
    name: "Hostile Courts",
    description: "Every faction starts with lowered credibility and heightened suspicion toward Delphi.",
    scoreMultiplier: 1.3
  },
  {
    id: "fragile_pythia",
    name: "Fragile Pythia",
    description: "The Pythia begins with diminished health and clarity. Trance is costly.",
    scoreMultiplier: 1.2
  },
  {
    id: "scarce_resources",
    name: "Scarce Resources",
    description: "Starting resources are halved and capacities are reduced.",
    scoreMultiplier: 1.35
  },
  {
    id: "aggressive_rivals",
    name: "Aggressive Rivals",
    description: "Rival oracles begin at elevated pressure and operate more frequently.",
    scoreMultiplier: 1.25
  },
  {
    id: "short_seasons",
    name: "Short Seasons",
    description: "Decline thresholds are tighter. The oracle must prove itself faster.",
    scoreMultiplier: 1.4
  }
];

export const burdenDefById: Record<BurdenId, BurdenDef> = Object.fromEntries(
  burdenDefs.map((b) => [b.id, b])
) as Record<BurdenId, BurdenDef>;

// --- Carryover Bonus ---

export type CarryoverBonusKind =
  | "starting_gold"
  | "starting_reputation"
  | "priest_skill"
  | "resource_capacity"
  | "consultation_quality";

export type CarryoverBonus = {
  id: string;
  kind: CarryoverBonusKind;
  value: number;
  sourceRunId: string;
};

// --- Run Record ---

export type RunRecord = {
  runId: string;
  originId: OriginId;
  seedText: string;
  startDay: number;
  endDay: number;
  finalScore: number;
  finalAge?: string;
  epitaph: string;
  completedAt: number;
};

// --- Lineage State ---

export type LineageState = {
  lineageId: string;
  runHistory: RunRecord[];
  totalRuns: number;
  unlockedOrigins: OriginId[];
  unlockedBurdens: BurdenId[];
  carryoverBonuses: CarryoverBonus[];
  lineageScore: number;
};

// --- Factory ---

export function createInitialLineageState(): LineageState {
  return {
    lineageId: `lineage-${Date.now().toString(36)}`,
    runHistory: [],
    totalRuns: 0,
    unlockedOrigins: ["upstart-shrine", "ancient-spring"],
    unlockedBurdens: [],
    carryoverBonuses: [],
    lineageScore: 0
  };
}

// --- Score thresholds for unlocking origins ---

const ORIGIN_UNLOCK_THRESHOLDS: Array<{ threshold: number; originId: OriginId }> = [
  { threshold: 200, originId: "cursed-oracle" },
  { threshold: 350, originId: "war-oracle" },
  { threshold: 500, originId: "gods-favourite" },
  { threshold: 650, originId: "merchant-oracle" },
  { threshold: 800, originId: "exiles-oracle" }
];

// --- Score thresholds for unlocking burdens ---

const BURDEN_UNLOCK_THRESHOLDS: Array<{ threshold: number; burdenId: BurdenId }> = [
  { threshold: 150, burdenId: "scarce_resources" },
  { threshold: 300, burdenId: "hostile_factions" },
  { threshold: 450, burdenId: "fragile_pythia" },
  { threshold: 550, burdenId: "no_trade" },
  { threshold: 700, burdenId: "aggressive_rivals" },
  { threshold: 850, burdenId: "short_seasons" }
];

// --- Carryover bonus computation ---

export function computeCarryoverBonuses(runHistory: RunRecord[]): CarryoverBonus[] {
  if (runHistory.length === 0) return [];

  const bonuses: CarryoverBonus[] = [];
  const cumulativeScore = runHistory.reduce((sum, r) => sum + r.finalScore, 0);
  const bestRun = [...runHistory].sort((a, b) => b.finalScore - a.finalScore)[0];
  const latestRun = runHistory[runHistory.length - 1];

  if (!bestRun || !latestRun) return [];

  // Gold bonus: cumulative score gives a starting gold bump
  if (cumulativeScore >= 100) {
    bonuses.push({
      id: `carryover-gold-${latestRun.runId}`,
      kind: "starting_gold",
      value: Math.min(80, Math.floor(cumulativeScore / 50) * 5),
      sourceRunId: latestRun.runId
    });
  }

  // Reputation bonus: best single-run score gives reputation head start
  if (bestRun.finalScore >= 300) {
    bonuses.push({
      id: `carryover-rep-${bestRun.runId}`,
      kind: "starting_reputation",
      value: Math.min(15, Math.floor(bestRun.finalScore / 100) * 2),
      sourceRunId: bestRun.runId
    });
  }

  // Priest skill bonus: total runs completed
  if (runHistory.length >= 3) {
    bonuses.push({
      id: `carryover-priest-${latestRun.runId}`,
      kind: "priest_skill",
      value: Math.min(20, runHistory.length * 2),
      sourceRunId: latestRun.runId
    });
  }

  // Resource capacity bonus: from cumulative score
  if (cumulativeScore >= 500) {
    bonuses.push({
      id: `carryover-capacity-${latestRun.runId}`,
      kind: "resource_capacity",
      value: Math.min(50, Math.floor(cumulativeScore / 200) * 10),
      sourceRunId: latestRun.runId
    });
  }

  // Consultation quality bonus: from best run
  if (bestRun.finalScore >= 600) {
    bonuses.push({
      id: `carryover-consult-${bestRun.runId}`,
      kind: "consultation_quality",
      value: Math.min(10, Math.floor(bestRun.finalScore / 200)),
      sourceRunId: bestRun.runId
    });
  }

  return bonuses;
}

// --- Record a completed run into lineage ---

export function recordRunInLineage(
  lineage: LineageState,
  artifact: LegacyArtifact,
  originId: OriginId,
  seedText: string,
  runId: string
): LineageState {
  const record: RunRecord = {
    runId,
    originId,
    seedText,
    startDay: 1,
    endDay: artifact.totalDays,
    finalScore: artifact.finalScore,
    finalAge: artifact.finalAge,
    epitaph: artifact.epitaph,
    completedAt: Date.now()
  };

  const nextHistory = [...lineage.runHistory, record];
  const nextTotalRuns = lineage.totalRuns + 1;
  const nextLineageScore = nextHistory.reduce((sum, r) => sum + r.finalScore, 0);

  // Unlock origins based on cumulative lineage score
  const nextUnlockedOrigins = [...lineage.unlockedOrigins];
  for (const entry of ORIGIN_UNLOCK_THRESHOLDS) {
    if (nextLineageScore >= entry.threshold && !nextUnlockedOrigins.includes(entry.originId)) {
      nextUnlockedOrigins.push(entry.originId);
    }
  }

  // Unlock burdens based on cumulative lineage score
  const nextUnlockedBurdens = [...lineage.unlockedBurdens];
  for (const entry of BURDEN_UNLOCK_THRESHOLDS) {
    if (nextLineageScore >= entry.threshold && !nextUnlockedBurdens.includes(entry.burdenId)) {
      nextUnlockedBurdens.push(entry.burdenId);
    }
  }

  const nextCarryoverBonuses = computeCarryoverBonuses(nextHistory);

  return {
    ...lineage,
    runHistory: nextHistory,
    totalRuns: nextTotalRuns,
    unlockedOrigins: nextUnlockedOrigins,
    unlockedBurdens: nextUnlockedBurdens,
    carryoverBonuses: nextCarryoverBonuses,
    lineageScore: nextLineageScore
  };
}
