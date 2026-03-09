import type { FactionId, DomainTag } from "./gameState";

/** A faction's political interpretation of a delivered prophecy */
export type ProphecyInterpretationBranch = {
  factionId: FactionId;
  reading: string;
  alignment: "supportive" | "hostile" | "neutral" | "exploitative";
  influenceShift: number;
  adopted: boolean;
};

/** Tracks a multi-year prophecy commitment */
export type ProphecyArc = {
  id: string;
  rootProphecyId: string;
  label: string;
  domain: DomainTag;
  factionId: FactionId;
  startDay: number;
  expectedEndDay: number;
  status: "active" | "fulfilled" | "failed" | "contradicted";
  milestones: ProphecyArcMilestone[];
  interpretationBranches: ProphecyInterpretationBranch[];
  followUpObligations: ProphecyFollowUp[];
};

export type ProphecyArcMilestone = {
  id: string;
  label: string;
  dueDay: number;
  completed: boolean;
  completedDay?: number;
  outcome?: string;
};

export type ProphecyFollowUp = {
  id: string;
  kind: "clarification" | "reaffirmation" | "amendment";
  dueDay: number;
  fulfilled: boolean;
  fulfilledDay?: number;
  consequenceProphecyId?: string;
};

/** Detected contradiction between two prophecies */
export type ProphecyContradiction = {
  id: string;
  prophecyIdA: string;
  prophecyIdB: string;
  domain: DomainTag;
  description: string;
  detectedDay: number;
  severity: "minor" | "major" | "catastrophic";
  resolved: boolean;
  resolutionReport?: string;
  credibilityImpact: number;
};

/** Root state for the prophecy arc system */
export type ProphecyArcState = {
  arcs: ProphecyArc[];
  contradictions: ProphecyContradiction[];
  totalArcsCompleted: number;
  totalContradictions: number;
  lastArcCheckDay: number;
};

/** Creates initial empty prophecy arc state */
export function createInitialProphecyArcState(): ProphecyArcState {
  return {
    arcs: [],
    contradictions: [],
    totalArcsCompleted: 0,
    totalContradictions: 0,
    lastArcCheckDay: 0
  };
}

/** Checks if two prophecy records contradict each other based on semantics */
export function detectContradiction(
  prophecyA: { id: string; semantics: Array<{ target: string; action: string; polarity: string; domain: string }> },
  prophecyB: { id: string; semantics: Array<{ target: string; action: string; polarity: string; domain: string }> },
  day: number
): ProphecyContradiction | null {
  for (const semA of prophecyA.semantics) {
    for (const semB of prophecyB.semantics) {
      if (
        semA.target === semB.target &&
        semA.domain === semB.domain &&
        semA.polarity !== semB.polarity &&
        (semA.action !== semB.action)
      ) {
        const severity =
          semA.polarity === "favorable" && semB.polarity === "warning" ? "major"
          : semA.action === "triumph" && semB.action === "fall" ? "catastrophic"
          : "minor";

        return {
          id: `contradiction-${prophecyA.id}-${prophecyB.id}`,
          prophecyIdA: prophecyA.id,
          prophecyIdB: prophecyB.id,
          domain: semA.domain as DomainTag,
          description: `Conflicting readings on ${semA.target}: ${semA.action} (${semA.polarity}) vs ${semB.action} (${semB.polarity})`,
          detectedDay: day,
          severity,
          resolved: false,
          credibilityImpact: severity === "catastrophic" ? -15 : severity === "major" ? -8 : -3
        };
      }
    }
  }
  return null;
}
