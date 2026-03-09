import type {
  ChallengeCompletion,
  LegacyRunSummary,
  OracleProfileSnapshot
} from "./ProfileRepository";

export const PROFILE_VERSION = 1;
export const MAX_PROFILE_CHARS = 500_000;

const MAX_COMPLETED_RUNS = 128;
const MAX_CHALLENGE_COMPLETIONS = 128;
const MAX_NOTABLE_EVENTS = 16;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isStringArray(value: unknown, max = MAX_NOTABLE_EVENTS): value is string[] {
  return Array.isArray(value) && value.length <= max && value.every(isString);
}

function isLegacyRunSummary(value: unknown): value is LegacyRunSummary {
  return isRecord(value)
    && isString(value.id)
    && isString(value.label)
    && isString(value.originId)
    && isFiniteNumber(value.worldSeed)
    && isString(value.completedAt)
    && isFiniteNumber(value.legacyScore)
    && isFiniteNumber(value.yearsRuled)
    && (value.finalReputationTier === undefined || isString(value.finalReputationTier))
    && isStringArray(value.notableEvents);
}

function isChallengeCompletion(value: unknown): value is ChallengeCompletion {
  return isRecord(value)
    && isString(value.id)
    && isFiniteNumber(value.seed)
    && isStringArray(value.burdenIds, MAX_NOTABLE_EVENTS)
    && isString(value.completedAt)
    && isFiniteNumber(value.score);
}

export function createEmptyProfileSnapshot(now = new Date().toISOString()): OracleProfileSnapshot {
  return {
    version: PROFILE_VERSION,
    unlockedOrigins: [],
    completedRuns: [],
    challengeCompletions: [],
    createdAt: now,
    updatedAt: now
  };
}

export function isOracleProfileSnapshot(value: unknown): value is OracleProfileSnapshot {
  return isRecord(value)
    && value.version === PROFILE_VERSION
    && (value.lineageName === undefined || isString(value.lineageName))
    && (value.preferredOriginId === undefined || isString(value.preferredOriginId))
    && (value.lastWorldSeed === undefined || isFiniteNumber(value.lastWorldSeed))
    && isStringArray(value.unlockedOrigins, MAX_COMPLETED_RUNS)
    && Array.isArray(value.completedRuns)
    && value.completedRuns.length <= MAX_COMPLETED_RUNS
    && value.completedRuns.every(isLegacyRunSummary)
    && Array.isArray(value.challengeCompletions)
    && value.challengeCompletions.length <= MAX_CHALLENGE_COMPLETIONS
    && value.challengeCompletions.every(isChallengeCompletion)
    && isString(value.createdAt)
    && isString(value.updatedAt);
}

export function normalizeProfileSnapshot(
  snapshot?: Partial<OracleProfileSnapshot>,
  now = new Date().toISOString()
): OracleProfileSnapshot {
  const base = createEmptyProfileSnapshot(now);

  return {
    version: PROFILE_VERSION,
    lineageName: typeof snapshot?.lineageName === "string" ? snapshot.lineageName : base.lineageName,
    preferredOriginId: typeof snapshot?.preferredOriginId === "string" ? snapshot.preferredOriginId : base.preferredOriginId,
    lastWorldSeed: Number.isFinite(snapshot?.lastWorldSeed) ? Math.round(snapshot!.lastWorldSeed as number) : base.lastWorldSeed,
    unlockedOrigins: [...(snapshot?.unlockedOrigins ?? base.unlockedOrigins)].filter(isString).slice(0, MAX_COMPLETED_RUNS),
    completedRuns: [...(snapshot?.completedRuns ?? base.completedRuns)].filter(isLegacyRunSummary).slice(0, MAX_COMPLETED_RUNS),
    challengeCompletions: [...(snapshot?.challengeCompletions ?? base.challengeCompletions)]
      .filter(isChallengeCompletion)
      .slice(0, MAX_CHALLENGE_COMPLETIONS),
    createdAt: typeof snapshot?.createdAt === "string" ? snapshot.createdAt : base.createdAt,
    updatedAt: typeof snapshot?.updatedAt === "string" ? snapshot.updatedAt : now
  };
}

export function serializeProfileSnapshot(snapshot: OracleProfileSnapshot): string {
  return JSON.stringify(snapshot);
}

export function deserializeProfileSnapshot(raw: string): OracleProfileSnapshot {
  if (raw.length > MAX_PROFILE_CHARS) {
    throw new Error("Oracle profile exceeds maximum allowed size");
  }

  const parsed = JSON.parse(raw) as unknown;
  if (!isOracleProfileSnapshot(parsed)) {
    throw new Error("Invalid Oracle profile snapshot");
  }

  return normalizeProfileSnapshot(parsed);
}
