export type LegacyRunSummary = {
  id: string;
  label: string;
  originId: string;
  worldSeed: number;
  completedAt: string;
  legacyScore: number;
  yearsRuled: number;
  finalReputationTier?: string;
  notableEvents: string[];
};

export type ChallengeCompletion = {
  id: string;
  seed: number;
  burdenIds: string[];
  completedAt: string;
  score: number;
};

export type OracleProfileSnapshot = {
  version: 1;
  lineageName?: string;
  preferredOriginId?: string;
  lastWorldSeed?: number;
  unlockedOrigins: string[];
  completedRuns: LegacyRunSummary[];
  challengeCompletions: ChallengeCompletion[];
  createdAt: string;
  updatedAt: string;
};

export interface ProfileRepository {
  loadProfile(): Promise<OracleProfileSnapshot>;
  saveProfile(snapshot: OracleProfileSnapshot): Promise<void>;
  resetProfile(): Promise<void>;
}
