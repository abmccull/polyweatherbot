import type { OriginId } from "@the-oracle/content";
import type { GameState } from "./gameState";
import type { BurdenId } from "./lineage";
import { computeLegacyScore } from "./legacy";

// --- Challenge Seed Types ---

export type ChallengeSeed = {
  id: string;
  seedText: string;
  originId: OriginId;
  burdens: BurdenId[];
  targetScore: number;
  description: string;
};

// --- Curated Challenge Scenarios ---

export const challengeSeeds: ChallengeSeed[] = [
  {
    id: "challenge-iron-road",
    seedText: "IRON-ROAD",
    originId: "war-oracle",
    burdens: ["no_trade", "hostile_factions"],
    targetScore: 400,
    description: "Generals demand prophecy, but every caravan route is blocked and every court is hostile. Survive through war patronage alone."
  },
  {
    id: "challenge-silent-spring",
    seedText: "SILENT-SPRING",
    originId: "ancient-spring",
    burdens: ["fragile_pythia", "scarce_resources"],
    targetScore: 350,
    description: "The sacred spring runs thin and the Pythia is frail. Nurse Delphi through lean seasons with ritual discipline."
  },
  {
    id: "challenge-crowded-altar",
    seedText: "CROWDED-ALTAR",
    originId: "gods-favourite",
    burdens: ["aggressive_rivals", "short_seasons"],
    targetScore: 500,
    description: "Apollo favors Delphi, but rival oracles are relentless and decline strikes early. Turn divine favor into lasting dominance."
  },
  {
    id: "challenge-exiles-gambit",
    seedText: "EXILES-GAMBIT",
    originId: "exiles-oracle",
    burdens: ["hostile_factions", "scarce_resources", "no_trade"],
    targetScore: 550,
    description: "Refugees built this shrine with nothing. Three burdens stack against a sanctuary that must leverage intrigue over wealth."
  },
  {
    id: "challenge-golden-cage",
    seedText: "GOLDEN-CAGE",
    originId: "merchant-oracle",
    burdens: ["fragile_pythia", "aggressive_rivals"],
    targetScore: 450,
    description: "Gold flows freely but the Pythia breaks under strain and rivals sense weakness. Wealth cannot replace divine authority."
  }
];

export const challengeSeedById: Record<string, ChallengeSeed> = Object.fromEntries(
  challengeSeeds.map((c) => [c.id, c])
) as Record<string, ChallengeSeed>;

// --- Validation ---

export type ChallengeRunResult = {
  valid: boolean;
  score: number;
  meetsTarget: boolean;
};

export function validateChallengeRun(state: GameState, challenge: ChallengeSeed): ChallengeRunResult {
  const valid =
    state.originId === challenge.originId &&
    state.worldSeedText === challenge.seedText;

  const score = computeLegacyScore(state);
  const meetsTarget = score >= challenge.targetScore;

  return { valid, score, meetsTarget };
}
