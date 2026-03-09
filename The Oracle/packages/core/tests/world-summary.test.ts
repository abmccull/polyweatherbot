import { describe, expect, it } from "vitest";

import { renderGameToText } from "../src/selectors";
import { advancePoliticalClimate } from "../src/simulation/events";
import { advanceCampaignState, syncCampaignState } from "../src/state/campaign";
import { createInitialState } from "../src/state/initialState";
import type { GameState } from "../src/state/gameState";

function buildWorldSummaryState(seed = 47): GameState {
  const initial = createInitialState(seed);
  const seeded: GameState = {
    ...initial,
    clock: {
      ...initial.clock,
      month: 5,
      day: 1
    },
    campaign: syncCampaignState(
      {
        ...initial.campaign,
        reputation: {
          ...initial.campaign.reputation,
          score: 24
        }
      },
      121
    ),
    factions: {
      ...initial.factions,
      corinth: {
        ...initial.factions.corinth,
        favour: 74,
        credibility: 77,
        debt: 4,
        currentAgenda: "trade",
        tradeAccess: true,
        treaties: ["athens", "miletus"],
        relations: {
          ...initial.factions.corinth.relations,
          athens: 60,
          miletus: 58,
          thebes: -42
        }
      },
      thebes: {
        ...initial.factions.thebes,
        favour: 32,
        credibility: 27,
        debt: 25,
        dependence: 44,
        tradeAccess: true,
        currentAgenda: "succession",
        activeConflicts: ["sparta"],
        relations: {
          ...initial.factions.thebes.relations,
          corinth: -42,
          sparta: -52
        }
      }
    }
  };
  const politicalUpdate = advancePoliticalClimate(seeded);
  const campaignUpdate = advanceCampaignState(
    {
      ...seeded,
      factions: politicalUpdate.factions,
      tradeOffers: politicalUpdate.tradeOffers
    },
    politicalUpdate.factions
  );

  return {
    ...seeded,
    factions: politicalUpdate.factions,
    tradeOffers: politicalUpdate.tradeOffers,
    campaign: campaignUpdate.campaign,
    eventFeed: [...campaignUpdate.feedItems, ...politicalUpdate.feedItems]
  };
}

describe("renderGameToText world summary", () => {
  it("surfaces hegemon, destabilization, and ideological drift without widening game state", () => {
    const payload = JSON.parse(renderGameToText(buildWorldSummaryState()));

    expect(payload.world_summary).toBeDefined();
    expect(payload.world_summary.hegemon).toContain("Corinth");
    expect(payload.world_summary.destabilized).toContain("Thebes");
    expect(payload.world_summary.ideologicalDrift.some((entry: string) => entry.includes(":"))).toBe(true);
    expect(Array.isArray(payload.world_summary.activeCrises)).toBe(true);
  });
});
