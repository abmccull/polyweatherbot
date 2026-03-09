import { describe, expect, it } from "vitest";

import { OracleRuntime } from "../src/commands/dispatcher";
import { advancePoliticalClimate, buildMonthlyTradeOffers } from "../src/simulation/events";
import { createBuildingAt } from "../src/simulation/updateDay";
import { advanceCampaignState, syncCampaignState } from "../src/state/campaign";
import { createInitialState } from "../src/state/initialState";
import type { GameState, TileSemantics } from "../src/state/gameState";

const DEFAULT_SEMANTICS: TileSemantics = {
  target: "alliance",
  action: "prosper",
  polarity: "favorable",
  ambiguity: "balanced",
  timeHorizon: "seasonal",
  domain: "economic"
};

function buildCampaignState(seed = 23): GameState {
  const initial = createInitialState(seed);
  const quarters = createBuildingAt("priest_quarters", { x: 28, y: 49 }, "building-campaign-quarters");
  const storehouse = createBuildingAt("storehouse", { x: 29, y: 49 }, "building-campaign-storehouse");
  const sanctum = {
    ...createBuildingAt("inner_sanctum", { x: 30, y: 49 }, "building-campaign-sanctum"),
    assignedPriestIds: ["priest-1"]
  };
  const brazier = {
    ...createBuildingAt("eternal_flame_brazier", { x: 31, y: 49 }, "building-campaign-brazier"),
    assignedPriestIds: ["priest-1"]
  };
  const granary = createBuildingAt("granary", { x: 32, y: 49 }, "building-campaign-granary");
  const kitchen = createBuildingAt("kitchen", { x: 33, y: 49 }, "building-campaign-kitchen");
  const market = createBuildingAt("agora_market", { x: 34, y: 49 }, "building-campaign-market");
  const xenon = createBuildingAt("xenon", { x: 35, y: 49 }, "building-campaign-xenon");

  return {
    ...initial,
    clock: {
      ...initial.clock,
      day: 30,
      month: 1,
      tick: initial.clock.ticksPerDay * 29 + 599,
      tickOfDay: initial.clock.ticksPerDay - 1
    },
    resources: {
      ...initial.resources,
      gold: {
        ...initial.resources.gold,
        amount: 320
      },
      incense: {
        ...initial.resources.incense,
        amount: 24
      },
      sacred_water: {
        ...initial.resources.sacred_water,
        amount: 54
      },
      grain: {
        ...initial.resources.grain,
        amount: 84
      },
      olive_oil: {
        ...initial.resources.olive_oil,
        amount: 32
      }
    },
    buildings: [quarters, storehouse, sanctum, brazier, granary, kitchen, market, xenon],
    priests: initial.priests.map((priest) => ({
      ...priest,
      homeBuildingId: quarters.id,
      currentAssignmentBuildingId: sanctum.id
    })),
    consultation: {
      ...initial.consultation,
      history: [
        {
          id: "prophecy-athens-1",
          factionId: "athens",
          dayIssued: 18,
          text: "Athens prospers if it honors Delphi.",
          tileIds: ["subject-treasury"],
          semantics: [DEFAULT_SEMANTICS],
          clarity: 77,
          value: 81,
          risk: 28,
          dueDay: 26,
          resolved: true,
          resolvedDay: 26,
          resolutionReport: "Athens hailed Delphi after the reading proved true.",
          credibilityDelta: 10
        }
      ]
    },
    campaign: syncCampaignState(
      {
        ...initial.campaign,
        reputation: {
          ...initial.campaign.reputation,
          score: 19
        },
        treasury: {
          ...initial.campaign.treasury,
          totalGoldInvested: 145,
          nextMilestoneGold: 150
        }
      },
      30
    )
  };
}

function advanceMonthSnapshot(state: GameState): GameState {
  const nextMonth = state.clock.month === 12 ? 1 : state.clock.month + 1;
  const nextYear = state.clock.month === 12 ? state.clock.year + 1 : state.clock.year;
  const monthState = {
    ...state,
    clock: {
      ...state.clock,
      day: 1,
      month: nextMonth,
      year: nextYear
    }
  };
  const politicalUpdate = advancePoliticalClimate(monthState);
  const campaignUpdate = advanceCampaignState(
    {
      ...monthState,
      factions: politicalUpdate.factions,
      tradeOffers: politicalUpdate.tradeOffers
    },
    politicalUpdate.factions
  );

  return {
    ...monthState,
    factions: politicalUpdate.factions,
    tradeOffers: politicalUpdate.tradeOffers,
    campaign: campaignUpdate.campaign,
    eventFeed: [...campaignUpdate.feedItems, ...politicalUpdate.feedItems, ...monthState.eventFeed].slice(0, 8)
  };
}

describe("Oracle politics", () => {
  it("updates faction politics and trade offers on month turnover", () => {
    const initial = createInitialState(11);
    const runtime = new OracleRuntime({
      ...initial,
      clock: {
        ...initial.clock,
        day: 30,
        month: 1,
        tick: initial.clock.ticksPerDay * 29 + 599,
        tickOfDay: initial.clock.ticksPerDay - 1
      }
    });
    const agendasBefore = Object.fromEntries(Object.values(initial.factions).map((faction) => [faction.id, faction.currentAgenda]));

    runtime.advanceTicks(1);

    const state = runtime.getState();
    expect(state.clock.month).toBe(2);
    expect(state.clock.day).toBe(1);
    expect(state.tradeOffers.length).toBeGreaterThan(0);
    expect(state.tradeOffers.every((offer) => state.factions[offer.factionId].tradeAccess)).toBe(true);
    expect(Object.values(state.factions).some((faction) => faction.lastOutcome?.includes("Month 2"))).toBe(true);
    expect(Object.values(state.factions).some((faction) => faction.history[0]?.includes("Month 2"))).toBe(true);
    expect(
      Object.values(state.factions).some(
        (faction) => faction.currentAgenda !== agendasBefore[faction.id] || faction.lastOutcome?.includes("generals")
          || faction.lastOutcome?.includes("merchants") || faction.lastOutcome?.includes("priests") || faction.lastOutcome?.includes("court")
      )
    ).toBe(true);
    expect(state.eventFeed.some((event) => event.text.includes("Month 2"))).toBe(true);
  });

  it("uses faction profiles to prioritize mercantile trade partners and better pricing", () => {
    const initial = createInitialState(13);
    const state = {
      ...initial,
      factions: {
        ...initial.factions,
        athens: {
          ...initial.factions.athens,
          favour: 56,
          credibility: 60,
          debt: 6,
          tradeAccess: true
        },
        corinth: {
          ...initial.factions.corinth,
          favour: 56,
          credibility: 60,
          debt: 6,
          tradeAccess: true
        },
        sparta: {
          ...initial.factions.sparta,
          tradeAccess: false
        },
        thebes: {
          ...initial.factions.thebes,
          tradeAccess: false
        },
        argos: {
          ...initial.factions.argos,
          tradeAccess: false
        },
        miletus: {
          ...initial.factions.miletus,
          tradeAccess: false
        },
        syracuse: {
          ...initial.factions.syracuse,
          tradeAccess: false
        },
        macedon: {
          ...initial.factions.macedon,
          tradeAccess: false
        }
      }
    };

    const offers = buildMonthlyTradeOffers(state);
    const corinthOffer = offers.find((offer) => offer.factionId === "corinth");
    const athensOffer = offers.find((offer) => offer.factionId === "athens");

    expect(offers[0]?.factionId).toBe("corinth");
    expect(corinthOffer).toBeDefined();
    expect(athensOffer).toBeDefined();
    expect(corinthOffer!.price).toBeLessThan(athensOffer!.price);
    expect(corinthOffer!.amount).toBeGreaterThan(athensOffer!.amount);
  });

  it("derives treaties and embargoes from bilateral relations and lets them affect trade access", () => {
    const initial = createInitialState(17);
    const result = advancePoliticalClimate({
      ...initial,
      factions: {
        ...initial.factions,
        athens: {
          ...initial.factions.athens,
          relations: {
            ...initial.factions.athens.relations,
            miletus: 66,
            sparta: -78
          },
          currentAgenda: "trade",
          tradeAccess: true,
          activeConflicts: ["sparta"]
        },
        miletus: {
          ...initial.factions.miletus,
          relations: {
            ...initial.factions.miletus.relations,
            athens: 66
          },
          currentAgenda: "trade",
          tradeAccess: true,
          activeConflicts: []
        },
        sparta: {
          ...initial.factions.sparta,
          relations: {
            ...initial.factions.sparta.relations,
            athens: -78
          },
          currentAgenda: "war",
          tradeAccess: true,
          activeConflicts: ["athens"]
        },
        syracuse: {
          ...initial.factions.syracuse,
          relations: {
            ...initial.factions.syracuse.relations,
            athens: -52
          },
          currentAgenda: "trade",
          tradeAccess: true
        }
      }
    });
    const athensOffer = result.tradeOffers.find((offer) => offer.factionId === "athens");
    const miletusOffer = result.tradeOffers.find((offer) => offer.factionId === "miletus");

    expect(result.factions.athens.treaties).toContain("miletus");
    expect(result.factions.miletus.treaties).toContain("athens");
    expect(result.factions.athens.embargoes).toContain("sparta");
    expect(result.factions.sparta.embargoes).toContain("athens");
    if (athensOffer && miletusOffer) {
      expect(athensOffer.price).toBeGreaterThanOrEqual(miletusOffer.price);
    } else {
      expect(athensOffer).toBeUndefined();
    }
    expect(result.factions.athens.lastOutcome).toContain("Treaties");
    expect(result.factions.athens.lastOutcome).toContain("Embargoes");
  });

  it("prices treaty-backed caravans better than embargo-bound rivals with similar stats", () => {
    const initial = createInitialState(19);
    const offers = buildMonthlyTradeOffers({
      ...initial,
      factions: {
        ...initial.factions,
        corinth: {
          ...initial.factions.corinth,
          favour: 58,
          credibility: 61,
          debt: 5,
          tradeAccess: true,
          treaties: ["argos", "miletus"],
          embargoes: []
        },
        thebes: {
          ...initial.factions.thebes,
          favour: 58,
          credibility: 61,
          debt: 5,
          tradeAccess: true,
          treaties: [],
          embargoes: ["sparta", "macedon"]
        },
        athens: {
          ...initial.factions.athens,
          tradeAccess: false
        },
        sparta: {
          ...initial.factions.sparta,
          tradeAccess: false
        },
        argos: {
          ...initial.factions.argos,
          tradeAccess: false
        },
        miletus: {
          ...initial.factions.miletus,
          tradeAccess: false
        },
        syracuse: {
          ...initial.factions.syracuse,
          tradeAccess: false
        },
        macedon: {
          ...initial.factions.macedon,
          tradeAccess: false
        }
      }
    });
    const corinthOffer = offers.find((offer) => offer.factionId === "corinth");
    const thebesOffer = offers.find((offer) => offer.factionId === "thebes");

    expect(corinthOffer).toBeDefined();
    expect(thebesOffer).toBeDefined();
    expect(corinthOffer!.price).toBeLessThan(thebesOffer!.price);
    expect(corinthOffer!.amount).toBeGreaterThan(thebesOffer!.amount);
  });

  it("advances reputation, treasury dedication, and world-map pressure state from the campaign layer", () => {
    const state = buildCampaignState(29);
    const politicalUpdate = advancePoliticalClimate(state);
    const { campaign, feedItems } = advanceCampaignState(
      {
        ...state,
        factions: politicalUpdate.factions,
        tradeOffers: politicalUpdate.tradeOffers
      },
      politicalUpdate.factions
    );

    expect(campaign.reputation.score).toBeGreaterThan(state.campaign.reputation.score);
    expect(campaign.reputation.currentTier).toBe("recognized");
    expect(campaign.reputation.unlockedBuildingIds).toContain("granary");
    expect(campaign.treasury.completed).toBe(1);
    expect(campaign.patronMilestones.some((entry) => entry.includes("treasury offering"))).toBe(true);
    expect(campaign.worldMap.activePressures.length).toBeGreaterThan(0);
    expect(campaign.worldMap.crisisChains.length).toBeGreaterThan(0);
    expect(campaign.winCondition.summary).toContain("Tier recognized");
    expect(feedItems.some((item) => item.text.includes("treasury offering"))).toBe(true);
  });

  it("resolves a crisis chain and completes Rising Oracle once Delphi is revered with a dedication", () => {
    const seeded = buildCampaignState(31);
    const quietFactions = {
      ...seeded.factions,
      athens: {
        ...seeded.factions.athens,
        tradeAccess: true,
        activeConflicts: [],
        embargoes: [],
        treaties: ["miletus" as const],
        credibility: 66,
        favour: 63,
        debt: 3
      },
      sparta: {
        ...seeded.factions.sparta,
        tradeAccess: true,
        activeConflicts: [],
        embargoes: [],
        treaties: ["argos" as const],
        credibility: 61,
        favour: 58,
        debt: 4
      }
    };
    const state = {
      ...seeded,
      factions: quietFactions,
      campaign: syncCampaignState(
        {
          ...seeded.campaign,
          reputation: {
            ...seeded.campaign.reputation,
            score: 48
          },
          treasury: {
            ...seeded.campaign.treasury,
            totalGoldInvested: 149,
            nextMilestoneGold: 150
          },
          worldMap: {
            ...seeded.campaign.worldMap,
            crisisChains: [
              {
                id: "crisis-athens-conflict",
                label: "Athens League Fracture",
                nodeId: "athens",
                factionId: "athens",
                stage: "active",
                pressure: 40,
                stepsCompleted: 2
              }
            ]
          }
        },
        30
      )
    };

    const { campaign, feedItems } = advanceCampaignState(state, quietFactions);

    expect(campaign.reputation.currentTier).toBe("revered");
    expect(campaign.treasury.completed).toBeGreaterThanOrEqual(1);
    expect(campaign.worldMap.crisisChains.some((chain) => chain.stage === "resolution")).toBe(true);
    expect(campaign.winCondition.completed).toBe(true);
    expect(campaign.winCondition.completedDay).toBeDefined();
    expect(feedItems.some((item) => item.text.includes("fulfilled"))).toBe(true);
  });

  it("applies campaign progression through the live month-turnover runtime path", () => {
    const runtime = new OracleRuntime(buildCampaignState(37));

    runtime.advanceTicks(1);

    const state = runtime.getState();
    expect(state.clock.month).toBe(2);
    expect(state.campaign.reputation.score).toBeGreaterThanOrEqual(20);
    expect(state.campaign.treasury.totalGoldInvested).toBeGreaterThan(145);
    expect(state.campaign.worldMap.activePressures.length).toBeGreaterThan(0);
    expect(state.eventFeed.some((event) => event.text.includes("treasury offering") || event.text.includes("regional crisis"))).toBe(true);
  });

  it("builds multi-month hegemon, destabilization, and philosophical drift into campaign outputs", () => {
    const seeded = buildCampaignState(41);
    let state: GameState = {
      ...seeded,
      factions: {
        ...seeded.factions,
        corinth: {
          ...seeded.factions.corinth,
          favour: 74,
          credibility: 76,
          debt: 4,
          currentAgenda: "trade",
          tradeAccess: true,
          relations: {
            ...seeded.factions.corinth.relations,
            athens: 48,
            miletus: 58,
            thebes: -34
          }
        },
        athens: {
          ...seeded.factions.athens,
          favour: 62,
          credibility: 66,
          debt: 8,
          currentAgenda: "trade",
          tradeAccess: true,
          relations: {
            ...seeded.factions.athens.relations,
            corinth: 48
          }
        },
        miletus: {
          ...seeded.factions.miletus,
          favour: 64,
          credibility: 68,
          debt: 5,
          currentAgenda: "trade",
          tradeAccess: true,
          relations: {
            ...seeded.factions.miletus.relations,
            corinth: 58
          }
        },
        thebes: {
          ...seeded.factions.thebes,
          favour: 34,
          credibility: 28,
          debt: 24,
          dependence: 46,
          tradeAccess: true,
          currentAgenda: "succession",
          activeConflicts: ["sparta", "athens"],
          relations: {
            ...buildCampaignState(41).factions.thebes.relations,
            corinth: -42,
            sparta: -55,
            athens: -48
          }
        }
      }
    };

    for (let index = 0; index < 5; index += 1) {
      state = advanceMonthSnapshot(state);
    }

    const histories = Object.values(state.factions).flatMap((faction) => faction.history);
    expect(histories.some((entry) => entry.includes("dominant bloc"))).toBe(true);
    expect(histories.some((entry) => entry.includes("government crisis") || entry.includes("revolutionary agitation"))).toBe(true);
    expect(histories.some((entry) => entry.includes("drifts toward"))).toBe(true);
    expect(state.campaign.worldMap.activePressures.some((pressure) => pressure.kind === "trade" || pressure.kind === "conflict")).toBe(true);
    expect(state.campaign.worldMap.crisisChains.some((chain) => chain.label.includes("Hegemon Squeeze") || chain.label.includes("Regime Crisis") || chain.label.includes("Creed Schism"))).toBe(true);
  });

  it("lets hegemon blocs improve partner trade while squeezing rival access", () => {
    const initial = createInitialState(43);
    const result = advancePoliticalClimate({
      ...initial,
      clock: {
        ...initial.clock,
        month: 4,
        day: 1
      },
      factions: {
        ...initial.factions,
        corinth: {
          ...initial.factions.corinth,
          favour: 76,
          credibility: 78,
          debt: 3,
          currentAgenda: "trade",
          tradeAccess: true,
          treaties: ["athens", "miletus"],
          relations: {
            ...initial.factions.corinth.relations,
            athens: 62,
            miletus: 66,
            thebes: -44
          }
        },
        athens: {
          ...initial.factions.athens,
          favour: 62,
          credibility: 65,
          debt: 6,
          currentAgenda: "trade",
          tradeAccess: true,
          treaties: ["corinth"],
          relations: {
            ...initial.factions.athens.relations,
            corinth: 62
          }
        },
        miletus: {
          ...initial.factions.miletus,
          favour: 63,
          credibility: 67,
          debt: 4,
          currentAgenda: "trade",
          tradeAccess: true,
          treaties: ["corinth"],
          relations: {
            ...initial.factions.miletus.relations,
            corinth: 66
          }
        },
        thebes: {
          ...initial.factions.thebes,
          favour: 45,
          credibility: 41,
          debt: 13,
          currentAgenda: "trade",
          tradeAccess: true,
          embargoes: ["corinth"],
          relations: {
            ...initial.factions.thebes.relations,
            corinth: -52
          }
        }
      }
    });
    const corinthOffer = result.tradeOffers.find((offer) => offer.factionId === "corinth");
    const athensOffer = result.tradeOffers.find((offer) => offer.factionId === "athens");
    const thebesOffer = result.tradeOffers.find((offer) => offer.factionId === "thebes");
    const campaignUpdate = advanceCampaignState(
      {
        ...buildCampaignState(43),
        clock: {
          ...buildCampaignState(43).clock,
          month: 4,
          day: 1
        },
        factions: result.factions,
        tradeOffers: result.tradeOffers
      },
      result.factions
    );

    expect(result.factions.corinth.history.some((entry) => entry.includes("dominant bloc"))).toBe(true);
    expect(result.factions.athens.tradeAccess).toBe(true);
    expect(corinthOffer).toBeDefined();
    expect(athensOffer).toBeDefined();
    expect(corinthOffer!.price).toBeLessThanOrEqual(athensOffer!.price);
    if (thebesOffer) {
      expect(athensOffer!.price).toBeLessThan(thebesOffer.price);
    } else {
      expect(result.factions.thebes.tradeAccess).toBe(false);
    }
    expect(campaignUpdate.campaign.worldMap.crisisChains.some((chain) => chain.label.includes("Hegemon Squeeze"))).toBe(true);
  });
});
