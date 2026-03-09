import { describe, expect, it } from "vitest";

import { selectConsultationInsights } from "../src/selectors";
import { maybeCreateConsultation } from "../src/simulation/events";
import { createInitialState } from "../src/state/initialState";
import type { ConsultationCurrent, GameState, OmenReport, WordTile } from "../src/state/gameState";

function buildConsultationState(
  omenReports: OmenReport[],
  tilePool: WordTile[],
  scorePreview: ConsultationCurrent["scorePreview"],
  placedTileIds: string[]
): GameState {
  const initial = createInitialState(23);

  return {
    ...initial,
    consultation: {
      mode: "open",
      history: [],
      current: {
        id: "consultation-test",
        factionId: "sparta",
        envoyName: "Spartan Envoy",
        mood: "wary",
        paymentOffered: 42,
        question: "Should Sparta advance before the harvest?",
        domain: "military",
        omenReports,
        tilePool,
        placedTileIds,
        scorePreview
      }
    }
  };
}

describe("consultation insights", () => {
  it("flags contradictory, fragile omens when a risky prophecy is assembled", () => {
    const omenReports: OmenReport[] = [
      {
        id: "omen-a",
        sourceRole: "Augur",
        text: "The birds fled across the ridge in disorder.",
        reliability: 52,
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "omen-b",
        sourceRole: "Flame Keeper",
        text: "The coals darkened around the treasury bowl.",
        reliability: 55,
        semantics: {
          target: "treasury",
          action: "withhold",
          polarity: "warning",
          ambiguity: "specific",
          timeHorizon: "seasonal",
          domain: "economic"
        }
      },
      {
        id: "omen-c",
        sourceRole: "Spring Warden",
        text: "The basin trembled before the laurel hymn was done.",
        reliability: 49,
        semantics: {
          target: "oracle",
          action: "fracture",
          polarity: "warning",
          ambiguity: "cryptic",
          timeHorizon: "yearly",
          domain: "spiritual"
        }
      }
    ];
    const tilePool: WordTile[] = [
      {
        id: "tile-subject",
        text: "the army",
        category: "subject",
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "specific",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "tile-action",
        text: "shall fall",
        category: "action",
        semantics: {
          target: "city",
          action: "fall",
          polarity: "warning",
          ambiguity: "specific",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "tile-modifier",
        text: "before the moon turns",
        category: "modifier",
        semantics: {
          target: "oracle",
          action: "fracture",
          polarity: "warning",
          ambiguity: "specific",
          timeHorizon: "immediate",
          domain: "spiritual"
        }
      }
    ];

    const insights = selectConsultationInsights(
      buildConsultationState(
        omenReports,
        tilePool,
        { clarity: 58, value: 66, risk: 86 },
        ["tile-subject", "tile-action", "tile-modifier"]
      )
    );

    expect(insights).toBeDefined();
    expect(insights!.consensus).toBe("contradictory");
    expect(insights!.reliabilityBand).toBe("fragile");
    expect(insights!.riskWarning).toContain("dangerously");
    expect(insights!.omenSummaryText).toContain("Average reliability");
  });

  it("recommends sharper delivery when strong omens align", () => {
    const omenReports: OmenReport[] = [
      {
        id: "omen-a",
        sourceRole: "Augur",
        text: "Three hawks turned together toward the coast.",
        reliability: 74,
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "omen-b",
        sourceRole: "Flame Keeper",
        text: "The flame leaned east and held its shape.",
        reliability: 76,
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "omen-c",
        sourceRole: "Spring Warden",
        text: "The basin rang clear against the stone.",
        reliability: 72,
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      }
    ];
    const tilePool: WordTile[] = [
      {
        id: "tile-subject",
        text: "the army",
        category: "subject",
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "tile-action",
        text: "shall triumph",
        category: "action",
        semantics: {
          target: "army",
          action: "triumph",
          polarity: "favorable",
          ambiguity: "balanced",
          timeHorizon: "seasonal",
          domain: "military"
        }
      },
      {
        id: "tile-seal",
        text: "under Apollo's favor",
        category: "seal",
        semantics: {
          target: "oracle",
          action: "endure",
          polarity: "favorable",
          ambiguity: "cryptic",
          timeHorizon: "yearly",
          domain: "spiritual"
        }
      }
    ];

    const insights = selectConsultationInsights(
      buildConsultationState(
        omenReports,
        tilePool,
        { clarity: 79, value: 84, risk: 42 },
        ["tile-subject", "tile-action", "tile-seal"]
      )
    );

    expect(insights).toBeDefined();
    expect(insights!.consensus).toBe("aligned");
    expect(insights!.reliabilityBand).toBe("clear");
    expect(insights!.riskWarning).toBeNull();
    expect(insights!.guidance).toContain("align cleanly");
    expect(insights!.omenSummaryText).toContain("Depth");
  });

  it("warns when the trance is shallow even if the wording is otherwise serviceable", () => {
    const state = buildConsultationState(
      [
        {
          id: "omen-a",
          sourceRole: "Dream Priest",
          text: "The dream returned with a laurel crown floating in dark water.",
          reliability: 66,
          semantics: {
            target: "alliance",
            action: "endure",
            polarity: "double",
            ambiguity: "cryptic",
            timeHorizon: "yearly",
            domain: "spiritual"
          }
        },
        {
          id: "omen-b",
          sourceRole: "Scholar",
          text: "An older verse matched the envoy's question only in its final line.",
          reliability: 63,
          semantics: {
            target: "alliance",
            action: "endure",
            polarity: "double",
            ambiguity: "balanced",
            timeHorizon: "yearly",
            domain: "spiritual"
          }
        }
      ],
      [
        {
          id: "tile-subject",
          text: "the alliance",
          category: "subject",
          semantics: {
            target: "alliance",
            action: "endure",
            polarity: "double",
            ambiguity: "balanced",
            timeHorizon: "yearly",
            domain: "spiritual"
          }
        },
        {
          id: "tile-action",
          text: "shall endure",
          category: "action",
          semantics: {
            target: "alliance",
            action: "endure",
            polarity: "double",
            ambiguity: "balanced",
            timeHorizon: "yearly",
            domain: "spiritual"
          }
        },
        {
          id: "tile-modifier",
          text: "beneath the laurel crown",
          category: "modifier",
          semantics: {
            target: "oracle",
            action: "endure",
            polarity: "favorable",
            ambiguity: "cryptic",
            timeHorizon: "yearly",
            domain: "spiritual"
          }
        }
      ],
      { clarity: 67, value: 71, risk: 56 },
      ["tile-subject", "tile-action", "tile-modifier"]
    );

    state.pythia = {
      ...state.pythia,
      tranceDepth: 34,
      mentalClarity: 58,
      needs: {
        ...state.pythia.needs,
        purification: 54,
        rest: 44
      }
    };

    const insights = selectConsultationInsights(state);

    expect(insights).toBeDefined();
    expect(insights!.guidance).toContain("trance");
    expect(insights!.riskWarning).toContain("shallow");
  });

  it("creates a richer deterministic consultation from the expanded omen families", () => {
    const state = createInitialState(23);
    state.clock.day = 15;
    state.clock.month = 2;
    state.clock.year = 1;
    state.pythia = {
      ...state.pythia,
      traits: ["visionary", "calculating"]
    };

    const consultation = maybeCreateConsultation(state);

    expect(consultation).toBeDefined();
    expect(consultation!.omenReports).toHaveLength(4);
    expect(new Set(consultation!.omenReports.map((omen) => omen.sourceRole)).size).toBeGreaterThanOrEqual(3);
    expect(consultation!.tilePool).toHaveLength(12);
    expect(new Set(consultation!.tilePool.map((tile) => tile.category))).toEqual(
      new Set(["subject", "action", "condition", "modifier", "seal"])
    );
    expect(consultation!.scorePreview).toEqual({ clarity: 0, value: 0, risk: 0 });
  });
});
