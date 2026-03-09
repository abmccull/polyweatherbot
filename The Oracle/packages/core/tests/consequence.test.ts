import { describe, expect, it } from "vitest";

import { reduceCommand } from "../src/reducers";
import { evaluateResolutionObservers, resolveConsequence, scorePlacedTiles } from "../src/simulation/events";
import { createInitialState } from "../src/state/initialState";
import { runSimulationTick } from "../src/simulation/updateDay";
import type { ConsequenceCase, ProphecyRecord, TileSemantics, WordTile } from "../src/state/gameState";

const sampleSemantics: TileSemantics = {
  target: "army",
  action: "triumph",
  polarity: "favorable",
  ambiguity: "balanced",
  timeHorizon: "seasonal",
  domain: "military"
};

describe("Prophecy systems", () => {
  it("scores a balanced prophecy meaningfully", () => {
    const tiles: WordTile[] = [
      { id: "a", text: "the army", category: "subject", semantics: sampleSemantics },
      { id: "b", text: "shall triumph", category: "action", semantics: sampleSemantics },
      { id: "c", text: "before harvest", category: "condition", semantics: sampleSemantics },
      { id: "d", text: "thus speaks the god", category: "seal", semantics: sampleSemantics }
    ];

    const score = scorePlacedTiles(tiles, 70);
    expect(score.clarity).toBeGreaterThan(60);
    expect(score.value).toBeGreaterThan(70);
    expect(score.risk).toBeLessThan(65);
  });

  it("lets Pythia traits and depth change the same prophecy score", () => {
    const tiles: WordTile[] = [
      { id: "a", text: "the alliance", category: "subject", semantics: { ...sampleSemantics, target: "alliance", action: "endure", domain: "spiritual", polarity: "double" } },
      { id: "b", text: "shall endure", category: "action", semantics: { ...sampleSemantics, target: "alliance", action: "endure", domain: "spiritual", polarity: "double" } },
      { id: "c", text: "beneath the laurel crown", category: "modifier", semantics: { ...sampleSemantics, target: "oracle", action: "endure", domain: "spiritual", ambiguity: "cryptic", polarity: "favorable", timeHorizon: "yearly" } }
    ];

    const lucidScore = scorePlacedTiles(tiles, {
      attunement: 70,
      mentalClarity: 76,
      tranceDepth: 68,
      needs: {
        purification: 28,
        rest: 24,
        pilgrimageCooldown: 0
      },
      traits: ["visionary", "calculating", "diplomatic"]
    });
    const strainedScore = scorePlacedTiles(tiles, {
      attunement: 62,
      mentalClarity: 49,
      tranceDepth: 34,
      needs: {
        purification: 74,
        rest: 71,
        pilgrimageCooldown: 0
      },
      traits: ["fragile"]
    });

    expect(lucidScore.clarity).toBeGreaterThan(strainedScore.clarity);
    expect(lucidScore.value).toBeGreaterThan(strainedScore.value);
    expect(lucidScore.risk).toBeLessThan(strainedScore.risk);
  });

  it("rewards matching consequence tags", () => {
    const consequence: ConsequenceCase = {
      id: "consequence-1",
      prophecyId: "prophecy-1",
      factionId: "athens",
      dueDay: 20,
      outcome: sampleSemantics,
      resolved: false
    };

    const resolution = resolveConsequence(consequence, [sampleSemantics]);
    expect(resolution.delta).toBeGreaterThan(0);
  });

  it("punishes a wrong specific prophecy harder than a cryptic one", () => {
    const consequence: ConsequenceCase = {
      id: "consequence-2",
      prophecyId: "prophecy-2",
      factionId: "sparta",
      dueDay: 20,
      outcome: sampleSemantics,
      resolved: false
    };
    const specificWrong: TileSemantics = {
      ...sampleSemantics,
      target: "city",
      action: "fall",
      ambiguity: "specific",
      polarity: "warning"
    };
    const crypticWrong: TileSemantics = {
      ...sampleSemantics,
      target: "oracle",
      action: "fracture",
      ambiguity: "cryptic",
      polarity: "double"
    };

    const specificResolution = resolveConsequence(consequence, [specificWrong]);
    const crypticResolution = resolveConsequence(consequence, [crypticWrong]);

    expect(specificResolution.delta).toBeLessThan(crypticResolution.delta);
  });

  it("makes rival factions react when a resolved prophecy vindicates their enemy", () => {
    const state = createInitialState();
    const reactions = evaluateResolutionObservers(
      state,
      "sparta",
      sampleSemantics,
      14
    );

    const athensReaction = reactions.find((reaction) => reaction.factionId === "athens");
    expect(athensReaction).toBeDefined();
    expect(athensReaction?.favourDelta).toBeLessThan(0);
    expect(athensReaction?.relationDelta).toBeLessThan(0);
    expect(athensReaction?.historyEntry).toContain("Sparta");
  });

  it("applies observer reactions when a due consequence resolves in the live sim", () => {
    const prophecy: ProphecyRecord = {
      id: "prophecy-live",
      factionId: "sparta",
      dayIssued: 1,
      text: "the army shall triumph before harvest thus speaks the god",
      tileIds: ["a", "b", "c", "d"],
      semantics: [sampleSemantics],
      clarity: 82,
      value: 88,
      risk: 36,
      dueDay: 1,
      resolved: false
    };
    const consequence: ConsequenceCase = {
      id: "consequence-live",
      prophecyId: prophecy.id,
      factionId: "sparta",
      dueDay: 1,
      outcome: sampleSemantics,
      resolved: false
    };
    let state = createInitialState();
    const athensBefore = state.factions.athens;
    state = {
      ...state,
      consultation: {
        ...state.consultation,
        history: [prophecy]
      },
      consequences: [consequence]
    };

    const nextState = runSimulationTick(state, 1).state;

    expect(nextState.consequences[0]?.resolved).toBe(true);
    expect(nextState.factions.athens.favour).toBeLessThan(athensBefore.favour);
    expect((nextState.factions.athens.relations.sparta ?? 0)).toBeLessThan(athensBefore.relations.sparta ?? 0);
    expect(nextState.factions.athens.history[0]).toContain("Sparta");
    expect(nextState.eventFeed.some((entry) => entry.text.includes("vindication") || entry.text.includes("resents"))).toBe(true);
  });

  it("records depth and interpretation when a prophecy is delivered", () => {
    let state = createInitialState();
    state = reduceCommand(state, { type: "InjectScenarioCommand", scenario: "consultation-ready" }).state;
    state = reduceCommand(state, { type: "StartConsultationCommand" }).state;

    const current = state.consultation.current;
    expect(current).toBeDefined();
    const chosenTiles = [
      current?.tilePool.find((tile) => tile.category === "subject"),
      current?.tilePool.find((tile) => tile.category === "action"),
      current?.tilePool.find((tile) => tile.category === "condition" || tile.category === "modifier"),
      current?.tilePool.find((tile) => tile.category === "seal")
    ].filter((tile): tile is NonNullable<typeof tile> => Boolean(tile));

    for (const tile of chosenTiles) {
      state = reduceCommand(state, { type: "PlaceProphecyTileCommand", tileId: tile.id }).state;
    }

    state = reduceCommand(state, { type: "DeliverProphecyCommand" }).state;

    expect(state.consultation.history[0]).toMatchObject({
      depthBand: expect.any(String),
      depth: expect.any(Number),
      omenConsensus: expect.any(String)
    });
    expect(state.consultation.history[0]?.scaffold?.length).toBe(3);
    expect(state.consultation.history[0]?.interpretation?.summary).toBeTruthy();
    expect(state.eventFeed.some((entry) => entry.text.includes("Sacred Record inscribed"))).toBe(true);
  });
});
