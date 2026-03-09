import { describe, expect, it } from "vitest";

import { reduceCommand } from "../src/reducers";
import type { TileSemantics, WordTile } from "../src/state/gameState";
import { createInitialState } from "../src/state/initialState";

describe("Oracle reducers", () => {
  it("requires Sacred Way before buildings can be placed", () => {
    const state = createInitialState();
    const result = reduceCommand(state, {
      type: "PlaceBuildingCommand",
      defId: "priest_quarters",
      tile: { x: 30, y: 50 }
    });

    expect(result.state.buildings).toHaveLength(0);
  });

  it("places roads and then buildings adjacent to them", () => {
    let state = createInitialState();
    state = reduceCommand(state, { type: "PlaceRoadCommand", tile: { x: 30, y: 50 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "priest_quarters", tile: { x: 30, y: 49 } }).state;

    expect(state.grid.roads).toEqual([{ x: 30, y: 50 }]);
    expect(state.buildings[0]?.defId).toBe("priest_quarters");
  });

  it("folds building startup stock into the precinct reserve ledger", () => {
    let state = createInitialState();
    state = reduceCommand(state, { type: "PlaceRoadCommand", tile: { x: 30, y: 50 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 30, y: 49 } }).state;

    expect(state.resources.olive_oil.amount).toBe(22);
    expect(state.resources.incense.amount).toBe(15);
    expect(state.resources.grain.amount).toBe(48);
  });

  it("keeps event feed ids unique across repeated build actions", () => {
    let state = createInitialState();
    state = reduceCommand(state, { type: "PlaceRoadCommand", tile: { x: 30, y: 50 } }).state;
    state = reduceCommand(state, { type: "PlaceRoadCommand", tile: { x: 31, y: 50 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 31, y: 49 } }).state;

    const ids = state.eventFeed.map((event) => event.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("purchases a trade offer into the storehouse", () => {
    let state = createInitialState();
    state.tradeOffers = [
      {
        id: "trade-1",
        factionId: "corinth",
        resourceId: "incense",
        amount: 6,
        price: 12
      }
    ];
    state = reduceCommand(state, { type: "PlaceRoadCommand", tile: { x: 30, y: 50 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 30, y: 49 } }).state;

    state = reduceCommand(state, { type: "PurchaseTradeOfferCommand", offerId: "trade-1" }).state;

    expect(state.tradeOffers).toHaveLength(0);
    expect(state.resources.gold.amount).toBeLessThan(120);
    expect(state.resources.incense.amount).toBe(21);
    expect(state.buildings[0]?.storedResources.incense).toBe(9);
  });

  it("lets the player rest the Pythia to reduce fatigue", () => {
    let state = createInitialState();
    state.pythia.needs.rest = 78;
    state.pythia.mentalClarity = 60;

    state = reduceCommand(state, { type: "RestPythiaCommand" }).state;

    expect(state.pythia.needs.rest).toBe(54);
    expect(state.pythia.mentalClarity).toBe(68);
    expect(state.eventFeed[0]?.text).toContain("quiet rest");
  });

  it("purifies the Pythia by spending ritual supplies", () => {
    let state = createInitialState();
    state.pythia.needs.purification = 76;

    state = reduceCommand(state, { type: "PurifyPythiaCommand" }).state;

    expect(state.resources.sacred_water.amount).toBe(28);
    expect(state.resources.incense.amount).toBe(11);
    expect(state.pythia.needs.purification).toBe(48);
    expect(state.pythia.attunement).toBe(72);
  });

  it("refuses purification when supplies are missing", () => {
    let state = createInitialState();
    state.resources.sacred_water.amount = 1;
    state.resources.incense.amount = 0;

    state = reduceCommand(state, { type: "PurifyPythiaCommand" }).state;

    expect(state.resources.sacred_water.amount).toBe(1);
    expect(state.resources.incense.amount).toBe(0);
    expect(state.eventFeed[0]?.text).toContain("Purification failed");
  });

  it("creates cross-faction diplomatic reactions after prophecy delivery", () => {
    const militaryFavorable: TileSemantics = {
      target: "army",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    };
    const tilePool: WordTile[] = [
      { id: "subject-army", text: "the army", category: "subject", semantics: militaryFavorable },
      { id: "action-triumph", text: "shall triumph", category: "action", semantics: militaryFavorable },
      { id: "condition-harvest", text: "before harvest", category: "condition", semantics: militaryFavorable },
      { id: "seal-phoebus", text: "so shines Phoebus", category: "seal", semantics: militaryFavorable }
    ];
    let state = createInitialState();
    state = {
      ...state,
      consultation: {
        mode: "open",
        history: [],
        current: {
          id: "consultation-manual",
          factionId: "sparta",
          envoyName: "Spartan Envoy",
          mood: "measured",
          paymentOffered: 40,
          question: "Will the hoplites advance?",
          domain: "military",
          omenReports: [],
          tilePool,
          placedTileIds: tilePool.map((tile) => tile.id),
          scorePreview: {
            clarity: 80,
            value: 88,
            risk: 42
          }
        }
      }
    };

    const initialAthensFavour = state.factions.athens.favour;
    const initialAthensSpartaRelation = state.factions.athens.relations.sparta ?? 0;
    const result = reduceCommand(state, { type: "DeliverProphecyCommand" }).state;

    expect(result.factions.athens.favour).toBeLessThan(initialAthensFavour);
    expect((result.factions.athens.relations.sparta ?? 0)).toBeLessThan(initialAthensSpartaRelation);
    expect(result.factions.athens.history[0]).toContain("Sparta");
    expect(result.eventFeed.some((entry) => entry.text.includes("Athens"))).toBe(true);
  });
});
