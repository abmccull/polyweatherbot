import { describe, expect, it } from "vitest";

import { reduceCommand } from "../src/reducers";
import { createInitialState } from "../src/state/initialState";
import { buildWorldGeneration } from "../src/state/worldGen";

describe("origin and world generation", () => {
  it("recreates the same world for the same seed and origin", () => {
    const left = buildWorldGeneration({
      seed: "Delphi-001",
      originId: "merchant-oracle"
    });
    const right = buildWorldGeneration({
      seed: "Delphi-001",
      originId: "merchant-oracle"
    });

    expect(right).toEqual(left);
  });

  it("creates distinct opening conditions for different origins on the same seed", () => {
    const merchant = createInitialState({
      seed: "Delphi-001",
      originId: "merchant-oracle"
    });
    const war = createInitialState({
      seed: "Delphi-001",
      originId: "war-oracle"
    });

    expect(merchant.originId).not.toBe(war.originId);
    expect(merchant.worldGeneration.summary).not.toBe(war.worldGeneration.summary);
    expect(merchant.resources.gold.amount).toBeGreaterThan(war.resources.gold.amount);
    expect(merchant.pythia.prestige).not.toBe(war.pythia.prestige);
    expect(merchant.campaign.worldMap.activePressures).not.toEqual(war.campaign.worldMap.activePressures);
  });

  it("starts a new run from the reducer using the selected seed and origin", () => {
    const initial = createInitialState({
      seed: "Delphi-001",
      originId: "upstart-shrine"
    });

    const result = reduceCommand(initial, {
      type: "StartNewRunCommand",
      seed: "Cinder-77",
      originId: "cursed-oracle"
    }).state;

    expect(result.worldSeedText).toBe("Cinder-77");
    expect(result.originId).toBe("cursed-oracle");
    expect(result.worldGeneration.originTitle).toContain("Cursed");
    expect(result.eventFeed[0]?.text).toContain("Cursed Oracle");
    expect(result.worldGeneration.summary).not.toBe(initial.worldGeneration.summary);
  });
});
