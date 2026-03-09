import { describe, expect, it } from "vitest";

import { createInitialExcavationState } from "../src/state/excavation";
import { createInitialState } from "../src/state/initialState";
import { reduceCommand } from "../src/reducers";

describe("excavation", () => {
  it("generates excavation sites from world seed", () => {
    const excavation = createInitialExcavationState(42);
    expect(excavation.sites.length).toBeGreaterThanOrEqual(2);
    expect(excavation.sites.length).toBeLessThanOrEqual(4);
    // First site should be discovered
    expect(excavation.sites[0]!.status).toBe("discovered");
    // Subsequent sites should be undiscovered
    if (excavation.sites.length > 1) {
      expect(excavation.sites[1]!.status).toBe("undiscovered");
    }
  });

  it("generates layers with valid contents", () => {
    const excavation = createInitialExcavationState(123);
    const validContents = new Set(["empty", "pottery", "relic", "sacred_fragment", "ancient_chamber"]);
    for (const site of excavation.sites) {
      expect(site.layers.length).toBe(site.maxDepth);
      for (const layer of site.layers) {
        expect(validContents.has(layer.contents)).toBe(true);
        expect(layer.revealed).toBe(false);
      }
    }
  });

  it("BeginExcavationCommand changes site status to excavating", () => {
    const state = createInitialState();
    const site = state.excavation!.sites.find((s) => s.status === "discovered");
    if (!site) {
      // Skip if no discovered sites (seed-dependent)
      return;
    }

    const { state: next } = reduceCommand(state, {
      type: "BeginExcavationCommand",
      siteId: site.id
    });

    const updatedSite = next.excavation!.sites.find((s) => s.id === site.id);
    expect(updatedSite!.status).toBe("excavating");
  });

  it("generates sacred sites", () => {
    const excavation = createInitialExcavationState(99);
    expect(excavation.sacredSites.length).toBeGreaterThanOrEqual(1);
    const validKinds = new Set(["oracle_spring", "earth_fissure", "ancient_altar", "sacred_grove"]);
    for (const sacred of excavation.sacredSites) {
      expect(validKinds.has(sacred.kind)).toBe(true);
      expect(sacred.discovered).toBe(false);
      expect(sacred.active).toBe(false);
    }
  });

  it("different seeds produce different site layouts", () => {
    const excavation1 = createInitialExcavationState(1);
    const excavation2 = createInitialExcavationState(9999);
    // While technically possible to collide, extremely unlikely
    const sites1 = excavation1.sites.map((s) => `${s.tile.x},${s.tile.y}`).join(";");
    const sites2 = excavation2.sites.map((s) => `${s.tile.x},${s.tile.y}`).join(";");
    expect(sites1).not.toBe(sites2);
  });
});
