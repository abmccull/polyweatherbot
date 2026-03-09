import { describe, expect, it } from "vitest";

import { selectCharacterRoster, selectCharacterSpotlight, selectPriestPoliticsOverview } from "../src/selectors";
import { normalizeCharactersState } from "../src/state/characters";
import { createInitialState } from "../src/state/initialState";

describe("recurring characters", () => {
  it("seeds a deterministic recurring roster from world and origin context", () => {
    const left = createInitialState({
      seed: "FR-DVG-8",
      originId: "merchant-oracle"
    });
    const right = createInitialState({
      seed: "FR-DVG-8",
      originId: "merchant-oracle"
    });
    const warOrigin = createInitialState({
      seed: "FR-DVG-8",
      originId: "war-oracle"
    });

    expect(left.characters).toEqual(right.characters);
    expect(left.characters?.roster.length).toBeGreaterThanOrEqual(5);
    expect(new Set(left.characters?.roster.map((character) => character.role))).toEqual(
      new Set(["merchant", "general", "envoy", "philosopher", "priest"])
    );
    expect(warOrigin.characters).not.toEqual(left.characters);
  });

  it("normalizes missing character scaffolding for older states", () => {
    const initial = createInitialState({
      seed: "legacy-character-backfill",
      originId: "ancient-spring"
    });

    const normalized = normalizeCharactersState({
      worldSeed: initial.worldSeed,
      worldGeneration: initial.worldGeneration,
      factions: initial.factions,
      characters: undefined
    });

    expect(normalized).toEqual(initial.characters);
    expect(normalized.roster.every((character) => character.memory.notableMoments.length > 0)).toBe(true);
  });

  it("exposes stable roster and spotlight summaries for later UI lanes", () => {
    const state = createInitialState({
      seed: "selector-surface",
      originId: "cursed-oracle"
    });

    const roster = selectCharacterRoster(state);
    const spotlight = selectCharacterSpotlight(state);
    const priestPolitics = selectPriestPoliticsOverview(state);
    const lastRosterEntry = roster[roster.length - 1];

    expect(roster.length).toBe(state.characters?.roster.length);
    expect(roster[0]?.prominence).toBeGreaterThanOrEqual(lastRosterEntry?.prominence ?? 0);
    expect(spotlight.length).toBeGreaterThan(0);
    expect(spotlight[0]?.memoryHook).toBeTruthy();
    expect(spotlight.every((entry) => roster.some((character) => character.id === entry.id))).toBe(true);
    expect(priestPolitics.featuredCharacters.length).toBeGreaterThan(0);
    expect(priestPolitics.featuredCharacters.every((character) => character.factionName !== undefined)).toBe(true);
  });
});
