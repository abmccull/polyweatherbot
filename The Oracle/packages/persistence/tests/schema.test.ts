import { describe, expect, it } from "vitest";

import { createInitialState } from "@the-oracle/core";

import { deserializeSnapshot, MAX_SNAPSHOT_CHARS, serializeSnapshot } from "../src/schema";

function omitStateKeys<T extends object, K extends keyof T>(value: T, ...keys: K[]): Omit<T, K> {
  const clone = { ...value } as Partial<T>;
  for (const key of keys) {
    delete clone[key];
  }
  return clone as Omit<T, K>;
}

describe("snapshot schema", () => {
  it("round-trips the snapshot payload", () => {
    const snapshot = {
      version: 1 as const,
      state: createInitialState(),
      recentEvents: []
    };

    const restored = deserializeSnapshot(serializeSnapshot(snapshot));
    expect(restored.version).toBe(1);
    expect(restored.state.clock.day).toBe(1);
  });

  it("backfills carrier logistics fields when loading older saves", () => {
    const state = createInitialState();
    const legacySnapshot = {
      version: 1 as const,
      state: {
        ...state,
        walkers: state.walkers.map((walker) =>
          walker.role === "carrier"
            ? {
                id: walker.id,
                role: walker.role,
                name: walker.name,
                tile: walker.tile,
                state: walker.state,
                path: walker.path,
                moveCooldown: walker.moveCooldown
              }
            : walker
        )
      },
      recentEvents: []
    };

    const restored = deserializeSnapshot(JSON.stringify(legacySnapshot));
    const carrier = restored.state.walkers.find((walker) => walker.role === "carrier");

    expect(carrier?.fatigue).toBeDefined();
    expect(carrier?.haulingSkill).toBeDefined();
    expect(carrier?.supplyRadius).toBeDefined();
  });

  it("backfills campaign scaffolding when loading older saves", () => {
    const state = createInitialState();
    const legacyState = omitStateKeys(state, "campaign", "worldSeedText", "originId", "worldGeneration");

    const restored = deserializeSnapshot(
      JSON.stringify({
        version: 1 as const,
        state: legacyState,
        recentEvents: []
      })
    );

    expect(restored.state.campaign.scenarioId).toBe("rising-oracle");
    expect(restored.state.campaign.reputation.currentTier).toBe("obscure");
    expect(restored.state.campaign.worldMap.nodes.length).toBeGreaterThan(0);
    expect(restored.state.worldSeedText).toBe(String(state.worldSeed));
    expect(restored.state.originId).toBe("upstart-shrine");
    expect(restored.state.worldGeneration.originId).toBe("upstart-shrine");
  });

  it("backfills recurring character scaffolding when loading older saves", () => {
    const state = createInitialState();
    const legacyState = omitStateKeys(state, "characters");

    const restored = deserializeSnapshot(
      JSON.stringify({
        version: 1 as const,
        state: legacyState,
        recentEvents: []
      })
    );

    expect(restored.state.characters?.roster.length).toBeGreaterThanOrEqual(5);
    expect(restored.state.characters?.spotlightCharacterIds.length).toBeGreaterThan(0);
    expect(restored.state.characters?.roster.every((character) => character.memory.visitCount === 0)).toBe(true);
  });

  it("backfills rival oracle scaffolding when loading older saves", () => {
    const state = createInitialState();
    const legacyState = omitStateKeys(state, "rivalOracles");

    const restored = deserializeSnapshot(
      JSON.stringify({
        version: 1 as const,
        state: legacyState,
        recentEvents: []
      })
    );

    expect(restored.state.rivalOracles?.roster.length).toBeGreaterThanOrEqual(4);
    expect(restored.state.rivalOracles?.spotlightRivalIds.length).toBeGreaterThan(0);
    expect(restored.state.rivalOracles?.incidents).toEqual([]);
  });

  it("backfills priest politics scaffolding when loading older saves", () => {
    const state = createInitialState();
    const legacyState = omitStateKeys(state, "priestPolitics");

    const restored = deserializeSnapshot(
      JSON.stringify({
        version: 1 as const,
        state: legacyState,
        recentEvents: []
      })
    );

    expect(restored.state.priestPolitics?.blocs).toHaveLength(4);
    expect(restored.state.priestPolitics?.featuredCharacterIds.length).toBeGreaterThan(0);
    expect(restored.state.priestPolitics?.priests[state.priests[0]!.id]?.note).toBeTruthy();
  });

  it("rejects malformed nested snapshot payloads", () => {
    const snapshot = {
      version: 1 as const,
      state: {
        ...createInitialState(),
        walkers: "not-an-array"
      },
      recentEvents: []
    };

    expect(() => deserializeSnapshot(JSON.stringify(snapshot))).toThrow("Invalid Oracle snapshot");
  });

  it("rejects unsupported snapshot versions", () => {
    const snapshot = {
      version: 99,
      state: createInitialState(),
      recentEvents: []
    };

    expect(() => deserializeSnapshot(JSON.stringify(snapshot))).toThrow("Invalid Oracle snapshot");
  });

  it("rejects oversized payloads before loading", () => {
    const oversized = "x".repeat(MAX_SNAPSHOT_CHARS + 1);

    expect(() => deserializeSnapshot(oversized)).toThrow("Oracle snapshot exceeds maximum allowed size");
  });
});
