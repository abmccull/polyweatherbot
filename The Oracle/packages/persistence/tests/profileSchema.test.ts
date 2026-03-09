import { describe, expect, it } from "vitest";

import {
  createEmptyProfileSnapshot,
  deserializeProfileSnapshot,
  MAX_PROFILE_CHARS,
  normalizeProfileSnapshot,
  serializeProfileSnapshot
} from "../src/profileSchema";

describe("profile schema", () => {
  it("round-trips a profile snapshot", () => {
    const profile = normalizeProfileSnapshot({
      lineageName: "House of Smoke",
      preferredOriginId: "merchant-oracle",
      lastWorldSeed: 77,
      unlockedOrigins: ["ancient-spring", "merchant-oracle"],
      completedRuns: [
        {
          id: "run-1",
          label: "Merchant Ascendant",
          originId: "merchant-oracle",
          worldSeed: 77,
          completedAt: "2026-03-06T00:00:00.000Z",
          legacyScore: 1820,
          yearsRuled: 12,
          finalReputationTier: "revered",
          notableEvents: ["Resolved the Grain Corridor crisis."]
        }
      ]
    });

    const restored = deserializeProfileSnapshot(serializeProfileSnapshot(profile));
    expect(restored.lineageName).toBe("House of Smoke");
    expect(restored.completedRuns[0]?.legacyScore).toBe(1820);
  });

  it("creates an empty profile when one does not exist", () => {
    const profile = createEmptyProfileSnapshot("2026-03-06T00:00:00.000Z");

    expect(profile.unlockedOrigins).toEqual([]);
    expect(profile.completedRuns).toEqual([]);
    expect(profile.challengeCompletions).toEqual([]);
  });

  it("normalizes partial legacy profile payloads", () => {
    const profile = normalizeProfileSnapshot({
      lineageName: "House of Echoes",
      unlockedOrigins: ["war-oracle"]
    }, "2026-03-06T00:00:00.000Z");

    expect(profile.lineageName).toBe("House of Echoes");
    expect(profile.unlockedOrigins).toEqual(["war-oracle"]);
    expect(profile.completedRuns).toEqual([]);
    expect(profile.challengeCompletions).toEqual([]);
  });

  it("rejects malformed profile payloads", () => {
    expect(() =>
      deserializeProfileSnapshot(JSON.stringify({
        version: 1,
        unlockedOrigins: "merchant-oracle",
        completedRuns: [],
        challengeCompletions: [],
        createdAt: "2026-03-06T00:00:00.000Z",
        updatedAt: "2026-03-06T00:00:00.000Z"
      }))
    ).toThrow("Invalid Oracle profile snapshot");
  });

  it("rejects oversized profile payloads", () => {
    expect(() => deserializeProfileSnapshot("x".repeat(MAX_PROFILE_CHARS + 1))).toThrow(
      "Oracle profile exceeds maximum allowed size"
    );
  });
});
