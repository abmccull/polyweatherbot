import { describe, expect, it } from "vitest";

import { createInitialLegacyState, computeLegacyScore, generateLegacyArtifact } from "../src/state/legacy";
import { createInitialState } from "../src/state/initialState";
import { advanceDecline } from "../src/simulation/decline";

describe("legacy & decline", () => {
  it("initializes legacy state in thriving phase", () => {
    const legacy = createInitialLegacyState();
    expect(legacy.phase).toBe("thriving");
    expect(legacy.declineSeverity).toBe(0);
    expect(legacy.comebackAttempts).toBe(0);
    expect(legacy.legacyScore).toBe(0);
  });

  it("computes a non-negative legacy score for a fresh game", () => {
    const state = createInitialState();
    const score = computeLegacyScore(state);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1000);
  });

  it("does not enter decline when conditions are healthy", () => {
    const state = createInitialState();
    // Fresh game has gold > 30, credibility > 25, reputation is obscure but not all
    // conditions met simultaneously for decline at day 1
    const advanced = advanceDecline(state);
    expect(advanced.legacy!.phase).toBe("thriving");
  });

  it("enters decline when credibility, gold, and reputation are all low", () => {
    const state = createInitialState();
    // Engineer decline conditions: low gold, low credibility, obscure reputation
    const declineState = {
      ...state,
      resources: {
        ...state.resources,
        gold: { ...state.resources.gold, amount: 5 }
      },
      factions: Object.fromEntries(
        Object.entries(state.factions).map(([id, f]) => [id, { ...f, credibility: 10 }])
      ) as typeof state.factions,
      campaign: {
        ...state.campaign,
        reputation: { ...state.campaign.reputation, currentTier: "obscure" as const, score: 0 }
      }
    };

    const result = advanceDecline(declineState);
    expect(result.legacy!.phase).toBe("declining");
    expect(result.legacy!.declineSeverity).toBe(1);
  });

  it("generates a legacy artifact with required fields", () => {
    const state = createInitialState();
    const artifact = generateLegacyArtifact(state);
    expect(artifact.totalDays).toBeGreaterThan(0);
    expect(artifact.totalYears).toBeGreaterThanOrEqual(0);
    expect(artifact.finalScore).toBeGreaterThanOrEqual(0);
    expect(artifact.finalReputation).toBeTruthy();
    expect(artifact.epitaph).toBeTruthy();
    expect(Array.isArray(artifact.majorProphecies)).toBe(true);
    expect(Array.isArray(artifact.notablePatrons)).toBe(true);
    expect(Array.isArray(artifact.turningPoints)).toBe(true);
    expect(Array.isArray(artifact.namedFigures)).toBe(true);
  });
});
