import { describe, expect, it } from "vitest";

import { createInitialAgeState, advanceAge } from "../src/state/ages";

describe("age transitions", () => {
  it("starts in the archaic age", () => {
    const age = createInitialAgeState();
    expect(age.currentAgeId).toBe("archaic");
    expect(age.currentAgeIndex).toBe(0);
    expect(age.ageHistory).toHaveLength(1);
  });

  it("transitions to classical age at year 3", () => {
    const age = createInitialAgeState();
    const advanced = advanceAge(age, 3, 61);
    expect(advanced.currentAgeId).toBe("classical");
    expect(advanced.currentAgeIndex).toBe(1);
    expect(advanced.ageHistory).toHaveLength(2);
    expect(advanced.ageHistory[1]!.ageId).toBe("classical");
    expect(advanced.ageHistory[1]!.startYear).toBe(3);
  });

  it("does not transition when year has not changed", () => {
    const age = createInitialAgeState();
    const sameYear = advanceAge(age, 1, 30);
    expect(sameYear.currentAgeId).toBe("archaic");
    expect(sameYear).toBe(age); // Reference equality — no change
  });

  it("skips ages when jumping many years forward", () => {
    const age = createInitialAgeState();
    // Jump to year 15 — should land on roman_shadow, skipping classical/hellenic/hellenistic
    const advanced = advanceAge(age, 15, 1);
    expect(advanced.currentAgeId).toBe("roman_shadow");
    expect(advanced.currentAgeIndex).toBe(4);
    // History should still only have 2 entries: archaic start + the final jump
    expect(advanced.ageHistory).toHaveLength(2);
  });

  it("stays in the same age if already past the threshold", () => {
    const age = createInitialAgeState();
    const atClassical = advanceAge(age, 4, 1);
    expect(atClassical.currentAgeId).toBe("classical");

    // Advance to year 5 — still classical (hellenic starts at 6)
    const still = advanceAge(atClassical, 5, 1);
    expect(still.currentAgeId).toBe("classical");
    expect(still.ageHistory).toHaveLength(2);
  });
});
