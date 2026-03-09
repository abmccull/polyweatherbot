import { ageDefs, type AgeId } from "@the-oracle/content";

export type AgeState = {
  currentAgeId: AgeId;
  currentAgeIndex: number;
  ageHistory: Array<{ ageId: AgeId; startYear: number; startDay: number }>;
  lastAgeCheckYear: number;
};

export function createInitialAgeState(): AgeState {
  return {
    currentAgeId: "archaic",
    currentAgeIndex: 0,
    ageHistory: [{ ageId: "archaic", startYear: 1, startDay: 1 }],
    lastAgeCheckYear: 1
  };
}

export function advanceAge(ageState: AgeState, currentYear: number, currentDay: number, endlessMode?: boolean): AgeState {
  if (currentYear === ageState.lastAgeCheckYear) {
    return ageState;
  }

  let nextIndex = ageState.currentAgeIndex;

  // Find the highest age whose threshold we've passed
  for (let i = ageState.currentAgeIndex + 1; i < ageDefs.length; i++) {
    if (currentYear >= ageDefs[i]!.yearThreshold) {
      nextIndex = i;
    } else {
      break;
    }
  }

  // Endless mode: when in roman_shadow for 5+ years, cycle back to classical
  if (endlessMode && ageState.currentAgeId === "roman_shadow") {
    const romanShadowDef = ageDefs.find((a) => a.id === "roman_shadow");
    if (romanShadowDef && currentYear >= romanShadowDef.yearThreshold + 5) {
      const classicalIndex = ageDefs.findIndex((a) => a.id === "classical");
      if (classicalIndex >= 0) {
        const classicalDef = ageDefs[classicalIndex]!;
        return {
          currentAgeId: classicalDef.id,
          currentAgeIndex: classicalIndex,
          ageHistory: [
            ...ageState.ageHistory,
            { ageId: classicalDef.id, startYear: currentYear, startDay: currentDay }
          ],
          lastAgeCheckYear: currentYear
        };
      }
    }
  }

  if (nextIndex === ageState.currentAgeIndex) {
    return {
      ...ageState,
      lastAgeCheckYear: currentYear
    };
  }

  const nextAge = ageDefs[nextIndex]!;
  return {
    currentAgeId: nextAge.id,
    currentAgeIndex: nextIndex,
    ageHistory: [
      ...ageState.ageHistory,
      { ageId: nextAge.id, startYear: currentYear, startDay: currentDay }
    ],
    lastAgeCheckYear: currentYear
  };
}
