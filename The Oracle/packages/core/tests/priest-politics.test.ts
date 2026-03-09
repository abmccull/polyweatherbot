import { describe, expect, it } from "vitest";

import { createBuildingAt } from "../src/simulation/updateDay";
import { selectPriestPoliticsOverview, selectPriestRosterInsights } from "../src/selectors";
import { createInitialState } from "../src/state/initialState";
import { advancePriestPoliticsState, normalizePriestPoliticsState } from "../src/state/priestPolitics";

describe("priest politics", () => {
  it("seeds deterministic priest profiles and council pressure", () => {
    const left = createInitialState({
      seed: "FR-DVG-6-priest",
      originId: "merchant-oracle"
    });
    const right = createInitialState({
      seed: "FR-DVG-6-priest",
      originId: "merchant-oracle"
    });
    const shifted = createInitialState({
      seed: "FR-DVG-6-priest-alt",
      originId: "merchant-oracle"
    });

    expect(left.priestPolitics).toEqual(right.priestPolitics);
    expect(left.priestPolitics?.blocs).toHaveLength(4);
    expect(left.priestPolitics?.featuredCharacterIds.length).toBeGreaterThan(0);
    expect(shifted.priestPolitics).not.toEqual(left.priestPolitics);
  });

  it("backfills missing politics state and filters stale priest records", () => {
    const state = createInitialState({
      seed: "legacy-priest-politics",
      originId: "ancient-spring"
    });

    const normalized = normalizePriestPoliticsState({
      ...state,
      priestPolitics: {
        ...state.priestPolitics!,
        priests: {
          "obsolete-priest": state.priestPolitics!.priests[state.priests[0]!.id]!
        }
      }
    });

    expect(Object.keys(normalized.priests)).toEqual([state.priests[0]!.id]);
    expect(normalized.featuredCharacterIds.length).toBeGreaterThan(0);
    expect(normalized.blocs.map((bloc) => bloc.id)).toEqual(["pythia", "rites", "patrons", "reformers"]);
  });

  it("raises visible pressure when rites go uncovered and the pythia is strained", () => {
    const state = createInitialState({
      seed: "pressure-visible",
      originId: "cursed-oracle"
    });
    const strained = {
      ...state,
      clock: {
        ...state.clock,
        day: state.clock.day + 1
      },
      buildings: [
        createBuildingAt("inner_sanctum", { x: 12, y: 12 }, "building-inner-sanctum")
      ],
      pythia: {
        ...state.pythia,
        attunement: 48,
        needs: {
          ...state.pythia.needs,
          rest: 82,
          purification: 88
        }
      }
    };

    const nextPolitics = advancePriestPoliticsState({
      ...strained,
      priestPolitics: state.priestPolitics
    });
    const overview = selectPriestPoliticsOverview({
      ...strained,
      priestPolitics: nextPolitics
    });
    const roster = selectPriestRosterInsights({
      ...strained,
      priestPolitics: nextPolitics
    });

    expect(nextPolitics.overallPressure).toBeGreaterThan(state.priestPolitics?.overallPressure ?? 0);
    expect(["restless", "fractured", "crisis"]).toContain(nextPolitics.status);
    expect(overview.currentIssue).toContain("rite");
    expect(roster[0]?.note).toBeTruthy();
    expect(roster[0]?.anchorName).toBeTruthy();
  });
});
