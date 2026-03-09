import { describe, expect, it } from "vitest";

import type { FactionId } from "@the-oracle/content";

import { OracleRuntime } from "../src/commands/dispatcher";
import { advancePhilosopherThreats } from "../src/simulation/philosophers";
import { createInitialState } from "../src/state/initialState";

describe("philosopher threats", () => {
  it("initializes deterministically from seed and origin-driven world state", () => {
    const left = createInitialState({
      seed: "Delphi-001",
      originId: "merchant-oracle"
    });
    const right = createInitialState({
      seed: "Delphi-001",
      originId: "merchant-oracle"
    });
    const warOrigin = createInitialState({
      seed: "Delphi-001",
      originId: "war-oracle"
    });

    expect(left.philosophers).toEqual(right.philosophers);
    expect(left.philosophers?.spotlightFactionIds.length).toBeGreaterThan(0);
    expect(left.philosophers?.byFaction.corinth.worldview.length).toBeGreaterThan(0);
    expect(warOrigin.philosophers).not.toEqual(left.philosophers);
  });

  it("escalates stressed factions into authored philosopher threats", () => {
    const initial = createInitialState(71);
    const result = advancePhilosopherThreats({
      ...initial,
      clock: {
        ...initial.clock,
        month: 4,
        day: 1
      },
      factions: {
        ...initial.factions,
        thebes: {
          ...initial.factions.thebes,
          favour: 34,
          credibility: 28,
          debt: 26,
          dependence: 54,
          currentAgenda: "succession",
          embargoes: ["corinth"],
          activeConflicts: ["sparta", "athens"]
        }
      },
      philosophers: {
        ...initial.philosophers!,
        byFaction: {
          ...initial.philosophers!.byFaction,
          thebes: {
            ...initial.philosophers!.byFaction.thebes,
            philosopherId: "anaxagoras",
            worldview: "Skeptical courts",
            influence: 54,
            suspicion: 46,
            pressure: 64,
            stage: "sect",
            active: true
          }
        }
      }
    });

    expect(result.philosophers.byFaction.thebes.pressure).toBeGreaterThan(64);
    expect(["sect", "crisis"]).toContain(result.philosophers.byFaction.thebes.stage);
    expect(result.factions.thebes.history[0]).toContain("legitimacy");
    expect(result.factions.thebes.credibility).toBeLessThan(initial.factions.thebes.credibility);
    expect(result.feedItems.some((item) => item.text.includes("Anaxagoras"))).toBe(true);
  });

  it("advances philosopher threats through the live month-turnover runtime path", () => {
    const initial = createInitialState(83);
    const spotlightFactionIds = ([
      "athens",
      ...initial.philosophers!.spotlightFactionIds.filter((factionId) => factionId !== "athens")
    ] as FactionId[]).slice(0, 3);
    const runtime = new OracleRuntime({
      ...initial,
      clock: {
        ...initial.clock,
        day: 30,
        month: 1,
        tick: initial.clock.ticksPerDay * 29 + initial.clock.ticksPerDay - 1,
        tickOfDay: initial.clock.ticksPerDay - 1
      },
      factions: {
        ...initial.factions,
        athens: {
          ...initial.factions.athens,
          favour: 39,
          credibility: 34,
          debt: 24,
          dependence: 48,
          currentAgenda: "trade",
          embargoes: ["sparta"],
          activeConflicts: ["sparta"]
        }
      },
      philosophers: {
        ...initial.philosophers!,
        spotlightFactionIds,
        byFaction: {
          ...initial.philosophers!.byFaction,
          athens: {
            ...initial.philosophers!.byFaction.athens,
            philosopherId: "gorgias",
            worldview: "Skeptical courts",
            influence: 50,
            suspicion: 42,
            pressure: 61,
            stage: "circle",
            active: true
          }
        }
      }
    });

    runtime.advanceTicks(1);

    const state = runtime.getState();
    expect(state.clock.month).toBe(2);
    expect(state.philosophers?.byFaction.athens.pressure).toBeGreaterThan(61);
    expect(["sect", "crisis"]).toContain(state.philosophers?.byFaction.athens.stage);
    expect(state.eventFeed.some((item) => item.text.includes("Gorgias"))).toBe(true);
  });
});
