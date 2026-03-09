import { describe, expect, it } from "vitest";

import { OracleRuntime } from "../src/commands/dispatcher";
import { renderGameToText, selectRivalOracleSummary } from "../src/selectors";
import { createInitialState } from "../src/state/initialState";

describe("rival oracles", () => {
  it("initializes deterministically with patrons, intel uncertainty, and operations", () => {
    const left = createInitialState({
      seed: "Rival-001",
      originId: "merchant-oracle"
    });
    const right = createInitialState({
      seed: "Rival-001",
      originId: "merchant-oracle"
    });

    expect(left.rivalOracles).toEqual(right.rivalOracles);
    expect(left.rivalOracles?.roster.length).toBeGreaterThanOrEqual(4);
    expect(left.rivalOracles?.roster.every((rival) => rival.operations.length > 0)).toBe(true);
    expect(left.rivalOracles?.roster.some((rival) => rival.visibility !== rival.intel)).toBe(true);
  });

  it("advances rival operations on day turnover and records espionage pressure", () => {
    const initial = createInitialState(71);
    const runtime = new OracleRuntime({
      ...initial,
      clock: {
        ...initial.clock,
        day: 14,
        tick: initial.clock.ticksPerDay * 13 + initial.clock.ticksPerDay - 1,
        tickOfDay: initial.clock.ticksPerDay - 1
      }
    });

    runtime.advanceTicks(1);

    const state = runtime.getState();
    expect(state.clock.day).toBe(15);
    expect(state.rivalOracles?.lastPressureDay).toBe(15);
    expect(state.rivalOracles?.incidents.length).toBeGreaterThan(0);
    expect(state.rivalOracles?.roster.some((rival) => rival.lastOperationDay === 15)).toBe(true);
    expect(state.eventFeed.some((item) => item.text.toLowerCase().includes("rival") || item.text.toLowerCase().includes("counter-rite"))).toBe(true);
  });

  it("surfaces rival pressure in text-state world summaries", () => {
    const state = createInitialState(19);
    const payload = JSON.parse(renderGameToText(state));
    const rivalSummary = selectRivalOracleSummary(state);

    expect(Array.isArray(payload.rival_oracles)).toBe(true);
    expect(payload.world_summary.rivalOraclePressure).toBe(state.rivalOracles?.totalPressure ?? 0);
    expect(payload.world_summary.leadingRivals.length).toBeGreaterThan(0);
    expect(rivalSummary[0]?.patronLabel).toBeTruthy();
  });
});
