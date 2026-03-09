import { describe, expect, it } from "vitest";

import { reduceCommand } from "../src/reducers";
import { createInitialState } from "../src/state/initialState";
import { runSimulationTick } from "../src/simulation/updateDay";

describe("Oracle chronicle", () => {
  it("records consequence resolution back onto the prophecy history", () => {
    let state = createInitialState(9);
    state = reduceCommand(state, { type: "InjectScenarioCommand", scenario: "consultation-ready" }).state;
    state = reduceCommand(state, { type: "StartConsultationCommand" }).state;

    const tileIds = state.consultation.current?.tilePool.slice(0, 4).map((tile) => tile.id) ?? [];
    for (const tileId of tileIds) {
      state = reduceCommand(state, { type: "PlaceProphecyTileCommand", tileId }).state;
    }
    state = reduceCommand(state, { type: "DeliverProphecyCommand" }).state;
    state = {
      ...state,
      consequences: state.consequences.map((entry, index) => (index === 0 ? { ...entry, dueDay: state.clock.day } : entry))
    };

    const tickResult = runSimulationTick(state, 1, {
      previousClock: state.clock
    });
    const prophecy = tickResult.state.consultation.history[0];

    expect(prophecy?.resolved).toBe(true);
    expect(prophecy?.resolvedDay).toBe(state.clock.day);
    expect(prophecy?.resolutionReport).toBeTruthy();
    expect(tickResult.state.factions[prophecy!.factionId].history[0]).toBe(prophecy?.resolutionReport);
  });
});
