import { describe, expect, it } from "vitest";

import { reduceCommand } from "../src/reducers";
import { createInitialState } from "../src/state/initialState";
import { runSimulationTick } from "../src/simulation/updateDay";

describe("Oracle logistics", () => {
  it("queues and delivers precinct resources to needy buildings", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 28, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "inner_sanctum", tile: { x: 29, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "castalian_spring", tile: { x: 30, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "eternal_flame_brazier", tile: { x: 31, y: 49 } }).state;

    const brazier = state.buildings.find((building) => building.defId === "eternal_flame_brazier");
    const sanctum = state.buildings.find((building) => building.defId === "inner_sanctum");
    expect(brazier).toBeDefined();
    expect(sanctum).toBeDefined();

    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.defId === "eternal_flame_brazier") {
          return {
            ...building,
            storedResources: {
              ...building.storedResources,
              olive_oil: 0
            }
          };
        }
        if (building.defId === "inner_sanctum") {
          return {
            ...building,
            storedResources: {
              ...building.storedResources,
              incense: 0,
              sacred_water: 0
            }
          };
        }
        return building;
      })
    };

    const result = runSimulationTick(state, 600);
    const nextState = result.state;
    const nextBrazier = nextState.buildings.find((building) => building.defId === "eternal_flame_brazier");
    const nextSanctum = nextState.buildings.find((building) => building.defId === "inner_sanctum");

    expect(nextBrazier?.storedResources.olive_oil ?? 0).toBeGreaterThan(0);
    expect(nextSanctum?.storedResources.incense ?? 0).toBeGreaterThan(0);
    expect(nextSanctum?.storedResources.sacred_water ?? 0).toBeGreaterThan(0);
    expect(nextState.walkers.find((walker) => walker.role === "carrier")?.assignedJobId).toBeUndefined();
  });

  it("balances stock between storehouses before precinct shortages become critical", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 27, y: 50 },
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 27, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 32, y: 49 } }).state;

    const [westStorehouse, eastStorehouse] = state.buildings.filter((building) => building.defId === "storehouse");
    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === westStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 12,
              incense: 8,
              grain: 16
            }
          };
        }
        if (building.id === eastStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 0,
              incense: 0,
              grain: 0
            }
          };
        }
        return building;
      })
    };

    const nextState = runSimulationTick(state, 1).state;
    const rebalanceJobs = nextState.resourceJobs.filter(
      (job) => job.sourceBuildingId === westStorehouse?.id && job.targetBuildingId === eastStorehouse?.id
    );

    expect(rebalanceJobs.length).toBeGreaterThan(0);
    expect(rebalanceJobs.some((job) => job.resourceId === "olive_oil")).toBe(true);
  });

  it("chooses the nearest stocked storehouse when feeding sacred buildings", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 26, y: 50 },
      { x: 27, y: 50 },
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 },
      { x: 33, y: 50 },
      { x: 34, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 26, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 32, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "inner_sanctum", tile: { x: 33, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "eternal_flame_brazier", tile: { x: 34, y: 49 } }).state;

    const [westStorehouse, eastStorehouse] = state.buildings.filter((building) => building.defId === "storehouse");
    const sanctum = state.buildings.find((building) => building.defId === "inner_sanctum");
    const brazier = state.buildings.find((building) => building.defId === "eternal_flame_brazier");
    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === westStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 8,
              incense: 8,
              grain: 10
            }
          };
        }
        if (building.id === eastStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 5,
              incense: 5,
              grain: 10
            }
          };
        }
        if (building.id === sanctum?.id) {
          return {
            ...building,
            storedResources: {
              incense: 0,
              sacred_water: 0
            }
          };
        }
        if (building.id === brazier?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 0
            }
          };
        }
        return building;
      })
    };

    const nextState = runSimulationTick(state, 1).state;
    const sanctumJob = nextState.resourceJobs.find((job) => job.targetBuildingId === sanctum?.id && job.resourceId === "incense");
    const brazierJob = nextState.resourceJobs.find((job) => job.targetBuildingId === brazier?.id && job.resourceId === "olive_oil");

    expect(sanctumJob?.sourceBuildingId).toBe(eastStorehouse?.id);
    expect(brazierJob?.sourceBuildingId).toBe(eastStorehouse?.id);
  });

  it("queues supply jobs for the new grain, altar, and olive chains", () => {
    let state = createInitialState();
    const roadTiles = Array.from({ length: 12 }, (_, index) => ({ x: 26 + index, y: 50 }));

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 26, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "granary", tile: { x: 27, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "kitchen", tile: { x: 28, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "animal_pen", tile: { x: 29, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "sacrificial_altar", tile: { x: 30, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "olive_press", tile: { x: 31, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "incense_store", tile: { x: 32, y: 49 } }).state;

    const storehouse = state.buildings.find((building) => building.defId === "storehouse");
    const kitchen = state.buildings.find((building) => building.defId === "kitchen");
    const animalPen = state.buildings.find((building) => building.defId === "animal_pen");
    const altar = state.buildings.find((building) => building.defId === "sacrificial_altar");
    const olivePress = state.buildings.find((building) => building.defId === "olive_press");
    const incenseStore = state.buildings.find((building) => building.defId === "incense_store");

    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === storehouse?.id) {
          return {
            ...building,
            storedResources: {
              grain: 20,
              incense: 10,
              olives: 8,
              papyrus: 4,
              olive_oil: 4
            }
          };
        }
        if (building.id === kitchen?.id) {
          return {
            ...building,
            storedResources: {
              grain: 0,
              bread: 0
            }
          };
        }
        if (building.id === animalPen?.id) {
          return {
            ...building,
            storedResources: {
              grain: 0,
              sacred_animals: 1
            }
          };
        }
        if (building.id === altar?.id) {
          return {
            ...building,
            assignedPriestIds: ["priest-1"],
            storedResources: {
              sacred_animals: 0,
              incense: 0
            }
          };
        }
        if (building.id === olivePress?.id) {
          return {
            ...building,
            storedResources: {
              olives: 0,
              olive_oil: 0
            }
          };
        }
        if (building.id === incenseStore?.id) {
          return {
            ...building,
            storedResources: {
              incense: 0,
              papyrus: 0
            }
          };
        }
        return building;
      })
    };

    const nextState = runSimulationTick(state, 1).state;

    expect(nextState.resourceJobs.some((job) => job.targetBuildingId === kitchen?.id && job.resourceId === "grain")).toBe(true);
    expect(nextState.resourceJobs.some((job) => job.targetBuildingId === olivePress?.id && job.resourceId === "olives")).toBe(true);
    expect(nextState.resourceJobs.some((job) => job.targetBuildingId === altar?.id && job.resourceId === "sacred_animals")).toBe(true);
    expect(nextState.resourceJobs.some((job) => job.targetBuildingId === incenseStore?.id && job.resourceId === "incense")).toBe(true);
  });

  it("processes kitchen, animal pen, olive press, and altar recipes once supplied", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "kitchen", tile: { x: 28, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "animal_pen", tile: { x: 29, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "olive_press", tile: { x: 30, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "sacrificial_altar", tile: { x: 31, y: 49 } }).state;

    const kitchen = state.buildings.find((building) => building.defId === "kitchen");
    const animalPen = state.buildings.find((building) => building.defId === "animal_pen");
    const olivePress = state.buildings.find((building) => building.defId === "olive_press");
    const altar = state.buildings.find((building) => building.defId === "sacrificial_altar");

    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === kitchen?.id) {
          return {
            ...building,
            storedResources: {
              grain: 6,
              bread: 0
            }
          };
        }
        if (building.id === animalPen?.id) {
          return {
            ...building,
            storedResources: {
              grain: 5,
              sacred_animals: 0
            }
          };
        }
        if (building.id === olivePress?.id) {
          return {
            ...building,
            storedResources: {
              olives: 5,
              olive_oil: 0
            }
          };
        }
        if (building.id === altar?.id) {
          return {
            ...building,
            assignedPriestIds: ["priest-1"],
            storedResources: {
              sacred_animals: 1.2,
              incense: 1.6
            }
          };
        }
        return building;
      })
    };

    const nextState = runSimulationTick(state, 1800).state;
    const nextKitchen = nextState.buildings.find((building) => building.id === kitchen?.id);
    const nextAnimalPen = nextState.buildings.find((building) => building.id === animalPen?.id);
    const nextOlivePress = nextState.buildings.find((building) => building.id === olivePress?.id);
    const nextAltar = nextState.buildings.find((building) => building.id === altar?.id);

    expect(nextKitchen?.storedResources.bread ?? 0).toBeGreaterThan(0);
    expect(nextAnimalPen?.storedResources.sacred_animals ?? 0).toBeGreaterThan(0);
    expect(nextOlivePress?.storedResources.olive_oil ?? 0).toBeGreaterThan(0);
    expect(nextAltar?.storedResources.sacred_animals ?? 100).toBeLessThan(1.2);
    expect(nextAltar?.condition ?? 0).toBeGreaterThanOrEqual(altar?.condition ?? 0);
  });

  it("adds extra carriers and keeps critical sacred jobs ahead of routine rebalancing", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 26, y: 50 },
      { x: 27, y: 50 },
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 },
      { x: 33, y: 50 },
      { x: 34, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 26, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 32, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "inner_sanctum", tile: { x: 33, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "eternal_flame_brazier", tile: { x: 34, y: 49 } }).state;

    const [westStorehouse, eastStorehouse] = state.buildings.filter((building) => building.defId === "storehouse");
    const sanctum = state.buildings.find((building) => building.defId === "inner_sanctum");
    const brazier = state.buildings.find((building) => building.defId === "eternal_flame_brazier");
    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === westStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 12,
              incense: 10,
              grain: 18
            }
          };
        }
        if (building.id === eastStorehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 0,
              incense: 0,
              grain: 0
            }
          };
        }
        if (building.id === sanctum?.id) {
          return {
            ...building,
            storedResources: {
              incense: 0,
              sacred_water: 0
            }
          };
        }
        if (building.id === brazier?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 0
            }
          };
        }
        return building;
      })
    };

    const nextState = runSimulationTick(state, 1).state;
    const carriers = nextState.walkers.filter((walker) => walker.role === "carrier");
    const assignedJobs = nextState.resourceJobs.filter((job) => job.assignedWalkerId);

    expect(carriers.length).toBeGreaterThan(1);
    expect(assignedJobs.length).toBeGreaterThan(0);
    expect(assignedJobs.every((job) => job.priority !== "routine")).toBe(true);
    expect(nextState.resourceJobs.some((job) => job.priority === "routine" && !job.assignedWalkerId)).toBe(true);
  });

  it("prefers fresher carriers with enough range for critical long-haul jobs", () => {
    let state = createInitialState();
    const roadTiles = [
      { x: 28, y: 50 },
      { x: 29, y: 50 },
      { x: 30, y: 50 },
      { x: 31, y: 50 },
      { x: 32, y: 50 },
      { x: 33, y: 50 },
      { x: 34, y: 50 }
    ];

    for (const tile of roadTiles) {
      state = reduceCommand(state, { type: "PlaceRoadCommand", tile }).state;
    }

    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "storehouse", tile: { x: 28, y: 49 } }).state;
    state = reduceCommand(state, { type: "PlaceBuildingCommand", defId: "eternal_flame_brazier", tile: { x: 34, y: 49 } }).state;

    const storehouse = state.buildings.find((building) => building.defId === "storehouse");
    const brazier = state.buildings.find((building) => building.defId === "eternal_flame_brazier");

    state = {
      ...state,
      buildings: state.buildings.map((building) => {
        if (building.id === storehouse?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 8,
              incense: 2,
              grain: 6
            }
          };
        }
        if (building.id === brazier?.id) {
          return {
            ...building,
            storedResources: {
              olive_oil: 0
            }
          };
        }
        return building;
      }),
      walkers: [
        ...state.walkers.filter((walker) => walker.role !== "carrier"),
        {
          id: "walker-carrier-weary",
          role: "carrier",
          name: "Weary Carrier",
          tile: { x: 28, y: 50 },
          state: "idle",
          path: [],
          moveCooldown: 0,
          fatigue: 84,
          haulingSkill: 40,
          supplyRadius: 3
        },
        {
          id: "walker-carrier-fresh",
          role: "carrier",
          name: "Fresh Carrier",
          tile: { x: 31, y: 50 },
          state: "idle",
          path: [],
          moveCooldown: 0,
          fatigue: 12,
          haulingSkill: 74,
          supplyRadius: 10
        }
      ]
    };

    const nextState = runSimulationTick(state, 1).state;
    const brazierJob = nextState.resourceJobs.find((job) => job.targetBuildingId === brazier?.id && job.resourceId === "olive_oil");
    const freshCarrier = nextState.walkers.find((walker) => walker.id === "walker-carrier-fresh");

    expect(brazierJob?.assignedWalkerId).toBe("walker-carrier-fresh");
    expect(freshCarrier?.fatigue ?? 0).toBeGreaterThan(12);
  });

  it("lets idle carriers recover fatigue between haul waves", () => {
    const state = {
      ...createInitialState(),
      walkers: createInitialState().walkers.map((walker) =>
        walker.role === "carrier"
          ? {
              ...walker,
              fatigue: 46
            }
          : walker
      )
    };

    const nextState = runSimulationTick(state, 30).state;
    const carrier = nextState.walkers.find((walker) => walker.role === "carrier");

    expect(carrier?.fatigue ?? 100).toBeLessThan(46);
  });
});
