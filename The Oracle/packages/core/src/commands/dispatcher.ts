import type { GameCommand } from "./types";
import type { GameEvent, GameSnapshot, GameState } from "../state/gameState";
import { reduceCommand } from "../reducers";
import { advanceClock } from "../simulation/clock";
import { runSimulationTick } from "../simulation/updateDay";

export interface RuntimeSaveRepository {
  loadSlot(slotId: string): Promise<GameSnapshot | null>;
  saveSlot(slotId: string, snapshot: GameSnapshot): Promise<void>;
}

export class OracleRuntime {
  private state: GameState;

  private listeners = new Set<() => void>();

  private recentEvents: GameEvent[] = [];

  constructor(initialState: GameState, private readonly saveRepository?: RuntimeSaveRepository) {
    this.state = initialState;
  }

  getState(): GameState {
    return this.state;
  }

  getSnapshot(): GameSnapshot {
    return {
      version: 1,
      state: this.state,
      recentEvents: this.recentEvents.slice(-40)
    };
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  private appendEvents(events: GameEvent[]): void {
    this.recentEvents = [...this.recentEvents, ...events].slice(-80);
  }

  dispatch(command: GameCommand): void {
    if (command.type === "AdvanceTickCommand") {
      this.advanceTicks(command.ticks);
      return;
    }

    const result = reduceCommand(this.state, command);
    this.state = result.state;
    this.appendEvents(result.events);
    this.emit();
  }

  advanceTicks(ticks: number): void {
    const safeTicks = Math.max(0, Math.floor(ticks));
    if (safeTicks === 0) {
      return;
    }

    for (let index = 0; index < safeTicks; index += 1) {
      const previousClock = this.state.clock;
      const clockResult = advanceClock(previousClock, 1);
      this.state = {
        ...this.state,
        clock: clockResult.clock
      };

      const sim = runSimulationTick(this.state, 1, {
        previousClock
      });
      this.state = sim.state;
      this.appendEvents(sim.events);
    }

    this.emit();
  }

  async save(slotId: string): Promise<void> {
    if (!this.saveRepository) {
      return;
    }
    await this.saveRepository.saveSlot(slotId, this.getSnapshot());
  }

  async load(slotId: string): Promise<void> {
    if (!this.saveRepository) {
      return;
    }
    const snapshot = await this.saveRepository.loadSlot(slotId);
    if (!snapshot) {
      return;
    }
    this.state = snapshot.state;
    this.recentEvents = snapshot.recentEvents;
    this.emit();
  }
}
