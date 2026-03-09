import type { GameState } from "./state/gameState";

/**
 * Creates a memoized selector that returns the cached value when the state
 * reference hasn't changed. Useful for avoiding redundant derived-state
 * computations in render loops.
 */
export function createMemoizedSelector<T>(
  selector: (state: GameState) => T
): (state: GameState) => T {
  let lastState: GameState | undefined;
  let lastResult: T;

  return (state: GameState): T => {
    if (state === lastState) {
      return lastResult;
    }
    lastState = state;
    lastResult = selector(state);
    return lastResult;
  };
}

/**
 * Applies multiple state transforms in sequence, passing the output of each
 * into the next. This avoids intermediate allocations when consumers would
 * otherwise chain several individual dispatch calls.
 */
export function batchStateUpdates(
  state: GameState,
  updates: Array<(s: GameState) => GameState>
): GameState {
  let current = state;
  for (const update of updates) {
    current = update(current);
  }
  return current;
}

/**
 * Simple wrapper that measures how long a synchronous function takes to
 * execute. Intended for profiling tick functions during development.
 */
export function measureTickPerformance(fn: () => void): { durationMs: number } {
  const start = performance.now();
  fn();
  const end = performance.now();
  return { durationMs: end - start };
}
