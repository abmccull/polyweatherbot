import {
  OracleRuntime,
  createInitialState,
  normalizeOriginId,
  renderGameToText,
  selectRunSetupOriginOptions,
  selectRunSetupPreview,
  type GameCommand,
  type GameState
} from "@the-oracle/core";
import type { OriginId } from "@the-oracle/content";
import { createPersistenceAdapters, type OracleDesktopBridge } from "./persistence";

const params = new URLSearchParams(window.location.search);
const seed = params.get("seed") ?? 7;
const originId = normalizeOriginId(params.get("origin") ?? undefined);
export const captureModeEnabled = params.get("capture") === "1" || window.navigator.webdriver === true;
const { saveRepository } = createPersistenceAdapters();
const runtime = new OracleRuntime(createInitialState({ seed, originId }), saveRepository);

export type RunSetupRequest = {
  seed?: number | string;
  originId?: OriginId;
};

function updateRunQuery(state: GameState, setupOpen = false) {
  const next = new URLSearchParams(window.location.search);
  next.set("seed", state.worldSeedText);
  next.set("origin", state.originId);
  if (setupOpen) {
    next.set("setup", "1");
  } else {
    next.delete("setup");
  }
  const query = next.toString();
  const nextUrl = query.length > 0 ? `${window.location.pathname}?${query}` : window.location.pathname;
  window.history.replaceState({}, "", nextUrl);
}

type ResolvedRunSetupRequest = {
  originId: OriginId;
  seed: string;
};

export type RunDebugContext = ResolvedRunSetupRequest & {
  href: string;
  origins: ReturnType<typeof selectRunSetupOriginOptions>;
  preview: ReturnType<typeof selectRunSetupPreview>;
  query: string;
  setupOpen: boolean;
};

function resolveRunSetupRequest(options?: RunSetupRequest, fallbackState: GameState = runtime.getState()): ResolvedRunSetupRequest {
  return {
    seed: String(options?.seed ?? fallbackState.worldSeedText),
    originId: normalizeOriginId(options?.originId ?? fallbackState.originId)
  };
}

function buildRunQueryString(options?: RunSetupRequest, setupOpen = false, fallbackState: GameState = runtime.getState()) {
  const resolved = resolveRunSetupRequest(options, fallbackState);
  const next = new URLSearchParams(window.location.search);
  next.set("seed", resolved.seed);
  next.set("origin", resolved.originId);
  if (setupOpen) {
    next.set("setup", "1");
  } else {
    next.delete("setup");
  }
  return next.toString();
}

export function getRunDebugContext(options?: RunSetupRequest, setupOpen = false, fallbackState: GameState = runtime.getState()): RunDebugContext {
  const resolved = resolveRunSetupRequest(options, fallbackState);
  const query = buildRunQueryString(resolved, setupOpen, fallbackState);
  const href = query.length > 0 ? `${window.location.origin}${window.location.pathname}?${query}` : `${window.location.origin}${window.location.pathname}`;

  return {
    ...resolved,
    setupOpen,
    query,
    href,
    preview: selectRunSetupPreview(resolved.seed, resolved.originId),
    origins: selectRunSetupOriginOptions(resolved.seed)
  };
}

declare global {
  interface Window {
    advanceTime: (ms: number) => void;
    render_game_to_text: () => string;
    oracleDesktop?: OracleDesktopBridge;
    __oracleDebug: {
      getState: () => GameState;
      injectScenario: (scenario: "foundation" | "low-incense" | "consultation-ready" | "logistics-lab" | "campaign-lab" | "world-map-lab") => void;
      save: (slotId?: string) => Promise<void>;
      load: (slotId?: string) => Promise<void>;
      dispatchCommand: (command: GameCommand) => void;
      startNewRun: (options?: RunSetupRequest) => void;
      getRunSetupPreview: (options?: RunSetupRequest) => ReturnType<typeof selectRunSetupPreview>;
      getRunSetupOrigins: (seed?: number | string) => ReturnType<typeof selectRunSetupOriginOptions>;
      getRunDebugContext: (options?: RunSetupRequest, setupOpen?: boolean) => RunDebugContext;
      captureFrame?: () => Promise<string | null>;
      getPrecinctArtDebug?: () => unknown;
      viewportForTile?: (tile: { x: number; y: number }) => { x: number; y: number };
    };
  }
}

window.advanceTime = (ms: number) => {
  const ticks = Math.max(1, Math.round(ms / 100));
  runtime.advanceTicks(ticks);
};

window.render_game_to_text = () => renderGameToText(runtime.getState());

window.__oracleDebug = {
  getState: () => structuredClone(runtime.getState()),
  injectScenario: (scenario) => {
    runtime.dispatch({ type: "InjectScenarioCommand", scenario });
  },
  save: async (slotId = "slot-1") => {
    await runtime.save(slotId);
  },
  load: async (slotId = "slot-1") => {
    await runtime.load(slotId);
  },
  dispatchCommand: (command) => {
    runtime.dispatch(command);
  },
  startNewRun: (options) => {
    runtime.dispatch({
      type: "StartNewRunCommand",
      seed: options?.seed,
      originId: options?.originId
    });
    updateRunQuery(runtime.getState(), false);
  },
  getRunSetupPreview: (options) => selectRunSetupPreview(options?.seed, options?.originId),
  getRunSetupOrigins: (seedValue) => selectRunSetupOriginOptions(seedValue),
  getRunDebugContext: (options, setupOpen) => getRunDebugContext(options, setupOpen),
  captureFrame: async () => {
    const canvas = document.querySelector<HTMLCanvasElement>("#precinct-canvas");
    if (!canvas) {
      return null;
    }
    return canvas.toDataURL("image/png");
  }
};

export function isRunSetupInitiallyOpen() {
  return params.get("setup") === "1";
}

export function syncRunQueryFromState(state: GameState, setupOpen = false) {
  updateRunQuery(state, setupOpen);
}

export function startNewRun(options?: RunSetupRequest) {
  runtime.dispatch({
    type: "StartNewRunCommand",
    seed: options?.seed,
    originId: options?.originId
  });
  updateRunQuery(runtime.getState(), false);
  return runtime.getState();
}

export function getRuntime() {
  return runtime;
}
