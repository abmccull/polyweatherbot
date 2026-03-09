import {
  renderGameToText,
  randomSeedText,
  selectSelectedBuilding,
  selectSelectedWalker,
  type BurdenId,
  type EspionageAgentCover,
  type EspionageOperationKind,
  type FactionId,
  type LegendaryConsultationId,
  type OriginId,
  type PlacementTool
} from "@the-oracle/core";
import {
  ConsultationOverlay,
  ErrorBoundary,
  EscapeMenu,
  HelpOverlay,
  GameDispatchContext,
  type GameDispatchActions,
  OracleHud,
  RunSetupPanel
} from "@the-oracle/ui";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { SeedReplayInspector } from "./dev/SeedReplayInspector";
import { getRunDebugContext, getRuntime, isRunSetupInitiallyOpen, startNewRun, syncRunQueryFromState } from "./runtime";

const TICK_INTERVAL_MS = 100;
const PERSISTENCE_NOTICE_MS = 4000;

type PersistenceNotice = {
  kind: "error" | "success";
  text: string;
};

function useOracleState() {
  const runtime = getRuntime();
  return useSyncExternalStore(
    (callback) => runtime.subscribe(callback),
    () => runtime.getState(),
    () => runtime.getState()
  );
}

export function App() {
  const state = useOracleState();
  const selectedBuilding = useMemo(() => selectSelectedBuilding(state), [state]);
  const selectedWalker = useMemo(() => selectSelectedWalker(state), [state]);
  const gameHostRef = useRef<HTMLDivElement | null>(null);
  const runtime = getRuntime();
  const lastAutosaveDayRef = useRef(state.lastAutosaveDay);
  const [persistenceNotice, setPersistenceNotice] = useState<PersistenceNotice | null>(null);
  const [runSetupOpen, setRunSetupOpen] = useState(() => isRunSetupInitiallyOpen());
  const [runSeed, setRunSeed] = useState(() => state.worldSeedText);
  const [runOriginId, setRunOriginId] = useState(() => state.originId);
  const [runDraftDirty, setRunDraftDirty] = useState(false);
  const [activeOverlay, setActiveOverlay] = useState<string | null>(null);
  const [escapeMenuOpen, setEscapeMenuOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const currentRunContext = useMemo(() => getRunDebugContext(undefined, false, state), [state]);
  const draftRunContext = useMemo(
    () => getRunDebugContext({ seed: runSeed, originId: runOriginId }, false, state),
    [runOriginId, runSeed, state]
  );
  const runOriginOptions = draftRunContext.origins;
  const runPreview = draftRunContext.preview;

  useEffect(() => {
    if (!persistenceNotice) {
      return;
    }

    const timer = window.setTimeout(() => {
      setPersistenceNotice((current) => (current === persistenceNotice ? null : current));
    }, PERSISTENCE_NOTICE_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [persistenceNotice]);

  useEffect(() => {
    if (!gameHostRef.current) {
      return;
    }
    let disposed = false;
    let destroyGame = () => {};

    void import("./game/createGame").then(({ createGame }) => {
      if (disposed || !gameHostRef.current) {
        return;
      }
      const game = createGame(gameHostRef.current);
      destroyGame = () => {
        game.destroy(true);
      };
    });

    return () => {
      disposed = true;
      destroyGame();
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const current = runtime.getState();
      if (!current.clock.paused && current.clock.speed > 0) {
        runtime.advanceTicks(current.clock.speed);
      }
    }, TICK_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [runtime]);

  useEffect(() => {
    if (state.lastAutosaveDay !== lastAutosaveDayRef.current) {
      lastAutosaveDayRef.current = state.lastAutosaveDay;
      void runtime.save("autosave").catch((error) => {
        console.error("Autosave failed", error);
        setPersistenceNotice({
          kind: "error",
          text: "Autosave failed. Existing progress is still in memory."
        });
      });
    }
    window.render_game_to_text = () => renderGameToText(runtime.getState());
  }, [runtime, state]);

  useEffect(() => {
    if (!runDraftDirty) {
      setRunSeed(state.worldSeedText);
      setRunOriginId(state.originId);
    }
  }, [runDraftDirty, state]);

  useEffect(() => {
    syncRunQueryFromState(state, runSetupOpen);
  }, [runSetupOpen, state]);

  const updateRunDraft = useCallback((seed: string, originId: OriginId) => {
    setRunSeed(seed);
    setRunOriginId(originId);
    setRunDraftDirty(seed !== state.worldSeedText || originId !== state.originId);
  }, [state.originId, state.worldSeedText]);

  const syncDraftToLiveRun = useCallback(() => {
    setRunSeed(state.worldSeedText);
    setRunOriginId(state.originId);
    setRunDraftDirty(false);
  }, [state.originId, state.worldSeedText]);

  const handleManualSave = useCallback(async () => {
    try {
      await runtime.save("slot-1");
      setPersistenceNotice({
        kind: "success",
        text: "Saved to slot 1."
      });
    } catch (error) {
      console.error("Manual save failed", error);
      setPersistenceNotice({
        kind: "error",
        text: "Save failed. Check the console for details."
      });
    }
  }, [runtime]);

  const handleManualLoad = useCallback(async () => {
    try {
      await runtime.load("slot-1");
      setRunDraftDirty(false);
      setRunSetupOpen(false);
    } catch (error) {
      console.error("Manual load failed", error);
      setPersistenceNotice({
        kind: "error",
        text: "Load failed because the save payload is invalid or unreadable."
      });
    }
  }, [runtime]);

  const handleStartRun = useCallback(() => {
    startNewRun({
      seed: runSeed,
      originId: runOriginId
    });
    setRunDraftDirty(false);
    setRunSetupOpen(false);
    setPersistenceNotice(null);
  }, [runOriginId, runSeed]);

  const openRunSetup = useCallback(() => {
    syncDraftToLiveRun();
    setRunSetupOpen(true);
  }, [syncDraftToLiveRun]);

  const handleSave = useCallback(() => { void handleManualSave(); }, [handleManualSave]);
  const handleLoad = useCallback(() => { void handleManualLoad(); }, [handleManualLoad]);

  // Consolidated keyboard handler
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      // Don't intercept when typing in inputs
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (event.key) {
        case "?":
          setHelpOpen((prev) => !prev);
          break;
        case "Escape":
          if (helpOpen) {
            setHelpOpen(false);
          } else if (escapeMenuOpen) {
            setEscapeMenuOpen(false);
          } else if (activeOverlay) {
            setActiveOverlay(null);
          } else {
            const current = runtime.getState();
            if (current.ui.activeTool !== "select") {
              runtime.dispatch({ type: "SetToolCommand", tool: "select" });
            } else {
              setEscapeMenuOpen(true);
            }
          }
          break;
        case "o":
        case "O":
          setActiveOverlay((prev) => (prev === "oracle" ? null : "oracle"));
          break;
        case "w":
        case "W":
          setActiveOverlay((prev) => (prev === "world" ? null : "world"));
          break;
        case "s":
        case "S":
          setActiveOverlay((prev) => (prev === "stores" ? null : "stores"));
          break;
        case "p":
        case "P":
          setActiveOverlay((prev) => (prev === "priests" ? null : "priests"));
          break;
        case "r":
        case "R":
          setActiveOverlay((prev) => (prev === "record" ? null : "record"));
          break;
        case "e":
        case "E":
          setActiveOverlay((prev) => (prev === "espionage" ? null : "espionage"));
          break;
        case "l":
        case "L":
          setActiveOverlay((prev) => (prev === "legacy" ? null : "legacy"));
          break;
        case "i":
        case "I":
          setActiveOverlay((prev) => (prev === "lineage" ? null : "lineage"));
          break;
        case "1":
          runtime.dispatch({ type: "SetGameSpeedCommand", speed: 1 });
          break;
        case "2":
          runtime.dispatch({ type: "SetGameSpeedCommand", speed: 2 });
          break;
        case "3":
          runtime.dispatch({ type: "SetGameSpeedCommand", speed: 3 });
          break;
        case " ":
          event.preventDefault();
          runtime.dispatch({
            type: "SetGameSpeedCommand",
            speed: runtime.getState().clock.speed === 0 ? 1 : 0
          });
          break;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [runtime, activeOverlay, escapeMenuOpen, helpOpen]);

  const dispatchActions: GameDispatchActions = useMemo(
    () => ({
      onSetTool: (tool: PlacementTool) => runtime.dispatch({ type: "SetToolCommand", tool }),
      onSetSpeed: (speed: 0 | 1 | 2 | 3) => runtime.dispatch({ type: "SetGameSpeedCommand", speed }),
      onAssignPriest: (priestId: string, buildingId: string) => runtime.dispatch({ type: "AssignPriestCommand", priestId, buildingId }),
      onStartConsultation: () => runtime.dispatch({ type: "StartConsultationCommand" }),
      onPurchaseTradeOffer: (offerId: string) => runtime.dispatch({ type: "PurchaseTradeOfferCommand", offerId }),
      onRestPythia: () => runtime.dispatch({ type: "RestPythiaCommand" }),
      onPurifyPythia: () => runtime.dispatch({ type: "PurifyPythiaCommand" }),
      onRepairBuilding: (buildingId: string) => runtime.dispatch({ type: "RepairBuildingCommand", buildingId }),
      onOpenRunSetup: openRunSetup,
      onSave: handleSave,
      onLoad: handleLoad,
      setActiveOverlay,
      onIssuePriestDecree: (decreeType: "calm" | "reform" | "investigate") => runtime.dispatch({ type: "IssuePriestDecreeCommand", decreeType }),
      onDismissPriest: (priestId: string) => runtime.dispatch({ type: "DismissPriestCommand", priestId }),
      onEndorseBloc: (blocId: string) => runtime.dispatch({ type: "EndorseBlocCommand", blocId }),
      onBeginExcavation: (siteId: string) => runtime.dispatch({ type: "BeginExcavationCommand", siteId }),
      onClaimRelic: (siteId: string, layerDepth: number) => runtime.dispatch({ type: "ClaimRelicCommand", siteId, layerDepth }),
      onActivateSacredSite: (siteId: string) => runtime.dispatch({ type: "ActivateSacredSiteCommand", siteId }),
      onLaunchEspionageOperation: (operationKind: EspionageOperationKind, agentId: string, targetId: string) =>
        runtime.dispatch({ type: "LaunchEspionageOperationCommand", operationKind, agentId, targetId }),
      onInvestigatePriest: (priestId: string) => runtime.dispatch({ type: "InvestigatePriestCommand", priestId }),
      onRecruitAgent: (cover: EspionageAgentCover, targetFactionId: FactionId) =>
        runtime.dispatch({ type: "RecruitAgentCommand", cover, targetFactionId }),
      onTriggerEndOfRun: () => runtime.dispatch({ type: "TriggerEndOfRunCommand" }),
      onBeginLegendaryConsultation: (consultationId: LegendaryConsultationId) =>
        runtime.dispatch({ type: "BeginLegendaryConsultationCommand", consultationId }),
      onAdvanceLegendaryStage: (consultationId: LegendaryConsultationId) =>
        runtime.dispatch({ type: "AdvanceLegendaryStageCommand", consultationId }),
      onStartNewLineageRun: (originId: OriginId, seedText: string, burdens: BurdenId[], endlessMode: boolean) =>
        runtime.dispatch({ type: "StartNewLineageRunCommand", originId, seedText, burdens, endlessMode }),
      onRecordLineageRun: () =>
        runtime.dispatch({ type: "RecordLineageRunCommand" })
    }),
    [handleLoad, handleSave, openRunSetup, runtime]
  );

  return (
    <ErrorBoundary>
    <GameDispatchContext.Provider value={dispatchActions}>
      <div className="app-shell">
        <div ref={gameHostRef} className="game-host" />
        {runSetupOpen ? (
          <div
            style={{
              position: "absolute",
              inset: 14,
              zIndex: 5,
              overflow: "auto"
            }}
          >
            <RunSetupPanel
              seed={runSeed}
              onSeedChange={(seed) => updateRunDraft(seed, runOriginId)}
              onSeedRandomize={() => updateRunDraft(randomSeedText(), runOriginId)}
              origins={runOriginOptions}
              selectedOriginId={runOriginId}
              onSelectOrigin={(originId) => updateRunDraft(runSeed, originId as typeof state.originId)}
              preview={runPreview}
              onStart={handleStartRun}
              note={`Live run ${state.worldGeneration.originTitle} · current seed ${state.worldSeedText}.`}
            />
          </div>
        ) : (
          <>
            <OracleHud
              state={state}
              selectedBuilding={selectedBuilding}
              selectedWalker={selectedWalker}
              activeOverlay={activeOverlay}
            />
            <ConsultationOverlay
              state={state}
              onToggleTile={(tileId, active) =>
                runtime.dispatch(active ? { type: "RemoveProphecyTileCommand", tileId } : { type: "PlaceProphecyTileCommand", tileId })
              }
              onDeliver={() => runtime.dispatch({ type: "DeliverProphecyCommand" })}
              onSave={handleSave}
              onLoad={handleLoad}
            />
            {escapeMenuOpen ? (
              <EscapeMenu
                onResume={() => setEscapeMenuOpen(false)}
                onSave={() => { void handleManualSave(); setEscapeMenuOpen(false); }}
                onLoad={() => { void handleManualLoad(); setEscapeMenuOpen(false); }}
                onNewRun={() => { openRunSetup(); setEscapeMenuOpen(false); }}
              />
            ) : null}
            {helpOpen ? <HelpOverlay onClose={() => setHelpOpen(false)} /> : null}
          </>
        )}
        <SeedReplayInspector
          currentContext={currentRunContext}
          draftContext={draftRunContext}
          runSetupOpen={runSetupOpen}
          state={state}
          onDraftOriginChange={(originId) => updateRunDraft(runSeed, originId)}
          onDraftSeedChange={(seed) => updateRunDraft(seed, runOriginId)}
          onOpenDraftInSetup={() => setRunSetupOpen(true)}
          onRandomizeSeed={() => updateRunDraft(randomSeedText(), runOriginId)}
          onResetDraftToLive={syncDraftToLiveRun}
          onStartDraftRun={handleStartRun}
        />
        {persistenceNotice ? (
          <div
            className="panel"
            role="status"
            style={{
              position: "absolute",
              right: 14,
              bottom: 14,
              zIndex: 4,
              maxWidth: 360,
              padding: "12px 14px",
              borderColor: persistenceNotice.kind === "error" ? "rgba(143, 43, 28, 0.55)" : "rgba(54, 93, 40, 0.45)"
            }}
          >
            {persistenceNotice.text}
          </div>
        ) : null}
      </div>
    </GameDispatchContext.Provider>
    </ErrorBoundary>
  );
}
