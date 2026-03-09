import { buildingDefs, resourceDefs } from "@the-oracle/content";
import type {
  BuildingInstance,
  GameState,
  WalkerInstance
} from "@the-oracle/core";
import { selectCurrentAge } from "@the-oracle/core";
import React, { useMemo, useState } from "react";

import { BottomToolbar } from "./BottomToolbar";
import { EspionagePanel } from "./EspionagePanel";
import { GameOverlay } from "./GameOverlay";
import { useGameDispatch } from "./GameDispatchContext";
import { OracleOverlayPanel } from "./OracleOverlayPanel";
import { OverlayTriggerStrip } from "./OverlayTriggerStrip";
import {
  PrecinctOverviewPanel,
  deriveSiteInspection,
  resolvePrecinctFocusTile
} from "./PrecinctOverviewPanel";
import { PrecinctArtThumb } from "./PrecinctArtThumb";
import { PriestsOverlayPanel } from "./PriestsOverlayPanel";
import { LegacyPanel } from "./LegacyPanel";
import { LineagePanel } from "./LineagePanel";
import { RecordOverlayPanel } from "./RecordOverlayPanel";
import { StoresOverlayPanel } from "./StoresOverlayPanel";
import { WorldOverlayPanel } from "./WorldOverlayPanel";
import { trendArrow } from "./SharedComponents";

const RESOURCE_ICON_MAP: Record<string, string> = {
  gold: "\u{25C9}",
  olive_oil: "\u{1F3FA}",
  incense: "\u{1F33F}",
  sacred_water: "\u{1F4A7}",
  grain: "\u{1F33E}",
  sacred_animals: "\u{1F411}",
  bread: "\u{1F35E}",
  olives: "\u{1FAD2}",
  papyrus: "\u{1F4C3}",
  scrolls: "\u{1F4DC}"
};

const SEASON_ICONS: Record<string, string> = {
  Spring: "\u{1F331}",
  Summer: "\u{2600}",
  Autumn: "\u{1F342}",
  Winter: "\u{2744}"
};

const OVERLAY_TITLES: Record<string, string> = {
  oracle: "Oracle",
  world: "World",
  stores: "Stores",
  priests: "Temple Council",
  espionage: "Espionage",
  record: "Sacred Record",
  legacy: "Legacy",
  lineage: "Lineage"
};

const OVERLAY_WIDTHS: Record<string, "narrow" | "medium" | "wide"> = {
  oracle: "medium",
  world: "wide",
  stores: "narrow",
  priests: "medium",
  espionage: "medium",
  record: "wide",
  legacy: "medium",
  lineage: "medium"
};

type OracleHudProps = {
  state: GameState;
  selectedBuilding?: BuildingInstance;
  selectedWalker?: WalkerInstance;
  activeOverlay: string | null;
};

export function OracleHud({
  state,
  selectedBuilding,
  selectedWalker,
  activeOverlay
}: OracleHudProps) {
  const dispatch = useGameDispatch();
  const [minimapVisible, setMinimapVisible] = useState(true);
  const consultationReady = state.consultation.mode === "pending";
  const currentAge = useMemo(() => selectCurrentAge(state), [state]);
  const firstPriest = state.priests[0];
  const hasPendingAlert = consultationReady || state.campaign?.worldMap?.crisisChains?.some((c: { stepsCompleted: number }) => c.stepsCompleted === 0);
  const needAssignment = selectedBuilding?.requiresPriest && selectedBuilding.assignedPriestIds.length === 0 && firstPriest;
  const activeBuildingJobs = selectedBuilding
    ? state.resourceJobs.filter((job) => job.sourceBuildingId === selectedBuilding.id || job.targetBuildingId === selectedBuilding.id)
    : [];
  const carrierCount = state.walkers.filter((walker) => walker.role === "carrier").length;
  const carrierFatigueValues = state.walkers
    .filter((walker) => walker.role === "carrier")
    .map((walker) => walker.fatigue ?? 0);
  const highestCarrierFatigue = carrierFatigueValues.length > 0
    ? Math.round(Math.max(...carrierFatigueValues))
    : 0;
  const logisticsRows = [...state.resourceJobs]
    .sort((left, right) => {
      const priorityWeight = { critical: 0, high: 1, routine: 2 } as const;
      const leftWeight = priorityWeight[left.priority];
      const rightWeight = priorityWeight[right.priority];
      return leftWeight - rightWeight || left.id.localeCompare(right.id);
    })
    .slice(0, 4);
  const focusTile = useMemo(
    () => resolvePrecinctFocusTile(state, selectedBuilding, selectedWalker),
    [selectedBuilding, selectedWalker, state]
  );
  const siteReading = useMemo(() => deriveSiteInspection(state, focusTile), [focusTile, state]);

  const ritualDefs = resourceDefs.filter((d) => d.category === "ritual" || d.category === "currency");
  const foodDefs = resourceDefs.filter((d) => d.category === "food" || d.category === "trade");

  // Handle M key for minimap toggle
  React.useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "m" || event.key === "M") {
        setMinimapVisible((prev) => !prev);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  function toggleOverlay(overlayId: string) {
    dispatch.setActiveOverlay(activeOverlay === overlayId ? null : overlayId);
  }

  function closeOverlay() {
    dispatch.setActiveOverlay(null);
  }

  function renderOverlayContent() {
    switch (activeOverlay) {
      case "oracle":
        return <OracleOverlayPanel state={state} />;
      case "world":
        return <WorldOverlayPanel state={state} />;
      case "stores":
        return <StoresOverlayPanel state={state} />;
      case "priests":
        return <PriestsOverlayPanel state={state} selectedBuilding={selectedBuilding} />;
      case "espionage":
        return <EspionagePanel state={state} />;
      case "record":
        return <RecordOverlayPanel state={state} />;
      case "legacy":
        return <LegacyPanel state={state} />;
      case "lineage":
        return <LineagePanel state={state} />;
      default:
        return null;
    }
  }

  return (
    <div className="oracle-ui">
      {/* Top Bar */}
      <header className="top-bar panel">
        <div className="top-bar-left">
          <div className="top-bar-chronicle">
            <div className="top-bar-kicker-row">
              <span className="top-bar-seal">{"\u{1F3DB}"}</span>
              <span className="top-bar-kicker">Temple Chronicle</span>
            </div>
            <div className="top-bar-calendar">
              <span className="top-bar-clock">
                Y{state.clock.year} M{state.clock.month} D{state.clock.day} {SEASON_ICONS[state.clock.season] ?? ""} {state.clock.season}
              </span>
              <span className="age-badge" id="age-badge" title={currentAge.description}>{currentAge.name}</span>
            </div>
          </div>
          <div className="top-bar-speed">
            {[0, 1, 2, 3].map((speed) => (
              <button
                key={speed}
                className={`metal-button ${state.clock.speed === speed ? "active" : ""}`}
                id={`speed-${speed}`}
                onClick={() => dispatch.onSetSpeed(speed as 0 | 1 | 2 | 3)}
                type="button"
              >
                {speed === 0 ? "||" : `${speed}x`}
              </button>
            ))}
          </div>
          <div className="resource-pills">
            {ritualDefs.map((def) => {
              const res = state.resources[def.id];
              if (!res) return null;
              return (
                <div key={def.id} className="resource-pill">
                  <span className={`resource-pill-icon ${def.id === "gold" ? "gold-icon" : ""}`}>{RESOURCE_ICON_MAP[def.id] ?? ""}</span>
                  <div className="resource-pill-stack">
                    <span className="resource-pill-label">{def.label}</span>
                    <span className="resource-pill-value">{res.amount.toFixed(0)}</span>
                  </div>
                  {trendArrow(res.trend)}
                </div>
              );
            })}
            <div className="resource-pill-divider" />
            {foodDefs.map((def) => {
              const res = state.resources[def.id];
              if (!res) return null;
              return (
                <div key={def.id} className="resource-pill">
                  <span className="resource-pill-icon">{RESOURCE_ICON_MAP[def.id] ?? ""}</span>
                  <div className="resource-pill-stack">
                    <span className="resource-pill-label">{def.label}</span>
                    <span className="resource-pill-value">{res.amount.toFixed(0)}</span>
                  </div>
                  {trendArrow(res.trend)}
                </div>
              );
            })}
          </div>
        </div>
        <div className="top-bar-right">
          <div className="hud-utility-pills">
            <div className="resource-pill" id="carrier-count-pill" data-testid="carrier-count-pill">
              <span className="resource-pill-icon">{"\u{1F69A}"}</span>
              <div className="resource-pill-stack">
                <span className="resource-pill-label">Carriers</span>
                <span className="resource-pill-value">{carrierCount}</span>
              </div>
            </div>
            <div className="resource-pill" id="carrier-strain-pill" data-testid="carrier-strain-pill">
              <span className="resource-pill-icon">{"\u{26A0}"}</span>
              <div className="resource-pill-stack">
                <span className="resource-pill-label">Strain</span>
                <span className="resource-pill-value">{highestCarrierFatigue}/100</span>
              </div>
            </div>
          </div>
          <div className="hud-action-group">
            <button className="metal-button" id="manual-save-btn" data-testid="manual-save-btn" onClick={dispatch.onSave} type="button">
              Save
            </button>
            <button className="metal-button" id="manual-load-btn" data-testid="manual-load-btn" onClick={dispatch.onLoad} type="button">
              Load
            </button>
            <button className="metal-button" id="new-run-btn" data-testid="new-run-btn" onClick={dispatch.onOpenRunSetup} type="button">
              New Run
            </button>
          </div>
        </div>
        {hasPendingAlert ? <span className="top-bar-alert" title="Pending events require attention" /> : null}
      </header>

      {/* Main Area */}
      <div className="oracle-ui-main">
        {/* Viewport */}
        <div className="viewport-area">
          {/* Consultation banner */}
          {consultationReady ? (
            <div className="consult-banner" id="envoy-btn" onClick={dispatch.onStartConsultation}>
              <span className="consult-banner-icon text-lg">{"\u{1F4DC}"}</span>
              <div className="consult-banner-copy">
                <span className="consult-banner-kicker">Supplicant at the gate</span>
                <strong>Envoy Approaching — {state.consultation.current ? state.factions[state.consultation.current.factionId].name : "Unknown"}</strong>
              </div>
              <span className="consult-banner-action">Receive Envoy {"\u2192"}</span>
            </div>
          ) : null}

          {/* Event log overlay */}
          {state.eventFeed.length > 0 ? (
            <div className="event-log-overlay" aria-live="polite" aria-label="Event feed">
              {state.eventFeed.slice(-4).map((event) => {
                const urgency = /collapse|destroy|revolt|death|killed|plague|famine|ruin/i.test(event.text)
                  ? "critical"
                  : /crisis|threat|decline|warning|danger|attack|war|rival oracle/i.test(event.text)
                    ? "important"
                    : /trade|build|harvest|income|pilgrim|offer|caravan/i.test(event.text)
                      ? "routine"
                      : "ambient";
                return (
                  <div key={event.id} className={`event-line event-${urgency}`}>
                    <span className="event-time">Day {event.day}</span>
                    <span className="event-text">{event.text}</span>
                  </div>
                );
              })}
            </div>
          ) : null}

          {logisticsRows.length > 0 ? (
            <div
              className="panel logistics-overlay"
              id="logistics-job-list"
              data-testid="logistics-job-list"
            >
              <div className="logistics-overlay-header">
                <div>
                  <span className="section-title">Active Logistics</span>
                  <div className="campaign-copy">Live queues across the sanctuary</div>
                </div>
                <span className="oracle-inline-chip">{logisticsRows.length} routes</span>
              </div>
              {logisticsRows.map((job, index) => (
                <div
                  key={job.id}
                  className={`logistics-overlay-row ${job.priority}`}
                  id={`job-row-${index}`}
                  data-testid={`logistics-job-row-${index}`}
                >
                  <div className="logistics-overlay-row-top">
                    <strong>{job.priority} · {resourceDefs.find((def) => def.id === job.resourceId)?.label ?? job.resourceId}</strong>
                    <span>{Math.round(job.amount)} units</span>
                  </div>
                  <span>{job.sourceBuildingId} → {job.targetBuildingId}</span>
                </div>
              ))}
            </div>
          ) : null}

          {/* Advisor banner */}
          {state.advisorMessages[0] ? (
            <div
              className={`advisor-banner ${state.advisorMessages[0].severity}`}
              style={{ pointerEvents: "auto", position: "absolute", top: consultationReady ? 52 : 10, left: "50%", transform: "translateX(-50%)", maxWidth: 400 }}
            >
              {state.advisorMessages[0].text}
            </div>
          ) : null}

          {/* Info card for selected entity */}
          {selectedBuilding ? (
            <div className="info-card-overlay">
              <div className="info-card-header">
                <div className="info-card-visual">
                  <PrecinctArtThumb
                    defId={selectedBuilding.defId}
                    alt={buildingDefs[selectedBuilding.defId].name}
                    className="info-card-art"
                  />
                  <div className="info-card-copy">
                    <span className="eyebrow">Selected structure</span>
                    <div className="headline">{buildingDefs[selectedBuilding.defId].name}</div>
                    <div className="inspector-subline">
                      {buildingDefs[selectedBuilding.defId].category} · {buildingDefs[selectedBuilding.defId].unlockTier ?? "base"}
                    </div>
                  </div>
                </div>
                <div className="info-card-pill-row">
                  <span className="oracle-inline-chip">Priests {selectedBuilding.assignedPriestIds.length}</span>
                  <span className="oracle-inline-chip">Jobs {activeBuildingJobs.length}</span>
                </div>
              </div>
              <div className="info-card-meter-block">
                Condition {selectedBuilding.condition.toFixed(1)} / {selectedBuilding.maxCondition}
                <div className="condition-bar-track">
                  <div
                    className={`condition-bar-fill ${
                      selectedBuilding.condition / selectedBuilding.maxCondition >= 0.8 ? "good"
                      : selectedBuilding.condition / selectedBuilding.maxCondition >= 0.5 ? "warn"
                      : "bad"
                    }`}
                    style={{ width: `${Math.min(100, (selectedBuilding.condition / selectedBuilding.maxCondition) * 100)}%` }}
                  />
                </div>
              </div>
              {selectedBuilding.condition < selectedBuilding.maxCondition * 0.95 ? (
                <button
                  className="oracle-button"
                  id="repair-building-btn"
                  onClick={() => dispatch.onRepairBuilding(selectedBuilding.id)}
                  type="button"
                >
                  Repair ({Math.ceil((selectedBuilding.maxCondition - selectedBuilding.condition) * 0.15)}g)
                </button>
              ) : null}
              <div className="inspector-reading-row">
                <span>Site {siteReading.toneLabel}</span>
                <span>{siteReading.integrity}</span>
              </div>
              <div className="inspector-metric-strip">
                <span>Approach {siteReading.approach}</span>
                <span>Sanctity {siteReading.sanctity}</span>
                <span>Shelter {siteReading.shelter}</span>
                <span>Strain {siteReading.strain}</span>
              </div>
              {buildingDefs[selectedBuilding.defId].recipes?.length ? (
                <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                  Recipes: {buildingDefs[selectedBuilding.defId].recipes?.map((r) => r.id.replace(/-/g, " ")).join(", ")}
                </div>
              ) : null}
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                Stored: {Object.entries(selectedBuilding.storedResources).map(([k, v]) => `${k}:${Number(v).toFixed(0)}`).join(" ") || "None"}
              </div>
              {needAssignment ? (
                <button
                  className="oracle-button"
                  id="assign-priest-btn"
                  onClick={() => dispatch.onAssignPriest(firstPriest.id, selectedBuilding.id)}
                  type="button"
                >
                  Assign {state.walkers.find((w) => w.id === firstPriest.walkerId)?.name ?? "Priest"}
                </button>
              ) : null}
            </div>
          ) : null}

          {selectedWalker && !selectedBuilding ? (
            <div className="info-card-overlay">
              <div className="info-card-header">
                <div className="info-card-copy">
                  <span className="eyebrow">Selected walker</span>
                  <div className="headline">{selectedWalker.name}</div>
                  <div className="inspector-subline">
                    {selectedWalker.role} · {selectedWalker.assignmentBuildingId ? `Assigned to ${selectedWalker.assignmentBuildingId}` : "No assignment"}
                  </div>
                </div>
                <div className="info-card-pill-row">
                  <span className="oracle-inline-chip">State {selectedWalker.state}</span>
                  <span className="oracle-inline-chip">Tile {selectedWalker.tile.x},{selectedWalker.tile.y}</span>
                </div>
              </div>
              <div className="info-card-meter-block">
                Carrying {selectedWalker.carrying ? `${selectedWalker.carrying} ${selectedWalker.carryingAmount?.toFixed(1) ?? 0}` : "Nothing"}
              </div>
              {selectedWalker.role === "carrier" ? (
                <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                  <span id="selected-carrier-fatigue">Fatigue {(selectedWalker.fatigue ?? 0).toFixed(0)}/100</span>
                  {" · "}
                  <span id="selected-carrier-skill">Hauling skill {selectedWalker.haulingSkill ?? 0}</span>
                  {" · "}
                  <span id="selected-carrier-radius">Supply radius {selectedWalker.supplyRadius ?? 0}</span>
                </div>
              ) : null}
              <div className="inspector-reading-row">
                <span>Ground {siteReading.toneLabel}</span>
                <span>{siteReading.integrity}</span>
              </div>
              <div className="inspector-metric-strip">
                <span>Approach {siteReading.approach}</span>
                <span>Sanctity {siteReading.sanctity}</span>
                <span>Shelter {siteReading.shelter}</span>
                <span>Strain {siteReading.strain}</span>
              </div>
            </div>
          ) : null}

          {!selectedBuilding && !selectedWalker && state.ui.hoveredTile ? (
            <div className="site-inspection-overlay" id="site-inspection-panel">
              <div className="site-inspection-header">
                <div>
                  <div className="section-title">Site Inspection</div>
                  <div className="headline">
                    Tile {state.ui.hoveredTile.x},{state.ui.hoveredTile.y}
                  </div>
                </div>
                <span className={`condition-badge ${siteReading.tone === "consecrated" || siteReading.tone === "attuned" ? "good" : siteReading.tone === "watchful" ? "warn" : "bad"}`}>
                  {siteReading.integrity}
                </span>
              </div>
              <div className="site-inspection-grid">
                <div className="site-inspection-stat">
                  <span>Approach</span>
                  <strong>{siteReading.approach}</strong>
                </div>
                <div className="site-inspection-stat">
                  <span>Sanctity</span>
                  <strong>{siteReading.sanctity}</strong>
                </div>
                <div className="site-inspection-stat">
                  <span>Shelter</span>
                  <strong>{siteReading.shelter}</strong>
                </div>
                <div className="site-inspection-stat strain">
                  <span>Strain</span>
                  <strong>{siteReading.strain}</strong>
                </div>
              </div>
              <div className="inspector-subline">{siteReading.note}</div>
            </div>
          ) : null}

          {/* Minimap overlay (M to toggle) */}
          {minimapVisible ? (
            <div className="minimap-overlay">
              <PrecinctOverviewPanel
                state={state}
                selectedBuilding={selectedBuilding}
                selectedWalker={selectedWalker}
              />
            </div>
          ) : null}
        </div>

        {/* Overlay Trigger Strip (replaces right sidebar) */}
        <OverlayTriggerStrip
          activeOverlay={activeOverlay}
          onToggleOverlay={toggleOverlay}
          notifications={{
            oracle: consultationReady,
            world: !!hasPendingAlert && !consultationReady
          }}
        />
      </div>

      {/* Bottom Toolbar */}
      <BottomToolbar
        activeTool={state.ui.activeTool}
        unlockedBuildingIds={state.campaign.reputation.unlockedBuildingIds}
        resources={state.resources}
        onSetTool={dispatch.onSetTool}
      />

      {/* Game Overlays */}
      {activeOverlay && OVERLAY_TITLES[activeOverlay] ? (
        <GameOverlay
          title={OVERLAY_TITLES[activeOverlay]}
          width={OVERLAY_WIDTHS[activeOverlay]}
          onClose={closeOverlay}
        >
          {renderOverlayContent()}
        </GameOverlay>
      ) : null}
    </div>
  );
}
