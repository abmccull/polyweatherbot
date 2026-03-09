import type { GameState } from "@the-oracle/core";
import { selectCurrentAge } from "@the-oracle/core";
import React, { useMemo, useState } from "react";
import { CharactersPanel } from "./CharactersPanel";
import { CredibilityPips } from "./SharedComponents";
import { CrisisChainsPanel } from "./CrisisChainsPanel";
import { ExcavationPanel } from "./ExcavationPanel";
import { PhilosopherThreatsPanel } from "./PhilosopherThreatsPanel";
import { PortraitArt, resolveFactionPortraitId } from "./PortraitArt";
import { RivalOraclesPanel } from "./RivalOraclesPanel";
import { WorldMapPanel } from "./WorldMapPanel";
import { WorldTimelinePanel } from "./WorldTimelinePanel";

function summarizeRelations(state: GameState, factionId: keyof GameState["factions"]) {
  const faction = state.factions[factionId];
  const relations = Object.entries(faction.relations ?? {}) as [keyof GameState["factions"], number][];
  if (relations.length === 0) return { ally: "None", rival: "None" };
  const ally = [...relations].sort((a, b) => b[1] - a[1])[0];
  const rival = [...relations].sort((a, b) => a[1] - b[1])[0];
  return {
    ally: ally ? `${state.factions[ally[0]].name}` : "None",
    rival: rival ? `${state.factions[rival[0]].name}` : "None"
  };
}

function AgePanel({ state }: { state: GameState }) {
  const currentAge = useMemo(() => selectCurrentAge(state), [state]);
  const ageHistory = state.age?.ageHistory ?? [{ ageId: "archaic", startYear: 1, startDay: 1 }];

  return (
    <div className="sidebar-block world-surface-section world-age-card" id="age-panel">
      <div className="world-card-header">
        <div>
          <div className="section-title">Age of the World</div>
          <div className="headline world-card-title">{currentAge.name}</div>
        </div>
        <span className="oracle-inline-chip">{currentAge.yearsInAge}y</span>
      </div>
      <div className="campaign-copy world-age-description">
        {currentAge.description}
      </div>
      {ageHistory.length > 1 ? (
        <div className="world-age-history">
          {ageHistory.map((entry, index) => (
            <div key={entry.ageId} className="world-age-history-row">
              <span className="world-age-history-index">{index > 0 ? ">>" : "I"}</span>
              <span>{entry.ageId.replace(/_/g, " ")} (year {entry.startYear})</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

const ATLAS_TABS = ["factions", "characters", "excavation", "threats"] as const;
type AtlasTab = typeof ATLAS_TABS[number];

const ATLAS_TAB_LABELS: Record<AtlasTab, string> = {
  factions: "Factions",
  characters: "Characters",
  excavation: "Excavation",
  threats: "Threats"
};

export function WorldOverlayPanel({ state }: { state: GameState }) {
  const [worldView, setWorldView] = useState<"atlas" | "timeline">("atlas");
  const [atlasTab, setAtlasTab] = useState<AtlasTab>("factions");
  const currentAge = useMemo(() => selectCurrentAge(state), [state]);
  const factions = Object.values(state.factions).sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div className="world-surface-shell">
      <section className="world-surface-hero">
        <div className="world-surface-hero-copy">
          <span className="eyebrow">Hellenic Theatre</span>
          <div className="headline world-surface-title">Atlas of the Greek World</div>
          <div className="campaign-copy world-surface-summary">
            Track factions, rival fronts, excavation pressure, and the age now ruling the world beyond Delphi.
          </div>
          <div className="world-surface-metrics">
            <div className="world-surface-metric">
              <span className="campaign-pill-label">Age</span>
              <strong>{currentAge.name}</strong>
            </div>
            <div className="world-surface-metric">
              <span className="campaign-pill-label">Factions</span>
              <strong>{factions.length}</strong>
            </div>
            <div className="world-surface-metric">
              <span className="campaign-pill-label">World View</span>
              <strong>{worldView === "atlas" ? "Atlas" : "Replay"}</strong>
            </div>
          </div>
        </div>
        <div className="world-surface-nav world-surface-nav-hero" role="tablist" aria-label="World sidebar views">
          <button
            className={`world-surface-button ${worldView === "atlas" ? "active" : ""}`}
            id="world-view-atlas"
            data-testid="world-view-atlas"
            onClick={() => setWorldView("atlas")}
            role="tab"
            type="button"
          >
            Atlas
          </button>
          <button
            className={`world-surface-button ${worldView === "timeline" ? "active" : ""}`}
            id="world-view-timeline"
            data-testid="world-view-timeline"
            onClick={() => setWorldView("timeline")}
            role="tab"
            type="button"
          >
            Replay
          </button>
        </div>
      </section>
      {worldView === "atlas" ? (
        <>
          <WorldMapPanel state={state} />
          <AgePanel state={state} />

          {/* Atlas sub-tabs */}
          <div className="oracle-tab-bar world-atlas-tab-bar" role="tablist" aria-label="Atlas sections">
            {ATLAS_TABS.map((tab) => (
              <button
                key={tab}
                className={`oracle-tab ${atlasTab === tab ? "active" : ""}`}
                data-testid={`world-atlas-tab-${tab}`}
                onClick={() => setAtlasTab(tab)}
                role="tab"
                aria-selected={atlasTab === tab}
                type="button"
              >
                {ATLAS_TAB_LABELS[tab]}
              </button>
            ))}
          </div>

          {atlasTab === "factions" ? (
            <div className="sidebar-block world-surface-section world-faction-section">
              <div className="world-card-header">
                <div>
                  <div className="section-title">Factions</div>
                  <div className="campaign-copy">Allies, rivals, and trade posture across the wider world.</div>
                </div>
                <span className="oracle-inline-chip">{factions.length} polities</span>
              </div>
              <div className="world-faction-grid">
              {factions.map((faction) => {
                const rel = summarizeRelations(state, faction.id);
                return (
                  <div key={faction.id} className="faction-row world-faction-card" id={`faction-card-${faction.id}`}>
                    <div className="world-faction-shell">
                      <PortraitArt
                        portraitId={resolveFactionPortraitId(faction.id, faction.profile)}
                        alt={faction.name}
                        className="portrait-frame portrait-frame-faction"
                        imgClassName="portrait-image"
                      />
                      <div className="world-faction-card-body">
                        <div className="faction-row-header">
                          <div className="world-faction-card-copy">
                            <span className="faction-row-name">{faction.name}</span>
                            <span className={`agenda-chip ${faction.currentAgenda}`}>{faction.currentAgenda}</span>
                          </div>
                          <CredibilityPips value={faction.credibility} />
                        </div>
                        <div className="world-faction-meta">
                          <span>Ally: {rel.ally}</span>
                          <span>Rival: {rel.rival}</span>
                        </div>
                        <div className="faction-row-details world-faction-stats" id={`faction-relations-${faction.id}`}>
                          <span>Favour {faction.favour}</span>
                          <span>Debt {faction.debt}</span>
                          <span>Trade {faction.tradeAccess ? "Open" : "Closed"}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
              </div>
            </div>
          ) : null}

          {atlasTab === "characters" ? (
            <div className="world-surface-section world-panel-frame">
              <CharactersPanel state={state} />
            </div>
          ) : null}

          {atlasTab === "excavation" ? (
            <div className="world-surface-section world-panel-frame">
              <ExcavationPanel state={state} />
            </div>
          ) : null}

          {atlasTab === "threats" ? (
            <div className="world-threats-stack">
              <PhilosopherThreatsPanel state={state} />
              <CrisisChainsPanel state={state} />
              <RivalOraclesPanel state={state} />
            </div>
          ) : null}
        </>
      ) : (
        <div className="world-surface-section world-panel-frame">
          <WorldTimelinePanel state={state} />
        </div>
      )}
    </div>
  );
}
