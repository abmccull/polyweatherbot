import type { GameState } from "@the-oracle/core";
import {
  selectExcavationOverview,
  selectRelicCollection,
  selectSacredSites
} from "@the-oracle/core";
import React from "react";

import { useGameDispatch } from "./GameDispatchContext";

function ProgressBar({ value, max, label }: { value: number; max: number; label?: string }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="stat-bar-row">
      {label ? <span className="stat-bar-label">{label}</span> : null}
      <div className="stat-bar-track">
        <div className="stat-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{Math.round(pct)}%</span>
    </div>
  );
}

export function ExcavationPanel({ state }: { state: GameState }) {
  const dispatch = useGameDispatch();

  const sites = selectExcavationOverview(state);
  const relics = selectRelicCollection(state);
  const sacredSites = selectSacredSites(state);

  if (!state.excavation) {
    return null;
  }

  return (
    <>
      {/* Excavation Sites */}
      <div className="sidebar-block">
        <div className="section-title">Excavation Sites</div>
        {sites.length === 0 ? (
          <div className="text-sm" style={{ color: "var(--text-dim)" }}>No sites discovered yet.</div>
        ) : null}
        {sites.map((site) => (
          <div key={site.id} className="excavation-site-card" style={{ marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong className="text-sm">
                {site.id.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              </strong>
              <span
                className={`condition-badge ${
                  site.status === "excavating" ? "warn" : site.status === "exhausted" ? "bad" : "good"
                } text-xs`}
              >
                {site.status.charAt(0).toUpperCase() + site.status.slice(1)}
              </span>
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              Tile ({site.tile.x}, {site.tile.y})
            </div>
            <ProgressBar value={site.depth} max={site.maxDepth} label="Depth" />
            <div className="text-xs" style={{ color: "var(--text-dim)", display: "flex", justifyContent: "space-between" }}>
              <span>Layers revealed: {site.revealedLayerCount}</span>
              {site.unclaimedRelicCount > 0 ? (
                <span style={{ color: "var(--gold)" }}>
                  Unclaimed relics: {site.unclaimedRelicCount}
                </span>
              ) : null}
            </div>
            {site.status === "discovered" ? (
              <button
                className="oracle-button"
                onClick={() => dispatch.onBeginExcavation(site.id)}
                type="button"
                style={{ marginTop: "0.3rem" }}
              >
                Begin Excavation
              </button>
            ) : null}
          </div>
        ))}
      </div>

      {/* Relics */}
      {relics.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Relics</div>
          {relics.map((relic) => (
            <div key={relic.id} className="history-row" style={{ marginBottom: "0.3rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong className="text-sm">{relic.name}</strong>
                <span
                  className={`condition-badge ${
                    relic.kind === "legendary" ? "warn" : relic.kind === "major" ? "good" : ""
                  } text-2xs`}
                >
                  {relic.kind}
                </span>
              </div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                {relic.domain} | {relic.effect.type.replace(/_/g, " ")} +{relic.effect.value}
              </div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                Found Day {relic.discoveredDay}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Sacred Sites */}
      {sacredSites.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Sacred Sites</div>
          {sacredSites.map((site) => (
            <div key={site.id} className="history-row" style={{ marginBottom: "0.3rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong className="text-sm">
                  {site.kind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </strong>
                <span
                  className={`condition-badge ${site.active ? "good" : "warn"} text-2xs`}
                >
                  {site.active ? "Active" : "Inactive"}
                </span>
              </div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                {site.bonuses.map((b) => `${b.type.replace(/_/g, " ")} +${b.value}`).join(", ")}
              </div>
              {!site.active ? (
                <button
                  className="oracle-button"
                  onClick={() => dispatch.onActivateSacredSite(site.id)}
                  type="button"
                  style={{ marginTop: "0.3rem" }}
                >
                  Activate (20g + 3 incense)
                </button>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </>
  );
}
