import type { GameState } from "@the-oracle/core";
import { selectRivalOracleSummary } from "@the-oracle/core";
import React, { useMemo } from "react";

const DISCOVERY_BADGE: Record<string, string> = {
  shadow: "",
  suspected: "warn",
  confirmed: "bad"
};

export function RivalOraclesPanel({ state }: { state: GameState }) {
  const rivals = useMemo(() => selectRivalOracleSummary(state), [state]);

  if (rivals.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No rival oracles detected.</div>;
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Rival Oracles</div>
      {rivals.map((r) => (
        <div key={r.id} className="rival-oracle-row">
          <div className="rival-oracle-header">
            <span className="rival-oracle-name">{r.name}</span>
            <span className={`condition-badge ${DISCOVERY_BADGE[r.discovery] ?? ""}`}>
              {r.discovery}
            </span>
          </div>
          <div className="text-xs" style={{ color: "var(--text-dim)" }}>
            {r.title} · {r.patronLabel}
          </div>
          <div className="rival-oracle-metrics">
            <div className="stat-bar-row">
              <span className="stat-bar-label">Pressure</span>
              <div className="stat-bar-track">
                <div className="stat-bar-fill" style={{ width: `${Math.min(100, r.pressure)}%` }} />
              </div>
              <span className="stat-bar-value">{Math.round(r.pressure)}</span>
            </div>
            <div className="stat-bar-row">
              <span className="stat-bar-label">Visibility</span>
              <div className="stat-bar-track">
                <div className="stat-bar-fill blue" style={{ width: `${Math.min(100, r.visibility)}%` }} />
              </div>
              <span className="stat-bar-value">{Math.round(r.visibility)}</span>
            </div>
            <div className="stat-bar-row">
              <span className="stat-bar-label">Intel</span>
              <div className="stat-bar-track">
                <div className="stat-bar-fill" style={{ width: `${Math.min(100, r.intel)}%` }} />
              </div>
              <span className="stat-bar-value">{Math.round(r.intel)}</span>
            </div>
          </div>
          {r.lastSummary ? (
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              {r.lastSummary}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}
