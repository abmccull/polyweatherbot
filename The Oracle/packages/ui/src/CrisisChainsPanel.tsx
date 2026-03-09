import type { GameState } from "@the-oracle/core";
import { selectCrisisChains } from "@the-oracle/core";
import React, { useMemo } from "react";

const STAGE_LABELS: Record<string, string> = {
  rumor: "Rumor",
  active: "Active",
  resolution: "Resolution"
};

export function CrisisChainsPanel({ state }: { state: GameState }) {
  const chains = useMemo(() => selectCrisisChains(state), [state]);

  if (chains.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No active crises.</div>;
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Crisis Chains</div>
      {chains.map((c) => (
        <div key={c.id} className="crisis-chain-row">
          <div className="crisis-chain-header">
            <span className="crisis-chain-label">{c.label}</span>
            <span className={`condition-badge ${c.stage === "active" ? "warn" : c.stage === "resolution" ? "bad" : ""}`}>
              {STAGE_LABELS[c.stage] ?? c.stage}
            </span>
          </div>
          {c.factionName ? (
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>{c.factionName} · Node {c.nodeId}</div>
          ) : (
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>Node {c.nodeId}</div>
          )}
          <div className="stat-bar-row">
            <span className="stat-bar-label">Pressure</span>
            <div className="stat-bar-track">
              <div className={`stat-bar-fill ${c.pressure > 70 ? "health" : ""}`} style={{ width: `${Math.min(100, c.pressure)}%` }} />
            </div>
            <span className="stat-bar-value">{Math.round(c.pressure)}</span>
          </div>
          <div className="text-xs" style={{ color: "var(--text-dim)" }}>Steps completed: {c.stepsCompleted}</div>
        </div>
      ))}
    </div>
  );
}
