import type { GameState } from "@the-oracle/core";
import { selectCarrierSummary } from "@the-oracle/core";
import React, { useMemo } from "react";

export function CarrierDetailPanel({ state }: { state: GameState }) {
  const summary = useMemo(() => selectCarrierSummary(state), [state]);
  const carriers = useMemo(
    () => state.walkers.filter((w) => w.role === "carrier"),
    [state.walkers]
  );

  if (carriers.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No carriers active.</div>;
  }

  return (
    <>
      <div className="sidebar-block">
        <div className="section-title">Carrier Overview</div>
        <div className="pop-grid">
          <div className="pop-stat">
            <span className="pop-stat-count">{summary.count}</span>
            <span className="pop-stat-label">Active</span>
          </div>
          <div className="pop-stat">
            <span className="pop-stat-count">{summary.activeJobs}</span>
            <span className="pop-stat-label">Jobs</span>
          </div>
          <div className="pop-stat">
            <span className="pop-stat-count">{summary.strainedCount}</span>
            <span className="pop-stat-label">Strained</span>
          </div>
          <div className="pop-stat">
            <span className="pop-stat-count">{Math.round(summary.averageFatigue)}</span>
            <span className="pop-stat-label">Avg Fatigue</span>
          </div>
        </div>
      </div>

      <div className="sidebar-block">
        <div className="section-title">Carrier Roster</div>
        {carriers.map((c) => (
          <div key={c.id} className="carrier-detail-row">
            <div className="carrier-detail-header">
              <span>{c.name}</span>
              <span className={`condition-badge ${(c.fatigue ?? 0) > 70 ? "bad" : (c.fatigue ?? 0) > 40 ? "warn" : "good"}`}>
                {(c.fatigue ?? 0) > 70 ? "Strained" : (c.fatigue ?? 0) > 40 ? "Tired" : "Fresh"}
              </span>
            </div>
            <div className="stat-bar-row">
              <span className="stat-bar-label">Fatigue</span>
              <div className="stat-bar-track">
                <div className="stat-bar-fill" style={{ width: `${Math.min(100, c.fatigue ?? 0)}%` }} />
              </div>
              <span className="stat-bar-value">{Math.round(c.fatigue ?? 0)}</span>
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              Skill {c.haulingSkill ?? 0} · Radius {c.supplyRadius ?? 0}
              {c.carrying ? ` · Hauling ${c.carrying} (${(c.carryingAmount ?? 0).toFixed(0)})` : " · Idle"}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
