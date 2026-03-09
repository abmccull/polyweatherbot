import type { GameState } from "@the-oracle/core";
import { selectPhilosopherThreats } from "@the-oracle/core";
import React, { useMemo } from "react";

const STAGE_COLORS: Record<string, string> = {
  dormant: "var(--text-dim)",
  rumor: "var(--text-mid)",
  circle: "var(--gold-dim)",
  sect: "var(--gold)",
  crisis: "var(--red)"
};

export function PhilosopherThreatsPanel({ state }: { state: GameState }) {
  const threats = useMemo(() => selectPhilosopherThreats(state), [state]);

  if (threats.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No philosopher movements detected.</div>;
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Philosopher Threats</div>
      {threats.map((t) => (
        <div key={t.factionId} className="philosopher-threat-row">
          <div className="philosopher-threat-header">
            <span className="philosopher-threat-name">{t.factionName}</span>
            <span className="condition-badge" style={{ color: STAGE_COLORS[t.stage] ?? "var(--text)" }}>
              {t.stage}
            </span>
          </div>
          <div className="philosopher-threat-details">
            <span>{t.worldview}</span>
            <span>Influence {t.influence}</span>
            <span>Suspicion {t.suspicion}</span>
          </div>
          <div className="stat-bar-row">
            <span className="stat-bar-label">Pressure</span>
            <div className="stat-bar-track">
              <div className="stat-bar-fill" style={{ width: `${Math.min(100, t.pressure)}%` }} />
            </div>
            <span className="stat-bar-value">{Math.round(t.pressure)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
