import type { GameState } from "@the-oracle/core";
import { selectConsequenceTracker } from "@the-oracle/core";
import React, { useMemo } from "react";

export function ConsequenceTrackerPanel({ state }: { state: GameState }) {
  const tracker = useMemo(() => selectConsequenceTracker(state), [state]);
  const factions = state.factions;

  if (tracker.pending.length === 0 && tracker.resolved.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No prophecy consequences tracked yet.</div>;
  }

  return (
    <>
      {tracker.urgentCount > 0 ? (
        <div className="advisor-banner warn text-xs" style={{ marginBottom: "8px" }}>
          {tracker.urgentCount} consequence{tracker.urgentCount > 1 ? "s" : ""} due within 30 days
        </div>
      ) : null}

      {tracker.pending.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Pending ({tracker.pending.length})</div>
          {tracker.pending.map((c) => (
            <div key={c.id} className="consequence-row pending">
              <div className="consequence-header">
                <span>{factions[c.factionId as keyof typeof factions]?.name ?? c.factionId}</span>
                <span className="condition-badge warn">Due Day {c.dueDay}</span>
              </div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                Prophecy {c.prophecyId.slice(0, 8)}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {tracker.resolved.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Resolved ({tracker.resolved.length})</div>
          {tracker.resolved.slice(0, 10).map((c) => (
            <div key={c.id} className="consequence-row resolved">
              <div className="consequence-header">
                <span>{factions[c.factionId as keyof typeof factions]?.name ?? c.factionId}</span>
                {c.credibilityDelta !== undefined ? (
                  <span className={`condition-badge ${c.credibilityDelta >= 0 ? "good" : "bad"}`}>
                    {c.credibilityDelta > 0 ? "+" : ""}{c.credibilityDelta}
                  </span>
                ) : null}
              </div>
              {c.report ? (
                <div className="text-xs" style={{ color: "var(--text-mid)" }}>{c.report}</div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </>
  );
}
