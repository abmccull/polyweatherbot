import type { GameState } from "@the-oracle/core";
import { selectCharacterRoster, selectCharacterSpotlight } from "@the-oracle/core";
import React, { useMemo } from "react";

const TONE_BADGE: Record<string, string> = {
  trusted: "good",
  watchful: "warn",
  volatile: "bad",
  cold: ""
};

function RelationshipBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div className="stat-bar-row">
      <span className="stat-bar-label">{label}</span>
      <div className="stat-bar-track">
        <div className="stat-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{Math.round(value)}</span>
    </div>
  );
}

export function CharactersPanel({ state }: { state: GameState }) {
  const roster = useMemo(() => selectCharacterRoster(state), [state]);
  const spotlight = useMemo(() => selectCharacterSpotlight(state), [state]);

  if (roster.length === 0) {
    return <div className="text-sm" style={{ color: "var(--text-dim)" }}>No notable characters encountered.</div>;
  }

  return (
    <>
      {spotlight.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Spotlight</div>
          {spotlight.map((ch) => (
            <div key={ch.id} className="character-spotlight-card">
              <div className="character-spotlight-header">
                <strong>{ch.displayName}</strong>
                <span className={`condition-badge ${TONE_BADGE[ch.tone] ?? ""}`}>{ch.tone}</span>
              </div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                {ch.role}{ch.factionName ? ` · ${ch.factionName}` : ""}
              </div>
              {ch.memoryHook ? (
                <div className="text-xs" style={{ color: "var(--text-mid)", fontStyle: "italic" }}>{ch.memoryHook}</div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      <div className="sidebar-block">
        <div className="section-title">Character Roster</div>
        {roster.map((ch) => (
          <div key={ch.id} className="character-roster-row">
            <div className="character-roster-header">
              <span className="character-roster-name">{ch.displayName}</span>
              <span className={`condition-badge ${TONE_BADGE[ch.tone] ?? ""}`}>{ch.tone}</span>
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              {ch.role}{ch.factionName ? ` · ${ch.factionName}` : ""} · Influence {ch.influence}
            </div>
            <div className="character-relationship-bars">
              <RelationshipBar label="Trust" value={ch.trust} />
              <RelationshipBar label="Fear" value={ch.fear} />
              <RelationshipBar label="Hostility" value={ch.hostility} />
              <RelationshipBar label="Familiarity" value={ch.familiarity} />
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
