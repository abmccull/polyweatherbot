import {
  selectLegacyStatus,
  selectLegacyArtifact,
  type GameState,
  type LegacyArtifact
} from "@the-oracle/core";
import React from "react";
import { useGameDispatch } from "./GameDispatchContext";

function ScoreBar({ score, max = 1000 }: { score: number; max?: number }) {
  const pct = Math.min(100, (score / max) * 100);
  return (
    <div className="stat-bar-row">
      <span className="stat-bar-label">Legacy Score</span>
      <div className="stat-bar-track">
        <div className="stat-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{Math.round(score)}</span>
    </div>
  );
}

function SeverityMeter({ severity, max = 10 }: { severity: number; max?: number }) {
  const pct = Math.min(100, (severity / max) * 100);
  const variant = pct >= 80 ? "bad" : pct >= 50 ? "warn" : "";
  return (
    <div className="stat-bar-row">
      <span className="stat-bar-label">Decline Severity</span>
      <div className="stat-bar-track">
        <div className={`stat-bar-fill ${variant}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{severity}/{max}</span>
    </div>
  );
}

function ArtifactView({ artifact }: { artifact: LegacyArtifact }) {
  return (
    <div className="legacy-artifact">
      <div className="legacy-epitaph">
        <p>{artifact.epitaph}</p>
      </div>

      <div className="legacy-stats">
        <div className="pop-stat">
          <span className="pop-stat-count">{artifact.totalYears}</span>
          <span className="pop-stat-label">Years</span>
        </div>
        <div className="pop-stat">
          <span className="pop-stat-count">{artifact.finalScore}</span>
          <span className="pop-stat-label">Score</span>
        </div>
        <div className="pop-stat">
          <span className="pop-stat-count">{artifact.majorProphecies.length}</span>
          <span className="pop-stat-label">Major Prophecies</span>
        </div>
      </div>

      <div className="text-xs" style={{ color: "var(--text-dim)", marginBottom: 6 }}>
        {artifact.finalAge} -- {artifact.finalReputation}
      </div>

      {artifact.majorProphecies.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Major Prophecies</div>
          {artifact.majorProphecies.map((p) => (
            <div key={p.prophecyId} className="history-row">
              <strong>{p.factionName}</strong>
              <span>{p.domain} -- {p.outcome}</span>
              <span>Credibility {p.credibilityImpact > 0 ? `+${p.credibilityImpact}` : p.credibilityImpact}</span>
            </div>
          ))}
        </div>
      ) : null}

      {artifact.notablePatrons.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Notable Patrons</div>
          {artifact.notablePatrons.map((p) => (
            <div key={p.factionName} className="history-row">
              <strong>{p.factionName}</strong>
              <span>{p.totalConsultations} consultations</span>
              <span>Credibility {p.finalCredibility} (peak {p.peakCredibility})</span>
            </div>
          ))}
        </div>
      ) : null}

      {artifact.turningPoints.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Turning Points</div>
          {artifact.turningPoints.map((t, i) => (
            <div key={i} className={`history-row ${t.impact}`}>
              <strong>Year {t.year}, Day {t.day}</strong>
              <span>{t.description}</span>
            </div>
          ))}
        </div>
      ) : null}

      {artifact.namedFigures.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Named Figures</div>
          {artifact.namedFigures.map((f, i) => (
            <div key={i} className="history-row">
              <strong>{f.name}</strong>
              <span>{f.role} -- {f.relationship}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function LegacyPanel({ state }: { state: GameState }) {
  const status = selectLegacyStatus(state);
  const artifact = selectLegacyArtifact(state);
  const dispatch = useGameDispatch();

  // Terminal / post-game: show full artifact with ceremony animation
  if (status.phase === "terminal" && artifact) {
    return (
      <div className="legacy-panel legacy-ceremony">
        <div className="sidebar-block ceremony-section" style={{ animationDelay: "0s" }}>
          <div className="section-title">Legacy of Delphi</div>
        </div>
        <div className="ceremony-section" style={{ animationDelay: "0.3s" }}>
          <div className="legacy-epitaph-ceremony">
            <p>{artifact.epitaph}</p>
          </div>
        </div>
        <div className="ceremony-section" style={{ animationDelay: "0.6s" }}>
          <div className="legacy-stats">
            <div className="pop-stat">
              <span className="pop-stat-count">{artifact.totalYears}</span>
              <span className="pop-stat-label">Years</span>
            </div>
            <div className="pop-stat">
              <span className="pop-stat-count">{artifact.finalScore}</span>
              <span className="pop-stat-label">Score</span>
            </div>
            <div className="pop-stat">
              <span className="pop-stat-count">{artifact.majorProphecies.length}</span>
              <span className="pop-stat-label">Major Prophecies</span>
            </div>
          </div>
          <div className="text-xs" style={{ color: "var(--text-dim)" }}>
            {artifact.finalAge} — {artifact.finalReputation}
          </div>
        </div>
        <div className="ceremony-section" style={{ animationDelay: "0.9s" }}>
          <ArtifactView artifact={artifact} />
        </div>
        <div className="ceremony-section" style={{ animationDelay: "1.2s" }}>
          <button
            className="oracle-button gold"
            style={{ width: "100%", marginTop: 8 }}
            onClick={() => dispatch.onOpenRunSetup()}
            type="button"
          >
            Start New Run
          </button>
        </div>
      </div>
    );
  }

  // Declining: show warning and severity
  if (status.phase === "declining") {
    return (
      <div className="legacy-panel">
        <div className="sidebar-block">
          <div className="section-title">Oracle in Decline</div>
          <div className="text-sm" style={{ color: "var(--text-dim)", marginBottom: 8 }}>
            The oracle's influence is fading. Resources contract each month, and pilgrims grow scarce.
            A major prophecy success can slow or reverse the decline.
          </div>
          <SeverityMeter severity={status.declineSeverity} />
          <ScoreBar score={status.scorePreview} />
          {status.comebackAttempts > 0 ? (
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              Comeback attempts: {status.comebackAttempts}
            </div>
          ) : null}
          <div className="text-xs" style={{ color: "var(--text-dim)", marginTop: 6 }}>
            Suggestions: deliver high-value prophecies, resolve crises, secure trade routes.
          </div>
          <button
            className="oracle-button"
            style={{ marginTop: 8 }}
            onClick={() => dispatch.onTriggerEndOfRun()}
            type="button"
          >
            End Run Early
          </button>
        </div>
      </div>
    );
  }

  // Thriving: show score preview and milestones
  return (
    <div className="legacy-panel">
      <div className="sidebar-block">
        <div className="section-title">Legacy Preview</div>
        <ScoreBar score={status.scorePreview} />
        <div className="text-xs" style={{ color: "var(--text-dim)", marginTop: 4 }}>
          Phase: {status.phase}
        </div>
      </div>
    </div>
  );
}
