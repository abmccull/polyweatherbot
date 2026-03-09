import React, { useState } from "react";
import {
  selectLineageOverview,
  selectChallengeSeeds,
  burdenDefs,
  type GameState,
  type LineageOverview,
  type ChallengeSeedView,
  type BurdenId
} from "@the-oracle/core";
import { useGameDispatch } from "./GameDispatchContext";

function ScoreBar({ label, score, max }: { label: string; score: number; max: number }) {
  const pct = Math.min(100, (score / max) * 100);
  return (
    <div className="stat-bar-row">
      <span className="stat-bar-label">{label}</span>
      <div className="stat-bar-track">
        <div className="stat-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{Math.round(score)}</span>
    </div>
  );
}

function RunHistoryList({ overview }: { overview: LineageOverview }) {
  if (overview.runHistory.length === 0) {
    return <div className="campaign-copy">No completed runs yet. Finish a run to record it here.</div>;
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Run History</div>
      {overview.runHistory
        .slice()
        .reverse()
        .map((run, i) => (
          <div key={run.runId ?? i} className="history-row">
            <strong>
              Run {overview.runHistory.length - i}: {run.originId}
            </strong>
            <span>Seed: {run.seedText}</span>
            <span>Score: {run.finalScore}</span>
            {run.finalAge ? <span>{run.finalAge}</span> : null}
            <span className="text-xs" style={{ color: "var(--text-dim)" }}>{run.epitaph}</span>
          </div>
        ))}
    </div>
  );
}

function UnlockedOriginsView({ origins }: { origins: string[] }) {
  return (
    <div className="sidebar-block">
      <div className="section-title">Unlocked Origins ({origins.length})</div>
      <div className="world-chip-row">
        {origins.map((id) => (
          <span key={id} className="world-chip tone-steady compact">
            {id.replace(/-/g, " ")}
          </span>
        ))}
      </div>
    </div>
  );
}

function UnlockedBurdensView({ burdens, selectedBurdens, onToggleBurden }: { burdens: string[]; selectedBurdens: Set<string>; onToggleBurden: (id: string) => void }) {
  if (burdens.length === 0) {
    return (
      <div className="sidebar-block">
        <div className="section-title">Burdens</div>
        <div className="campaign-copy">
          No burdens unlocked yet. Raise your lineage score to unlock challenge modifiers.
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Burdens for Next Run ({selectedBurdens.size}/{burdens.length})</div>
      {burdenDefs
        .filter((b) => burdens.includes(b.id))
        .map((b) => (
          <label key={b.id} className="burden-toggle-row">
            <input
              type="checkbox"
              checked={selectedBurdens.has(b.id)}
              onChange={() => onToggleBurden(b.id)}
              className="burden-checkbox"
            />
            <div className="burden-toggle-info">
              <strong>{b.name}</strong>
              <span className="text-xs">{b.description}</span>
              <span className="text-xs" style={{ color: "var(--gold)" }}>
                Score x{b.scoreMultiplier}
              </span>
            </div>
          </label>
        ))}
    </div>
  );
}

function CarryoverBonusesView({ overview }: { overview: LineageOverview }) {
  if (overview.carryoverBonuses.length === 0) {
    return (
      <div className="sidebar-block">
        <div className="section-title">Carryover Bonuses</div>
        <div className="campaign-copy">
          Complete runs to earn bonuses that carry into your next run.
        </div>
      </div>
    );
  }

  const kindLabels: Record<string, string> = {
    starting_gold: "Starting Gold",
    starting_reputation: "Starting Reputation",
    priest_skill: "Priest Skill",
    resource_capacity: "Resource Capacity",
    consultation_quality: "Consultation Quality"
  };

  return (
    <div className="sidebar-block">
      <div className="section-title">Carryover Bonuses (Next Run)</div>
      {overview.carryoverBonuses.map((bonus) => (
        <div key={bonus.id} className="history-row">
          <strong>{kindLabels[bonus.kind] ?? bonus.kind}</strong>
          <span>+{bonus.value}</span>
        </div>
      ))}
    </div>
  );
}

function ChallengeSeedBrowser({ challenges }: { challenges: ChallengeSeedView[] }) {
  return (
    <div className="sidebar-block">
      <div className="section-title">Challenge Seeds</div>
      {challenges.map((entry) => {
        const c = entry.challenge;
        return (
          <div
            key={c.id}
            className={`history-row ${entry.completed ? "positive" : ""}`}
          >
            <strong>{c.id.replace("challenge-", "").replace(/-/g, " ")}</strong>
            <span>{c.description}</span>
            <span>
              Origin: {c.originId.replace(/-/g, " ")} | Seed: {c.seedText}
            </span>
            <span>
              Target: {c.targetScore}
              {entry.bestScore !== undefined ? ` | Best: ${entry.bestScore}` : ""}
              {entry.completed ? " | COMPLETED" : ""}
            </span>
            <span className="text-xs" style={{ color: "var(--text-dim)" }}>
              Burdens: {c.burdens.length > 0 ? c.burdens.join(", ") : "none"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function LineagePanel({ state }: { state: GameState }) {
  const overview = selectLineageOverview(state);
  const challenges = selectChallengeSeeds(state);
  const dispatch = useGameDispatch();
  const [selectedBurdens, setSelectedBurdens] = useState<Set<string>>(new Set());

  const toggleBurden = (id: string) => {
    setSelectedBurdens((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const milestones = [500, 1000, 2000, 3500, 5000];

  return (
    <div className="lineage-panel">
      <div className="sidebar-block">
        <div className="section-title">Lineage</div>
        <div className="legacy-stats">
          <div className="pop-stat">
            <span className="pop-stat-count">{overview.totalRuns}</span>
            <span className="pop-stat-label">Total Runs</span>
          </div>
          <div className="pop-stat">
            <span className="pop-stat-count">{overview.lineageScore}</span>
            <span className="pop-stat-label">Lineage Score</span>
          </div>
        </div>
        <ScoreBar label="Lineage Score" score={overview.lineageScore} max={5000} />
        <div className="lineage-milestones">
          {milestones.map((m) => (
            <span key={m} className={`lineage-milestone ${overview.lineageScore >= m ? "reached" : ""}`}>
              {m}
            </span>
          ))}
        </div>
      </div>

      <UnlockedOriginsView origins={overview.unlockedOrigins} />
      <UnlockedBurdensView burdens={overview.unlockedBurdens} selectedBurdens={selectedBurdens} onToggleBurden={toggleBurden} />
      <CarryoverBonusesView overview={overview} />

      {overview.totalRuns > 0 ? (
        <div className="sidebar-block">
          <button
            className="oracle-button gold"
            style={{ width: "100%" }}
            onClick={() => {
              const origin = overview.unlockedOrigins[0] ?? "classical-delphi";
              dispatch.onStartNewLineageRun(
                origin as Parameters<typeof dispatch.onStartNewLineageRun>[0],
                `lineage-${overview.totalRuns + 1}`,
                [...selectedBurdens] as BurdenId[],
                false
              );
            }}
            type="button"
          >
            Launch New Run
          </button>
        </div>
      ) : null}

      <RunHistoryList overview={overview} />
      <ChallengeSeedBrowser challenges={challenges} />
    </div>
  );
}
