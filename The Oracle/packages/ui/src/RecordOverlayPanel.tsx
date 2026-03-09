import { selectActiveProphecyArcs, selectProphecyContradictions, type GameState } from "@the-oracle/core";
import React from "react";
import { SacredRecordPanel } from "./SacredRecordPanel";

export function RecordOverlayPanel({ state }: { state: GameState }) {
  const activeArcs = selectActiveProphecyArcs(state);
  const contradictions = selectProphecyContradictions(state);

  return (
    <div className="record-overlay">
      <SacredRecordPanel state={state} />

      {contradictions.length > 0 ? (
        <div className="sidebar-block prophecy-contradictions" id="prophecy-contradictions">
          <div className="section-title">Contradictions</div>
          {contradictions.map((c) => (
            <div key={c.id} className={`contradiction-row contradiction-${c.severity}`}>
              <div className="contradiction-header">
                <span className={`contradiction-severity-chip ${c.severity}`}>{c.severity}</span>
                <span>{c.domain}</span>
              </div>
              <p>{c.description}</p>
              <span className="contradiction-meta">
                Detected Day {c.detectedDay} · Credibility {c.credibilityImpact}
              </span>
            </div>
          ))}
        </div>
      ) : null}

      {activeArcs.length > 0 ? (
        <div className="sidebar-block prophecy-arcs" id="prophecy-arcs">
          <div className="section-title">Active Prophecy Arcs</div>
          {activeArcs.map((arc) => (
            <div key={arc.id} className="arc-card">
              <div className="arc-card-header">
                <strong>{arc.label}</strong>
                <span>{arc.factionName}</span>
              </div>
              <div className="arc-card-meta">
                <span>{arc.domain}</span>
                <span>{arc.daysRemaining} days remaining</span>
                <span>Milestones {arc.milestoneProgress}</span>
              </div>
              <div className="arc-milestone-track">
                <div
                  className="arc-milestone-fill"
                  style={{ width: `${arc.totalMilestones > 0 ? (arc.completedMilestones / arc.totalMilestones) * 100 : 0}%` }}
                />
              </div>
              {arc.milestones.map((m) => (
                <div key={m.id} className={`arc-milestone-row ${m.completed ? "completed" : "pending"}`}>
                  <span>{m.label}</span>
                  <span>{m.completed ? (m.outcome ?? "Done") : `Due day ${m.dueDay}`}</span>
                </div>
              ))}
              {arc.interpretationBranches.length > 0 ? (
                <div className="arc-interpretations">
                  <span className="arc-interpretations-label">Faction Readings</span>
                  {arc.interpretationBranches.map((branch) => (
                    <div key={branch.factionId} className={`arc-interpretation-row ${branch.alignment}`}>
                      <span className={`arc-alignment-chip ${branch.alignment}`}>{branch.alignment}</span>
                      <p>{branch.reading}</p>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
