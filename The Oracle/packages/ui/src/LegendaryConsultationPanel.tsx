import type { GameState } from "@the-oracle/core";
import {
  selectAvailableLegendaryConsultations,
  selectLegendaryProgress
} from "@the-oracle/core";
import React from "react";

import { useGameDispatch } from "./GameDispatchContext";

export function LegendaryConsultationPanel({ state }: { state: GameState }) {
  const {
    onBeginLegendaryConsultation,
    onAdvanceLegendaryStage
  } = useGameDispatch();

  const available = selectAvailableLegendaryConsultations(state);
  const { inProgress, completed } = selectLegendaryProgress(state);

  if (available.length === 0 && inProgress.length === 0 && completed.length === 0) {
    return (
      <div className="sidebar-block">
        <div className="section-title">Legendary Consultations</div>
        <div className="text-sm" style={{ color: "var(--text-dim)", fontStyle: "italic" }}>
          No legendary figures have sought the oracle yet. They will come as the oracle's renown grows.
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar-block">
      <div className="section-title">Legendary Consultations</div>

      {/* In-progress consultations */}
      {inProgress.map(({ def, progress }) => {
        const currentStage = def.stages[progress.currentStage];
        return (
          <div key={def.id} className="envoy-card legendary-card" style={{ marginBottom: "8px" }} data-figure={def.figure.toLowerCase().replace(/\s+/g, "-")}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong style={{ color: "var(--gold)" }}>{def.figure}</strong>
              <div className="legendary-progress-dots">
                {def.stages.map((_, i) => (
                  <span key={i} className={`legendary-dot ${i < progress.currentStage ? "complete" : i === progress.currentStage ? "current" : "locked"}`} />
                ))}
              </div>
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)", marginBottom: "4px" }}>
              {def.name}
            </div>
            {currentStage ? (
              <>
                <p className="text-sm" style={{ lineHeight: 1.5, margin: "4px 0" }}>
                  {currentStage.prompt}
                </p>
                <div className="advisor-banner info text-xs" style={{ marginBottom: "6px" }}>
                  {currentStage.hint}
                </div>
                {currentStage.depthRequirement ? (
                  <div className="legendary-gate text-xs">
                    Requires {currentStage.depthRequirement} depth prophecy
                  </div>
                ) : null}
              </>
            ) : null}
            <button
              className="oracle-button gold"
              onClick={() => onAdvanceLegendaryStage(def.id)}
              type="button"
              style={{ marginTop: "6px" }}
            >
              Complete Stage
            </button>
          </div>
        );
      })}

      {/* Available (not started) consultations */}
      {available
        .filter(({ progress }) => !progress)
        .map(({ def }) => (
          <div key={def.id} className="envoy-card" style={{ marginBottom: "8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong>{def.figure}</strong>
              <span className="text-xs" style={{ color: "var(--text-dim)" }}>
                {def.stages.length} {def.stages.length === 1 ? "stage" : "stages"}
              </span>
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              {def.name}
            </div>
            <p className="text-sm" style={{ lineHeight: 1.4, margin: "4px 0" }}>
              {def.description}
            </p>
            <div className="text-xs" style={{ color: "var(--text-dim)", marginBottom: "4px" }}>
              Reward: +{def.reward.goldBonus} gold, +{def.reward.reputationBonus} reputation, +{def.reward.credibilityBonus} credibility
              {def.reward.specialEffect ? ` (${def.reward.specialEffect.replace(/_/g, " ")})` : ""}
            </div>
            <button
              className="oracle-button"
              onClick={() => onBeginLegendaryConsultation(def.id)}
              type="button"
            >
              Begin Consultation
            </button>
          </div>
        ))}

      {/* Completed consultations */}
      {completed.length > 0 ? (
        <>
          <div className="section-title" style={{ marginTop: "8px" }}>Completed</div>
          {completed.map(({ def, progress }) => (
            <div key={def.id} className="history-row" style={{ opacity: 0.8 }}>
              <strong>{def.figure}</strong>
              <span className="text-xs" style={{ color: "var(--text-dim)" }}>
                {def.name} — Day {progress.completedDay ?? progress.startDay}
              </span>
              <span className="text-xs">
                +{def.reward.goldBonus}g, +{def.reward.reputationBonus} rep, +{def.reward.credibilityBonus} cred
              </span>
            </div>
          ))}
        </>
      ) : null}
    </div>
  );
}
