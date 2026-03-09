import { selectConsultationInsights, type GameState } from "@the-oracle/core";
import React from "react";

import { PortraitArt, resolveFactionPortraitId } from "./PortraitArt";

type ConsultationOverlayProps = {
  state: GameState;
  onToggleTile: (tileId: string, active: boolean) => void;
  onDeliver: () => void;
  onCancel?: () => void;
  onSave: () => void;
  onLoad: () => void;
};

function ScoreMeter({ label, value, variant }: { label: string; value: number; variant: string }) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div className="score-meter">
      <span className="score-meter-label">{label}</span>
      <div className="score-meter-track">
        <div className={`score-meter-fill ${variant}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function MiniStatBar({ label, value, max = 100 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
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

export function ConsultationOverlay({ state, onToggleTile, onDeliver, onCancel, onSave, onLoad }: ConsultationOverlayProps) {
  const current = state.consultation.current;
  if (!current || state.consultation.mode !== "open") {
    return null;
  }

  const tileGroups = [
    { category: "subject", label: "Subjects" },
    { category: "action", label: "Actions" },
    { category: "condition", label: "Conditions" },
    { category: "modifier", label: "Modifiers" },
    { category: "seal", label: "Seals" }
  ] as const;
  const placed = new Set(current.placedTileIds);
  const placedTiles = current.tilePool.filter((tile) => placed.has(tile.id));
  const canDeliver = current.placedTileIds.length >= 3;
  const faction = state.factions[current.factionId];
  const envoyPortraitId = resolveFactionPortraitId(current.factionId, faction.profile);
  const pythiaPortraitId = "hierophant_portrait";
  const insights = selectConsultationInsights(state);
  const guidance = insights?.guidance ?? "Balance specificity and ambiguity before you deliver the answer.";
  const riskWarning = insights?.riskWarning ?? null;
  const depthValue = insights?.depth ?? current.scorePreview.depth ?? 0;
  const depthBand = insights?.depthBand ?? current.scorePreview.depthBand ?? "shallow";
  const depthText = insights?.depthText ?? "The reading still needs a stronger inner shape.";
  const scaffold = insights?.scaffold ?? [];
  const pythiaNote =
    state.pythia.tranceDepth < 42
      ? "The trance is shallow. Build a survivable answer before you make it sharp."
      : state.pythia.needs.purification >= 68
        ? "Purification strain is high. Cleaner, broader phrasing will travel better."
        : state.pythia.mentalClarity >= 74
          ? "The Pythia is lucid. A strong spine can carry this reading."
          : "The Pythia is steady, but not inexhaustible.";

  return (
    <div className="consultation-overlay">
      <div className="pm-container">
        {/* Header */}
        <div className="pm-header">
          <div className="pm-header-mark">
            <span className="pm-header-icon">{"\u{1F3DB}"}</span>
          </div>
          <div className="pm-header-copy">
            <span className="pm-header-kicker">Delphic consultation</span>
            <span className="pm-header-title">Prophecy of Apollo</span>
          </div>
          <div className="pm-header-status">
            <span className="pm-header-faction">{faction.name}</span>
            <span className="pm-header-subline">{current.envoyName} seeks the god's answer</span>
          </div>
        </div>

        {/* Body */}
        <div className="pm-body">
          {/* Left — Supplicant */}
          <div className="pm-left">
            <div className="pm-section-card consultation-supplicant-card">
              <div className="section-title">Supplicant</div>
              <div className="consultation-portrait-card">
                <PortraitArt
                  portraitId={envoyPortraitId}
                  alt={current.envoyName}
                  className="portrait-frame portrait-frame-tall"
                  imgClassName="portrait-image"
                />
                <div className="consultation-supplicant-copy">
                  <strong>{current.envoyName}</strong>
                  <div className="text-sm consultation-muted">{faction.name}</div>
                  <div className="consultation-meta">
                    <span className="consultation-meta-chip">Mood: {current.mood}</span>
                    <span className="consultation-meta-chip">Payment: {current.paymentOffered}g</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="pm-section-card consultation-question-card">
              <div className="section-title">The Question</div>
              <p className="text-sm consultation-question-text">{current.question}</p>
            </div>

            <div className="pm-section-card consultation-pythia-card">
              <div className="section-title">Pythia</div>
              <div className="consultation-pythia-header">
                <PortraitArt
                  portraitId={pythiaPortraitId}
                  alt={state.pythia.name}
                  className="portrait-frame portrait-frame-compact"
                  imgClassName="portrait-image"
                />
                <div className="consultation-pythia-copy">
                  <strong>{state.pythia.name}</strong>
                  <span className="consultation-muted">Voice of the tripod</span>
                </div>
              </div>
              <div className="stat-bar-group">
                <MiniStatBar label="Attune" value={state.pythia.attunement} />
                <MiniStatBar label="Health" value={state.pythia.physicalHealth} />
                <MiniStatBar label="Clarity" value={state.pythia.mentalClarity} />
                <MiniStatBar label="Trance" value={state.pythia.tranceDepth} />
                <MiniStatBar label="Purify" value={state.pythia.needs.purification} />
              </div>
              <div className="advisor-banner info text-xs" id="consultation-depth-note">
                {pythiaNote}
              </div>
            </div>
          </div>

          {/* Center — Tile Builder */}
          <div className="pm-center">
            <div className="consultation-builder-header">
              <div className="consultation-meta">
                <span className="consultation-meta-chip">Selected {placedTiles.length} / {current.tilePool.length}</span>
                <span className="consultation-meta-chip">Build a clear spine, then soften or seal it.</span>
              </div>
              <div className={`depth-readout ${depthBand}`} id="consultation-depth-readout">
                <strong>Prophecy depth {Math.round(depthValue)} · {depthBand}</strong>
                <span>{depthText}</span>
              </div>
            </div>

            {/* Omen summary */}
            <div className="advisor-banner info" id="consultation-omen-summary">
              {insights?.omenSummaryText ?? "No omen summary available."}
            </div>

            {/* Omen reports */}
            <div className="consultation-omen-grid">
              {current.omenReports.map((omen, index) => (
                <article key={omen.id} className="omen-card" id={`consultation-omen-${index}`}>
                  <strong>{omen.sourceRole}</strong>
                  <p className="text-sm omen-card-text">{omen.text}</p>
                  <div className="text-xs omen-card-meta" id={`consultation-omen-${index}-reliability`}>
                    Reliability {omen.reliability.toFixed(1)} · {omen.reliability >= 70 ? "clear" : omen.reliability >= 58 ? "steady" : "fragile"}
                  </div>
                  <div className="text-xs omen-card-meta" id={`consultation-omen-${index}-semantics`}>
                    {omen.semantics.domain} · {omen.semantics.target} · {omen.semantics.polarity}
                  </div>
                </article>
              ))}
            </div>

            {/* Tile categories */}
            {tileGroups.map((group) => (
              <section key={group.category} className="consultation-tile-section">
                <div className="consultation-section-header">
                  <div className="section-title">{group.label}</div>
                  <span className="campaign-copy">
                    {current.tilePool.filter((tile) => tile.category === group.category).length} tiles
                  </span>
                </div>
                <div className="tile-pool">
                  {current.tilePool
                    .filter((tile) => tile.category === group.category)
                    .map((tile) => {
                      const active = placed.has(tile.id);
                      return (
                        <button
                          key={tile.id}
                          className={`tile-button ${active ? "active" : ""}`}
                          id={`prophecy-tile-${tile.id}`}
                          onClick={() => onToggleTile(tile.id, active)}
                          type="button"
                        >
                          {tile.text}
                        </button>
                      );
                    })}
                </div>
              </section>
            ))}

            {/* Assembled prophecy */}
            <div className="consultation-stage-card">
              <div className="consultation-section-header">
                <div className="section-title">Assembled Prophecy</div>
                <span className="campaign-copy">Click a tile to remove it from the reading.</span>
              </div>
              <div className="assembled-prophecy" id="consultation-assembled-prophecy">
                {placedTiles.length > 0 ? (
                  placedTiles.map((tile) => (
                    <button
                      key={tile.id}
                      className="tile-button active consultation-assembled-tile"
                      onClick={() => onToggleTile(tile.id, true)}
                      type="button"
                    >
                      {tile.text} ×
                    </button>
                  ))
                ) : (
                  <span className="consultation-empty-copy">Arrange the tiles to shape the god's answer.</span>
                )}
              </div>
            </div>

            <div className="consultation-stage-card">
              <div className="consultation-section-header">
                <div className="section-title">Reading Scaffold</div>
                <span className="campaign-copy">Track the spine, hinge, and final seal before delivery.</span>
              </div>
              <div className="consultation-scaffold" id="consultation-scaffold">
                {scaffold.map((part) => (
                  <div key={part.kind} className={`scaffold-row ${part.state}`} id={`consultation-scaffold-${part.kind}`}>
                    <span className="scaffold-label">{part.label}</span>
                    <strong className="scaffold-text">{part.text}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Scores */}
          <div className="pm-right">
            <div className={`pm-section-card consultation-verdict-card ${depthBand}`}>
              <span className="consultation-verdict-kicker">Reading posture</span>
              <strong>Depth {Math.round(depthValue)} · {depthBand}</strong>
              <p className="consultation-verdict-copy">{depthText}</p>
            </div>

            <div className="pm-section-card consultation-score-card">
              <div className="section-title">Reading Quality</div>
              <div className="score-row consultation-score-column" id="consultation-score-row">
                <ScoreMeter label="Clarity" value={current.scorePreview.clarity} variant="clarity" />
                <ScoreMeter label="Value" value={current.scorePreview.value} variant="value" />
                <ScoreMeter label="Risk" value={current.scorePreview.risk} variant="risk" />
                <ScoreMeter label="Depth" value={depthValue} variant="depth" />
              </div>
            </div>

            <div className="pm-section-card consultation-guidance-card">
              <div className="section-title">Oracle Guidance</div>
              <div className="advisor-banner info text-xs" id="consultation-guidance-text">
                {guidance}
              </div>
              {riskWarning ? (
                <div className="advisor-banner warn text-xs" id="consultation-risk-warning">
                  {riskWarning}
                </div>
              ) : null}
            </div>

            <div className="pm-section-card consultation-traits-card">
              <div className="section-title">Traits</div>
              <div className="text-xs consultation-traits-text">{state.pythia.traits.join(", ")}</div>
            </div>

            {onCancel ? (
              <div className="consultation-dismiss-area">
                <button className="oracle-button" id="consultation-cancel-btn" onClick={onCancel} type="button" style={{ width: "100%" }}>
                  Dismiss Envoy
                </button>
              </div>
            ) : null}
          </div>
        </div>

        {/* Footer */}
        <div className="pm-footer">
          <span className="pm-footer-hint">
            Select at least 3 tiles. Remove tiles from the prophecy zone by clicking them.
          </span>
          <div className="pm-footer-actions">
            <button
              className="metal-button"
              id="consultation-save-btn"
              data-testid="consultation-save-btn"
              onClick={onSave}
              type="button"
            >
              Save Reading
            </button>
            <button
              className="metal-button"
              id="consultation-load-btn"
              data-testid="consultation-load-btn"
              onClick={onLoad}
              type="button"
            >
              Load Reading
            </button>
          </div>
          <button
            className="oracle-button gold"
            disabled={!canDeliver}
            id="deliver-prophecy-btn"
            onClick={onDeliver}
            type="button"
          >
            Utter the Prophecy
          </button>
        </div>
      </div>
    </div>
  );
}
