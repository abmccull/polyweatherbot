import { WorldMapPanel } from "./WorldMapPanel";
import type { RunSetupOriginOption, RunSetupWorldPreview, WorldFactionShare, WorldHistoryEntry, WorldMetric, WorldPressureSummary } from "./worldPreview";
import { clampMeter, toneClass } from "./worldPreview";

type RunSetupPanelProps = {
  title?: string;
  eyebrow?: string;
  seed: string;
  onSeedChange?: (seed: string) => void;
  onSeedRandomize?: () => void;
  origins: RunSetupOriginOption[];
  selectedOriginId?: string;
  onSelectOrigin?: (originId: string) => void;
  preview?: RunSetupWorldPreview;
  onStart?: () => void;
  startLabel?: string;
  disabled?: boolean;
  note?: string;
  idPrefix?: string;
};

function renderMetricCard(metric: WorldMetric) {
  const meter = clampMeter(metric.meter);

  return (
    <div className={`setup-metric-card ${toneClass(metric.tone)}`}>
      <div className="section-title">{metric.label}</div>
      <div className="headline">{metric.value}</div>
      {metric.detail ? <div className="campaign-copy">{metric.detail}</div> : null}
      {typeof meter === "number" ? (
        <div className="campaign-meter">
          <div className={`campaign-meter-fill ${toneClass(metric.tone)}`} style={{ width: `${meter}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function renderFactionMix(items: WorldFactionShare[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">Faction mix not generated yet.</div>;
  }

  return (
    <div className="world-chip-row">
      {items.map((item) => (
        <div key={item.id} className={`world-chip ${toneClass(item.tone)}`}>
          <strong>{item.label}</strong>
          {item.value ? <span>{item.value}</span> : null}
          {item.detail ? <span>{item.detail}</span> : null}
        </div>
      ))}
    </div>
  );
}

function renderPressures(items: WorldPressureSummary[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No global pressures surfaced for this seed.</div>;
  }

  return (
    <div className="setup-pressure-list">
      {items.map((item) => (
        <div key={item.id} className={`pressure-row ${item.severity ?? "watchful"}`}>
          <span>{item.label}</span>
          <span>{item.value ?? item.detail ?? "Watch"}</span>
        </div>
      ))}
    </div>
  );
}

function renderHistory(items: WorldHistoryEntry[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No world history preview yet.</div>;
  }

  return (
    <div className="history-list">
      {items.map((item) => (
        <div key={item.id} className={`history-row ${toneClass(item.tone)}`}>
          <strong>{item.label}</strong>
          {item.detail ? <span>{item.detail}</span> : null}
        </div>
      ))}
    </div>
  );
}

export function RunSetupPanel({
  title = "New Run",
  eyebrow = "Found the Next Oracle",
  seed,
  onSeedChange,
  onSeedRandomize,
  origins,
  selectedOriginId,
  onSelectOrigin,
  preview,
  onStart,
  startLabel = "Begin the Omen Cycle",
  disabled = false,
  note,
  idPrefix = "run-setup"
}: RunSetupPanelProps) {
  const selectedOrigin = origins.find((origin) => origin.id === selectedOriginId) ?? origins[0];

  return (
    <section className="run-setup-panel panel">
      <div className="run-setup-header">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <div className="headline">{title}</div>
        </div>
        <div className="run-setup-actions">
          <label className="seed-field" htmlFor={`${idPrefix}-seed-input`}>
            <span className="section-title">Seed</span>
            <input
              id={`${idPrefix}-seed-input`}
              data-testid={`${idPrefix}-seed-input`}
              className="seed-input"
              onChange={(event) => onSeedChange?.(event.target.value)}
              placeholder="Delphi-001"
              type="text"
              value={seed}
            />
          </label>
          <button
            className="metal-button"
            id={`${idPrefix}-seed-randomize`}
            data-testid={`${idPrefix}-seed-randomize`}
            onClick={() => onSeedRandomize?.()}
            type="button"
          >
            Randomize
          </button>
          <button
            className="oracle-button"
            id={`${idPrefix}-start-btn`}
            data-testid={`${idPrefix}-start-btn`}
            disabled={disabled}
            onClick={() => onStart?.()}
            type="button"
          >
            {startLabel}
          </button>
        </div>
      </div>
      <div className="run-setup-shell">
        <div className="run-setup-column">
          <div className="section-title">Choose an Origin</div>
          <div className="run-setup-origin-grid">
            {origins.map((origin) => {
              const active = origin.id === selectedOrigin?.id;
              return (
                <button
                  key={origin.id}
                  className={`setup-origin-card ${active ? "active" : ""}`}
                  disabled={origin.disabled}
                  id={`${idPrefix}-origin-${origin.id}`}
                  data-testid={`${idPrefix}-origin-${origin.id}`}
                  onClick={() => onSelectOrigin?.(origin.id)}
                  type="button"
                >
                  <span className="section-title">{origin.title ?? origin.label}</span>
                  {origin.subtitle ? <span className="inspector-subline">{origin.subtitle}</span> : null}
                  <span className="campaign-copy">{origin.summary}</span>
                  <span className="campaign-copy">Climate {origin.climate}</span>
                  <span className="campaign-copy">Divine Mood {origin.divineMood}</span>
                  <span className="campaign-copy">Oracle Density {origin.oracleDensity}</span>
                  <span className="campaign-copy">Faction Mix {origin.factionMix}</span>
                  {origin.tags?.length ? (
                    <span className="world-chip-row">
                      {origin.tags.map((tag) => (
                        <span key={tag} className="world-chip tone-steady compact">
                          {tag}
                        </span>
                      ))}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
          {selectedOrigin ? (
            <div className="campaign-list">
              <div className="section-title">Selected Origin</div>
              <div className="headline">{selectedOrigin.title ?? selectedOrigin.label}</div>
              <div className="campaign-copy">{selectedOrigin.summary}</div>
            </div>
          ) : null}
        </div>
        <div className="run-setup-column run-setup-preview-column">
          <div className="section-title">World Preview</div>
          {preview ? (
            <>
              <div className="setup-preview-summary">
                <div className="headline">{preview.title ?? "Generated World"}</div>
                <div className="campaign-copy">{preview.summary}</div>
              </div>
              <div className="run-setup-metric-grid">
                {renderMetricCard(preview.climate)}
                {renderMetricCard(preview.divineMood)}
                {renderMetricCard(preview.oracleDensity)}
              </div>
              <div className="campaign-list">
                <div className="section-title">Faction Mix</div>
                {renderFactionMix(preview.factionMix)}
              </div>
              <div className="campaign-list">
                <div className="section-title">World Pressures</div>
                {renderPressures(preview.pressures ?? [])}
              </div>
              <div className="campaign-list">
                <div className="section-title">World History</div>
                {renderHistory(preview.history ?? [])}
              </div>
              {preview.map ? <WorldMapPanel preview={preview.map} /> : null}
              {preview.note ? <div className="campaign-copy">{preview.note}</div> : null}
            </>
          ) : (
            <div className="campaign-copy">
              Supply the generated preview payload to show climate, divine mood, oracle density, faction spread, and regional map summaries here.
            </div>
          )}
          {note ? <div className="campaign-copy">{note}</div> : null}
        </div>
      </div>
    </section>
  );
}
