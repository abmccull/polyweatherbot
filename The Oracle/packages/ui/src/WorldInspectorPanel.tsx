import type { WorldFactionShare, WorldHistoryEntry, WorldInspectorRegion, WorldPressureSummary } from "./worldPreview";
import { toneClass } from "./worldPreview";

type WorldInspectorPanelProps = {
  title?: string;
  eyebrow?: string;
  summary?: string;
  regions: WorldInspectorRegion[];
  selectedRegionId?: string;
  onSelectRegion?: (regionId: string) => void;
  footerNote?: string;
  emptyMessage?: string;
  idPrefix?: string;
};

function renderFactionMix(items: WorldFactionShare[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No faction spread recorded yet.</div>;
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

function renderHistory(items: WorldHistoryEntry[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No historical markers surfaced for this region.</div>;
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

function renderPressures(items: WorldPressureSummary[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No active world pressures highlighted.</div>;
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

export function WorldInspectorPanel({
  title = "World Inspector",
  eyebrow = "Deep Read",
  summary,
  regions,
  selectedRegionId,
  onSelectRegion,
  footerNote,
  emptyMessage = "World-generation detail has not been supplied yet.",
  idPrefix = "world-inspector"
}: WorldInspectorPanelProps) {
  const selectedRegion = regions.find((region) => region.id === selectedRegionId) ?? regions[0];

  return (
    <section className="world-inspector-panel panel">
      <div className="run-setup-header">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <div className="headline">{title}</div>
        </div>
        {summary ? <div className="campaign-copy">{summary}</div> : null}
      </div>
      {regions.length === 0 || !selectedRegion ? (
        <div className="campaign-copy">{emptyMessage}</div>
      ) : (
        <div className="world-inspector-shell">
          <div className="world-inspector-region-list" role="tablist" aria-label="World regions">
            {regions.map((region) => {
              const active = region.id === selectedRegion.id;
              return (
                <button
                  key={region.id}
                  className={`inspector-region-button ${active ? "active" : ""}`}
                  id={`${idPrefix}-region-${region.id}`}
                  data-testid={`${idPrefix}-region-${region.id}`}
                  onClick={() => onSelectRegion?.(region.id)}
                  type="button"
                >
                  <span className="section-title">{region.label}</span>
                  {region.subtitle ? <span className="inspector-subline">{region.subtitle}</span> : null}
                </button>
              );
            })}
          </div>
          <div className="world-inspector-detail">
            <div className="world-focus-grid">
              <div className="world-focus-card">
                <div className="section-title">Regional Read</div>
                <div className="headline">{selectedRegion.label}</div>
                <div className="campaign-copy">{selectedRegion.summary}</div>
              </div>
              <div className="world-focus-card">
                <div className="section-title">Climate</div>
                <div className="campaign-copy">{selectedRegion.climate ?? "Awaiting climate roll"}</div>
              </div>
              <div className="world-focus-card">
                <div className="section-title">Hegemon</div>
                <div className="campaign-copy">{selectedRegion.hegemon ?? "No hegemon resolved yet"}</div>
              </div>
              <div className="world-focus-card">
                <div className="section-title">Philosophy</div>
                <div className="campaign-copy">{selectedRegion.philosophy ?? "No dominant school surfaced yet"}</div>
              </div>
              <div className="world-focus-card">
                <div className="section-title">Divine Mood</div>
                <div className="campaign-copy">{selectedRegion.divineMood ?? "Pending divine weather"}</div>
              </div>
              <div className="world-focus-card">
                <div className="section-title">Oracle Density</div>
                <div className="campaign-copy">{selectedRegion.oracleDensity ?? "No density estimate yet"}</div>
              </div>
            </div>
            <div className="campaign-list">
              <div className="section-title">Pressure Pattern</div>
              <div className="campaign-copy">{selectedRegion.pressure ?? "Pressure summary pending"}</div>
              {renderPressures(selectedRegion.pressures ?? [])}
            </div>
            <div className="campaign-list">
              <div className="section-title">Faction Mix</div>
              {renderFactionMix(selectedRegion.factionMix ?? [])}
            </div>
            <div className="campaign-list">
              <div className="section-title">World History</div>
              {renderHistory(selectedRegion.history ?? [])}
            </div>
            {footerNote ? <div className="campaign-copy">{footerNote}</div> : null}
          </div>
        </div>
      )}
    </section>
  );
}
