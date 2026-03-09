import { selectSacredRecordEntries, type GameState } from "@the-oracle/core";
import React, { useEffect, useState } from "react";

type SacredRecordPanelProps = {
  state: GameState;
};

const FILTERS = [
  { id: "all", label: "All" },
  { id: "awaiting", label: "Awaiting" },
  { id: "resolved", label: "Resolved" }
] as const;

type RecordFilter = typeof FILTERS[number]["id"];

export function SacredRecordPanel({ state }: SacredRecordPanelProps) {
  const entries = selectSacredRecordEntries(state);
  const [filter, setFilter] = useState<RecordFilter>("all");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | undefined>(entries[0]?.id);

  const filteredEntries = entries.filter((entry) => {
    if (filter === "awaiting" && entry.status !== "awaiting") return false;
    if (filter === "resolved" && entry.status === "awaiting") return false;
    if (search) {
      const q = search.toLowerCase();
      return entry.factionName.toLowerCase().includes(q) || entry.text.toLowerCase().includes(q) || entry.title.toLowerCase().includes(q);
    }
    return true;
  });

  useEffect(() => {
    if (!filteredEntries.some((entry) => entry.id === selectedId)) {
      setSelectedId(filteredEntries[0]?.id);
    }
  }, [filteredEntries, selectedId]);

  const selected = filteredEntries.find((entry) => entry.id === selectedId) ?? filteredEntries[0];

  if (entries.length === 0) {
    return (
      <div className="sidebar-block sacred-record-empty" id="sacred-record-panel">
        <div className="section-title">Sacred Record</div>
        <div className="record-empty-copy">No prophecies have been inscribed yet.</div>
      </div>
    );
  }

  return (
    <div className="sacred-record-panel" id="sacred-record-panel">
      <div className="sidebar-block">
        <div className="section-title">Sacred Record</div>
        <div className="record-filter-row">
          {FILTERS.map((option) => (
            <button
              key={option.id}
              className={`metal-button ${filter === option.id ? "active" : ""}`}
              id={`sacred-record-filter-${option.id}`}
              onClick={() => setFilter(option.id)}
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>
        <input
          className="record-search-input"
          type="text"
          placeholder="Search prophecies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          id="sacred-record-search"
        />
      </div>

      <div className="sidebar-block record-list">
        {filteredEntries.map((entry, index) => (
          <button
            key={entry.id}
            className={`record-list-row ${selected?.id === entry.id ? "active" : ""}`}
            id={`sacred-record-entry-${index}`}
            onClick={() => setSelectedId(entry.id)}
            type="button"
          >
            <div className="record-list-row-top">
              <strong>{entry.factionName}</strong>
              <span className={`record-status-chip ${entry.status}`}>{entry.status}</span>
            </div>
            <span>Day {entry.dayIssued} · {entry.depthBand} depth</span>
            <span>{entry.text}</span>
          </button>
        ))}
      </div>

      {selected ? (
        <div className="sidebar-block record-detail" id="sacred-record-detail">
          <div className="record-detail-header">
            <div>
              <div className="section-title">Selected Record</div>
              <strong>{selected.title}</strong>
            </div>
            <span className={`record-status-chip ${selected.status}`}>{selected.status}</span>
          </div>

          <div className="record-depth-card">
            <div className="record-depth-headline">
              <strong>Depth {selected.depth}</strong>
              <span>{selected.depthBand}</span>
            </div>
            <div className="record-depth-track">
              <div className={`record-depth-fill ${selected.depthBand}`} style={{ width: `${selected.depth}%` }} />
            </div>
            <p>{selected.depthText}</p>
          </div>

          <div className="record-meta-grid">
            <span>Issued Day {selected.dayIssued}</span>
            <span>Due Day {selected.dueDay}</span>
            <span>Omens {selected.omenConsensus} · {selected.omenReliability}</span>
            <span>Scores C{selected.clarity} V{selected.value} R{selected.risk}</span>
          </div>

          <div className="record-quote">{selected.text}</div>

          <div className="record-scaffold">
            {selected.scaffold.map((part) => (
              <div key={part.kind} className={`record-scaffold-row ${part.state}`}>
                <span>{part.label}</span>
                <strong>{part.text}</strong>
              </div>
            ))}
          </div>

          <div className="record-interpretation">
            <div>
              <span className="record-interpretation-label">Reading</span>
              <p>{selected.interpretation.summary}</p>
            </div>
            <div>
              <span className="record-interpretation-label">Political use</span>
              <p>{selected.interpretation.politicalReading}</p>
            </div>
            <div>
              <span className="record-interpretation-label">Caution</span>
              <p>{selected.interpretation.caution}</p>
            </div>
            <div>
              <span className="record-interpretation-label">Window</span>
              <p>{selected.interpretation.fulfillmentWindow}</p>
            </div>
            <div>
              <span className="record-interpretation-label">Rival margin</span>
              <p>{selected.interpretation.rivalContext ?? "No rival oracle pressure is recorded against this reading."}</p>
            </div>
          </div>

          <div className="record-resolution">
            <span className="record-interpretation-label">Outcome</span>
            <p>
              {selected.resolutionReport
                ? `${selected.resolutionReport}${selected.credibilityDelta !== null ? ` Credibility ${selected.credibilityDelta > 0 ? `+${selected.credibilityDelta}` : selected.credibilityDelta}.` : ""}`
                : "Awaiting consequence. The record remains open to later judgment."}
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
