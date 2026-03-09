import type { GameState } from "@the-oracle/core";
import { getAbsoluteDay, selectChronicleEntries, selectRivalOracleSummary } from "@the-oracle/core";

import type { WorldTone } from "./worldPreview";
import { toneClass } from "./worldPreview";

type TimelineItem = {
  id: string;
  lane: string;
  stamp: string;
  label: string;
  detail?: string;
  tone: WorldTone;
  sortValue: number;
};

function crisisTone(pressure: number): WorldTone {
  if (pressure >= 75) {
    return "critical";
  }
  if (pressure >= 58) {
    return "rising";
  }
  if (pressure >= 40) {
    return "watchful";
  }
  return "steady";
}

function keywordTone(text: string): WorldTone {
  const normalized = text.toLowerCase();
  if (
    normalized.includes("crisis")
    || normalized.includes("war")
    || normalized.includes("embargo")
    || normalized.includes("resent")
    || normalized.includes("bristles")
    || normalized.includes("failed")
  ) {
    return "critical";
  }
  if (
    normalized.includes("treaty")
    || normalized.includes("trade")
    || normalized.includes("accord")
    || normalized.includes("vindicated")
    || normalized.includes("favorable")
  ) {
    return "rising";
  }
  if (normalized.includes("watch") || normalized.includes("rumor")) {
    return "watchful";
  }
  return "steady";
}

function parseHistoryStamp(text: string): { stamp: string; sortValue: number } | null {
  const dayMatch = text.match(/\bDay (\d+)\b/i);
  if (dayMatch) {
    return {
      stamp: `Day ${dayMatch[1]}`,
      sortValue: Number(dayMatch[1])
    };
  }

  const monthMatch = text.match(/\bMonth (\d+)\b/i);
  if (monthMatch) {
    const month = Number(monthMatch[1]);
    return {
      stamp: `Month ${month}`,
      sortValue: month * 30
    };
  }

  return null;
}

function buildHorizonItems(state: GameState): TimelineItem[] {
  const absoluteDay = getAbsoluteDay(state.clock);
  const crisisItems = state.campaign.worldMap.crisisChains.slice(0, 3).map((chain) => {
    const node = state.campaign.worldMap.nodes.find((entry) => entry.id === chain.nodeId);
    const factionLabel = chain.factionId ? state.factions[chain.factionId].name : node?.label ?? "Unknown front";
    const stamp = chain.resolvedDay ? `Day ${chain.resolvedDay}` : chain.stage === "active" ? "Live" : chain.stage === "resolution" ? "Resolving" : "Rumor";

    return {
      id: `horizon-crisis-${chain.id}`,
      lane: "Crisis Front",
      stamp,
      label: chain.label,
      detail: `${factionLabel} · ${node?.label ?? chain.nodeId} · pressure ${chain.pressure}`,
      tone: crisisTone(chain.pressure),
      sortValue: chain.resolvedDay ?? absoluteDay + 0.5
    };
  });

  const omenItems = state.consultation.history
    .filter((entry) => !entry.resolved)
    .sort((left, right) => left.dueDay - right.dueDay)
    .slice(0, 3)
    .map((entry) => ({
      id: `horizon-prophecy-${entry.id}`,
      lane: "Due Soon",
      stamp: `Day ${entry.dueDay}`,
      label: `${state.factions[entry.factionId].name} omen ripens`,
      detail: entry.text,
      tone: entry.dueDay <= absoluteDay + 3 ? "critical" as const : "watchful" as const,
      sortValue: entry.dueDay
    }));

  const rivalItems = selectRivalOracleSummary(state)
    .filter((rival) => rival.pressure >= 38)
    .slice(0, 3)
    .map((rival) => ({
      id: `horizon-rival-${rival.id}`,
      lane: "Espionage",
      stamp: rival.discovery === "confirmed" ? "Confirmed" : rival.discovery === "suspected" ? "Suspected" : "Shadow",
      label: `${rival.name} presses ${rival.targetRegionLabel}`,
      detail: rival.lastSummary ?? `${rival.operationLabel} · ${rival.patronLabel}`,
      tone: rival.pressure >= 72 ? "critical" as const : rival.pressure >= 52 ? "rising" as const : "watchful" as const,
      sortValue: absoluteDay + rival.pressure / 100
    }));

  return [...crisisItems, ...omenItems, ...rivalItems];
}

function buildReplayItems(state: GameState): TimelineItem[] {
  const rivalReplayItems: TimelineItem[] = (state.rivalOracles?.incidents ?? [])
    .slice(0, 4)
    .map((incident) => ({
      id: `timeline-rival-${incident.id}`,
      lane: "Shadow War",
      stamp: `Day ${incident.day}`,
      label: incident.discovery === "shadow" ? "Obscured rival activity" : "Rival oracle activity",
      detail: incident.summary,
      tone: incident.pressureDelta >= 11 ? "critical" : incident.pressureDelta >= 8 ? "rising" : "watchful",
      sortValue: incident.day
    }));
  const chronicleItems: TimelineItem[] = selectChronicleEntries(state).map((entry) => ({
    id: `timeline-chronicle-${entry.id}`,
    lane: entry.kind === "consequence" ? "Consequence" : "Prophecy",
    stamp: `Day ${entry.day}`,
    label: entry.title,
    detail: entry.text,
    tone: entry.kind === "consequence"
      ? (entry.delta !== null && entry.delta < 0 ? "critical" : "watchful")
      : (entry.delta !== null && entry.delta > 0 ? "rising" : "steady"),
    sortValue: entry.day
  }));

  const feedItems: TimelineItem[] = state.eventFeed.slice(-6).map((entry) => ({
    id: `timeline-feed-${entry.id}`,
    lane: "Field Feed",
    stamp: `Day ${entry.day}`,
    label: entry.text,
    tone: keywordTone(entry.text),
    sortValue: entry.day
  }));

  const factionItems: TimelineItem[] = Object.values(state.factions)
    .flatMap((faction) => {
      const items: TimelineItem[] = [];

      [faction.lastOutcome, ...faction.history.slice(0, 1)]
        .filter((entry): entry is string => Boolean(entry))
        .forEach((entry, index) => {
          const stamp = parseHistoryStamp(entry);
          if (!stamp) {
            return;
          }

          items.push({
            id: `timeline-faction-${faction.id}-${index}`,
            lane: "Faction Echo",
            stamp: stamp.stamp,
            label: faction.name,
            detail: entry,
            tone: keywordTone(entry),
            sortValue: stamp.sortValue
          });
        });

      return items;
    })
    .sort((left, right) => right.sortValue - left.sortValue || left.label.localeCompare(right.label))
    .slice(0, 4);

  const seen = new Set<string>();

  return [...rivalReplayItems, ...chronicleItems, ...feedItems, ...factionItems]
    .filter((entry) => {
      const signature = `${entry.stamp}:${entry.label}:${entry.detail ?? ""}`;
      if (seen.has(signature)) {
        return false;
      }
      seen.add(signature);
      return true;
    })
    .sort((left, right) => right.sortValue - left.sortValue || left.label.localeCompare(right.label))
    .slice(0, 10);
}

function buildOriginItems(state: GameState): TimelineItem[] {
  const generated = state.worldGeneration.history.slice(0, 4).map((entry, index) => ({
    id: `timeline-origin-${entry.id}`,
    lane: index === 0 ? "Founding" : "Seed Echo",
    stamp: "Origin",
    label: entry.label,
    detail: entry.detail,
    tone: entry.tone ?? "steady",
    sortValue: -100 - index
  }));

  if (generated.length > 0) {
    return generated;
  }

  return [
    {
      id: "timeline-origin-summary",
      lane: "Founding",
      stamp: "Origin",
      label: state.worldGeneration.originTitle,
      detail: state.worldGeneration.summary,
      tone: state.worldGeneration.climate.tone ?? "steady",
      sortValue: -100
    }
  ];
}

function renderTimelineRows(items: TimelineItem[]) {
  if (items.length === 0) {
    return <div className="campaign-copy">No replay beats recorded yet.</div>;
  }

  return (
    <div className="world-timeline-list">
      {items.map((item) => (
        <div key={item.id} className={`world-timeline-row ${toneClass(item.tone)}`}>
          <div className="world-timeline-stamp">
            <strong>{item.stamp}</strong>
            <span>{item.lane}</span>
          </div>
          <div className="world-timeline-body">
            <strong>{item.label}</strong>
            {item.detail ? <span>{item.detail}</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

export function WorldTimelinePanel({ state }: { state: GameState }) {
  const absoluteDay = getAbsoluteDay(state.clock);
  const horizonItems = buildHorizonItems(state);
  const replayItems = buildReplayItems(state);
  const originItems = buildOriginItems(state);
  const openProphecies = state.consultation.history.filter((entry) => !entry.resolved).length;
  const rivalPressure = state.rivalOracles?.totalPressure ?? 0;

  return (
    <section className="world-timeline-panel">
      <div className="section-title">World Replay</div>
      <div className="campaign-copy">
        Day {absoluteDay} in {state.worldGeneration.originTitle}. Replay the current fronts, recent omens, and the founding pressure that shaped this run.
      </div>
      <div className="campaign-kpis">
        <div className="campaign-pill">
          <span className="campaign-pill-label">Today</span>
          <strong>Day {absoluteDay}</strong>
        </div>
        <div className="campaign-pill">
          <span className="campaign-pill-label">Open Crises</span>
          <strong>{state.campaign.worldMap.crisisChains.length}</strong>
        </div>
        <div className="campaign-pill">
          <span className="campaign-pill-label">Pending Omens</span>
          <strong>{openProphecies}</strong>
        </div>
        <div className="campaign-pill">
          <span className="campaign-pill-label">Rival Pressure</span>
          <strong>{rivalPressure}</strong>
        </div>
        <div className="campaign-pill">
          <span className="campaign-pill-label">Field Notes</span>
          <strong>{state.eventFeed.length}</strong>
        </div>
      </div>
      <div className="campaign-list">
        <div className="section-title">Immediate Horizon</div>
        {renderTimelineRows(horizonItems)}
      </div>
      <div className="campaign-list">
        <div className="section-title">Recent Replay</div>
        {renderTimelineRows(replayItems)}
      </div>
      <div className="campaign-list">
        <div className="section-title">Founding Pressure</div>
        {renderTimelineRows(originItems)}
      </div>
    </section>
  );
}
