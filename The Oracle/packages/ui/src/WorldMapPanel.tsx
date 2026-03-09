import { buildingDefs } from "@the-oracle/content";
import type { GameState, ReputationTierId } from "@the-oracle/core";
import { selectRivalOracleSummary } from "@the-oracle/core";
import React, { useEffect, useState } from "react";

import { PrecinctArtThumb } from "./PrecinctArtThumb";
import type {
  WorldFactionShare,
  WorldHistoryEntry,
  WorldMapLinkPreview,
  WorldMapNodePreview,
  WorldMapPreview,
  WorldPressureSummary,
  WorldTone
} from "./worldPreview";
import { toneClass } from "./worldPreview";

type WorldMapPanelProps =
  | {
      state: GameState;
      preview?: never;
      onSelectNode?: never;
    }
  | {
      preview: WorldMapPreview;
      state?: never;
      onSelectNode?: (nodeId: string) => void;
    };

type CampaignHeaderMeta = {
  reputationTier: string;
  reputationScore: number;
  dedications: number;
  progressToNext: number;
  nextTierCopy: string;
  dedicationProgress: number;
  treasuryCopy: string;
  latestMilestone?: string;
};

type LinkedNodeSummary = {
  id: string;
  label: string;
  kind: NonNullable<WorldMapLinkPreview["kind"]>;
  pressure?: string;
  unrest?: string;
};

function tierLabel(tier: ReputationTierId): string {
  return tier.charAt(0).toUpperCase() + tier.slice(1);
}

function metricNumber(value?: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function nodeTone(node: WorldMapNodePreview): WorldTone {
  const pressure = metricNumber(node.pressure);
  const unrest = metricNumber(node.unrest);
  const intensity = Math.max(pressure, unrest);

  if (intensity >= 75) {
    return "critical";
  }
  if (intensity >= 58) {
    return "rising";
  }
  if (intensity >= 40) {
    return "watchful";
  }
  return "steady";
}

function linkKindLabel(kind: WorldMapLinkPreview["kind"]): string {
  switch (kind) {
    case "sea":
      return "Sea lane";
    case "mountain":
      return "Mountain pass";
    case "pilgrim":
      return "Pilgrim road";
    default:
      return "Road";
  }
}

function renderFactionMix(items: WorldFactionShare[]) {
  if (items.length === 0) {
    return null;
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
    return null;
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

function pressureSeverityClass(pressure: WorldPressureSummary): string {
  if (pressure.severity === "critical") {
    return "critical";
  }
  if (pressure.severity === "rising") {
    return "rising";
  }
  return "watchful";
}

function buildCampaignHeaderMeta(state: GameState): CampaignHeaderMeta {
  const thresholds = state.campaign.reputation.thresholds;
  const currentThreshold = thresholds[state.campaign.reputation.currentTier];
  const nextThreshold = state.campaign.reputation.nextTier
    ? thresholds[state.campaign.reputation.nextTier]
    : currentThreshold + 1;
  const progressToNext = nextThreshold > currentThreshold
    ? Math.min(100, ((state.campaign.reputation.score - currentThreshold) / (nextThreshold - currentThreshold)) * 100)
    : 100;
  const dedicationProgress = state.campaign.treasury.nextMilestoneGold > 0
    ? Math.min(100, (state.campaign.treasury.totalGoldInvested / state.campaign.treasury.nextMilestoneGold) * 100)
    : 100;

  return {
    reputationTier: tierLabel(state.campaign.reputation.currentTier),
    reputationScore: state.campaign.reputation.score,
    dedications: state.campaign.treasury.completed,
    progressToNext,
    nextTierCopy: state.campaign.reputation.nextTier
      ? `Toward ${tierLabel(state.campaign.reputation.nextTier)} at ${nextThreshold}`
      : "Highest known reputation tier reached",
    dedicationProgress,
    treasuryCopy: `${state.campaign.treasury.totalGoldInvested} / ${state.campaign.treasury.nextMilestoneGold} gold invested toward the next dedication`,
    latestMilestone: state.campaign.patronMilestones[0]
  };
}

function buildCampaignPreview(state: GameState): WorldMapPreview {
  const unlocked = new Set(state.campaign.reputation.unlockedBuildingIds);
  const nextUnlocks = Object.values(buildingDefs)
    .filter((building) => !unlocked.has(building.id))
    .sort((left, right) => left.costGold - right.costGold || left.name.localeCompare(right.name))
    .slice(0, 3)
    .map((building) => ({
      id: building.id,
      label: building.name,
      detail: building.unlockTier ? tierLabel(building.unlockTier) : "Now",
      artDefId: building.id
    }));
  const regionMap = new Map(state.worldGeneration.regions.map((region) => [region.id, region]));
  const rivalSummaries = selectRivalOracleSummary(state);
  const rivalPressures: WorldPressureSummary[] = rivalSummaries
    .filter((rival) => rival.pressure >= 38)
    .slice(0, 4)
    .map((rival) => {
      const tone: WorldTone = rival.pressure >= 72 ? "critical" : rival.pressure >= 52 ? "rising" : "watchful";

      return {
        id: `rival-pressure-${rival.id}`,
        label: rival.name,
        value: `pressure ${rival.pressure}`,
        detail: `${rival.operationLabel} · ${rival.patronLabel}`,
        severity: tone,
        tone,
        factionLabel: rival.patronLabel,
        nodeId: rival.targetRegionId
      };
    });

  return {
    title: `Campaign Atlas · ${state.worldGeneration.originTitle}`,
    nodes: state.campaign.worldMap.nodes.map((node) => {
      const region = regionMap.get(node.id);
      const localRivals = rivalPressures.filter((pressure) => pressure.nodeId === node.id);

      return {
        id: node.id,
        label: node.label,
        position: node.position,
        summary: region?.summary ?? `Pressure ${node.pressure} · Unrest ${node.unrest} · Links ${node.connectedNodeIds.length}`,
        pressure: String(node.pressure),
        unrest: String(node.unrest),
        controllingFactionLabel: node.controllingFactionId ? state.factions[node.controllingFactionId].name : "No dominant faction recorded",
        climate: region?.climate,
        hegemon: region?.hegemon,
        philosophy: region?.philosophy,
        divineMood: region?.divineMood,
        oracleDensity: region?.oracleDensity,
        connectedNodeIds: node.connectedNodeIds,
        history: region?.history,
        factionMix: region?.factionMix,
        pressureTags: [...(region?.pressures ?? []), ...localRivals]
      };
    }),
    links: state.campaign.worldMap.links.map((link) => ({
      id: link.id,
      fromNodeId: link.fromNodeId,
      toNodeId: link.toNodeId,
      kind: link.kind
    })),
    activePressures: [
      ...state.campaign.worldMap.activePressures.map((pressure) => {
        const tone: WorldTone = pressure.severity === "critical" ? "critical" : pressure.severity === "rising" ? "rising" : "watchful";

        return {
          id: pressure.id,
          label: state.factions[pressure.factionId].name,
          value: `${pressure.kind} ${pressure.value}`,
          detail: pressure.nodeId,
          severity: tone,
          tone,
          factionLabel: state.factions[pressure.factionId].name,
          nodeId: pressure.nodeId
        };
      }),
      ...rivalPressures
    ],
    crisisChains: [
      ...state.campaign.worldMap.crisisChains.map((chain) => ({
        id: chain.id,
        label: chain.label,
        detail: `${chain.stage} · pressure ${chain.pressure}`,
        tone: (chain.pressure >= 8 ? "critical" : chain.pressure >= 5 ? "rising" : "watchful") as WorldTone
      })),
      ...state.worldGeneration.history.slice(0, 2)
    ].slice(0, 5),
    summary: `Seed ${state.worldSeedText} · ${state.worldGeneration.cityStateCount} city-states · ${state.worldGeneration.politicalClimate}`,
    selectedNodeId: state.campaign.worldMap.selectedNodeId,
    winCondition: {
      label: state.campaign.winCondition.label,
      summary: state.campaign.winCondition.summary,
      completed: state.campaign.winCondition.completed
    },
    nextUnlocks: [
      ...nextUnlocks,
      ...state.worldGeneration.challengeTags.slice(0, 2).map((tag) => ({
        id: `challenge-${tag}`,
        label: tag.replace(/-/g, " "),
        detail: "Origin challenge tag"
      }))
    ].slice(0, 5)
  };
}

function buildLinkedNodes(preview: WorldMapPreview, nodeId?: string): LinkedNodeSummary[] {
  if (!nodeId) {
    return [];
  }

  const linkedNodes: LinkedNodeSummary[] = [];

  for (const link of preview.links ?? []) {
    if (link.fromNodeId !== nodeId && link.toNodeId !== nodeId) {
      continue;
    }

    const targetId = link.fromNodeId === nodeId ? link.toNodeId : link.fromNodeId;
    const node = preview.nodes.find((entry) => entry.id === targetId);
    if (!node) {
      continue;
    }

    linkedNodes.push({
      id: `${nodeId}-${link.id}-${targetId}`,
      label: node.label,
      kind: link.kind ?? "road",
      pressure: node.pressure,
      unrest: node.unrest
    });
  }

  return linkedNodes
    .sort((left, right) => metricNumber(right.pressure) - metricNumber(left.pressure) || left.label.localeCompare(right.label));
}

function renderSelectedNode(selectedNode: WorldMapNodePreview, linkedNodes: LinkedNodeSummary[]) {
  const tone = toneClass(nodeTone(selectedNode));

  return (
    <article className="world-map-focus">
      <div className={`world-focus-heading ${tone}`}>
        <div className="world-focus-heading-copy">
          <span className="world-focus-kicker">Selected polis</span>
          <div className="headline">{selectedNode.label}</div>
          <div className="world-focus-meta">
            <span>Unrest {selectedNode.unrest ?? "?"}</span>
            <span>{linkedNodes.length} routes</span>
            {selectedNode.controllingFactionLabel ? <span>{selectedNode.controllingFactionLabel}</span> : null}
          </div>
        </div>
        <span className={`world-focus-badge ${tone}`}>Pressure {selectedNode.pressure ?? "?"}</span>
      </div>
      {selectedNode.summary ? <div className="campaign-copy world-focus-copy">{selectedNode.summary}</div> : null}
      <div className="world-focus-grid">
        <div className="world-focus-card">
          <div className="section-title">Pressure</div>
          <div className="campaign-copy">{selectedNode.pressure ?? "Unknown"}</div>
        </div>
        <div className="world-focus-card">
          <div className="section-title">Unrest</div>
          <div className="campaign-copy">{selectedNode.unrest ?? "Unknown"}</div>
        </div>
        <div className="world-focus-card">
          <div className="section-title">Routes</div>
          <div className="campaign-copy">{linkedNodes.length}</div>
        </div>
        {selectedNode.climate ? (
          <div className="world-focus-card">
            <div className="section-title">Climate</div>
            <div className="campaign-copy">{selectedNode.climate}</div>
          </div>
        ) : null}
        {selectedNode.hegemon ? (
          <div className="world-focus-card">
            <div className="section-title">Hegemon</div>
            <div className="campaign-copy">{selectedNode.hegemon}</div>
          </div>
        ) : null}
        {selectedNode.philosophy ? (
          <div className="world-focus-card">
            <div className="section-title">Philosophy</div>
            <div className="campaign-copy">{selectedNode.philosophy}</div>
          </div>
        ) : null}
        {selectedNode.oracleDensity ? (
          <div className="world-focus-card">
            <div className="section-title">Oracle Density</div>
            <div className="campaign-copy">{selectedNode.oracleDensity}</div>
          </div>
        ) : null}
        {selectedNode.divineMood ? (
          <div className="world-focus-card">
            <div className="section-title">Divine Mood</div>
            <div className="campaign-copy">{selectedNode.divineMood}</div>
          </div>
        ) : null}
      </div>
      {linkedNodes.length > 0 ? (
        <div className="campaign-list">
          <div className="section-title">Connected Routes</div>
          <div className="world-route-list">
            {linkedNodes.map((node) => (
              <div key={node.id} className="world-route-row">
                <div>
                  <strong>{node.label}</strong>
                  <span>{linkKindLabel(node.kind)}</span>
                </div>
                <span>
                  P {node.pressure ?? "?"} · U {node.unrest ?? "?"}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {selectedNode.factionMix?.length ? (
        <div className="campaign-list">
          <div className="section-title">Faction Mix</div>
          {renderFactionMix(selectedNode.factionMix)}
        </div>
      ) : null}
      {selectedNode.pressureTags?.length ? (
        <div className="campaign-list">
          <div className="section-title">Pressure Summary</div>
          <div className="world-chip-row">
            {selectedNode.pressureTags.map((pressure) => (
              <div key={pressure.id} className={`world-chip ${toneClass(pressure.tone)}`}>
                <strong>{pressure.label}</strong>
                {pressure.value ? <span>{pressure.value}</span> : null}
                {pressure.detail ? <span>{pressure.detail}</span> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {selectedNode.history?.length ? (
        <div className="campaign-list">
          <div className="section-title">Regional History</div>
          {renderHistory(selectedNode.history)}
        </div>
      ) : null}
    </article>
  );
}

export function WorldMapPanel(props: WorldMapPanelProps) {
  const isStatePanel = typeof props.state !== "undefined";
  const preview = isStatePanel ? buildCampaignPreview(props.state) : props.preview;
  const campaignHeader = isStatePanel ? buildCampaignHeaderMeta(props.state) : undefined;
  const defaultNodeId = preview.selectedNodeId ?? preview.nodes[0]?.id ?? "";
  const nodeIdsSignature = preview.nodes.map((node) => node.id).join("|");
  const [focusedNodeId, setFocusedNodeId] = useState(defaultNodeId);

  useEffect(() => {
    setFocusedNodeId(defaultNodeId);
  }, [defaultNodeId, nodeIdsSignature]);

  const selectedNode = preview.nodes.find((node) => node.id === focusedNodeId) ?? preview.nodes[0];
  const linkedNodes = buildLinkedNodes(preview, selectedNode?.id);
  const hotspotCount = preview.nodes.filter((node) => nodeTone(node) === "critical" || nodeTone(node) === "rising").length;
  const criticalFrontCount = (preview.activePressures ?? []).filter((pressure) => pressure.severity === "critical").length;
  const rivalFrontCount = (preview.activePressures ?? []).filter((pressure) => pressure.id.startsWith("rival-pressure-")).length;
  const strategicNodes = [...preview.nodes]
    .sort((left, right) => {
      const leftScore = buildLinkedNodes(preview, left.id).length * 10 + metricNumber(left.pressure) + metricNumber(left.unrest);
      const rightScore = buildLinkedNodes(preview, right.id).length * 10 + metricNumber(right.pressure) + metricNumber(right.unrest);
      return rightScore - leftScore || left.label.localeCompare(right.label);
    })
    .slice(0, 5);

  const selectNode = (nodeId: string) => {
    if ("onSelectNode" in props && typeof props.onSelectNode === "function") {
      props.onSelectNode(nodeId);
      return;
    }
    setFocusedNodeId(nodeId);
  };

  return (
    <section className="world-map-panel">
      <div className="world-map-hero">
        <div className="world-map-hero-copy">
          <div className="world-map-kicker-row">
            <span className="world-map-seal">{"\u{1F30D}"}</span>
            <span className="world-map-kicker">Campaign Atlas</span>
          </div>
          <div className="headline world-map-title">{preview.title ?? "World Atlas"}</div>
          <div className="campaign-copy world-map-summary">
            {preview.summary ?? "Chart the rival fronts, sacred roads, and rising unrest across the Greek world."}
          </div>
        </div>
        {campaignHeader?.latestMilestone ? (
          <div className="world-map-hero-aside">
            <span className="world-map-kicker">Latest dedication</span>
            <strong>{campaignHeader.latestMilestone}</strong>
          </div>
        ) : null}
      </div>
      {campaignHeader ? (
        <div className="world-map-status-card">
          <div className="campaign-kpis">
            <div className="campaign-pill">
              <span className="campaign-pill-label">Tier</span>
              <strong>{campaignHeader.reputationTier}</strong>
            </div>
            <div className="campaign-pill">
              <span className="campaign-pill-label">Score</span>
              <strong>{campaignHeader.reputationScore}</strong>
            </div>
            <div className="campaign-pill">
              <span className="campaign-pill-label">Dedications</span>
              <strong>{campaignHeader.dedications}</strong>
            </div>
          </div>
          <div className="campaign-meter-card">
            <div className="campaign-meter">
              <div className="campaign-meter-fill" style={{ width: `${campaignHeader.progressToNext}%` }} />
            </div>
            <div className="campaign-meter-copy">{campaignHeader.nextTierCopy}</div>
          </div>
          <div className="campaign-list world-map-treasury-card">
            <div className="section-title">Treasury Progress</div>
            <div className="campaign-copy">{campaignHeader.treasuryCopy}</div>
            <div className="campaign-meter campaign-meter-secondary">
              <div className="campaign-meter-fill campaign-meter-fill-secondary" style={{ width: `${campaignHeader.dedicationProgress}%` }} />
            </div>
          </div>
        </div>
      ) : preview.summary ? (
        <div className="campaign-copy">{preview.summary}</div>
      ) : null}
      <div className="world-map-stage">
        <div className="world-map-stage-card">
          <div className="world-map-stage-header">
            <div className="world-map-stage-copy">
              <div className="section-title">Regional Theatre</div>
              <div className="campaign-copy">Select a polis to inspect its routes, pressures, and recent omens.</div>
              <div className="world-graph-overview campaign-kpis">
                <div className="campaign-pill">
                  <span className="campaign-pill-label">Nodes</span>
                  <strong>{preview.nodes.length}</strong>
                </div>
                <div className="campaign-pill">
                  <span className="campaign-pill-label">Routes</span>
                  <strong>{preview.links?.length ?? 0}</strong>
                </div>
                <div className="campaign-pill">
                  <span className="campaign-pill-label">Hotspots</span>
                  <strong>{hotspotCount}</strong>
                </div>
                <div className="campaign-pill">
                  <span className="campaign-pill-label">Critical Fronts</span>
                  <strong>{criticalFrontCount}</strong>
                </div>
                <div className="campaign-pill">
                  <span className="campaign-pill-label">Rival Fronts</span>
                  <strong>{rivalFrontCount}</strong>
                </div>
              </div>
            </div>
            <div className="world-map-legend">
              <span className="world-map-legend-item"><span className="world-map-swatch road" /> Road</span>
              <span className="world-map-legend-item"><span className="world-map-swatch sea" /> Sea</span>
              <span className="world-map-legend-item"><span className="world-map-swatch mountain" /> Mountain</span>
              <span className="world-map-legend-item"><span className="world-map-swatch pilgrim" /> Pilgrim</span>
              <span className="world-map-legend-item"><span className="world-map-dot watchful" /> Watchful</span>
              <span className="world-map-legend-item"><span className="world-map-dot rising" /> Rising</span>
              <span className="world-map-legend-item"><span className="world-map-dot critical" /> Critical</span>
            </div>
          </div>
          <svg className="world-map-svg" viewBox="0 0 100 80" role="img" aria-label="Campaign world map">
            {(preview.links ?? []).map((link) => {
              const from = preview.nodes.find((node) => node.id === link.fromNodeId);
              const to = preview.nodes.find((node) => node.id === link.toNodeId);
              if (!from || !to) {
                return null;
              }

              return (
                <line
                  key={link.id}
                  x1={from.position.x}
                  y1={from.position.y}
                  x2={to.position.x}
                  y2={to.position.y}
                  className={`world-map-link ${link.kind ?? "road"}`}
                />
              );
            })}
            {(preview.activePressures ?? []).map((pressure) => {
              const node = preview.nodes.find((entry) => entry.id === pressure.nodeId);
              if (!node) {
                return null;
              }

              return (
                <circle
                  key={pressure.id}
                  cx={node.position.x + 2.2}
                  cy={node.position.y - 2.2}
                  r={pressure.severity === "critical" ? 2.5 : pressure.severity === "rising" ? 2 : 1.5}
                  className={`world-map-pressure ${pressureSeverityClass(pressure)}`}
                />
              );
            })}
            {preview.nodes.map((node) => {
              const selected = node.id === selectedNode?.id;
              const interactive = preview.nodes.length > 1;
              const nodeClass = `world-map-node ${selected ? "selected" : ""} ${interactive ? "clickable" : ""}`;
              const labelClass = `world-map-label ${interactive ? "clickable" : ""}`;

              return (
                <g
                  key={node.id}
                  className={interactive ? "world-map-node-group clickable" : "world-map-node-group"}
                  onClick={interactive ? () => selectNode(node.id) : undefined}
                >
                  <circle
                    cx={node.position.x}
                    cy={node.position.y}
                    r={selected ? 4.5 : 3.3}
                    className={nodeClass}
                  />
                  <text x={node.position.x} y={node.position.y - 6.2} textAnchor="middle" className={labelClass}>
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        {selectedNode ? renderSelectedNode(selectedNode, linkedNodes) : null}
      </div>
      <div className="world-network-grid world-network-grid-primary">
        <div className="campaign-list world-deck-card">
          <div className="world-card-header">
            <div className="section-title">Strategic Nodes</div>
            <span className="oracle-inline-chip">Top 5</span>
          </div>
          <div className="world-node-ledger">
            {strategicNodes.map((node) => {
              const selected = node.id === selectedNode?.id;
              return (
                <button
                  key={node.id}
                  className={`world-node-ledger-row ${selected ? "selected" : ""}`}
                  onClick={() => selectNode(node.id)}
                  type="button"
                >
                  <div>
                    <strong>{node.label}</strong>
                    <span>{buildLinkedNodes(preview, node.id).length} routes</span>
                  </div>
                  <span className={`world-node-ledger-metric ${toneClass(nodeTone(node))}`}>
                    P {node.pressure ?? "?"} · U {node.unrest ?? "?"}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        {preview.winCondition ? (
          <div className={`campaign-list world-deck-card win-condition-card ${preview.winCondition.completed ? "complete" : "pending"}`}>
            <div className="world-card-header">
              <div className="section-title">Win Condition</div>
              <span className="oracle-inline-chip">{preview.winCondition.completed ? "Complete" : "Pending"}</span>
            </div>
            <div className="campaign-copy">{preview.winCondition.label}</div>
            {preview.winCondition.summary ? <div className="campaign-copy">{preview.winCondition.summary}</div> : null}
          </div>
        ) : null}
      </div>
      <div className="world-network-grid world-network-grid-secondary">
        <div className="campaign-list world-deck-card">
          <div className="world-card-header">
            <div className="section-title">Active Pressures</div>
            <span className="oracle-inline-chip">{(preview.activePressures ?? []).length}</span>
          </div>
          {(preview.activePressures ?? []).length === 0 ? <div className="campaign-copy">No regional pressures yet.</div> : null}
          {(preview.activePressures ?? []).map((pressure) => (
            <div key={pressure.id} className={`pressure-row ${pressureSeverityClass(pressure)}`}>
              <span>{pressure.label}</span>
              <span>{pressure.value ?? pressure.detail ?? "Watch"}</span>
              {pressure.factionLabel ? <strong>{pressure.factionLabel}</strong> : null}
            </div>
          ))}
        </div>
        <div className="campaign-list world-deck-card">
          <div className="world-card-header">
            <div className="section-title">Crisis Chains</div>
            <span className="oracle-inline-chip">{(preview.crisisChains ?? []).length}</span>
          </div>
          {(preview.crisisChains ?? []).length === 0 ? <div className="campaign-copy">No chains have ignited.</div> : null}
          {(preview.crisisChains ?? []).map((chain) => (
            <div key={chain.id} className={`history-row ${toneClass(chain.tone)}`}>
              <strong>{chain.label}</strong>
              {chain.detail ? <span>{chain.detail}</span> : null}
            </div>
          ))}
        </div>
        <div className="campaign-list world-deck-card">
          <div className="world-card-header">
            <div className="section-title">Next Unlocks</div>
            <span className="oracle-inline-chip">{(preview.nextUnlocks ?? []).length}</span>
          </div>
          {(preview.nextUnlocks ?? []).length === 0 ? <div className="campaign-copy">All current precinct builds are unlocked.</div> : null}
          {(preview.nextUnlocks ?? []).map((entry) => (
            <div key={entry.id} className="unlock-row">
              <span className="unlock-row-main">
                {entry.artDefId ? (
                  <PrecinctArtThumb
                    defId={entry.artDefId}
                    alt=""
                    className="unlock-art"
                  />
                ) : null}
                <span className="unlock-copy">
                  <span>{entry.label}</span>
                  {entry.artDefId ? <span className="campaign-copy">Precinct build</span> : null}
                </span>
              </span>
              <span>{entry.detail ?? "Pending"}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
