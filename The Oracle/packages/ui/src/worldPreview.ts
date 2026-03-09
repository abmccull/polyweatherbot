import type { BuildingDefId } from "@the-oracle/core";

export type WorldTone = "steady" | "watchful" | "rising" | "critical";

export type WorldMetric = {
  label: string;
  value: string;
  detail?: string;
  meter?: number;
  tone?: WorldTone;
};

export type WorldFactionShare = {
  id: string;
  label: string;
  value?: string;
  detail?: string;
  tone?: WorldTone;
};

export type WorldPressureSummary = {
  id: string;
  label: string;
  value?: string;
  detail?: string;
  tone?: WorldTone;
  severity?: "watchful" | "rising" | "critical";
  factionLabel?: string;
  nodeId?: string;
};

export type WorldHistoryEntry = {
  id: string;
  label: string;
  detail?: string;
  tone?: WorldTone;
  artDefId?: BuildingDefId;
};

export type WorldMapNodePreview = {
  id: string;
  label: string;
  position: { x: number; y: number };
  summary?: string;
  pressure?: string;
  unrest?: string;
  controllingFactionLabel?: string;
  climate?: string;
  hegemon?: string;
  philosophy?: string;
  divineMood?: string;
  oracleDensity?: string;
  connectedNodeIds?: string[];
  history?: WorldHistoryEntry[];
  factionMix?: WorldFactionShare[];
  pressureTags?: WorldPressureSummary[];
};

export type WorldMapLinkPreview = {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  kind?: "road" | "sea" | "mountain" | "pilgrim";
};

export type WorldMapPreview = {
  title?: string;
  summary?: string;
  nodes: WorldMapNodePreview[];
  links?: WorldMapLinkPreview[];
  activePressures?: WorldPressureSummary[];
  crisisChains?: WorldHistoryEntry[];
  selectedNodeId?: string;
  winCondition?: {
    label: string;
    summary?: string;
    completed?: boolean;
  };
  nextUnlocks?: WorldHistoryEntry[];
};

export type WorldInspectorRegion = {
  id: string;
  label: string;
  subtitle?: string;
  summary: string;
  climate?: string;
  hegemon?: string;
  philosophy?: string;
  divineMood?: string;
  oracleDensity?: string;
  pressure?: string;
  pressures?: WorldPressureSummary[];
  history?: WorldHistoryEntry[];
  factionMix?: WorldFactionShare[];
};

export type RunSetupOriginOption = {
  id: string;
  label: string;
  title?: string;
  subtitle?: string;
  summary: string;
  climate: string;
  divineMood: string;
  oracleDensity: string;
  factionMix: string;
  tags?: string[];
  disabled?: boolean;
};

export type RunSetupWorldPreview = {
  title?: string;
  summary: string;
  climate: WorldMetric;
  divineMood: WorldMetric;
  oracleDensity: WorldMetric;
  factionMix: WorldFactionShare[];
  history?: WorldHistoryEntry[];
  pressures?: WorldPressureSummary[];
  map?: WorldMapPreview;
  note?: string;
};

export function clampMeter(value?: number): number | undefined {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }

  return Math.max(0, Math.min(100, value));
}

export function toneClass(tone?: WorldTone): string {
  return tone ? `tone-${tone}` : "tone-steady";
}
