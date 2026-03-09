import {
  DEFAULT_ORIGIN_ID,
  factionDefs,
  originDefById,
  originDefs,
  type FactionAgenda,
  type FactionId,
  type FactionProfile,
  type OriginDef,
  type OriginId
} from "@the-oracle/content";

type MetricBand = {
  label: string;
  detail: string;
  meter: number;
};

type WorldLinkLayout = {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  kind: "road" | "sea" | "mountain" | "pilgrim";
};

type WorldRegionLayout = {
  id: string;
  label: string;
  position: { x: number; y: number };
  tags: ("sanctuary" | "city" | "port" | "pass")[];
  defaultFactionId?: FactionId;
  connectedNodeIds: string[];
};

type RunSetupSeverity = "watchful" | "rising" | "critical";
type WorldTone = "steady" | "watchful" | "rising" | "critical";

export type WorldMetricPreview = {
  label: string;
  value: string;
  detail?: string;
  meter?: number;
  tone?: WorldTone;
};

export type WorldFactionSharePreview = {
  id: string;
  label: string;
  value?: string;
  detail?: string;
  tone?: WorldTone;
};

export type WorldPressureSummaryPreview = {
  id: string;
  label: string;
  value?: string;
  detail?: string;
  tone?: WorldTone;
  severity?: RunSetupSeverity;
  factionLabel?: string;
  nodeId?: string;
};

export type WorldHistoryEntryPreview = {
  id: string;
  label: string;
  detail?: string;
  tone?: WorldTone;
};

export type WorldMapNodePreviewData = {
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
  history?: WorldHistoryEntryPreview[];
  factionMix?: WorldFactionSharePreview[];
  pressureTags?: WorldPressureSummaryPreview[];
};

export type WorldMapLinkPreviewData = {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  kind?: "road" | "sea" | "mountain" | "pilgrim";
};

export type WorldMapPreviewData = {
  title?: string;
  summary?: string;
  nodes: WorldMapNodePreviewData[];
  links?: WorldMapLinkPreviewData[];
  activePressures?: WorldPressureSummaryPreview[];
  crisisChains?: WorldHistoryEntryPreview[];
  selectedNodeId?: string;
  winCondition?: {
    label: string;
    summary?: string;
    completed?: boolean;
  };
  nextUnlocks?: WorldHistoryEntryPreview[];
};

export type RunSetupOriginOptionData = {
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

export type RunSetupWorldPreviewData = {
  title?: string;
  summary: string;
  climate: WorldMetricPreview;
  divineMood: WorldMetricPreview;
  oracleDensity: WorldMetricPreview;
  factionMix: WorldFactionSharePreview[];
  history?: WorldHistoryEntryPreview[];
  pressures?: WorldPressureSummaryPreview[];
  map?: WorldMapPreviewData;
  note?: string;
};

export type GeneratedFactionSetup = {
  id: FactionId;
  profile: FactionProfile;
  agenda: FactionAgenda;
  tradeAccess: boolean;
  credibilityDelta: number;
  favourDelta: number;
  dependenceDelta: number;
  debtDelta: number;
  influence: number;
  note: string;
};

export type GeneratedRegionState = {
  id: string;
  label: string;
  summary: string;
  tags: ("sanctuary" | "city" | "port" | "pass")[];
  position: { x: number; y: number };
  connectedNodeIds: string[];
  controllingFactionId?: FactionId;
  climate: string;
  hegemon: string;
  philosophy: string;
  divineMood: string;
  oracleDensity: string;
  unrest: number;
  pressure: number;
  factionMix: WorldFactionSharePreview[];
  pressures: WorldPressureSummaryPreview[];
  history: WorldHistoryEntryPreview[];
};

export type WorldGenerationState = {
  seed: number;
  seedText: string;
  originId: OriginId;
  originTitle: string;
  cityStateCount: number;
  politicalClimate: string;
  economicClimate: string;
  economic: WorldMetricPreview;
  climate: WorldMetricPreview;
  divineMood: WorldMetricPreview;
  oracleDensity: WorldMetricPreview;
  factionMix: WorldFactionSharePreview[];
  pressures: WorldPressureSummaryPreview[];
  history: WorldHistoryEntryPreview[];
  challengeTags: string[];
  note: string;
  summary: string;
  scoreModifier: number;
  advisorIntro: string;
  uniqueMechanic: string;
  disabledSystems: string[];
  factions: GeneratedFactionSetup[];
  regions: GeneratedRegionState[];
};

export type NewRunOptions = {
  seed?: number | string;
  originId?: OriginId;
};

const PROFILE_ORDER: FactionProfile[] = ["martial", "mercantile", "devout", "scheming"];
const AGENDA_ORDER: FactionAgenda[] = ["war", "trade", "faith", "succession"];

const CLIMATE_BANDS: MetricBand[] = [
  { label: "Storm-lashed", detail: "Roads are brittle and the countryside keeps breaking rhythm.", meter: 18 },
  { label: "Lean", detail: "Harvests survive, but every caravan feels important.", meter: 34 },
  { label: "Temperate", detail: "Travel and offerings flow without dramatic seasonal shocks.", meter: 53 },
  { label: "Abundant", detail: "Pilgrim roads stay open and surplus moves more easily.", meter: 71 },
  { label: "Golden", detail: "The wider Greek world feels unusually generous and mobile.", meter: 88 }
];

const DIVINE_MOOD_BANDS: MetricBand[] = [
  { label: "Wrathful", detail: "The god answers sharply and cities grow superstitious.", meter: 16 },
  { label: "Austere", detail: "Rites matter more than charisma in this world.", meter: 32 },
  { label: "Watchful", detail: "Apollo listens, but he is not eager to forgive drift.", meter: 50 },
  { label: "Favorable", detail: "Visions travel farther and envoys arrive hopeful.", meter: 72 },
  { label: "Radiant", detail: "Delphi sits inside a visibly blessed religious moment.", meter: 90 }
];

const ORACLE_DENSITY_BANDS: MetricBand[] = [
  { label: "Isolated", detail: "Delphi speaks into a sparse sacred network and stands alone.", meter: 20 },
  { label: "Sparse", detail: "Rival shrines exist, but they are weak or distant.", meter: 38 },
  { label: "Layered", detail: "The Greek world contains several competing sacred voices.", meter: 56 },
  { label: "Crowded", detail: "Pilgrims and patrons compare sanctuaries constantly.", meter: 74 },
  { label: "Myth-thick", detail: "The world is dense with omens, rivals, and sacred rumor.", meter: 90 }
];

const ECONOMIC_BANDS: MetricBand[] = [
  { label: "Austere", detail: "Offerings are cautious and trade routes pinch often.", meter: 24 },
  { label: "Frugal", detail: "Wealth exists, but it is politically conditional.", meter: 40 },
  { label: "Balanced", detail: "Markets and sanctuaries are both viable sources of leverage.", meter: 56 },
  { label: "Prosperous", detail: "Caravans move quickly and patrons spend earlier.", meter: 74 },
  { label: "Booming", detail: "Ports, markets, and donor houses are eager to speculate.", meter: 90 }
];

const PHILOSOPHY_LABELS = [
  "Heroic cult",
  "Civic piety",
  "Mercantile pragmatism",
  "Mystic reform",
  "Ancestor law",
  "Skeptical courts",
  "Military austerity",
  "Festival populism"
] as const;

export const DEFAULT_WORLD_REGION_LAYOUTS: ReadonlyArray<WorldRegionLayout> = [
  {
    id: "delphi",
    label: "Delphi",
    position: { x: 42, y: 28 },
    tags: ["sanctuary", "city"],
    connectedNodeIds: ["corinth", "thebes"]
  },
  {
    id: "athens",
    label: "Athens",
    position: { x: 63, y: 45 },
    tags: ["city", "port"],
    defaultFactionId: "athens",
    connectedNodeIds: ["corinth", "miletus"]
  },
  {
    id: "sparta",
    label: "Sparta",
    position: { x: 28, y: 58 },
    tags: ["city", "pass"],
    defaultFactionId: "sparta",
    connectedNodeIds: ["argos", "corinth"]
  },
  {
    id: "corinth",
    label: "Corinth",
    position: { x: 51, y: 48 },
    tags: ["city", "port"],
    defaultFactionId: "corinth",
    connectedNodeIds: ["delphi", "athens", "sparta", "argos"]
  },
  {
    id: "thebes",
    label: "Thebes",
    position: { x: 53, y: 34 },
    tags: ["city"],
    defaultFactionId: "thebes",
    connectedNodeIds: ["delphi", "athens"]
  }
];

export const DEFAULT_WORLD_LINKS: ReadonlyArray<WorldLinkLayout> = [
  { id: "delphi-corinth", fromNodeId: "delphi", toNodeId: "corinth", kind: "road" },
  { id: "delphi-thebes", fromNodeId: "delphi", toNodeId: "thebes", kind: "road" },
  { id: "corinth-athens", fromNodeId: "corinth", toNodeId: "athens", kind: "sea" },
  { id: "corinth-sparta", fromNodeId: "corinth", toNodeId: "sparta", kind: "road" },
  { id: "athens-thebes", fromNodeId: "athens", toNodeId: "thebes", kind: "road" }
];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hashUint(value: number): number {
  let hash = value | 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x21f0aaad);
  hash = Math.imul(hash ^ (hash >>> 15), 0x735a2d97);
  return (hash ^ (hash >>> 15)) >>> 0;
}

function mixSeed(seed: number, salt: number): number {
  return hashUint(seed ^ Math.imul(salt, 0x9e3779b1));
}

function unit(seed: number, salt: number): number {
  return mixSeed(seed, salt) / 0xffffffff;
}

function integer(seed: number, salt: number, min: number, max: number): number {
  return min + Math.floor(unit(seed, salt) * (max - min + 1));
}

function hashSeedText(value: string): number {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

function titleCase(value: string): string {
  return value
    .split("-")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function toneFromMeter(meter: number): WorldTone {
  if (meter <= 22) {
    return "critical";
  }
  if (meter <= 42) {
    return "watchful";
  }
  if (meter >= 74) {
    return "rising";
  }
  return "steady";
}

function severityFromMeter(meter: number): RunSetupSeverity {
  if (meter >= 80) {
    return "critical";
  }
  if (meter >= 60) {
    return "rising";
  }
  return "watchful";
}

function metricFromBand(label: string, band: MetricBand): WorldMetricPreview {
  return {
    label,
    value: band.label,
    detail: band.detail,
    meter: band.meter,
    tone: toneFromMeter(band.meter)
  };
}

function normalizeSeedText(input?: number | string): string {
  if (typeof input === "number" && Number.isFinite(input)) {
    return String(Math.abs(Math.trunc(input)));
  }
  if (typeof input === "string" && input.trim().length > 0) {
    return input.trim();
  }
  return "7";
}

export function normalizeSeedInput(input?: number | string): { seed: number; seedText: string } {
  const seedText = normalizeSeedText(input);
  const asNumber = Number(seedText);

  if (Number.isFinite(asNumber) && seedText === String(asNumber)) {
    return {
      seed: Math.abs(Math.trunc(asNumber)),
      seedText
    };
  }

  return {
    seed: hashSeedText(seedText),
    seedText
  };
}

export function randomSeedText(): string {
  const value = Math.floor(Math.random() * 0xffffffff).toString(36).toUpperCase();
  return `ORACLE-${value.padStart(6, "0").slice(0, 6)}`;
}

export function normalizeOriginId(originId?: string): OriginId {
  if (originId && originId in originDefById) {
    return originId as OriginId;
  }
  return DEFAULT_ORIGIN_ID;
}

function originFor(originId?: string): OriginDef {
  return originDefById[normalizeOriginId(originId)];
}

function bandIndex(seed: number, salt: number, bias: number, length: number): number {
  const base = Math.floor(unit(seed, salt) * length);
  return clamp(base + Math.round(bias), 0, length - 1);
}

function profileWeightMap(seed: number, origin: OriginDef) {
  const war = origin.worldBias?.war ?? 0;
  const trade = origin.worldBias?.trade ?? 0;
  const faith = origin.worldBias?.faith ?? 0;
  const intrigue = origin.worldBias?.intrigue ?? 0;

  return {
    martial: 28 + integer(seed, 31, 0, 18) + war * 10 + (origin.worldBias?.unrest ?? 0) * 4,
    mercantile: 28 + integer(seed, 41, 0, 18) + trade * 10 + (origin.worldBias?.economy ?? 0) * 6,
    devout: 28 + integer(seed, 53, 0, 18) + faith * 10 + (origin.worldBias?.divineMood ?? 0) * 7,
    scheming: 28 + integer(seed, 67, 0, 18) + intrigue * 10 + (origin.worldBias?.pressure ?? 0) * 5
  } satisfies Record<FactionProfile, number>;
}

function factionMixFromWeights(weights: Record<FactionProfile, number>): WorldFactionSharePreview[] {
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;

  return PROFILE_ORDER.map((profile) => {
    const percent = Math.round((weights[profile] / total) * 100);
    const detail =
      profile === "martial"
        ? "War leagues and garrison politics"
        : profile === "mercantile"
          ? "Ports, caravans, and donor houses"
          : profile === "devout"
            ? "Temple blocs and ritual authority"
            : "Court factions, spies, and succession games";

    return {
      id: profile,
      label: titleCase(profile),
      value: `${percent}%`,
      detail,
      tone: percent >= 34 ? "rising" : percent <= 18 ? "watchful" : "steady"
    };
  });
}

function weightedProfile(seed: number, salt: number, baseProfile: FactionProfile, weights: Record<FactionProfile, number>): FactionProfile {
  let winner = baseProfile;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const profile of PROFILE_ORDER) {
    const baseBonus = profile === baseProfile ? 12 : 0;
    const score = weights[profile] + baseBonus + unit(seed, salt + PROFILE_ORDER.indexOf(profile) * 19) * 24;
    if (score > bestScore) {
      bestScore = score;
      winner = profile;
    }
  }

  return winner;
}

function agendaForProfile(profile: FactionProfile, seed: number, salt: number): FactionAgenda {
  const preferred: FactionAgenda =
    profile === "martial"
      ? "war"
      : profile === "mercantile"
        ? "trade"
        : profile === "devout"
          ? "faith"
          : "succession";

  if (unit(seed, salt) >= 0.74) {
    return AGENDA_ORDER[integer(seed, salt + 3, 0, AGENDA_ORDER.length - 1)] ?? preferred;
  }

  return preferred;
}

function buildFactionSetups(
  seed: number,
  origin: OriginDef,
  weights: Record<FactionProfile, number>,
  economyBand: MetricBand,
  divineBand: MetricBand
): GeneratedFactionSetup[] {
  return factionDefs.map((faction, index) => {
    const profile = weightedProfile(seed, 101 + index * 17, faction.profile, weights);
    const agenda = agendaForProfile(profile, seed, 151 + index * 17);
    const tradeRoll = unit(seed, 211 + index * 13);
    const tradeBias = (origin.worldBias?.trade ?? 0) * 0.1 + (economyBand.meter - 50) / 180;
    const tradeAccess = tradeRoll + tradeBias >= 0.34;
    const influence = clamp(
      Math.round(42 + unit(seed, 251 + index * 11) * 38 + (weights[profile] - 28) * 0.35 + (origin.worldBias?.pressure ?? 0) * 3),
      18,
      94
    );
    const credibilityDelta = Math.round((divineBand.meter - 50) / 10 + (profile === "devout" ? 2 : 0) - (origin.worldBias?.pressure ?? 0));
    const favourDelta = Math.round((economyBand.meter - 50) / 9 + (profile === "mercantile" ? 3 : 0));
    const dependenceDelta = Math.round((origin.worldBias?.pressure ?? 0) * 2 + (agenda === "war" ? 3 : agenda === "faith" ? 2 : 0));
    const debtDelta = Math.round((50 - economyBand.meter) / 9 + (agenda === "war" ? 2 : 0) - (tradeAccess ? 2 : 0));

    return {
      id: faction.id,
      profile,
      agenda,
      tradeAccess,
      credibilityDelta,
      favourDelta,
      dependenceDelta,
      debtDelta,
      influence,
      note: `${titleCase(profile)} posture with ${agenda} priorities.`
    };
  });
}

function dominantProfile(weights: Record<FactionProfile, number>): FactionProfile {
  return PROFILE_ORDER.reduce((winner, profile) => (
    weights[profile] > weights[winner] ? profile : winner
  ), PROFILE_ORDER[0]);
}

function buildPoliticalClimate(origin: OriginDef, weights: Record<FactionProfile, number>, divineBand: MetricBand, economyBand: MetricBand): string {
  const dominant = dominantProfile(weights);
  const pressureBias = origin.worldBias?.pressure ?? 0;

  if (dominant === "martial") {
    return pressureBias >= 1 ? "League wars and coercive diplomacy dominate the season." : "Generals and border leagues set the political tone.";
  }
  if (dominant === "mercantile") {
    return economyBand.meter >= 70
      ? "Ports, caravan houses, and donor networks are setting policy."
      : "Trade blocs bargain nervously as cities race for leverage.";
  }
  if (dominant === "devout") {
    return divineBand.meter >= 70
      ? "Temple politics and ritual legitimacy are shaping every alliance."
      : "Priestly councils are powerful, but they are policing each other closely.";
  }

  return "Courts, succession intrigues, and private patronage carry more weight than public vows.";
}

function buildGlobalHistory(seed: number, origin: OriginDef, regions: GeneratedRegionState[]): WorldHistoryEntryPreview[] {
  const hottest = [...regions].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))[0];
  const calmest = [...regions].sort((left, right) => left.pressure - right.pressure || left.id.localeCompare(right.id))[0];

  return [
    {
      id: "founding-memory",
      label: `${origin.title ?? origin.label} takes root`,
      detail: origin.uniqueMechanic,
      tone: "steady"
    },
    {
      id: "regional-flashpoint",
      label: `${hottest?.label ?? "The region"} begins as the first flashpoint`,
      detail: hottest?.summary,
      tone: hottest && hottest.pressure >= 72 ? "critical" : "watchful"
    },
    {
      id: "quiet-power",
      label: `${calmest?.label ?? "A quieter city"} hides the most patient influence`,
      detail: calmest?.philosophy,
      tone: "steady"
    },
    {
      id: "seed-marker",
      label: `Seed ${normalizeSeedText(seed)} remembers this opening shape`,
      detail: "The same origin and seed will recreate this world profile exactly.",
      tone: "watchful"
    }
  ];
}

function buildRegionPressures(
  layout: WorldRegionLayout,
  seed: number,
  origin: OriginDef,
  pressureBase: number,
  controllingFactionLabel?: string
): WorldPressureSummaryPreview[] {
  const portBias = layout.tags.includes("port") ? 8 : 0;
  const sanctuaryBias = layout.tags.includes("sanctuary") ? 6 : 0;
  const unrestBias = (origin.worldBias?.unrest ?? 0) * 7;
  const tradeBias = (origin.worldBias?.trade ?? 0) * 6;
  const warBias = (origin.worldBias?.war ?? 0) * 7;
  const doctrineBias = (origin.worldBias?.intrigue ?? 0) * 6;

  const pressures: WorldPressureSummaryPreview[] = [
    {
      id: `${layout.id}-pressure-pilgrim`,
      label: "Pilgrim traffic",
      value: `${clamp(Math.round(32 + pressureBase * 0.35 + sanctuaryBias - warBias), 10, 98)}`,
      detail: "shifting demand",
      tone: "watchful" as const,
      severity: severityFromMeter(clamp(Math.round(32 + pressureBase * 0.35 + sanctuaryBias - warBias), 0, 100)),
      nodeId: layout.id
    },
    {
      id: `${layout.id}-pressure-trade`,
      label: "Trade route strain",
      value: `${clamp(Math.round(28 + pressureBase * 0.4 + portBias - tradeBias), 10, 98)}`,
      detail: controllingFactionLabel ? `${controllingFactionLabel} route leverage` : "regional markets",
      tone: "rising" as const,
      severity: severityFromMeter(clamp(Math.round(28 + pressureBase * 0.4 + portBias - tradeBias), 0, 100)),
      nodeId: layout.id
    },
    {
      id: `${layout.id}-pressure-doctrine`,
      label: "Doctrinal agitation",
      value: `${clamp(Math.round(24 + pressureBase * 0.3 + doctrineBias + unrestBias), 10, 98)}`,
      detail: layout.tags.includes("city") ? "schools and councils" : "shrines and rumor",
      tone: "critical" as const,
      severity: severityFromMeter(clamp(Math.round(24 + pressureBase * 0.3 + doctrineBias + unrestBias), 0, 100)),
      nodeId: layout.id
    }
  ];

  return pressures.sort((left, right) => Number(right.value) - Number(left.value));
}

function buildRegionHistory(layout: WorldRegionLayout, seed: number, philosophy: string, pressure: number): WorldHistoryEntryPreview[] {
  return [
    {
      id: `${layout.id}-history-${seed}-1`,
      label: `${layout.label} leans into ${philosophy.toLowerCase()}`,
      detail: "This local culture will shape how envoys interpret Delphi.",
      tone: "steady"
    },
    {
      id: `${layout.id}-history-${seed}-2`,
      label: pressure >= 70 ? "A political rupture is already brewing" : "Local elites are still balancing against each other",
      detail: layout.tags.includes("port") ? "Ports and caravan houses are amplifying rumors quickly." : "Internal factions are still testing one another.",
      tone: pressure >= 70 ? "critical" : "watchful"
    }
  ];
}

function buildRegions(
  seed: number,
  origin: OriginDef,
  climateBand: MetricBand,
  divineBand: MetricBand,
  densityBand: MetricBand,
  factionSetups: GeneratedFactionSetup[],
  factionMix: WorldFactionSharePreview[]
): GeneratedRegionState[] {
  const rankedFactions = [...factionSetups].sort((left, right) => right.influence - left.influence || left.id.localeCompare(right.id));

  return DEFAULT_WORLD_REGION_LAYOUTS.map((layout, index) => {
    const preferredFaction = layout.defaultFactionId
      ? factionSetups.find((faction) => faction.id === layout.defaultFactionId)
      : rankedFactions[0];
    const fallbackFaction = rankedFactions[index % rankedFactions.length];
    const controller = preferredFaction ?? fallbackFaction;
    const pressure = clamp(
      Math.round(22 + unit(seed, 301 + index * 29) * 52 + (origin.worldBias?.pressure ?? 0) * 8 + (origin.worldBias?.war ?? 0) * 4),
      8,
      96
    );
    const unrest = clamp(
      Math.round(18 + unit(seed, 341 + index * 31) * 44 + (origin.worldBias?.unrest ?? 0) * 9 + (controller?.agenda === "war" ? 8 : 0)),
      6,
      98
    );
    const philosophy = PHILOSOPHY_LABELS[integer(seed, 401 + index * 17, 0, PHILOSOPHY_LABELS.length - 1)] ?? PHILOSOPHY_LABELS[0];
    const hegemon = controller ? `${titleCase(controller.profile)} ${titleCase(controller.agenda)} bloc` : "Fragmented councils";
    const regionalMix = factionMix
      .map((entry, mixIndex) => ({
        ...entry,
        value: `${clamp(Number((entry.value ?? "0").replace("%", "")) + integer(seed, 451 + index * 13 + mixIndex * 7, -6, 6), 8, 62)}%`
      }))
      .sort((left, right) => Number((right.value ?? "0").replace("%", "")) - Number((left.value ?? "0").replace("%", "")))
      .slice(0, 3);
    const pressures = buildRegionPressures(layout, seed + index * 101, origin, pressure, controller ? titleCase(controller.id) : undefined);

    return {
      id: layout.id,
      label: layout.label,
      summary: `${layout.label} opens under ${hegemon.toLowerCase()} with pressure ${pressure} and unrest ${unrest}.`,
      tags: layout.tags,
      position: layout.position,
      connectedNodeIds: layout.connectedNodeIds,
      controllingFactionId: controller?.id,
      climate: climateBand.label,
      hegemon,
      philosophy,
      divineMood: divineBand.label,
      oracleDensity: densityBand.label,
      unrest,
      pressure,
      factionMix: regionalMix,
      pressures,
      history: buildRegionHistory(layout, seed, philosophy, pressure)
    };
  });
}

function buildGlobalPressures(
  seed: number,
  origin: OriginDef,
  economicBand: MetricBand,
  divineBand: MetricBand,
  densityBand: MetricBand,
  regions: GeneratedRegionState[]
): WorldPressureSummaryPreview[] {
  const hottest = [...regions].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id)).slice(0, 3);
  const global: WorldPressureSummaryPreview[] = [
    {
      id: "pressure-economic",
      label: "Market weather",
      value: economicBand.label,
      detail: economicBand.detail,
      tone: toneFromMeter(economicBand.meter),
      severity: severityFromMeter(Math.abs(economicBand.meter - 50) + 36)
    },
    {
      id: "pressure-divine",
      label: "Divine weather",
      value: divineBand.label,
      detail: origin.advisorIntro,
      tone: toneFromMeter(divineBand.meter),
      severity: severityFromMeter(Math.abs(divineBand.meter - 50) + 34)
    },
    {
      id: "pressure-density",
      label: "Rival oracle field",
      value: densityBand.label,
      detail: densityBand.detail,
      tone: toneFromMeter(densityBand.meter),
      severity: severityFromMeter(Math.abs(densityBand.meter - 50) + 34)
    },
    ...hottest.map((region) => ({
      id: `pressure-region-${region.id}`,
      label: region.label,
      value: `${region.pressure}`,
      detail: region.summary,
      tone: (region.pressure >= 72 ? "critical" : region.pressure >= 58 ? "rising" : "watchful") as WorldTone,
      severity: (region.pressure >= 72 ? "critical" : region.pressure >= 58 ? "rising" : "watchful") as RunSetupSeverity,
      nodeId: region.id,
      factionLabel: region.controllingFactionId
    }))
  ];

  return global.slice(0, 5 + integer(seed, 499, 0, 1));
}

export function buildWorldGeneration(options?: NewRunOptions): WorldGenerationState {
  const { seed, seedText } = normalizeSeedInput(options?.seed);
  const origin = originFor(options?.originId);
  const climateBand = CLIMATE_BANDS[bandIndex(seed, 11, origin.worldBias?.climate ?? 0, CLIMATE_BANDS.length)] ?? CLIMATE_BANDS[2];
  const economicBand = ECONOMIC_BANDS[bandIndex(seed, 23, origin.worldBias?.economy ?? 0, ECONOMIC_BANDS.length)] ?? ECONOMIC_BANDS[2];
  const divineBand = DIVINE_MOOD_BANDS[bandIndex(seed, 37, origin.worldBias?.divineMood ?? 0, DIVINE_MOOD_BANDS.length)] ?? DIVINE_MOOD_BANDS[2];
  const densityBand = ORACLE_DENSITY_BANDS[bandIndex(seed, 43, origin.worldBias?.oracleDensity ?? 0, ORACLE_DENSITY_BANDS.length)] ?? ORACLE_DENSITY_BANDS[2];
  const weights = profileWeightMap(seed, origin);
  const factionMix = factionMixFromWeights(weights);
  const factions = buildFactionSetups(seed, origin, weights, economicBand, divineBand);
  const regions = buildRegions(seed, origin, climateBand, divineBand, densityBand, factions, factionMix);
  const history = buildGlobalHistory(seed, origin, regions);
  const pressures = buildGlobalPressures(seed, origin, economicBand, divineBand, densityBand, regions);
  const cityStateCount = integer(seed, 59, 7, 14) + Math.max(0, origin.worldBias?.trade ?? 0);
  const politicalClimate = buildPoliticalClimate(origin, weights, divineBand, economicBand);
  const summary = `${origin.title ?? origin.label} opens in a ${economicBand.label.toLowerCase()} economy with ${divineBand.label.toLowerCase()} divine weather, ${densityBand.label.toLowerCase()} rival oracles, and ${politicalClimate.toLowerCase()}`;
  const scoreModifier = origin.scoreModifier ?? 0;
  const note = `Seed ${seedText} · ${cityStateCount} notable city-states in play · score modifier ${scoreModifier >= 0 ? "+" : ""}${scoreModifier}.`;

  return {
    seed,
    seedText,
    originId: origin.id,
    originTitle: origin.title ?? origin.label,
    cityStateCount,
    politicalClimate,
    economicClimate: economicBand.label,
    economic: metricFromBand("Economic Climate", economicBand),
    climate: metricFromBand("Climate", climateBand),
    divineMood: metricFromBand("Divine Mood", divineBand),
    oracleDensity: metricFromBand("Oracle Density", densityBand),
    factionMix,
    pressures,
    history,
    challengeTags: [...(origin.challengeTags ?? [])],
    note,
    summary,
    scoreModifier,
    advisorIntro: origin.advisorIntro,
    uniqueMechanic: origin.uniqueMechanic,
    disabledSystems: [...(origin.disabledSystems ?? [])],
    factions,
    regions
  };
}

export function normalizeWorldGenerationState(value?: Partial<WorldGenerationState>): WorldGenerationState {
  return buildWorldGeneration({
    seed: value?.seedText ?? value?.seed ?? 7,
    originId: value?.originId ?? DEFAULT_ORIGIN_ID
  });
}

export function buildRunSetupPreview(options?: NewRunOptions): RunSetupWorldPreviewData {
  const world = buildWorldGeneration(options);
  const selectedRegion = [...world.regions].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))[0] ?? world.regions[0];

  return {
    title: `${world.originTitle} · Seed ${world.seedText}`,
    summary: world.summary,
    climate: world.climate,
    divineMood: world.divineMood,
    oracleDensity: world.oracleDensity,
    factionMix: world.factionMix,
    history: world.history,
    pressures: world.pressures,
    map: {
      title: "Generated Greek World",
      summary: `${world.cityStateCount} city-states · ${world.politicalClimate}`,
      nodes: world.regions.map((region) => ({
        id: region.id,
        label: region.label,
        position: region.position,
        summary: region.summary,
        pressure: String(region.pressure),
        unrest: String(region.unrest),
        controllingFactionLabel: region.controllingFactionId ? titleCase(region.controllingFactionId) : "No stable hegemon",
        climate: region.climate,
        hegemon: region.hegemon,
        philosophy: region.philosophy,
        divineMood: region.divineMood,
        oracleDensity: region.oracleDensity,
        connectedNodeIds: region.connectedNodeIds,
        history: region.history,
        factionMix: region.factionMix,
        pressureTags: region.pressures
      })),
      links: [...DEFAULT_WORLD_LINKS],
      activePressures: world.pressures,
      crisisChains: world.history,
      selectedNodeId: selectedRegion?.id,
      winCondition: {
        label: `${world.originTitle} opening`,
        summary: world.uniqueMechanic,
        completed: false
      },
      nextUnlocks: world.disabledSystems.map((system, index) => ({
        id: `system-${index}`,
        label: system.replace(/-/g, " "),
        detail: "Starts constrained under this origin.",
        tone: "watchful"
      }))
    },
    note: world.note
  };
}

export function buildRunSetupOriginOptions(seed?: number | string): RunSetupOriginOptionData[] {
  return originDefs.map((origin) => {
    const preview = buildRunSetupPreview({
      seed,
      originId: origin.id
    });

    return {
      id: origin.id,
      label: origin.label,
      title: origin.title,
      subtitle: origin.subtitle,
      summary: origin.summary,
      climate: preview.climate.value,
      divineMood: preview.divineMood.value,
      oracleDensity: preview.oracleDensity.value,
      factionMix: preview.factionMix[0]?.label ?? "Balanced",
      tags: [...(origin.challengeTags ?? [])]
    };
  });
}
