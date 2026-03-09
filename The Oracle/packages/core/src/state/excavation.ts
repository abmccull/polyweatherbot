import type { Coord, DomainTag } from "./gameState";

// --- Relic effects ---

export type RelicEffectType =
  | "credibility_bonus"
  | "resource_production"
  | "consultation_quality"
  | "priest_skill"
  | "omen_strength";

export type RelicEffect = {
  type: RelicEffectType;
  value: number;
};

// --- Relics ---

export type RelicKind = "minor" | "major" | "legendary";

export type Relic = {
  id: string;
  name: string;
  kind: RelicKind;
  domain: DomainTag;
  effect: RelicEffect;
  discoveredDay: number;
};

// --- Excavation layers ---

export type ExcavationLayerContents =
  | "empty"
  | "pottery"
  | "relic"
  | "sacred_fragment"
  | "ancient_chamber";

export type ExcavationLayer = {
  depth: number;
  contents: ExcavationLayerContents;
  revealed: boolean;
  relicId?: string;
};

// --- Excavation sites ---

export type ExcavationSiteStatus =
  | "undiscovered"
  | "discovered"
  | "excavating"
  | "exhausted";

export type ExcavationSite = {
  id: string;
  tile: Coord;
  depth: number;
  maxDepth: number;
  status: ExcavationSiteStatus;
  layers: ExcavationLayer[];
  assignedPriestId?: string;
};

// --- Sacred sites ---

export type SacredSiteKind =
  | "oracle_spring"
  | "earth_fissure"
  | "ancient_altar"
  | "sacred_grove";

export type SacredSite = {
  id: string;
  tile: Coord;
  kind: SacredSiteKind;
  discovered: boolean;
  active: boolean;
  bonuses: RelicEffect[];
};

// --- Root state ---

export type ExcavationState = {
  sites: ExcavationSite[];
  relics: Relic[];
  sacredSites: SacredSite[];
};

// --- Deterministic seeded RNG helpers ---

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

const LAYER_CONTENTS_TABLE: ExcavationLayerContents[] = [
  "empty",
  "empty",
  "pottery",
  "pottery",
  "relic",
  "sacred_fragment",
  "ancient_chamber"
];

const RELIC_NAMES: Record<DomainTag, string[]> = {
  military: [
    "Bronze Spearhead of Ares",
    "Shield Fragment of Hector",
    "Helm of the Fallen Champion"
  ],
  economic: [
    "Golden Amphora of Hermes",
    "Merchant's Seal of Corinth",
    "Coin Hoard of Croesus"
  ],
  spiritual: [
    "Tripod Fragment of the Pythia",
    "Laurel Crown of Apollo",
    "Omphalos Stone Shard"
  ]
};

const SACRED_SITE_KINDS: SacredSiteKind[] = [
  "oracle_spring",
  "earth_fissure",
  "ancient_altar",
  "sacred_grove"
];

const DOMAINS: DomainTag[] = ["military", "economic", "spiritual"];

function generateLayers(rng: () => number, maxDepth: number, siteIndex: number): ExcavationLayer[] {
  const layers: ExcavationLayer[] = [];
  for (let depth = 1; depth <= maxDepth; depth += 1) {
    const roll = rng();
    // Deeper layers have rarer finds
    const index = Math.min(
      LAYER_CONTENTS_TABLE.length - 1,
      Math.floor(roll * (LAYER_CONTENTS_TABLE.length - 1) + depth * 0.3)
    );
    const contents = LAYER_CONTENTS_TABLE[Math.min(index, LAYER_CONTENTS_TABLE.length - 1)];
    layers.push({
      depth,
      contents,
      revealed: false,
      relicId: contents === "relic" || contents === "sacred_fragment"
        ? `relic-site${siteIndex}-d${depth}`
        : undefined
    });
  }
  return layers;
}

function generateRelicForLayer(
  rng: () => number,
  relicId: string,
  contents: ExcavationLayerContents,
  depth: number
): Relic {
  const domain = DOMAINS[Math.floor(rng() * DOMAINS.length)];
  const names = RELIC_NAMES[domain];
  const name = names[Math.floor(rng() * names.length)];
  const kind: RelicKind = contents === "sacred_fragment"
    ? "legendary"
    : depth >= 4
      ? "major"
      : "minor";
  const effectTypes: RelicEffectType[] = [
    "credibility_bonus",
    "resource_production",
    "consultation_quality",
    "priest_skill",
    "omen_strength"
  ];
  const effectType = effectTypes[Math.floor(rng() * effectTypes.length)];
  const baseValue = kind === "legendary" ? 15 : kind === "major" ? 8 : 4;
  const value = baseValue + Math.floor(rng() * 5);

  return {
    id: relicId,
    name,
    kind,
    domain,
    effect: { type: effectType, value },
    discoveredDay: 0 // Will be set when claimed
  };
}

export function createInitialExcavationState(worldSeed: number): ExcavationState {
  const rng = seededRandom(worldSeed ^ 0xdead);
  const siteCount = 2 + Math.floor(rng() * 3); // 2-4 sites
  const sites: ExcavationSite[] = [];
  const pregenRelics: Relic[] = [];

  for (let i = 0; i < siteCount; i += 1) {
    const maxDepth = 3 + Math.floor(rng() * 4); // 3-6 layers
    const tileX = 25 + Math.floor(rng() * 10);
    const tileY = 45 + Math.floor(rng() * 10);
    const layers = generateLayers(rng, maxDepth, i);

    // Pre-generate relics for layers that contain them
    for (const layer of layers) {
      if (layer.relicId && (layer.contents === "relic" || layer.contents === "sacred_fragment")) {
        pregenRelics.push(
          generateRelicForLayer(rng, layer.relicId, layer.contents, layer.depth)
        );
      }
    }

    const status: ExcavationSiteStatus = i === 0 ? "discovered" : "undiscovered";
    sites.push({
      id: `excavation-site-${i}`,
      tile: { x: tileX, y: tileY },
      depth: 0,
      maxDepth,
      status,
      layers
    });
  }

  // Generate 1-2 sacred sites
  const sacredSiteCount = 1 + Math.floor(rng() * 2);
  const sacredSites: SacredSite[] = [];
  for (let i = 0; i < sacredSiteCount; i += 1) {
    const kind = SACRED_SITE_KINDS[Math.floor(rng() * SACRED_SITE_KINDS.length)];
    const bonusType: RelicEffectType = kind === "oracle_spring"
      ? "consultation_quality"
      : kind === "earth_fissure"
        ? "omen_strength"
        : kind === "ancient_altar"
          ? "credibility_bonus"
          : "priest_skill";

    sacredSites.push({
      id: `sacred-site-${i}`,
      tile: {
        x: 26 + Math.floor(rng() * 8),
        y: 46 + Math.floor(rng() * 8)
      },
      kind,
      discovered: false,
      active: false,
      bonuses: [
        { type: bonusType, value: 10 + Math.floor(rng() * 8) }
      ]
    });
  }

  return {
    sites,
    relics: pregenRelics,
    sacredSites
  };
}
