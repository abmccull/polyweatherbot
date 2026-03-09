import { namedCharacterDefs, type FactionId, type NamedCharacterDef } from "@the-oracle/content";

import type { CharactersState, FactionState, GameState, NamedCharacterState } from "./gameState";

function hash(seed: number): number {
  let value = seed + 0x6d2b79f5;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function averageRelation(faction: FactionState): number {
  const relations = Object.values(faction.relations);
  if (relations.length === 0) {
    return 0;
  }

  return relations.reduce((sum, value) => sum + value, 0) / relations.length;
}

function factionInfluence(faction: FactionState): number {
  return (
    faction.credibility * 0.34
    + faction.favour * 0.28
    + averageRelation(faction) * 0.16
    + faction.treaties.length * 5
    - faction.embargoes.length * 4
    - faction.activeConflicts.length * 3
    + (faction.tradeAccess ? 6 : -4)
    - faction.debt * 0.18
    - faction.dependence * 0.1
  );
}

function regionForFaction(
  state: Pick<GameState, "worldGeneration">,
  factionId?: FactionId,
  slot = 0
): GameState["worldGeneration"]["regions"][number] | undefined {
  const controlled = factionId
    ? state.worldGeneration.regions
      .filter((region) => region.controllingFactionId === factionId)
      .sort((left, right) => right.pressure - left.pressure || right.unrest - left.unrest || left.id.localeCompare(right.id))
    : [];
  const fallback = [...state.worldGeneration.regions]
    .sort((left, right) => right.pressure - left.pressure || right.unrest - left.unrest || left.id.localeCompare(right.id));

  return controlled[slot] ?? controlled[0] ?? fallback[slot] ?? fallback[0];
}

function delphiRegion(state: Pick<GameState, "worldGeneration">) {
  return state.worldGeneration.regions.find((region) => region.id === "delphi") ?? regionForFaction(state);
}

function desiredInitialCount(
  state: Pick<GameState, "worldGeneration" | "factions">,
  definition: NamedCharacterDef
): number {
  const tradeFactions = Object.values(state.factions).filter((faction) => faction.tradeAccess).length;
  const warFactions = Object.values(state.factions).filter((faction) => faction.currentAgenda === "war").length;
  const highTensionFactions = Object.values(state.factions).filter((faction) => faction.activeConflicts.length > 0 || faction.embargoes.length > 0).length;
  const economicMeter = state.worldGeneration.economic.meter ?? 50;
  const divineMeter = state.worldGeneration.divineMood.meter ?? 50;
  const densityMeter = state.worldGeneration.oracleDensity.meter ?? 50;

  switch (definition.role) {
    case "merchant":
      return clamp(definition.minInitialCount + (economicMeter >= 68 || tradeFactions >= 5 ? 1 : 0), definition.minInitialCount, definition.maxInitialCount);
    case "envoy":
      return clamp(definition.minInitialCount + (highTensionFactions >= 4 || state.worldGeneration.cityStateCount >= 7 ? 1 : 0), definition.minInitialCount, definition.maxInitialCount);
    case "priest":
      return clamp(definition.minInitialCount + (divineMeter >= 70 || densityMeter >= 72 ? 1 : 0), definition.minInitialCount, definition.maxInitialCount);
    case "general":
      return clamp(definition.minInitialCount + (warFactions >= 4 ? 1 : 0), definition.minInitialCount, definition.maxInitialCount);
    case "legendary":
      return 0;
    default:
      return definition.minInitialCount;
  }
}

function roleAffinity(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">,
  definition: NamedCharacterDef,
  faction: FactionState,
  slot: number
): number {
  const region = regionForFaction(state, faction.id, slot);
  let score = factionInfluence(faction);

  if (definition.preferredAgendas?.includes(faction.currentAgenda)) {
    score += 16;
  }
  if (definition.preferredProfiles?.includes(faction.profile)) {
    score += 14;
  }
  if (definition.role === "merchant") {
    score += faction.tradeAccess ? 12 : -10;
    score += faction.currentAgenda === "trade" ? 10 : 0;
    score -= faction.embargoes.length * 3;
  }
  if (definition.role === "general") {
    score += faction.activeConflicts.length * 11;
    score += faction.currentAgenda === "war" ? 12 : 0;
  }
  if (definition.role === "envoy") {
    score += faction.treaties.length * 7 + faction.embargoes.length * 5;
    score += faction.credibility * 0.14;
  }
  if (definition.role === "philosopher") {
    score += (region?.pressure ?? 0) * 0.45 + (region?.unrest ?? 0) * 0.3;
    score += Math.max(0, 55 - faction.credibility) * 0.28;
    score += Math.max(0, faction.dependence - 20) * 0.16;
  }
  if (definition.role === "priest") {
    score += faction.currentAgenda === "faith" ? 12 : 0;
    score += state.worldGeneration.divineMood.meter ? (state.worldGeneration.divineMood.meter - 50) * 0.24 : 0;
    score += state.worldGeneration.oracleDensity.meter ? (state.worldGeneration.oracleDensity.meter - 50) * 0.16 : 0;
  }

  score += hash(state.worldSeed + slot * 37 + faction.name.length * 11 + definition.id.length * 17) * 6;
  return score;
}

function selectFactionIds(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">,
  definition: NamedCharacterDef,
  count: number
): FactionId[] {
  const ranked = Object.values(state.factions)
    .map((faction) => ({
      factionId: faction.id,
      score: roleAffinity(state, definition, faction, 0)
    }))
    .sort((left, right) => right.score - left.score || left.factionId.localeCompare(right.factionId))
    .map((entry) => entry.factionId);

  if (ranked.length === 0 || count === 0) {
    return [];
  }

  return Array.from({ length: count }, (_, index) => ranked[index % ranked.length]!);
}

function pickFromPool(pool: readonly string[], seed: number, salt: number): string {
  return pool[Math.floor(hash(seed + salt) * pool.length) % pool.length] ?? pool[0] ?? "";
}

function buildDisplayName(title: string, name: string, epithet: string): string {
  return [title, name, epithet].filter(Boolean).join(" ");
}

function characterImpression(
  definition: NamedCharacterDef,
  faction: FactionState | undefined,
  regionLabel: string | undefined,
  state: Pick<GameState, "worldGeneration">
): string {
  switch (definition.role) {
    case "merchant":
      return `${faction?.name ?? "A trading house"} watches the ${regionLabel ?? "regional"} routes for Delphi's next season of patrons.`;
    case "general":
      return `${faction?.name ?? "A campaigning faction"} measures troop movement around ${regionLabel ?? "the passes"}.`;
    case "envoy":
      return `${faction?.name ?? "A foreign court"} keeps a seal at hand for the next petition to Apollo.`;
    case "philosopher":
      return `A sharp-tongued school gathers around ${regionLabel ?? "the busiest court"}, testing Delphi's authority.`;
    case "priest":
      return `Temple rites stay calibrated to a ${state.worldGeneration.divineMood.value.toLowerCase()} divine mood.`;
    default:
      return `Rumor claims a figure of consequence may one day cross Delphi's threshold.`;
  }
}

function createCharacterState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">,
  definition: NamedCharacterDef,
  slot: number,
  factionId?: FactionId
): NamedCharacterState {
  const faction = factionId ? state.factions[factionId] : undefined;
  const region = definition.role === "priest"
    ? delphiRegion(state)
    : definition.role === "philosopher"
      ? regionForFaction(state, factionId, slot)
      : regionForFaction(state, factionId);
  const seedBasis = state.worldSeed
    + slot * 101
    + definition.id.length * 53
    + (faction?.name.length ?? 0) * 19
    + (region?.label.length ?? 0) * 23;
  const title = pickFromPool(definition.titlePool, seedBasis, 7);
  const name = pickFromPool(definition.firstNamePool, seedBasis, 17);
  const epithet = pickFromPool(definition.epithetPool, seedBasis, 29);
  const influence = clamp(
    Math.round(
      definition.baseInfluence
      + (faction ? roleAffinity(state, definition, faction, slot) * 0.12 : 0)
      + (region?.pressure ?? 0) * (definition.role === "philosopher" ? 0.16 : 0.05)
      + (definition.role === "priest" ? (state.worldGeneration.divineMood.meter ?? 50) * 0.08 : 0)
      - 10
    ),
    18,
    96
  );
  const prominence = clamp(
    Math.round(
      definition.baseProminence
      + (faction ? factionInfluence(faction) * 0.1 : 0)
      + (region?.unrest ?? 0) * (definition.role === "general" ? 0.12 : 0.05)
      + hash(seedBasis + 43) * 8
      - 4
    ),
    18,
    98
  );
  const trust = clamp(
    Math.round(
      (definition.role === "priest" ? 22 : definition.role === "envoy" ? 15 : definition.role === "merchant" ? 12 : 8)
      + (faction?.favour ?? 50) * 0.12
      - (faction?.debt ?? 0) * 0.05
      + (region?.id === "delphi" ? 6 : 0)
    ),
    0,
    100
  );
  const fear = clamp(
    Math.round(
      (definition.role === "general" ? 22 : definition.role === "philosopher" ? 16 : definition.role === "legendary" ? 30 : 8)
      + (region?.pressure ?? 0) * 0.12
      + (faction?.activeConflicts.length ?? 0) * 4
    ),
    0,
    100
  );
  const hostility = clamp(
    Math.round(
      (definition.role === "general" ? (faction?.activeConflicts.length ?? 0) * 7 : 0)
      + (definition.role === "merchant" && faction && !faction.tradeAccess ? 10 : 0)
      + (definition.role === "philosopher" ? Math.max(0, 58 - (faction?.credibility ?? 58)) * 0.14 : 0)
      - trust * 0.18
      + (faction?.embargoes.length ?? 0) * 2
    ),
    0,
    100
  );
  const familiarity = clamp(
    Math.round(
      (definition.role === "priest" ? 24 : definition.role === "envoy" ? 18 : definition.role === "merchant" ? 16 : 10)
      + (region?.id === "delphi" ? 8 : 0)
      + hash(seedBasis + 59) * 6
    ),
    0,
    100
  );
  const impression = characterImpression(definition, faction, region?.label, state);

  return {
    id: `character-${definition.id}-${factionId ?? region?.id ?? "delphi"}-${slot + 1}`,
    defId: definition.id,
    role: definition.role,
    cadence: definition.cadence,
    name,
    title,
    epithet,
    displayName: buildDisplayName(title, name, epithet),
    homeFactionId: factionId,
    anchorRegionId: region?.id,
    influence,
    prominence,
    status: definition.role === "legendary" ? "legendary" : "active",
    tags: [...new Set([
      ...definition.initialTags,
      faction?.currentAgenda,
      faction?.profile,
      region?.id
    ].filter((value): value is string => Boolean(value)))],
    relationship: {
      trust,
      fear,
      hostility,
      familiarity
    },
    memory: {
      knownSinceDay: 1,
      visitCount: 0,
      successfulVisits: 0,
      rebuffedVisits: 0,
      lastLocationId: region?.id,
      lastImpression: impression,
      notableMoments: [impression]
    }
  };
}

function characterSpotlightScore(character: NamedCharacterState): number {
  return (
    character.prominence * 0.52
    + character.influence * 0.26
    + character.relationship.familiarity * 0.14
    + character.relationship.trust * 0.08
    - character.relationship.hostility * 0.12
  );
}

function sortRoster(left: NamedCharacterState, right: NamedCharacterState): number {
  return right.prominence - left.prominence
    || right.influence - left.influence
    || left.role.localeCompare(right.role)
    || left.id.localeCompare(right.id);
}

function selectSpotlightCharacterIds(roster: NamedCharacterState[]): string[] {
  return [...roster]
    .sort((left, right) => characterSpotlightScore(right) - characterSpotlightScore(left) || sortRoster(left, right))
    .slice(0, 3)
    .map((character) => character.id);
}

export function createInitialCharactersState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions">
): CharactersState {
  const roster = namedCharacterDefs.flatMap((definition) => {
    const count = desiredInitialCount(state, definition);
    const factionIds = selectFactionIds(state, definition, count);

    return Array.from({ length: count }, (_, slot) => createCharacterState(state, definition, slot, factionIds[slot]));
  }).sort(sortRoster);

  return {
    roster,
    spotlightCharacterIds: selectSpotlightCharacterIds(roster)
  };
}

export function normalizeCharactersState(
  state: Pick<GameState, "worldSeed" | "worldGeneration" | "factions" | "characters">
): CharactersState {
  const fallback = createInitialCharactersState(state);
  if (!state.characters) {
    return fallback;
  }

  const fallbackById = new Map(fallback.roster.map((character) => [character.id, character]));
  const roster = state.characters.roster.map((character) => {
    const matchedFallback = fallbackById.get(character.id)
      ?? fallback.roster.find((entry) => entry.defId === character.defId && entry.homeFactionId === character.homeFactionId)
      ?? fallback.roster[0]!;

    return {
      ...matchedFallback,
      ...character,
      displayName: character.displayName ?? buildDisplayName(character.title, character.name, character.epithet),
      tags: character.tags?.length ? [...new Set(character.tags)] : matchedFallback.tags,
      relationship: {
        ...matchedFallback.relationship,
        ...character.relationship
      },
      memory: {
        ...matchedFallback.memory,
        ...character.memory,
        notableMoments: character.memory?.notableMoments?.length
          ? character.memory.notableMoments.slice(0, 6)
          : matchedFallback.memory.notableMoments
      }
    };
  });

  const knownIds = new Set(roster.map((character) => character.id));
  for (const fallbackCharacter of fallback.roster) {
    if (!knownIds.has(fallbackCharacter.id)) {
      roster.push(fallbackCharacter);
    }
  }

  roster.sort(sortRoster);
  const spotlightCharacterIds = state.characters.spotlightCharacterIds?.filter((characterId) => roster.some((entry) => entry.id === characterId));

  return {
    roster,
    spotlightCharacterIds: spotlightCharacterIds?.length ? spotlightCharacterIds : selectSpotlightCharacterIds(roster)
  };
}
